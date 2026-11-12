"""End-to-end reproducible OutGuard-MED training, evaluation, and verification pipeline."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from outguard.config import PipelineConfig
from outguard.data.simulator import generate_benchmark_dataset
from outguard.evaluation.error_analysis import run_slice_error_analysis
from outguard.evaluation.metrics import evaluate_model_comprehensive
from outguard.evaluation.plots import generate_all_plots
from outguard.experimentation.ab_testing import run_online_ab_simulation
from outguard.features.extractor import (
    OutGuardFeatureExtractor,
    verify_online_offline_parity,
)
from outguard.models.baselines import (
    LegacyDLPRuleBaseline,
    StandardizedLogisticSpecialist,
)
from outguard.models.ensemble import OutGuardHybridEnsemble
from outguard.monitoring.drift_and_oncall import run_drift_and_oncall_triage
from outguard.schemas import ALL_FEATURES, GRAPH_FEATURES, LEXICAL_FEATURES
from outguard.serving.app import RealtimeScoringContext


def run_full_pipeline(
    config_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root)
    artifacts_dir = root / "artifacts"
    data_dir = artifacts_dir / "data"
    assets_dir = root / "docs" / "assets"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    cfg = PipelineConfig.from_yaml(config_path)

    # 1. Data & Labeling Workflow Generation
    events_df, feedback_df, brand_map = generate_benchmark_dataset(cfg)

    # 2. Point-in-Time Causal Feature Backfill + Online/Offline Parity Audit
    extractor = OutGuardFeatureExtractor(cfg, domain_to_brand=brand_map)
    feat_df, snapshots = extractor.build_offline_feature_table(events_df)

    # Persist partitioned Parquet feature store tables
    feat_df.to_parquet(data_dir / "point_in_time_features.parquet", index=False)
    feedback_df.to_parquet(data_dir / "feedback_labels.parquet", index=False)

    parity_report = verify_online_offline_parity(
        events_df, feat_df, snapshots["before_test"], extractor, sample_size=500
    )

    # 3. Chronological Split Preparation
    train_df = feat_df[feat_df["split"] == "train"].reset_index(drop=True)
    val_df = feat_df[feat_df["split"] == "val"].reset_index(drop=True)
    test_df = feat_df[feat_df["split"] == "test"].reset_index(drop=True)

    X_train = train_df[ALL_FEATURES].to_numpy(dtype=np.float64)
    # Train uses realistic observed feedback labels weighted by label_confidence!
    y_train_obs = train_df["observed_train_label"].to_numpy(dtype=np.int32)
    w_train = train_df["label_confidence"].to_numpy(dtype=np.float64)

    X_val = val_df[ALL_FEATURES].to_numpy(dtype=np.float64)
    y_val = val_df["true_label"].to_numpy(dtype=np.int32)

    X_test = test_df[ALL_FEATURES].to_numpy(dtype=np.float64)
    y_test = test_df["true_label"].to_numpy(dtype=np.int32)
    vip_test = test_df["sender_is_vip"].to_numpy(dtype=bool)

    # 4. Hypothesis-Driven Model Progression (H0 -> H4)
    h0 = LegacyDLPRuleBaseline()
    h1 = StandardizedLogisticSpecialist(GRAPH_FEATURES, C=0.8, seed=cfg.seed).fit(
        X_train, y_train_obs, w_train
    )
    h2 = StandardizedLogisticSpecialist(
        GRAPH_FEATURES + LEXICAL_FEATURES, C=0.8, seed=cfg.seed
    ).fit(X_train, y_train_obs, w_train)

    ensemble = OutGuardHybridEnsemble(cfg).fit(
        X_train, y_train_obs, w_train, X_val, y_val
    )
    assert ensemble.thresholds is not None

    val_scores_map = {
        "H0_Legacy_DLP": h0.predict_proba(X_val),
        "H1_Graph_Only": h1.predict_proba(X_val),
        "H2_Graph_Lexical": h2.predict_proba(X_val),
        "H3_MultiModal_LGBM": ensemble.predict_lgbm_only(X_val),
        "H4_OutGuard_Hybrid": ensemble.predict_hybrid_proba(X_val),
    }
    test_scores_map = {
        "H0_Legacy_DLP": h0.predict_proba(X_test),
        "H1_Graph_Only": h1.predict_proba(X_test),
        "H2_Graph_Lexical": h2.predict_proba(X_test),
        "H3_MultiModal_LGBM": ensemble.predict_lgbm_only(X_test),
        "H4_OutGuard_Hybrid": ensemble.predict_hybrid_proba(X_test),
    }

    ablation_results: list[dict[str, Any]] = []
    for m_name in val_scores_map:
        fixed_t = ensemble.thresholds if m_name == "H4_OutGuard_Hybrid" else None
        res = evaluate_model_comprehensive(
            name=m_name,
            y_val=y_val,
            val_scores=val_scores_map[m_name],
            y_test=y_test,
            test_scores=test_scores_map[m_name],
            vip_mask_test=vip_test,
            block_max_fpr=cfg.thresholds.block_max_fpr,
            nudge_max_fpr=cfg.thresholds.nudge_max_fpr,
            bootstrap_rounds=cfg.thresholds.bootstrap_rounds,
            fixed_thresholds=fixed_t,
        )
        ablation_results.append(res)

    # 5. Export ONNX + Go Real-Time Bundle & Verify Cross-Runtime Parity
    onnx_report = ensemble.export_onnx_and_verify(
        artifacts_dir / "med_lgbm.onnx", X_test
    )
    # Include both high-risk positives and tricky benign cases in Go golden vectors
    pos_sample = X_test[y_test == 1][:15]
    neg_sample = X_test[y_test == 0][:15]
    X_golden = np.vstack([pos_sample, neg_sample])
    go_bundle_info = ensemble.export_go_bundle(
        artifacts_dir / "model_bundle.json", X_golden
    )
    parity_report["onnx_verification"] = onnx_report
    parity_report["go_bundle"] = go_bundle_info

    # 6. Benchmark Single-Event Python Online Latency (Feature Extraction + H4 Scoring)
    bench_state = snapshots["before_test"].clone_snapshot()
    bench_events = (
        events_df[events_df["split"] == "test"]
        .sort_values("timestamp_epoch", kind="mergesort")
        .head(300)
        .to_dict(orient="records")
    )
    latencies_ms: list[float] = []
    for ev in bench_events:
        t0 = time.perf_counter()
        fdict = extractor.extract_single_event(ev, bench_state, mutate_state=True)
        fvec = np.array([fdict[k] for k in ALL_FEATURES], dtype=np.float64)
        _ = ensemble.score_single_detailed(fvec)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    py_latency_stats = {
        "p50_ms": round(float(np.percentile(latencies_ms, 50)), 3),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 3),
        "p99_ms": round(float(np.percentile(latencies_ms, 99)), 3),
        "mean_ms": round(float(np.mean(latencies_ms)), 3),
    }

    # 7. Slice Error Analysis, Online A/B Experiment, and Post-Launch Drift Triage
    slice_report = run_slice_error_analysis(
        test_df=test_df,
        scores_by_model={
            "H0_Legacy_DLP": test_scores_map["H0_Legacy_DLP"],
            "H3_MultiModal_LGBM": test_scores_map["H3_MultiModal_LGBM"],
            "H4_OutGuard_Hybrid": test_scores_map["H4_OutGuard_Hybrid"],
        },
        h4_nudge_threshold=ensemble.thresholds.nudge_threshold,
        h4_block_threshold=ensemble.thresholds.block_threshold,
    )

    h2_metrics = next(m for m in ablation_results if m["model_name"] == "H2_Graph_Lexical")
    h4_metrics = next(m for m in ablation_results if m["model_name"] == "H4_OutGuard_Hybrid")

    ab_report = run_online_ab_simulation(
        cfg=cfg.experimentation,
        v1_nudge_recall=h2_metrics["test_realized_nudge_recall"],
        v1_nudge_fpr=h2_metrics["test_realized_nudge_fpr"],
        v2_nudge_recall=h4_metrics["test_realized_nudge_recall"],
        v2_nudge_fpr=h4_metrics["test_realized_nudge_fpr"],
        base_misdirected_rate=cfg.simulation.base_misdirected_rate,
        seed=cfg.seed,
    )

    drift_report = run_drift_and_oncall_triage(
        events_df=events_df,
        feat_df=feat_df,
        ensemble=ensemble,
        mon_cfg=cfg.monitoring,
    )

    # 8. Generate All 6 Diagnostic Plots
    plot_paths = generate_all_plots(
        assets_dir=assets_dir,
        y_test=y_test,
        scores_by_model=test_scores_map,
        ablation_list=ablation_results,
        slice_report=slice_report,
        ab_report=ab_report,
        drift_report=drift_report,
        nudge_tau=ensemble.thresholds.nudge_threshold,
        block_tau=ensemble.thresholds.block_threshold,
    )

    # Save live FastAPI serving runtime context
    serving_ctx = RealtimeScoringContext(
        cfg=cfg,
        extractor=extractor,
        state=snapshots["final"],
        ensemble=ensemble,
        shadow_mode=False,
    )
    with open(artifacts_dir / "serving_runtime.pkl", "wb") as fh:
        pickle.dump(serving_ctx, fh)

    total_sec = round(time.perf_counter() - t_start, 2)

    # Feature importance ranking from LightGBM gain
    importances = ensemble.lgbm.booster_.feature_importance(importance_type="gain")
    total_gain = max(1.0, float(np.sum(importances)))
    top_features = dict(
        sorted(
            {
                ALL_FEATURES[i]: round(float(importances[i] / total_gain * 100.0), 2)
                for i in range(len(ALL_FEATURES))
            }.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[:12]
    )

    summary = {
        "pipeline_runtime_sec": total_sec,
        "dataset_counts": feat_df.groupby("split")["true_label"]
        .agg(["count", "sum", "mean"])
        .round(5)
        .to_dict("index"),
        "signal_combiner_weights": {
            "graph_specialist_logit_weight": round(float(ensemble.meta_combiner.coef_[0][0]), 4),
            "lexical_coherence_logit_weight": round(float(ensemble.meta_combiner.coef_[0][1]), 4),
            "dlp_context_logit_weight": round(float(ensemble.meta_combiner.coef_[0][2]), 4),
            "lgbm_interaction_logit_weight": round(float(ensemble.meta_combiner.coef_[0][3]), 4),
            "intercept": round(float(ensemble.meta_combiner.intercept_[0]), 4),
        },
        "top_12_features_by_gain_pct": top_features,
        "champion_h4_metrics": h4_metrics,
        "python_online_latency_ms": py_latency_stats,
        "online_offline_parity_passed": parity_report["parity_passed"],
        "onnx_parity_passed": onnx_report["onnx_parity_passed"],
        "all_slice_regression_gates_passed": slice_report["all_gates_passed"],
        "ab_ship_recommendation": ab_report["ship_recommendation"],
        "drift_incident_detected": drift_report["alert_triggered"],
        "generated_plots": [Path(p).name for p in plot_paths],
    }

    _write_json(artifacts_dir / "metrics_summary.json", summary)
    _write_json(artifacts_dir / "ablation_results.json", ablation_results)
    _write_json(artifacts_dir / "slice_analysis.json", slice_report)
    _write_json(artifacts_dir / "ab_experiment_report.json", ab_report)
    _write_json(artifacts_dir / "drift_incident_report.json", drift_report)
    _write_json(artifacts_dir / "parity_verification.json", parity_report)

    return summary


def _write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
