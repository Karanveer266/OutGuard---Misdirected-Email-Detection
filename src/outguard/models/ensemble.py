"""H3 LightGBM, H4 Calibrated Signal-Averaging Hybrid Ensemble, ONNX & Go Exporter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import onnxruntime as ort
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from outguard.config import PipelineConfig
from outguard.models.baselines import (
    StandardizedLogisticSpecialist,
    build_specialist_suite,
)
from outguard.models.guardrails import (
    apply_guardrails_batch,
    apply_guardrails_single,
    explain_top_risk_factors,
)
from outguard.models.thresholds import ActionThresholds, optimize_soc_thresholds
from outguard.schemas import ALL_FEATURES, PolicyAction


def _safe_logit(p: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -25.0, 25.0)))


def _make_lgbm(cfg: PipelineConfig, seed_offset: int = 0) -> lgb.LGBMClassifier:
    p = cfg.model.lgbm
    return lgb.LGBMClassifier(
        n_estimators=p.n_estimators,
        learning_rate=p.learning_rate,
        num_leaves=p.num_leaves,
        max_depth=p.max_depth,
        min_child_samples=p.min_child_samples,
        subsample=p.subsample,
        subsample_freq=1,
        colsample_bytree=p.colsample_bytree,
        reg_alpha=p.reg_alpha,
        reg_lambda=p.reg_lambda,
        scale_pos_weight=p.scale_pos_weight,
        random_state=cfg.seed + seed_offset,
        n_jobs=4,
        verbose=-1,
    )


class OutGuardHybridEnsemble:
    """End-to-end H4 Misdirected Email Detection ensemble.

    Combines 3 orthogonal domain-specialist scorers (Graph, Lexical-Coherence,
    DLP-Context) with a Multi-Modal LightGBM interaction model via out-of-fold
    log-odds signal averaging, smooth tie-free Isotonic+Platt calibration, and
    deterministic precision/suppression guardrails.
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.specialists: dict[str, StandardizedLogisticSpecialist] = build_specialist_suite(
            cfg.seed
        )
        self.lgbm: lgb.LGBMClassifier = _make_lgbm(cfg)
        self.meta_combiner = LogisticRegression(C=1.0, random_state=cfg.seed)
        self.platt_calibrator = LogisticRegression(C=2.0, random_state=cfg.seed)
        self.iso_calibrator = IsotonicRegression(y_min=1e-5, y_max=0.9999, out_of_bounds="clip")
        self.iso_blend_weight: float = 0.25
        self.thresholds: ActionThresholds | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train_obs: np.ndarray,
        sample_weight: np.ndarray,
        X_val: np.ndarray,
        y_val_true: np.ndarray,
    ) -> "OutGuardHybridEnsemble":
        # 1. 5-fold OOF stacking on training split to learn unbiased signal-combiner weights
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.cfg.seed)
        oof_logits = np.zeros((len(X_train), 4), dtype=np.float64)

        spec_names = [
            "graph_specialist",
            "lexical_coherence_specialist",
            "dlp_context_specialist",
        ]

        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train_obs)):
            fold_specs = build_specialist_suite(self.cfg.seed + 10 * fold_idx)
            for s_i, s_name in enumerate(spec_names):
                fold_specs[s_name].fit(
                    X_train[tr_idx], y_train_obs[tr_idx], sample_weight[tr_idx]
                )
                oof_logits[va_idx, s_i] = _safe_logit(
                    fold_specs[s_name].predict_proba(X_train[va_idx])
                )

            fold_lgbm = _make_lgbm(self.cfg, seed_offset=fold_idx + 1)
            fold_lgbm.fit(
                X_train[tr_idx], y_train_obs[tr_idx], sample_weight=sample_weight[tr_idx]
            )
            oof_logits[va_idx, 3] = _safe_logit(
                fold_lgbm.predict_proba(X_train[va_idx])[:, 1]
            )

        self.meta_combiner.fit(oof_logits, y_train_obs, sample_weight=sample_weight)

        # 2. Fit full-train specialists & LightGBM backbone
        for s_name in spec_names:
            self.specialists[s_name].fit(X_train, y_train_obs, sample_weight)
        self.lgbm.fit(X_train, y_train_obs, sample_weight=sample_weight)

        # 3. Fit smooth tie-free calibration (Isotonic + Platt blend) on held-out validation
        val_meta_z = self._raw_meta_logit(X_val)
        self.platt_calibrator.fit(val_meta_z.reshape(-1, 1), y_val_true)
        self.iso_calibrator.fit(val_meta_z, y_val_true)

        # 4. Optimize two-tier SOC FPR thresholds (NUDGE <= 0.25% FPR, BLOCK <= 0.05% FPR)
        # First estimate base calibrated thresholds, then refine post-guardrails
        val_cal_probs = self._calibrated_ml_proba(X_val)
        prelim = optimize_soc_thresholds(
            y_val_true,
            val_cal_probs,
            nudge_max_fpr=self.cfg.thresholds.nudge_max_fpr,
            block_max_fpr=self.cfg.thresholds.block_max_fpr,
        )
        val_guarded, _ = apply_guardrails_batch(
            X_val,
            val_cal_probs,
            self.cfg.guardrails,
            prelim.nudge_threshold,
            prelim.block_threshold,
        )
        self.thresholds = optimize_soc_thresholds(
            y_val_true,
            val_guarded,
            nudge_max_fpr=self.cfg.thresholds.nudge_max_fpr,
            block_max_fpr=self.cfg.thresholds.block_max_fpr,
        )
        return self

    def _specialist_and_lgbm_probs(self, X: np.ndarray) -> np.ndarray:
        p_graph = self.specialists["graph_specialist"].predict_proba(X)
        p_lex = self.specialists["lexical_coherence_specialist"].predict_proba(X)
        p_dlp = self.specialists["dlp_context_specialist"].predict_proba(X)
        p_lgbm = self.lgbm.predict_proba(X)[:, 1]
        return np.column_stack([p_graph, p_lex, p_dlp, p_lgbm])

    def _raw_meta_logit(self, X: np.ndarray) -> np.ndarray:
        sub_probs = self._specialist_and_lgbm_probs(X)
        sub_logits = _safe_logit(sub_probs)
        coefs = self.meta_combiner.coef_[0]
        intercept = float(self.meta_combiner.intercept_[0])
        return sub_logits @ coefs + intercept

    def _calibrated_ml_proba(self, X: np.ndarray) -> np.ndarray:
        z = self._raw_meta_logit(X)
        p_iso = np.interp(
            z, self.iso_calibrator.X_thresholds_, self.iso_calibrator.y_thresholds_
        )
        platt_a = float(self.platt_calibrator.coef_[0][0])
        platt_b = float(self.platt_calibrator.intercept_[0])
        p_platt = _sigmoid(platt_a * z + platt_b)
        return self.iso_blend_weight * p_iso + (1.0 - self.iso_blend_weight) * p_platt

    def predict_lgbm_only(self, X: np.ndarray) -> np.ndarray:
        """H3: Raw Multi-Modal LightGBM probability without signal fusion or guardrails."""
        return self.lgbm.predict_proba(X)[:, 1]

    def predict_hybrid_proba(self, X: np.ndarray) -> np.ndarray:
        """H4: Full calibrated signal-averaging + deterministic guardrails score."""
        assert self.thresholds is not None, "Model must be fitted first"
        cal = self._calibrated_ml_proba(X)
        guarded, _ = apply_guardrails_batch(
            X,
            cal,
            self.cfg.guardrails,
            self.thresholds.nudge_threshold,
            self.thresholds.block_threshold,
        )
        return guarded

    def score_single_detailed(
        self, feat_vec: np.ndarray, allowlisted_domain_hit: bool = False
    ) -> dict[str, Any]:
        """Detailed single-event inference for real-time FastAPI serving."""
        assert self.thresholds is not None
        X = feat_vec.reshape(1, -1)
        sub_probs = self._specialist_and_lgbm_probs(X)[0]
        cal_prob = float(self._calibrated_ml_proba(X)[0])
        final_prob, rule = apply_guardrails_single(
            feat_vec,
            cal_prob,
            self.cfg.guardrails,
            self.thresholds.nudge_threshold,
            self.thresholds.block_threshold,
            allowlisted_domains_hit=allowlisted_domain_hit,
        )
        if final_prob >= self.thresholds.block_threshold:
            action = PolicyAction.HOLD_BLOCK
        elif final_prob >= self.thresholds.nudge_threshold:
            action = PolicyAction.NUDGE_USER
        else:
            action = PolicyAction.ALLOW

        factors = explain_top_risk_factors(feat_vec)
        return {
            "calibrated_risk_score": round(float(final_prob), 6),
            "raw_lgbm_prob": round(float(sub_probs[3]), 6),
            "graph_specialist_prob": round(float(sub_probs[0]), 6),
            "lexical_specialist_prob": round(float(sub_probs[1]), 6),
            "dlp_specialist_prob": round(float(sub_probs[2]), 6),
            "action": action,
            "triggered_guardrail": rule,
            "top_risk_factors": factors,
        }

    def export_onnx_and_verify(
        self, onnx_path: str | Path, X_sample: np.ndarray
    ) -> dict[str, Any]:
        """Export LightGBM backbone to ONNX and verify numerical parity in onnxruntime."""
        onnx_path = Path(onnx_path)
        initial_types = [("features", FloatTensorType([None, len(ALL_FEATURES)]))]
        onnx_model = onnxmltools.convert_lightgbm(
            self.lgbm,
            initial_types=initial_types,
            target_opset=15,
            zipmap=False,
        )
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        with open(onnx_path, "wb") as fh:
            fh.write(onnx_model.SerializeToString())

        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        sess = ort.InferenceSession(
            str(onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )
        X32 = X_sample[:256].astype(np.float32)
        ort_out = sess.run(None, {"features": X32})
        # Extract class-1 probability from ONNX ZipMap or probability tensor
        probs_raw = ort_out[1]
        if isinstance(probs_raw, list) and isinstance(probs_raw[0], dict):
            ort_probs = np.array([d[1] for d in probs_raw], dtype=np.float64)
        else:
            ort_probs = np.asarray(probs_raw, dtype=np.float64)[:, 1]

        py_probs = self.lgbm.predict_proba(X32)[:, 1]
        max_diff = float(np.max(np.abs(ort_probs - py_probs)))
        return {
            "onnx_path": str(onnx_path),
            "samples_verified": len(X32),
            "max_abs_diff_float32": max_diff,
            "onnx_parity_passed": bool(max_diff < 1e-5),
        }

    def export_go_bundle(
        self, bundle_path: str | Path, X_golden: np.ndarray
    ) -> dict[str, Any]:
        """Export portable JSON model bundle for the Go real-time microservice."""
        assert self.thresholds is not None
        bundle_path = Path(bundle_path)
        booster_dump = self.lgbm.booster_.dump_model()
        trees = [t["tree_structure"] for t in booster_dump["tree_info"]]

        golden_cases = []
        for i in range(min(30, len(X_golden))):
            vec = X_golden[i]
            res = self.score_single_detailed(vec)
            golden_cases.append(
                {
                    "features": [float(v) for v in vec],
                    "expected_score": res["calibrated_risk_score"],
                    "expected_lgbm_prob": res["raw_lgbm_prob"],
                    "expected_action": res["action"].value,
                    "expected_guardrail": res["triggered_guardrail"] or "",
                }
            )

        bundle = {
            "version": "1.0.0",
            "feature_names": ALL_FEATURES,
            "specialists": {
                k: v.export_dict() for k, v in self.specialists.items()
            },
            "lgbm_trees": trees,
            "meta_combiner": {
                "coef": [float(c) for c in self.meta_combiner.coef_[0]],
                "intercept": float(self.meta_combiner.intercept_[0]),
            },
            "calibration": {
                "iso_weight": self.iso_blend_weight,
                "iso_x": [float(x) for x in self.iso_calibrator.X_thresholds_],
                "iso_y": [float(y) for y in self.iso_calibrator.y_thresholds_],
                "platt_a": float(self.platt_calibrator.coef_[0][0]),
                "platt_b": float(self.platt_calibrator.intercept_[0]),
            },
            "guardrails": self.cfg.guardrails.model_dump(),
            "thresholds": self.thresholds.to_dict(),
            "golden_test_vectors": golden_cases,
        }
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        with open(bundle_path, "w", encoding="utf-8") as fh:
            json.dump(bundle, fh, indent=2)
        return {"bundle_path": str(bundle_path), "num_trees": len(trees), "golden_vectors": len(golden_cases)}
