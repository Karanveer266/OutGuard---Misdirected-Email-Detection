"""Population Stability Index (PSI) drift detector and MED On-Call triage engine."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from outguard.config import MonitoringConfig
from outguard.models.ensemble import OutGuardHybridEnsemble
from outguard.schemas import ALL_FEATURES


def compute_psi(ref: np.ndarray, cur: np.ndarray, n_bins: int = 10) -> float:
    """Compute Population Stability Index (PSI) between reference and current distributions."""
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) <= 2:
        # Binary or near-constant feature: use fixed two-bin split at midpoint
        edges = np.array([-np.inf, 0.5, np.inf])
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf

    ref_cnt, _ = np.histogram(ref, bins=edges)
    cur_cnt, _ = np.histogram(cur, bins=edges)

    ref_p = np.clip(ref_cnt / max(1, len(ref)), 1e-4, 1.0)
    cur_p = np.clip(cur_cnt / max(1, len(cur)), 1e-4, 1.0)
    psi = float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))
    return max(0.0, psi)


def run_drift_and_oncall_triage(
    events_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    ensemble: OutGuardHybridEnsemble,
    mon_cfg: MonitoringConfig,
) -> dict[str, Any]:
    """Detect post-launch M&A subsidiary domain shift, triage root cause, and verify mitigation."""
    assert ensemble.thresholds is not None
    nudge_tau = ensemble.thresholds.nudge_threshold

    ref_df = feat_df[feat_df["split"] == "test"].reset_index(drop=True)
    cur_df = feat_df[feat_df["split"] == "drift"].reset_index(drop=True)
    cur_events = events_df[events_df["split"] == "drift"].sort_values(
        "timestamp_epoch", kind="mergesort"
    ).reset_index(drop=True)

    X_ref = ref_df[ALL_FEATURES].to_numpy(dtype=np.float64)
    X_cur = cur_df[ALL_FEATURES].to_numpy(dtype=np.float64)

    ref_scores = ensemble.predict_hybrid_proba(X_ref)
    cur_scores = ensemble.predict_hybrid_proba(X_cur)

    # 1. Global & per-tenant PSI and alert-rate anomalies
    feature_psi: dict[str, float] = {}
    for idx, fname in enumerate(ALL_FEATURES):
        feature_psi[fname] = round(compute_psi(X_ref[:, idx], X_cur[:, idx]), 5)

    # Tenant-specific PSI on impacted tenant (`tenant_finsight`)
    ref_fin = ref_df[ref_df["tenant_id"] == "tenant_finsight"]
    cur_fin = cur_df[cur_df["tenant_id"] == "tenant_finsight"]
    X_ref_fin = ref_fin[ALL_FEATURES].to_numpy(dtype=np.float64)
    X_cur_fin = cur_fin[ALL_FEATURES].to_numpy(dtype=np.float64)
    fin_ref_scores = ensemble.predict_hybrid_proba(X_ref_fin)
    fin_cur_scores = ensemble.predict_hybrid_proba(X_cur_fin)

    tenant_feature_psi = {
        fname: round(compute_psi(X_ref_fin[:, idx], X_cur_fin[:, idx]), 5)
        for idx, fname in enumerate(ALL_FEATURES)
    }
    score_psi_finsight = round(compute_psi(fin_ref_scores, fin_cur_scores), 5)

    ref_fin_alert_rate = float(np.mean(fin_ref_scores >= nudge_tau))
    cur_fin_alert_rate = float(np.mean(fin_cur_scores >= nudge_tau))
    cur_fin_benign = cur_fin["true_label"].to_numpy() == 0
    pre_mitigation_benign_fpr = float(np.mean(fin_cur_scores[cur_fin_benign] >= nudge_tau))

    # 2. Automated root-cause domain attribution on flagged alerts in drift window
    flagged_domain_counts: Counter[str] = Counter()
    all_drift_fin_domains: Counter[str] = Counter()
    for idx, row in cur_events.iterrows():
        if row["tenant_id"] != "tenant_finsight":
            continue
        doms = {r["domain"].lower() for r in row["recipients"] if r["domain"].lower() != row["tenant_domain"].lower()}
        for d in doms:
            all_drift_fin_domains[d] += 1
            if cur_scores[idx] >= nudge_tau:
                flagged_domain_counts[d] += 1

    culprit_domain, culprit_volume = all_drift_fin_domains.most_common(1)[0]

    # 3. Verify immediate safe-launch guardrail mitigation (allowlisting verified M&A subsidiary)
    mitigated_scores = cur_scores.copy()
    for idx, row in cur_events.iterrows():
        if row["tenant_id"] == "tenant_finsight":
            doms = {r["domain"].lower() for r in row["recipients"]}
            if culprit_domain in doms:
                res = ensemble.score_single_detailed(
                    X_cur[idx], allowlisted_domain_hit=True
                )
                mitigated_scores[idx] = res["calibrated_risk_score"]

    fin_mask = (cur_df["tenant_id"] == "tenant_finsight").to_numpy()
    fin_mit_scores = mitigated_scores[fin_mask]
    post_mitigation_benign_fpr = float(np.mean(fin_mit_scores[cur_fin_benign] >= nudge_tau))
    drift_pos_mask = cur_df["true_label"].to_numpy() == 1
    post_mitigation_recall = float(np.mean(mitigated_scores[drift_pos_mask] >= nudge_tau))

    top_drifted = dict(
        sorted(tenant_feature_psi.items(), key=lambda kv: kv[1], reverse=True)[:6]
    )

    alert_triggered = (
        score_psi_finsight >= mon_cfg.psi_warning_threshold
        or max(tenant_feature_psi.values()) >= mon_cfg.psi_critical_threshold
        or cur_fin_alert_rate >= ref_fin_alert_rate * mon_cfg.alert_rate_spike_multiplier
    )

    return {
        "incident_id": "INC-2026-1112-MED-04",
        "alert_triggered": bool(alert_triggered),
        "impacted_tenant": "tenant_finsight",
        "diagnosed_root_cause_domain": culprit_domain,
        "culprit_domain_event_volume": int(culprit_volume),
        "culprit_domain_flagged_alerts": int(flagged_domain_counts[culprit_domain]),
        "tenant_score_psi": score_psi_finsight,
        "top_tenant_drifted_features_psi": top_drifted,
        "global_feature_psi_top5": dict(
            sorted(feature_psi.items(), key=lambda kv: kv[1], reverse=True)[:5]
        ),
        "finsight_alert_rate_reference": round(ref_fin_alert_rate, 5),
        "finsight_alert_rate_drift_unmitigated": round(cur_fin_alert_rate, 5),
        "finsight_benign_fpr_pre_mitigation": round(pre_mitigation_benign_fpr, 5),
        "finsight_benign_fpr_post_mitigation": round(post_mitigation_benign_fpr, 5),
        "overall_drift_window_recall_post_mitigation": round(post_mitigation_recall, 4),
        "recommended_remediation": (
            f"Add verified M&A subsidiary '{culprit_domain}' to tenant_finsight "
            "trusted domain graph allowlist and schedule nightly graph warm-start backfill."
        ),
    }
