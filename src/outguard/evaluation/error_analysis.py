"""Targeted slice error analysis, regression gates, and residual FP/FN diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _threshold_at_target_recall(
    y_true: np.ndarray, scores: np.ndarray, target_recall: float = 0.85
) -> float:
    pos_scores = np.sort(scores[y_true == 1])
    idx = max(0, int(np.floor((1.0 - target_recall) * len(pos_scores))))
    return float(pos_scores[min(idx, len(pos_scores) - 1)])


def run_slice_error_analysis(
    test_df: pd.DataFrame,
    scores_by_model: dict[str, np.ndarray],
    h4_nudge_threshold: float,
    h4_block_threshold: float,
) -> dict[str, Any]:
    """Compare benign slice FPRs and threat archetype recall across model iterations."""
    y_test = test_df["true_label"].to_numpy(dtype=np.int32)
    slices = test_df["eval_slice"].to_numpy()
    m_types = test_df["misdirection_type"].to_numpy()

    # 1. Benign slice FPR comparison at matched 85% recall operating point
    matched_taus = {
        m_name: _threshold_at_target_recall(y_test, sc, target_recall=0.85)
        for m_name, sc in scores_by_model.items()
    }

    benign_slices = [
        "vip_first_contact_sensitive",
        "recruiter_sales_cold_outreach",
        "nickname_salutation",
        "cc_warm_introduction",
        "personal_self_forward",
        "routine_reply_chain",
        "standard_business",
    ]
    benign_slice_report: dict[str, Any] = {}
    for sl in benign_slices:
        mask = (y_test == 0) & (slices == sl)
        cnt = int(np.sum(mask))
        if cnt == 0:
            continue
        entry: dict[str, Any] = {"count": cnt}
        for m_name, sc in scores_by_model.items():
            fpr_85 = float(np.mean(sc[mask] >= matched_taus[m_name]))
            entry[f"{m_name}_fpr_at_85rec"] = round(fpr_85, 5)
        entry["H4_prod_nudge_fpr"] = round(
            float(np.mean(scores_by_model["H4_OutGuard_Hybrid"][mask] >= h4_nudge_threshold)),
            5,
        )
        benign_slice_report[sl] = entry

    # 2. Threat archetype recall at H4 production Nudge and Block thresholds
    archetypes = [
        "AUTOCOMPLETE_COLLISION",
        "EXTERNAL_THREAD_LEAK",
        "SALUTATION_MISMATCH",
        "WRONG_TENANT_DIVERGENCE",
        "DOMAIN_TYPO_LEAK",
    ]
    archetype_report: dict[str, Any] = {}
    h4_scores = scores_by_model["H4_OutGuard_Hybrid"]
    h0_scores = scores_by_model["H0_Legacy_DLP"]
    h0_tau = np.quantile(h0_scores[y_test == 0], 1.0 - 0.0025)

    for arch in archetypes:
        mask = (y_test == 1) & (m_types == arch)
        cnt = int(np.sum(mask))
        if cnt == 0:
            continue
        archetype_report[arch] = {
            "count": cnt,
            "H0_recall_at_025fpr": round(float(np.mean(h0_scores[mask] >= h0_tau)), 4),
            "H4_nudge_recall": round(float(np.mean(h4_scores[mask] >= h4_nudge_threshold)), 4),
            "H4_block_recall": round(float(np.mean(h4_scores[mask] >= h4_block_threshold)), 4),
        }

    # 3. Inspect concrete residual False Positives and False Negatives for H4
    fp_mask = (y_test == 0) & (h4_scores >= h4_nudge_threshold)
    fn_mask = (y_test == 1) & (h4_scores < h4_nudge_threshold)

    fp_rows = (
        test_df.assign(h4_score=h4_scores)[fp_mask]
        .sort_values("h4_score", ascending=False)
        .head(5)
    )
    fn_rows = (
        test_df.assign(h4_score=h4_scores)[fn_mask]
        .sort_values("h4_score", ascending=True)
        .head(5)
    )

    residual_fps = [
        {
            "event_id": r["event_id"],
            "eval_slice": r["eval_slice"],
            "sender_department": r["sender_department"],
            "score": round(float(r["h4_score"]), 4),
            "first_time_ext": float(r["graph_any_first_time_external"]),
            "dlp_blast": round(float(r["dlp_composite_blast_radius"]), 3),
        }
        for _, r in fp_rows.iterrows()
    ]
    residual_fns = [
        {
            "event_id": r["event_id"],
            "misdirection_type": r["misdirection_type"],
            "score": round(float(r["h4_score"]), 4),
            "prefix_gap": round(float(r["lex_autocomplete_prefix_collision_gap"]), 3),
            "dlp_blast": round(float(r["dlp_composite_blast_radius"]), 3),
            "body_other_org": float(r["sem_body_mentions_other_known_org"]),
        }
        for _, r in fn_rows.iterrows()
    ]

    # Automated CI regression gates
    gates = {
        "vip_benign_nudge_fpr_below_0_5pct": benign_slice_report["vip_first_contact_sensitive"]["H4_prod_nudge_fpr"] <= 0.006,
        "nickname_salutation_nudge_fpr_below_0_5pct": benign_slice_report["nickname_salutation"]["H4_prod_nudge_fpr"] <= 0.005,
        "recruiter_cold_outreach_nudge_fpr_below_0_5pct": benign_slice_report["recruiter_sales_cold_outreach"]["H4_prod_nudge_fpr"] <= 0.005,
        "wrong_tenant_known_contact_recall_above_75pct": archetype_report["WRONG_TENANT_DIVERGENCE"]["H4_nudge_recall"] >= 0.75,
    }

    return {
        "benign_slice_fpr_comparison": benign_slice_report,
        "threat_archetype_recall": archetype_report,
        "residual_false_positives_sample": residual_fps,
        "residual_false_negatives_sample": residual_fns,
        "regression_gates": gates,
        "all_gates_passed": all(gates.values()),
    }
