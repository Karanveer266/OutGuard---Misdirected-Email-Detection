"""Low-FPR detection metrics, calibration error, and SOC operational KPIs."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from outguard.models.thresholds import (
    ActionThresholds,
    bootstrap_recall_at_fpr_ci,
    optimize_soc_thresholds,
    select_threshold_for_max_fpr,
)


def expected_calibration_error(
    y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE) over probability bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs <= hi if i == n_bins - 1 else probs < hi)
        cnt = int(np.sum(mask))
        if cnt == 0:
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(probs[mask]))
        ece += (cnt / n) * abs(acc - conf)
    return float(ece)


def evaluate_model_comprehensive(
    name: str,
    y_val: np.ndarray,
    val_scores: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
    vip_mask_test: np.ndarray,
    block_max_fpr: float = 0.0005,
    nudge_max_fpr: float = 0.0025,
    bootstrap_rounds: int = 300,
    fixed_thresholds: ActionThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate a candidate model using validation-locked SOC thresholds on test."""
    thresholds = fixed_thresholds or optimize_soc_thresholds(
        y_val, val_scores, nudge_max_fpr=nudge_max_fpr, block_max_fpr=block_max_fpr
    )

    # Oracle curve metrics on test at exact FPR operating points + bootstrap CIs
    tau_005 = select_threshold_for_max_fpr(y_test, test_scores, block_max_fpr)
    tau_025 = select_threshold_for_max_fpr(y_test, test_scores, nudge_max_fpr)
    rec_at_005 = float(np.mean(test_scores[y_test == 1] >= tau_005))
    rec_at_025 = float(np.mean(test_scores[y_test == 1] >= tau_025))

    ci_005 = bootstrap_recall_at_fpr_ci(
        y_test, test_scores, block_max_fpr, n_rounds=bootstrap_rounds
    )
    ci_025 = bootstrap_recall_at_fpr_ci(
        y_test, test_scores, nudge_max_fpr, n_rounds=bootstrap_rounds
    )

    # Validation-locked threshold operational performance on held-out test
    neg_mask = y_test == 0
    pos_mask = y_test == 1
    pred_block = test_scores >= thresholds.block_threshold
    pred_nudge_or_block = test_scores >= thresholds.nudge_threshold

    realized_block_fpr = float(np.mean(pred_block[neg_mask]))
    realized_nudge_fpr = float(np.mean(pred_nudge_or_block[neg_mask]))
    realized_block_recall = float(np.mean(pred_block[pos_mask]))
    realized_nudge_recall = float(np.mean(pred_nudge_or_block[pos_mask]))

    block_prec = (
        float(np.sum(pred_block & pos_mask) / np.sum(pred_block))
        if np.sum(pred_block) > 0
        else 1.0
    )
    nudge_prec = (
        float(np.sum(pred_nudge_or_block & pos_mask) / np.sum(pred_nudge_or_block))
        if np.sum(pred_nudge_or_block) > 0
        else 1.0
    )

    vip_benign = neg_mask & vip_mask_test
    vip_fpr = (
        float(np.mean(pred_nudge_or_block[vip_benign]))
        if np.any(vip_benign)
        else 0.0
    )

    return {
        "model_name": name,
        "pr_auc": round(float(average_precision_score(y_test, test_scores)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, test_scores)), 4),
        "brier_score": round(float(brier_score_loss(y_test, np.clip(test_scores, 0.0, 1.0))), 5),
        "ece": round(expected_calibration_error(y_test, np.clip(test_scores, 0.0, 1.0)), 5),
        "recall_at_fpr_0_05pct": round(rec_at_005, 4),
        "recall_at_fpr_0_05pct_ci95": [ci_005["ci95_low"], ci_005["ci95_high"]],
        "recall_at_fpr_0_25pct": round(rec_at_025, 4),
        "recall_at_fpr_0_25pct_ci95": [ci_025["ci95_low"], ci_025["ci95_high"]],
        "val_locked_block_threshold": round(thresholds.block_threshold, 5),
        "val_locked_nudge_threshold": round(thresholds.nudge_threshold, 5),
        "test_realized_block_fpr": round(realized_block_fpr, 5),
        "test_realized_block_precision": round(block_prec, 4),
        "test_realized_block_recall": round(realized_block_recall, 4),
        "test_realized_nudge_fpr": round(realized_nudge_fpr, 5),
        "test_realized_nudge_precision": round(nudge_prec, 4),
        "test_realized_nudge_recall": round(realized_nudge_recall, 4),
        "soc_false_alerts_per_10k_benign": round(realized_nudge_fpr * 10_000.0, 2),
        "vip_benign_fpr": round(vip_fpr, 5),
    }
