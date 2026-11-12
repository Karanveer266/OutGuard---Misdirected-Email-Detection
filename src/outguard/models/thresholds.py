"""Neyman-Pearson low-FPR threshold selection under SOC alert-budget constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ActionThresholds:
    nudge_threshold: float
    block_threshold: float
    val_nudge_fpr: float
    val_block_fpr: float
    val_nudge_recall: float
    val_block_recall: float

    def to_dict(self) -> dict[str, float]:
        return {
            "nudge_threshold": round(self.nudge_threshold, 6),
            "block_threshold": round(self.block_threshold, 6),
            "val_nudge_fpr": round(self.val_nudge_fpr, 6),
            "val_block_fpr": round(self.val_block_fpr, 6),
            "val_nudge_recall": round(self.val_nudge_recall, 6),
            "val_block_recall": round(self.val_block_recall, 6),
        }


def select_threshold_for_max_fpr(
    y_true: np.ndarray, scores: np.ndarray, max_fpr: float
) -> float:
    """Find the lowest decision threshold tau such that empirical benign FPR <= max_fpr."""
    neg_scores = np.sort(scores[y_true == 0])
    n_neg = len(neg_scores)
    if n_neg == 0:
        return 0.5
    max_allowed_fp = int(np.floor(max_fpr * n_neg))
    if max_allowed_fp <= 0:
        return float(min(0.9999, neg_scores[-1] + 1e-6))
    idx = n_neg - max_allowed_fp - 1
    tau = float(neg_scores[max(0, idx)] + 1e-7)
    return min(0.9999, tau)


def optimize_soc_thresholds(
    y_val: np.ndarray,
    val_scores: np.ndarray,
    nudge_max_fpr: float = 0.0025,
    block_max_fpr: float = 0.0005,
) -> ActionThresholds:
    """Calibrate two-tier action thresholds (NUDGE_USER vs HOLD_BLOCK) on validation data."""
    nudge_tau = select_threshold_for_max_fpr(y_val, val_scores, nudge_max_fpr)
    block_tau = select_threshold_for_max_fpr(y_val, val_scores, block_max_fpr)
    if block_tau <= nudge_tau:
        block_tau = min(0.9999, nudge_tau + 0.02)

    neg_mask = y_val == 0
    pos_mask = y_val == 1
    nudge_fpr = float(np.mean(val_scores[neg_mask] >= nudge_tau))
    block_fpr = float(np.mean(val_scores[neg_mask] >= block_tau))
    nudge_rec = float(np.mean(val_scores[pos_mask] >= nudge_tau))
    block_rec = float(np.mean(val_scores[pos_mask] >= block_tau))

    return ActionThresholds(
        nudge_threshold=nudge_tau,
        block_threshold=block_tau,
        val_nudge_fpr=nudge_fpr,
        val_block_fpr=block_fpr,
        val_nudge_recall=nudge_rec,
        val_block_recall=block_rec,
    )


def bootstrap_recall_at_fpr_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_fpr: float,
    n_rounds: int = 300,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute stratified bootstrap 95% confidence interval for Recall @ target FPR."""
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)

    point_tau = select_threshold_for_max_fpr(y_true, scores, target_fpr)
    point_recall = float(np.mean(scores[pos_idx] >= point_tau))

    recalls = np.empty(n_rounds, dtype=np.float64)
    for b in range(n_rounds):
        b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        b_idx = np.concatenate([b_pos, b_neg])
        b_y = y_true[b_idx]
        b_s = scores[b_idx]
        tau = select_threshold_for_max_fpr(b_y, b_s, target_fpr)
        recalls[b] = float(np.mean(scores[b_pos] >= tau))

    return {
        "point_estimate": round(point_recall, 4),
        "ci95_low": round(float(np.percentile(recalls, 2.5)), 4),
        "ci95_high": round(float(np.percentile(recalls, 97.5)), 4),
    }
