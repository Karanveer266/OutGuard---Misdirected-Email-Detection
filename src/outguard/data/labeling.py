"""Production labeling workflow resolver for outbound misdirected email telemetry.

In real enterprise MED deployments, training labels arrive asynchronously from
multiple noisy and delayed feedback channels:
  1. Inline Nudge Remediation: user cancels send or edits recipient after prompt.
  2. Rapid Recall & Resend Heuristic: sender recalls outbound email within 5 mins
     and immediately sends near-identical subject/body to a different recipient.
  3. SOC / DLP Analyst Adjudication: confirmed incident or false-positive review.
  4. Hurried User Override ("Send Anyway"): noisy negative signal when users dismiss
     warnings under time pressure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthesize_labeling_workflow(
    events_df: pd.DataFrame,
    noise_rate: float = 0.015,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic multi-source feedback table and confidence weights."""
    rng = np.random.default_rng(seed + 101)
    n = len(events_df)
    true_y = events_df["true_label"].to_numpy(dtype=np.int32)
    splits = events_df["split"].to_numpy()

    observed_y = true_y.copy()
    confidence = np.ones(n, dtype=np.float64)
    sources = np.full(n, "implicit_clean_delivery", dtype=object)
    delay_minutes = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if splits[i] != "train":
            # Validation and test splits use SOC-audited golden labels for unbiased evaluation
            sources[i] = "soc_golden_audit" if true_y[i] == 1 else "verified_benign_audit"
            confidence[i] = 1.0
            delay_minutes[i] = 0.0
            continue

        if true_y[i] == 1:
            u = rng.random()
            if u < 0.44:
                sources[i] = "nudge_recipient_corrected"
                confidence[i] = 0.96
                delay_minutes[i] = float(rng.uniform(0.1, 1.5))
            elif u < 0.72:
                sources[i] = "recall_and_resend_5m"
                confidence[i] = 0.92
                delay_minutes[i] = float(rng.uniform(1.0, 5.0))
            elif u < 0.91:
                sources[i] = "soc_confirmed_incident"
                confidence[i] = 1.00
                delay_minutes[i] = float(rng.uniform(30.0, 720.0))
            else:
                # Subtle misdirection dismissed by hurried sender ("Send Anyway") -> false negative in train labels
                observed_y[i] = 0
                sources[i] = "nudge_dismissed_hurried_override"
                confidence[i] = 0.55
                delay_minutes[i] = float(rng.uniform(0.1, 0.8))
        else:
            u = rng.random()
            if u < noise_rate * 0.45:
                # Cautious user cancelled draft for non-misdirected reason (e.g., typo in body)
                observed_y[i] = 1
                sources[i] = "nudge_cancelled_draft_edit_noise"
                confidence[i] = 0.60
                delay_minutes[i] = float(rng.uniform(0.2, 2.0))
            elif u < 0.04:
                sources[i] = "soc_fp_verified_benign"
                confidence[i] = 1.00
                delay_minutes[i] = float(rng.uniform(15.0, 360.0))
            else:
                sources[i] = "implicit_clean_delivery"
                confidence[i] = 0.90
                delay_minutes[i] = 0.0

    feedback_df = pd.DataFrame(
        {
            "event_id": events_df["event_id"].values,
            "split": splits,
            "true_label": true_y,
            "observed_train_label": observed_y,
            "label_confidence": np.round(confidence, 4),
            "label_source": sources,
            "feedback_delay_minutes": np.round(delay_minutes, 2),
        }
    )
    return feedback_df
