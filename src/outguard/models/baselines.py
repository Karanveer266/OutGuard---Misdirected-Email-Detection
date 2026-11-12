"""Baseline models (H0, H1, H2) and standardized domain-specialist scorers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from outguard.schemas import (
    ALL_FEATURES,
    DLP_FEATURES,
    GRAPH_FEATURES,
    LEXICAL_FEATURES,
    SEMANTIC_FEATURES,
)

FEAT_IDX = {name: idx for idx, name in enumerate(ALL_FEATURES)}


class LegacyDLPRuleBaseline:
    """H0: Traditional static DLP + first-time-external heuristic score.

    Represents legacy outbound DLP appliances that combine static sensitivity
    rules with naive first-time-recipient and exact-name checks (lacking
    communication graph decay, cold-outreach role normalization, nickname
    resolution, or autocomplete prefix collision modeling).
    """

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        first_ext = X[:, FEAT_IDX["graph_any_first_time_external"]]
        freemail = X[:, FEAT_IDX["sem_is_freemail_external"]]
        blast = X[:, FEAT_IDX["dlp_composite_blast_radius"]]
        fin = X[:, FEAT_IDX["dlp_has_financial_wire_or_pricing"]]
        has_sal = X[:, FEAT_IDX["lex_has_personal_salutation"]]
        # Legacy regex greeting check ignores nickname equivalence!
        nick_used = X[:, FEAT_IDX["lex_salutation_nickname_match"]]
        sal_mismatch = X[:, FEAT_IDX["lex_salutation_mismatch_flag"]]
        naive_sal_flag = np.clip(sal_mismatch + 0.85 * nick_used * has_sal, 0.0, 1.0)

        raw_logit = (
            -4.10
            + 2.15 * first_ext
            + 1.10 * freemail
            + 2.85 * blast
            + 0.90 * fin
            + 1.65 * naive_sal_flag
        )
        return 1.0 / (1.0 + np.exp(-np.clip(raw_logit, -20.0, 20.0)))


class StandardizedLogisticSpecialist:
    """L2-regularized logistic scorer over a specific feature subset.

    Exports exact mean/scale/weights/intercept so the Go real-time service
    computes identical specialist probabilities to double-precision tolerance.
    """

    def __init__(self, feature_names: list[str], C: float = 0.8, seed: int = 42) -> None:
        self.feature_names = list(feature_names)
        self.col_indices = [FEAT_IDX[f] for f in self.feature_names]
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(
            C=C,
            max_iter=600,
            class_weight={0: 1.0, 1: 3.5},
            random_state=seed,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "StandardizedLogisticSpecialist":
        X_sub = X[:, self.col_indices]
        X_scaled = self.scaler.fit_transform(X_sub)
        self.clf.fit(X_scaled, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_sub = X[:, self.col_indices]
        X_scaled = self.scaler.transform(X_sub)
        return self.clf.predict_proba(X_scaled)[:, 1]

    def export_dict(self) -> dict[str, Any]:
        return {
            "feature_indices": self.col_indices,
            "feature_names": self.feature_names,
            "mean": [float(x) for x in self.scaler.mean_],
            "scale": [float(x) for x in self.scaler.scale_],
            "coef": [float(x) for x in self.clf.coef_[0]],
            "intercept": float(self.clf.intercept_[0]),
        }


def build_specialist_suite(seed: int = 42) -> dict[str, StandardizedLogisticSpecialist]:
    """Create the 3 orthogonal specialist scorers used for signal averaging."""
    return {
        "graph_specialist": StandardizedLogisticSpecialist(GRAPH_FEATURES, C=0.8, seed=seed),
        "lexical_coherence_specialist": StandardizedLogisticSpecialist(
            LEXICAL_FEATURES + SEMANTIC_FEATURES, C=0.8, seed=seed + 1
        ),
        "dlp_context_specialist": StandardizedLogisticSpecialist(
            DLP_FEATURES + SEMANTIC_FEATURES, C=0.8, seed=seed + 2
        ),
    }
