"""Unified point-in-time feature extractor and online/offline parity verifier."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from outguard.config import PipelineConfig
from outguard.features.lexical import extract_lexical_features
from outguard.features.semantic_dlp import extract_semantic_and_dlp_features
from outguard.features.state_store import PointInTimeGraphState
from outguard.schemas import ALL_FEATURES


def extract_graph_features(
    event: dict[str, Any], state: PointInTimeGraphState
) -> dict[str, float]:
    """Compute all 10 point-in-time communication graph features BEFORE state update."""
    ts = float(event["timestamp_epoch"])
    sender = event["sender_email"].lower()
    tenant_id = event["tenant_id"]
    tenant_dom = event["tenant_domain"].lower()
    recips = event["recipients"]

    prior_counts: list[float] = []
    decay_affs: list[float] = []
    days_since_list: list[float] = []
    org_dom_counts: list[float] = []
    org_dom_shares: list[float] = []
    any_first_ext = 0.0

    tenant_total_ext = max(1, state.tenant_ext_total[tenant_id])

    for r in recips:
        r_email = r["email"].lower()
        r_dom = r["domain"].lower()
        edge = state.get_edge(sender, r_email)
        cnt = float(edge.count) if edge is not None else 0.0
        aff = state.decayed_affinity(edge, ts)
        days_since = (
            min(180.0, max(0.0, (ts - edge.last_ts) / 86400.0))
            if (edge is not None and edge.count > 0)
            else 180.0
        )

        prior_counts.append(math.log1p(cnt))
        decay_affs.append(aff)
        days_since_list.append(days_since)

        if r_dom != tenant_dom:
            if cnt == 0.0:
                any_first_ext = 1.0
            dom_cnt = float(state.tenant_domains[tenant_id].get(r_dom, 0))
            org_dom_counts.append(math.log1p(dom_cnt))
            org_dom_shares.append(dom_cnt / tenant_total_ext)
        else:
            org_dom_counts.append(math.log1p(float(tenant_total_ext)))
            org_dom_shares.append(1.0)

    # Co-recipient Jaccard across multi-recipient threads
    if len(recips) <= 1:
        min_jaccard = 1.0
    else:
        emails = [r["email"].lower() for r in recips]
        jaccards = []
        for i in range(len(emails)):
            for j in range(i + 1, len(emails)):
                jaccards.append(state.co_recipient_jaccard(emails[i], emails[j]))
        min_jaccard = min(jaccards) if jaccards else 0.0

    cold_rate, ext_ratio = state.get_sender_rates(sender)
    min_raw_cnt = min((state.get_edge(sender, r["email"].lower()).count if state.get_edge(sender, r["email"].lower()) else 0) for r in recips)
    is_verified_reply = (
        1.0
        if (
            bool(event.get("is_reply", False))
            and int(event.get("thread_prior_messages", 0)) >= 3
            and min_raw_cnt >= 2
        )
        else 0.0
    )

    return {
        "graph_min_recipient_prior_count": round(min(prior_counts), 6),
        "graph_min_recipient_decay_affinity": round(min(decay_affs), 6),
        "graph_min_recipient_days_since_last": round(max(days_since_list), 6),
        "graph_any_first_time_external": any_first_ext,
        "graph_min_org_domain_prior_count": round(min(org_dom_counts), 6),
        "graph_min_org_domain_share": round(min(org_dom_shares), 6),
        "graph_co_recipient_jaccard_min": round(min_jaccard, 6),
        "graph_sender_cold_outreach_rate": round(cold_rate, 6),
        "graph_sender_external_ratio": round(ext_ratio, 6),
        "graph_is_reply_chain_verified": is_verified_reply,
    }


class OutGuardFeatureExtractor:
    """Shared feature extraction engine for both batch backfill and online API."""

    def __init__(
        self,
        cfg: PipelineConfig,
        domain_to_brand: dict[str, str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.domain_to_brand = domain_to_brand

    def create_empty_state(self) -> PointInTimeGraphState:
        return PointInTimeGraphState(
            half_life_days=self.cfg.features.recency_half_life_days,
            cold_outreach_smoothing=self.cfg.features.cold_outreach_prior_smoothing,
        )

    def extract_single_event(
        self,
        event: dict[str, Any],
        state: PointInTimeGraphState,
        mutate_state: bool = False,
    ) -> dict[str, float]:
        """Compute all 34 features at event timestamp. Optionally advance state."""
        g_feats = extract_graph_features(event, state)
        l_feats = extract_lexical_features(
            event, state, min_prefix_len=self.cfg.features.min_prefix_len
        )
        s_feats = extract_semantic_and_dlp_features(
            event, state, domain_to_brand=self.domain_to_brand
        )
        merged = {**g_feats, **l_feats, **s_feats}
        ordered = {k: float(merged[k]) for k in ALL_FEATURES}
        if mutate_state:
            state.update(event)
        return ordered

    def build_offline_feature_table(
        self, events_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, PointInTimeGraphState]]:
        """Run strictly causal chronological feature backfill across all splits.

        Returns:
            features_df: DataFrame aligned with events_df containing ALL_FEATURES + metadata.
            boundary_snapshots: State snapshots captured at split boundaries (e.g., before 'test')
                                for online/offline skew verification and live serving warm-start.
        """
        # Enforce strict chronological ordering
        sorted_df = events_df.sort_values("timestamp_epoch", kind="mergesort").reset_index(drop=True)
        state = self.create_empty_state()
        snapshots: dict[str, PointInTimeGraphState] = {}
        seen_splits: set[str] = set()

        feature_rows: list[dict[str, float]] = []
        records = sorted_df.to_dict(orient="records")

        for rec in records:
            sp = rec["split"]
            if sp not in seen_splits:
                snapshots[f"before_{sp}"] = state.clone_snapshot()
                seen_splits.add(sp)
            feats = self.extract_single_event(rec, state, mutate_state=True)
            feature_rows.append(feats)

        snapshots["final"] = state.clone_snapshot()
        feat_only_df = pd.DataFrame(feature_rows, columns=ALL_FEATURES)
        meta_cols = [
            "event_id",
            "timestamp_epoch",
            "split",
            "tenant_id",
            "sender_email",
            "sender_department",
            "sender_is_vip",
            "true_label",
            "observed_train_label",
            "label_confidence",
            "label_source",
            "misdirection_type",
            "eval_slice",
        ]
        out_df = pd.concat([sorted_df[meta_cols], feat_only_df], axis=1)
        return out_df, snapshots


def verify_online_offline_parity(
    events_df: pd.DataFrame,
    offline_features_df: pd.DataFrame,
    before_test_snapshot: PointInTimeGraphState,
    extractor: OutGuardFeatureExtractor,
    sample_size: int = 500,
) -> dict[str, Any]:
    """Replay test events through the online single-event path and verify exact parity."""
    test_events = (
        events_df[events_df["split"] == "test"]
        .sort_values("timestamp_epoch", kind="mergesort")
        .head(sample_size)
    )
    offline_sub = (
        offline_features_df[offline_features_df["split"] == "test"]
        .sort_values("timestamp_epoch", kind="mergesort")
        .head(sample_size)
    )

    online_state = before_test_snapshot.clone_snapshot()
    online_rows = [
        extractor.extract_single_event(rec, online_state, mutate_state=True)
        for rec in test_events.to_dict(orient="records")
    ]
    online_mat = pd.DataFrame(online_rows, columns=ALL_FEATURES).to_numpy(dtype=np.float64)
    offline_mat = offline_sub[ALL_FEATURES].to_numpy(dtype=np.float64)

    abs_diff = np.abs(online_mat - offline_mat)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))
    per_feature_max = {
        feat: float(np.max(abs_diff[:, idx]))
        for idx, feat in enumerate(ALL_FEATURES)
    }
    return {
        "events_verified": int(len(test_events)),
        "num_features": len(ALL_FEATURES),
        "max_abs_discrepancy": max_diff,
        "mean_abs_discrepancy": mean_diff,
        "parity_passed": bool(max_diff < 1e-9),
        "top_feature_discrepancies": dict(
            sorted(per_feature_max.items(), key=lambda kv: kv[1], reverse=True)[:5]
        ),
    }
