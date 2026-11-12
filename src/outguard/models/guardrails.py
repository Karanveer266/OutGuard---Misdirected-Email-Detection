"""Deterministic precision detectors and safe-launch FP suppression guardrails.

Complements probabilistic ML models to guarantee zero-tolerance protection on
unambiguous high-severity leaks while suppressing high-visibility false positives
on recruiter outreach, CC warm introductions, employee self-archiving, and
verified ongoing threads.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from outguard.config import GuardrailConfig
from outguard.schemas import ALL_FEATURES

FEAT_IDX = {name: idx for idx, name in enumerate(ALL_FEATURES)}


def apply_guardrails_single(
    feat_vec: np.ndarray,
    base_prob: float,
    cfg: GuardrailConfig,
    nudge_threshold: float,
    block_threshold: float,
    allowlisted_domains_hit: bool = False,
) -> tuple[float, str | None]:
    """Apply deterministic rules & suppression guardrails to a single 34-d feature vector."""
    if allowlisted_domains_hit:
        return min(base_prob * 0.12, nudge_threshold * 0.45), "GUARDRAIL_SUBSIDIARY_ALLOWLIST"

    has_ssn = feat_vec[FEAT_IDX["dlp_has_ssn_or_tax_id"]] > 0.5
    has_secret = feat_vec[FEAT_IDX["dlp_has_api_or_cloud_secret"]] > 0.5
    has_card = feat_vec[FEAT_IDX["dlp_has_payment_card_luhn"]] > 0.5
    hard_dlp = has_ssn or has_secret or has_card
    blast = float(feat_vec[FEAT_IDX["dlp_composite_blast_radius"]])

    prefix_gap = float(feat_vec[FEAT_IDX["lex_autocomplete_prefix_collision_gap"]])
    decay_aff = float(feat_vec[FEAT_IDX["graph_min_recipient_decay_affinity"]])
    lookalike = feat_vec[FEAT_IDX["lex_domain_is_lookalike_unseen"]] > 0.5
    att_mismatch = feat_vec[FEAT_IDX["sem_attachment_org_mismatch"]] > 0.5
    body_other_org = feat_vec[FEAT_IDX["sem_body_mentions_other_known_org"]] > 0.5
    sal_mismatch = feat_vec[FEAT_IDX["lex_salutation_mismatch_flag"]] > 0.5
    warm_intro = feat_vec[FEAT_IDX["lex_has_cc_warm_intro"]] > 0.5
    self_webmail = float(feat_vec[FEAT_IDX["lex_self_personal_webmail_match"]])
    cold_rate = float(feat_vec[FEAT_IDX["graph_sender_cold_outreach_rate"]])
    verified_reply = feat_vec[FEAT_IDX["graph_is_reply_chain_verified"]] > 0.5
    mixed_thread = feat_vec[FEAT_IDX["sem_mixed_internal_external_thread"]] > 0.5

    first_ext = feat_vec[FEAT_IDX["graph_any_first_time_external"]] > 0.5

    # 1. Precision-first escalation detectors
    if (
        hard_dlp
        and first_ext
        and prefix_gap >= cfg.hard_block_pii_prefix_gap_min
        and (body_other_org or att_mismatch or sal_mismatch)
    ):
        floor = max(block_threshold + 0.04, 0.94)
        return max(base_prob, min(0.995, floor)), "RULE_CRITICAL_PII_PREFIX_COLLISION"

    if lookalike and first_ext and blast >= 0.22:
        floor = max(block_threshold + 0.02, 0.90) if hard_dlp else max(nudge_threshold + 0.05, 0.78)
        return max(base_prob, min(0.99, floor)), "RULE_LOOKALIKE_TYPOSQUAT_SENSITIVE"

    if att_mismatch and hard_dlp and not warm_intro and (first_ext or body_other_org):
        floor = max(block_threshold + 0.03, 0.92)
        return max(base_prob, min(0.992, floor)), "RULE_WRONG_ORG_ATTACHMENT_PII"

    # 2. Safe-launch FP suppression guardrails
    if (
        self_webmail >= cfg.self_personal_email_name_sim_min
        and not hard_dlp
        and not att_mismatch
        and not body_other_org
    ):
        return base_prob * 0.12, "GUARDRAIL_SELF_PERSONAL_WEBMAIL"

    if (
        warm_intro
        and not hard_dlp
        and not att_mismatch
        and not sal_mismatch
    ):
        return base_prob * 0.18, "GUARDRAIL_CC_WARM_INTRO_EXEMPTION"

    if (
        cold_rate >= cfg.recruiter_cold_outreach_rate_min
        and not hard_dlp
        and not att_mismatch
        and not body_other_org
        and not sal_mismatch
        and not lookalike
    ):
        return base_prob * 0.10, "GUARDRAIL_HIGH_COLD_OUTREACH_ROLE"

    if (
        verified_reply
        and not mixed_thread
        and not att_mismatch
        and not body_other_org
        and not sal_mismatch
    ):
        return base_prob * 0.22, "GUARDRAIL_VERIFIED_REPLY_THREAD"

    org_dom_cnt = float(feat_vec[FEAT_IDX["graph_min_org_domain_prior_count"]])
    if (
        (decay_aff >= 0.35 or org_dom_cnt >= 3.6)
        and not mixed_thread
        and not att_mismatch
        and not body_other_org
        and not sal_mismatch
        and not lookalike
    ):
        return base_prob * 0.22, "GUARDRAIL_ESTABLISHED_PARTNER_COHERENT"

    return base_prob, None


def apply_guardrails_batch(
    X: np.ndarray,
    base_probs: np.ndarray,
    cfg: GuardrailConfig,
    nudge_threshold: float,
    block_threshold: float,
) -> tuple[np.ndarray, list[str | None]]:
    """Apply guardrails identically across an evaluation matrix."""
    adj = np.empty_like(base_probs, dtype=np.float64)
    rules: list[str | None] = []
    for i in range(len(base_probs)):
        p, r = apply_guardrails_single(
            X[i], float(base_probs[i]), cfg, nudge_threshold, block_threshold
        )
        adj[i] = p
        rules.append(r)
    return adj, rules


def explain_top_risk_factors(feat_vec: np.ndarray) -> list[str]:
    """Generate human-readable risk explanations for SOC & end-user nudge banners."""
    reasons: list[str] = []
    if feat_vec[FEAT_IDX["lex_autocomplete_prefix_collision_gap"]] >= 0.35:
        reasons.append("Recipient shares prefix/name with a higher-affinity contact on another domain")
    if feat_vec[FEAT_IDX["sem_attachment_org_mismatch"]] > 0.5:
        reasons.append("Attachment filename references a different organization than recipient domain")
    if feat_vec[FEAT_IDX["sem_body_mentions_other_known_org"]] > 0.5:
        reasons.append("Email body references internal-only or third-party organization context")
    if feat_vec[FEAT_IDX["lex_salutation_mismatch_flag"]] > 0.5:
        reasons.append("Personal salutation does not match any recipient name or known nickname")
    if feat_vec[FEAT_IDX["lex_domain_is_lookalike_unseen"]] > 0.5:
        reasons.append("Unseen external domain is within 1-2 edits of a trusted partner domain")
    if (
        feat_vec[FEAT_IDX["sem_mixed_internal_external_thread"]] > 0.5
        and feat_vec[FEAT_IDX["graph_co_recipient_jaccard_min"]] < 0.05
    ):
        reasons.append("External recipient has near-zero historical co-occurrence with internal thread")
    if feat_vec[FEAT_IDX["dlp_composite_blast_radius"]] >= 0.35:
        reasons.append("Payload contains high-sensitivity PII, credentials, payment card, or confidential doc")
    if feat_vec[FEAT_IDX["graph_any_first_time_external"]] > 0.5 and not reasons:
        reasons.append("First-time outbound communication to external recipient")
    return reasons[:3]
