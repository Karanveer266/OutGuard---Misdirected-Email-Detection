"""Canonical schemas and feature definitions for OutGuard-MED."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class PolicyAction(str, Enum):
    ALLOW = "ALLOW"
    NUDGE_USER = "NUDGE_USER"
    HOLD_BLOCK = "HOLD_BLOCK"


class MisdirectionType(str, Enum):
    BENIGN = "BENIGN"
    AUTOCOMPLETE_COLLISION = "AUTOCOMPLETE_COLLISION"
    EXTERNAL_THREAD_LEAK = "EXTERNAL_THREAD_LEAK"
    SALUTATION_MISMATCH = "SALUTATION_MISMATCH"
    WRONG_TENANT_DIVERGENCE = "WRONG_TENANT_DIVERGENCE"
    DOMAIN_TYPO_LEAK = "DOMAIN_TYPO_LEAK"


GRAPH_FEATURES: list[str] = [
    "graph_min_recipient_prior_count",
    "graph_min_recipient_decay_affinity",
    "graph_min_recipient_days_since_last",
    "graph_any_first_time_external",
    "graph_min_org_domain_prior_count",
    "graph_min_org_domain_share",
    "graph_co_recipient_jaccard_min",
    "graph_sender_cold_outreach_rate",
    "graph_sender_external_ratio",
    "graph_is_reply_chain_verified",
]

LEXICAL_FEATURES: list[str] = [
    "lex_has_personal_salutation",
    "lex_salutation_best_name_sim",
    "lex_salutation_nickname_match",
    "lex_salutation_mismatch_flag",
    "lex_has_cc_warm_intro",
    "lex_autocomplete_prefix_collision_gap",
    "lex_autocomplete_same_firstname_diff_domain",
    "lex_domain_typosquat_min_dist",
    "lex_domain_is_lookalike_unseen",
    "lex_self_personal_webmail_match",
]

SEMANTIC_FEATURES: list[str] = [
    "sem_external_domain_count",
    "sem_recipient_domain_entropy",
    "sem_mixed_internal_external_thread",
    "sem_body_mentions_other_known_org",
    "sem_attachment_org_mismatch",
    "sem_topic_recipient_cosine_sim",
    "sem_recipient_count_log",
    "sem_is_freemail_external",
]

DLP_FEATURES: list[str] = [
    "dlp_has_ssn_or_tax_id",
    "dlp_has_payment_card_luhn",
    "dlp_has_api_or_cloud_secret",
    "dlp_has_financial_wire_or_pricing",
    "dlp_confidential_attachment_score",
    "dlp_composite_blast_radius",
]

ALL_FEATURES: list[str] = (
    GRAPH_FEATURES + LEXICAL_FEATURES + SEMANTIC_FEATURES + DLP_FEATURES
)
assert len(ALL_FEATURES) == 34, f"Expected 34 features, got {len(ALL_FEATURES)}"


class RecipientInfo(BaseModel):
    email: str
    display_name: str
    domain: str
    field: Literal["to", "cc", "bcc"] = "to"


class OutboundEmailEvent(BaseModel):
    event_id: str
    timestamp_epoch: float
    split: Literal["warmup", "train", "val", "test", "drift"]
    tenant_id: str
    tenant_domain: str
    sender_email: str
    sender_name: str
    sender_department: str
    sender_is_vip: bool = False
    is_reply: bool = False
    thread_prior_messages: int = 0
    recipients: list[RecipientInfo]
    subject: str
    body: str
    attachment_names: list[str] = Field(default_factory=list)
    # Ground truth & evaluation slice metadata (never exposed to feature extractor!)
    true_label: int = 0
    observed_train_label: int = 0
    label_confidence: float = 1.0
    label_source: str = "organic_clean"
    misdirection_type: str = MisdirectionType.BENIGN.value
    eval_slice: str = "standard_business"


class ScoringRequest(BaseModel):
    event_id: str
    timestamp_epoch: float
    tenant_id: str
    tenant_domain: str
    sender_email: str
    sender_name: str
    sender_department: str = "General"
    sender_is_vip: bool = False
    is_reply: bool = False
    thread_prior_messages: int = 0
    recipients: list[RecipientInfo]
    subject: str
    body: str
    attachment_names: list[str] = Field(default_factory=list)


class ScoringResponse(BaseModel):
    event_id: str
    calibrated_risk_score: float
    raw_lgbm_prob: float
    graph_specialist_prob: float
    lexical_specialist_prob: float
    dlp_blast_radius: float
    action: PolicyAction
    triggered_guardrail: str | None = None
    top_risk_factors: list[str] = Field(default_factory=list)
    remediation_banner: str | None = None
    latency_ms: float = 0.0
