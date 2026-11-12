"""Semantic entity-domain coherence and Luhn/PII/secret DLP feature extractors."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from outguard.data.catalog import FREEMAIL_DOMAINS, PARTNER_ORGS, TENANTS
from outguard.features.state_store import PointInTimeGraphState

SSN_RE = re.compile(r"\b[2-8]\d{2}-\d{2}-\d{4}\b")
CARD_CANDIDATE_RE = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
SECRET_KEY_RE = re.compile(r"\b(?:AKIA[0-9A-Z]{16}|sk_live_[0-9a-zA-Z]{16,})\b")
FINANCIAL_WIRE_RE = re.compile(
    r"\b(?:routing\s*#?\d{6,9}|wire|retainer|discount\s+tier|annual\s+commitment|\$\d{2,3},\d{3})\b",
    re.IGNORECASE,
)
INTERNAL_ONLY_RE = re.compile(r"\binternal\s+only\b", re.IGNORECASE)

CONFIDENTIAL_ATT_KEYWORDS = {
    "confidential": 0.45,
    "ssn": 0.50,
    "payroll": 0.40,
    "secrets": 0.55,
    "iam": 0.35,
    "audit": 0.30,
    "redline": 0.28,
    "retainer": 0.28,
    "internal": 0.35,
}

FREEMAIL_SET = frozenset(FREEMAIL_DOMAINS)

DEFAULT_DOMAIN_TO_BRAND: dict[str, str] = {
    d.lower(): b.lower() for d, b, _ in PARTNER_ORGS
}
for _, _tdom, _tbrand in TENANTS:
    DEFAULT_DOMAIN_TO_BRAND[_tdom.lower()] = _tbrand.lower()


def verify_luhn(card_text: str) -> bool:
    digits = [int(c) for c in card_text if c.isdigit()]
    if len(digits) != 16:
        return False
    total = 0
    for idx, d in enumerate(reversed(digits)):
        if idx % 2 == 1:
            dbl = d * 2
            total += dbl - 9 if dbl > 9 else dbl
        else:
            total += d
    return total % 10 == 0


def extract_semantic_and_dlp_features(
    event: dict[str, Any],
    state: PointInTimeGraphState,
    domain_to_brand: dict[str, str] | None = None,
) -> dict[str, float]:
    """Compute 8 semantic/coherence features and 6 DLP blast-radius features."""
    brand_map = (
        {k.lower(): v.lower() for k, v in domain_to_brand.items()}
        if domain_to_brand
        else DEFAULT_DOMAIN_TO_BRAND
    )
    partner_brands = {b.lower(): d.lower() for d, b, _ in PARTNER_ORGS}

    tenant_id = event["tenant_id"]
    tenant_dom = event["tenant_domain"].lower()
    subject = event.get("subject", "")
    body = event.get("body", "")
    attachments: list[str] = event.get("attachment_names", [])
    recips = event["recipients"]

    dom_counts = Counter(r["domain"].lower() for r in recips)
    total_recips = max(1, len(recips))
    ext_domains = [d for d in dom_counts if d != tenant_dom]
    has_internal = any(d == tenant_dom for d in dom_counts)
    has_external = len(ext_domains) > 0

    entropy = 0.0
    for cnt in dom_counts.values():
        p = cnt / total_recips
        if p > 0:
            entropy -= p * math.log2(p)

    mixed_thread = 1.0 if (has_internal and has_external) else 0.0
    is_freemail_ext = 1.0 if any(d in FREEMAIL_SET for d in ext_domains) else 0.0

    # Allowed brand tokens for the recipients actually on this email
    allowed_brands = {brand_map[d] for d in dom_counts if d in brand_map}
    sender_brand = brand_map.get(tenant_dom, "")

    text_lower = f"{subject} {body}".lower()
    att_lower = " ".join(attachments).lower()

    body_other_org = 0.0
    att_org_mismatch = 0.0

    if has_external:
        for p_brand, _ in partner_brands.items():
            if p_brand in allowed_brands:
                continue
            if p_brand in text_lower:
                body_other_org = 1.0
            if p_brand in att_lower:
                att_org_mismatch = 1.0

        # Check explicit "internal only" leak or internal-secret attachment on external thread
        if INTERNAL_ONLY_RE.search(body):
            body_other_org = 1.0
        if sender_brand and sender_brand in att_lower and ("internal" in att_lower or "ssn" in att_lower):
            att_org_mismatch = 1.0

    # Topic cosine similarity against historical domain communication profile
    tokens = state.tokenize_content(subject, body, attachments)
    if ext_domains:
        cos_sims = [
            state.topic_cosine_similarity(tenant_id, d, tokens) for d in ext_domains
        ]
        min_topic_cos = min(cos_sims)
    else:
        min_topic_cos = state.topic_cosine_similarity(tenant_id, tenant_dom, tokens)

    # Structured DLP detectors
    combined_scan = f"{subject}\n{body}"
    has_ssn = 1.0 if SSN_RE.search(combined_scan) else 0.0
    has_luhn_card = 0.0
    for cand in CARD_CANDIDATE_RE.findall(combined_scan):
        if verify_luhn(cand):
            has_luhn_card = 1.0
            break
    has_secret = 1.0 if SECRET_KEY_RE.search(combined_scan) else 0.0
    has_fin_wire = 1.0 if FINANCIAL_WIRE_RE.search(combined_scan) else 0.0

    att_score = 0.0
    for att in attachments:
        al = att.lower()
        for kw, weight in CONFIDENTIAL_ATT_KEYWORDS.items():
            if kw in al:
                att_score += weight
        if al.endswith((".xlsx", ".json", ".csv")):
            att_score += 0.12
    att_score = min(1.0, att_score)

    blast_radius = min(
        1.0,
        0.42 * has_ssn
        + 0.40 * has_luhn_card
        + 0.45 * has_secret
        + 0.24 * has_fin_wire
        + 0.35 * att_score,
    )

    return {
        "sem_external_domain_count": float(len(ext_domains)),
        "sem_recipient_domain_entropy": round(entropy, 6),
        "sem_mixed_internal_external_thread": mixed_thread,
        "sem_body_mentions_other_known_org": body_other_org,
        "sem_attachment_org_mismatch": att_org_mismatch,
        "sem_topic_recipient_cosine_sim": round(min_topic_cos, 6),
        "sem_recipient_count_log": round(math.log1p(total_recips), 6),
        "sem_is_freemail_external": is_freemail_ext,
        "dlp_has_ssn_or_tax_id": has_ssn,
        "dlp_has_payment_card_luhn": has_luhn_card,
        "dlp_has_api_or_cloud_secret": has_secret,
        "dlp_has_financial_wire_or_pricing": has_fin_wire,
        "dlp_confidential_attachment_score": round(att_score, 6),
        "dlp_composite_blast_radius": round(blast_radius, 6),
    }
