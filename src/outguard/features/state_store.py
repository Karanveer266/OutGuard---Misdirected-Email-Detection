"""Point-in-time causal communication graph & address-book state store.

Shared identically by the offline Parquet backfill pipeline and the online
real-time scoring engine to eliminate training/serving skew and prevent
future-information (time-travel) leakage.
"""

from __future__ import annotations

import copy
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9]{2,}")
STOP_TOKENS = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "from", "have", "are",
        "was", "were", "will", "your", "our", "please", "attached", "find",
        "review", "updated", "best", "regards", "thanks", "dear", "hello",
        "let", "know", "looking", "forward", "tomorrow", "before", "into",
    }
)


def _zero_triple() -> list[int]:
    return [0, 0, 0]


@dataclass
class EdgeStats:
    count: int = 0
    decay_weight: float = 0.0
    last_ts: float = 0.0
    display_name: str = ""
    domain: str = ""


class PointInTimeGraphState:
    """Strictly causal rolling communication graph and topic profile store."""

    def __init__(
        self,
        half_life_days: float = 14.0,
        cold_outreach_smoothing: float = 5.0,
    ) -> None:
        self.half_life_sec = half_life_days * 86400.0
        self.ln2 = math.log(2.0)
        self.cold_smoothing = cold_outreach_smoothing

        # (sender_email, recipient_email) -> EdgeStats
        self.sender_recipient: dict[tuple[str, str], EdgeStats] = {}
        # sender_email -> set of known external recipient_emails
        self.sender_address_book: dict[str, set[str]] = defaultdict(set)
        # sender_email -> [total_ext_sends, first_time_ext_sends, total_sends]
        self.sender_totals: dict[str, list[int]] = defaultdict(_zero_triple)
        # tenant_id -> Counter(external_domain -> count)
        self.tenant_domains: dict[str, Counter[str]] = defaultdict(Counter)
        self.tenant_ext_total: Counter[str] = Counter()
        # recipient_email -> total inbound-from-org count & Counter(co_recipient_email)
        self.recip_total: Counter[str] = Counter()
        self.co_recip_counts: dict[tuple[str, str], int] = defaultdict(int)
        # (tenant_id, recipient_domain) -> Counter(content_tokens)
        self.domain_topics: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    def decayed_affinity(self, edge: EdgeStats | None, current_ts: float) -> float:
        if edge is None or edge.count == 0:
            return 0.0
        dt = max(0.0, current_ts - edge.last_ts)
        factor = math.exp(-self.ln2 * dt / self.half_life_sec)
        return edge.decay_weight * factor

    def get_edge(self, sender_email: str, recip_email: str) -> EdgeStats | None:
        return self.sender_recipient.get((sender_email, recip_email))

    def get_sender_rates(self, sender_email: str) -> tuple[float, float]:
        """Return (smoothed_cold_outreach_rate, external_send_ratio)."""
        ext_sends, first_ext, total_sends = self.sender_totals[sender_email]
        # Prior expectation of 0.12 cold outreach rate with smoothing
        cold_rate = (first_ext + 0.12 * self.cold_smoothing) / (
            ext_sends + self.cold_smoothing
        )
        ext_ratio = (ext_sends + 1.0) / (total_sends + 2.0)
        return cold_rate, ext_ratio

    def co_recipient_jaccard(self, r1: str, r2: str) -> float:
        if r1 == r2:
            return 1.0
        c1, c2 = self.recip_total[r1], self.recip_total[r2]
        if c1 == 0 or c2 == 0:
            return 0.0
        pair = (r1, r2) if r1 < r2 else (r2, r1)
        inter = self.co_recip_counts.get(pair, 0)
        union = c1 + c2 - inter
        return inter / union if union > 0 else 0.0

    def topic_cosine_similarity(
        self, tenant_id: str, recip_domain: str, text_tokens: list[str]
    ) -> float:
        dom_cnt = self.tenant_domains[tenant_id].get(recip_domain, 0)
        profile = self.domain_topics.get((tenant_id, recip_domain))
        # Return neutral prior (0.35) when domain lacks established topic history
        if dom_cnt < 3 or not profile or not text_tokens:
            return 0.35
        msg_counts = Counter(text_tokens)
        dot = sum(cnt * profile.get(tok, 0) for tok, cnt in msg_counts.items())
        if dot == 0:
            return 0.0
        norm_msg = math.sqrt(sum(v * v for v in msg_counts.values()))
        norm_prof = math.sqrt(sum(v * v for v in profile.values()))
        if norm_msg == 0.0 or norm_prof == 0.0:
            return 0.0
        return dot / (norm_msg * norm_prof)

    @staticmethod
    def tokenize_content(subject: str, body: str, attachments: list[str]) -> list[str]:
        combined = f"{subject} {body} {' '.join(attachments)}".lower()
        return [
            t for t in TOKEN_RE.findall(combined)
            if t not in STOP_TOKENS and not t.isdigit()
        ][:60]

    def update(self, event: dict[str, Any]) -> None:
        """Advance state AFTER point-in-time features have been extracted."""
        ts = float(event["timestamp_epoch"])
        sender = event["sender_email"].lower()
        tenant_id = event["tenant_id"]
        tenant_dom = event["tenant_domain"].lower()
        recips = event["recipients"]

        recip_emails: list[str] = []
        any_ext = False
        any_first_ext = False

        for r in recips:
            r_email = r["email"].lower()
            r_dom = r["domain"].lower()
            r_name = r.get("display_name", "")
            recip_emails.append(r_email)

            key = (sender, r_email)
            existing = self.sender_recipient.get(key)
            is_ext = r_dom != tenant_dom
            if is_ext:
                any_ext = True
                if existing is None or existing.count == 0:
                    any_first_ext = True
                self.sender_address_book[sender].add(r_email)
                self.tenant_domains[tenant_id][r_dom] += 1
                self.tenant_ext_total[tenant_id] += 1

            cur_decay = self.decayed_affinity(existing, ts)
            if existing is None:
                self.sender_recipient[key] = EdgeStats(
                    count=1,
                    decay_weight=cur_decay + 1.0,
                    last_ts=ts,
                    display_name=r_name,
                    domain=r_dom,
                )
            else:
                existing.count += 1
                existing.decay_weight = cur_decay + 1.0
                existing.last_ts = ts
                if r_name:
                    existing.display_name = r_name

        totals = self.sender_totals[sender]
        totals[2] += 1
        if any_ext:
            totals[0] += 1
        if any_first_ext:
            totals[1] += 1

        unique_recips = sorted(set(recip_emails))
        for r_email in unique_recips:
            self.recip_total[r_email] += 1
        for i in range(len(unique_recips)):
            for j in range(i + 1, len(unique_recips)):
                pair = (unique_recips[i], unique_recips[j])
                self.co_recip_counts[pair] += 1

        tokens = self.tokenize_content(
            event.get("subject", ""),
            event.get("body", ""),
            event.get("attachment_names", []),
        )
        if tokens:
            seen_doms = {r["domain"].lower() for r in recips}
            for dom in seen_doms:
                prof = self.domain_topics[(tenant_id, dom)]
                prof.update(tokens)
                # Bound memory per domain profile in long-running online service
                if len(prof) > 250:
                    for k, v in list(prof.items()):
                        if v <= 1:
                            del prof[k]

    def clone_snapshot(self) -> "PointInTimeGraphState":
        clone = PointInTimeGraphState.__new__(PointInTimeGraphState)
        clone.half_life_sec = self.half_life_sec
        clone.ln2 = self.ln2
        clone.cold_smoothing = self.cold_smoothing
        clone.sender_recipient = {
            k: EdgeStats(v.count, v.decay_weight, v.last_ts, v.display_name, v.domain)
            for k, v in self.sender_recipient.items()
        }
        clone.sender_address_book = defaultdict(
            set, {k: set(v) for k, v in self.sender_address_book.items()}
        )
        clone.sender_totals = defaultdict(
            _zero_triple, {k: list(v) for k, v in self.sender_totals.items()}
        )
        clone.tenant_domains = defaultdict(
            Counter, {k: Counter(v) for k, v in self.tenant_domains.items()}
        )
        clone.tenant_ext_total = Counter(self.tenant_ext_total)
        clone.recip_total = Counter(self.recip_total)
        clone.co_recip_counts = defaultdict(int, self.co_recip_counts)
        clone.domain_topics = defaultdict(
            Counter, {k: Counter(v) for k, v in self.domain_topics.items()}
        )
        return clone
