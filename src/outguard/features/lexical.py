"""Lexical, identity, nickname, autocomplete-collision, and typosquat algorithms."""

from __future__ import annotations

from functools import lru_cache
import re
from typing import Any

from outguard.data.catalog import CLEAN_FIRST_NAMES, FREEMAIL_DOMAINS
from outguard.features.state_store import PointInTimeGraphState

SALUTATION_RE = re.compile(
    r"^\s*(?:hi|hello|dear|hey|good\s+(?:morning|afternoon|evening))\s+"
    r"(?:dr\.?\s+|mr\.?\s+|ms\.?\s+|mrs\.?\s+|prof\.?\s+)?([A-Za-z][A-Za-z\-]{1,22})\b",
    re.IGNORECASE,
)

GENERIC_SALUTATIONS = frozenset(
    {"team", "all", "everyone", "folks", "there", "partners", "colleagues", "board", "group"}
)

WARM_INTRO_RE = re.compile(
    r"\b(?:looping\s+in|introducing|adding|cc'?ing|connecting\s+you\s+with)\s+([A-Za-z]{2,20})\b",
    re.IGNORECASE,
)

FREEMAIL_SET = frozenset(FREEMAIL_DOMAINS)


def _build_nickname_equivalence() -> dict[str, set[str]]:
    equiv: dict[str, set[str]] = {}
    for canonical, nicks in CLEAN_FIRST_NAMES:
        group = {canonical.lower()} | {n.lower() for n in nicks}
        for member in group:
            equiv.setdefault(member, set()).update(group)
    return equiv


NICKNAME_EQUIV = _build_nickname_equivalence()


@lru_cache(maxsize=65536)
def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1) -> float:
    """Compute exact Jaro-Winkler string similarity in [0.0, 1.0]."""
    a, b = s1.lower().strip(), s2.lower().strip()
    if a == b:
        return 1.0 if a else 0.0
    len_a, len_b = len(a), len(b)
    if len_a == 0 or len_b == 0:
        return 0.0

    match_dist = max(len_a, len_b) // 2 - 1
    if match_dist < 0:
        match_dist = 0

    a_matches = [False] * len_a
    b_matches = [False] * len_b
    matches = 0
    transpositions = 0

    for i in range(len_a):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, len_b)
        for j in range(start, end):
            if b_matches[j] or a[i] != b[j]:
                continue
            a_matches[i] = True
            b_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len_a):
        if not a_matches[i]:
            continue
        while not b_matches[k]:
            k += 1
        if a[i] != b[k]:
            transpositions += 1
        k += 1

    jaro = (
        (matches / len_a)
        + (matches / len_b)
        + ((matches - transpositions / 2.0) / matches)
    ) / 3.0

    prefix = 0
    for c1, c2 in zip(a[:4], b[:4]):
        if c1 == c2:
            prefix += 1
        else:
            break
    return min(1.0, jaro + prefix * p * (1.0 - jaro))


@lru_cache(maxsize=65536)
def damerau_levenshtein(s1: str, s2: str, max_cap: int = 6) -> int:
    """Compute Optimal String Alignment (Damerau-Levenshtein) edit distance."""
    a, b = s1.lower(), s2.lower()
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) >= max_cap:
        return max_cap

    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j

    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            val = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
            if (
                i > 1
                and j > 1
                and a[i - 1] == b[j - 2]
                and a[i - 2] == b[j - 1]
            ):
                val = min(val, dp[i - 2][j - 2] + 1)
            dp[i][j] = val
    return min(max_cap, dp[la][lb])


def extract_lexical_features(
    event: dict[str, Any],
    state: PointInTimeGraphState,
    min_prefix_len: int = 4,
) -> dict[str, float]:
    """Compute all 10 lexical & identity-alignment features causally."""
    ts = float(event["timestamp_epoch"])
    sender_email = event["sender_email"].lower()
    sender_name = event.get("sender_name", "").strip()
    tenant_id = event["tenant_id"]
    tenant_dom = event["tenant_domain"].lower()
    body = event.get("body", "")
    recips = event["recipients"]

    # 1. Salutation extraction & nickname-aware matching
    sal_match = SALUTATION_RE.search(body[:160])
    greet_tok = sal_match.group(1).lower() if sal_match else ""
    has_personal_sal = 1.0 if (greet_tok and greet_tok not in GENERIC_SALUTATIONS) else 0.0

    best_name_sim = 1.0
    nickname_matched = 0.0
    if has_personal_sal > 0.5:
        best_name_sim = 0.0
        greet_equivs = NICKNAME_EQUIV.get(greet_tok, {greet_tok})
        for r in recips:
            disp = r.get("display_name", "").strip()
            parts = [p.lower() for p in re.split(r"[\s._\-]+", disp) if p]
            local_parts = [
                p.lower()
                for p in re.split(r"[._\-0-9]+", r["email"].split("@")[0])
                if p
            ]
            for cand in parts + local_parts:
                sim = jaro_winkler_similarity(greet_tok, cand)
                if sim > best_name_sim:
                    best_name_sim = sim
                if cand in greet_equivs:
                    nickname_matched = 1.0
                    best_name_sim = max(best_name_sim, 0.97)

    sal_mismatch = 1.0 if (has_personal_sal > 0.5 and best_name_sim < 0.80) else 0.0

    # 2. CC warm introduction check ("looping in Marcus...")
    intro_match = WARM_INTRO_RE.search(body[:350])
    has_cc_warm_intro = 0.0
    if intro_match:
        intro_name = intro_match.group(1).lower()
        intro_equivs = NICKNAME_EQUIV.get(intro_name, {intro_name})
        for r in recips:
            if r.get("field", "to") in ("cc", "to"):
                first = r.get("display_name", "").split(" ")[0].lower()
                if first in intro_equivs or jaro_winkler_similarity(intro_name, first) > 0.88:
                    has_cc_warm_intro = 1.0
                    break

    # 3. Autocomplete prefix & homonym collision against sender's causal address book
    known_book = state.sender_address_book.get(sender_email, set())
    max_collision_gap = 0.0
    same_fn_diff_dom = 0.0

    for r in recips:
        r_email = r["email"].lower()
        r_dom = r["domain"].lower()
        if r_dom == tenant_dom:
            continue
        r_aff = state.decayed_affinity(state.get_edge(sender_email, r_email), ts)
        r_prefix = r_email[:min_prefix_len]
        r_first = r.get("display_name", "").split(" ")[0].lower()

        for cand_email in known_book:
            if cand_email == r_email:
                continue
            cand_edge = state.get_edge(sender_email, cand_email)
            if cand_edge is None or cand_edge.domain == r_dom:
                continue
            cand_aff = state.decayed_affinity(cand_edge, ts)
            if cand_aff <= r_aff:
                continue

            shares_prefix = cand_email[:min_prefix_len] == r_prefix
            cand_first = cand_edge.display_name.split(" ")[0].lower() if cand_edge.display_name else ""
            shares_first = bool(r_first and cand_first and r_first == cand_first)

            if shares_prefix or shares_first:
                gap = (cand_aff - r_aff) / (cand_aff + 1.2)
                if gap > max_collision_gap:
                    max_collision_gap = gap
                if shares_first:
                    same_fn_diff_dom = 1.0

    # 4. Typosquat / fat-finger domain edit distance against tenant's trusted domains
    trusted_domains = [
        dom for dom, cnt in state.tenant_domains[tenant_id].items() if cnt >= 3
    ] + list(FREEMAIL_SET) + [tenant_dom]

    min_typo_dist = 6.0
    is_lookalike_unseen = 0.0
    for r in recips:
        r_dom = r["domain"].lower()
        if r_dom == tenant_dom or r_dom in FREEMAIL_SET:
            continue
        prior_dom_cnt = state.tenant_domains[tenant_id].get(r_dom, 0)
        for t_dom in trusted_domains:
            if t_dom == r_dom:
                continue
            d = damerau_levenshtein(r_dom, t_dom, max_cap=6)
            if 1 <= d < min_typo_dist:
                min_typo_dist = float(d)
            if d <= 2 and prior_dom_cnt == 0:
                is_lookalike_unseen = 1.0

    # 5. Self personal webmail match (benign employee self-forwarding vs contractor leak)
    self_webmail_match = 0.0
    sender_local_clean = re.sub(r"[0-9]+", "", sender_email.split("@")[0])
    for r in recips:
        r_dom = r["domain"].lower()
        if r_dom in FREEMAIL_SET:
            r_local_clean = re.sub(r"[0-9]+", "", r["email"].lower().split("@")[0])
            name_sim = jaro_winkler_similarity(
                sender_name, r.get("display_name", "")
            )
            local_sim = jaro_winkler_similarity(sender_local_clean, r_local_clean)
            if max(name_sim, local_sim) >= 0.88 and len(recips) == 1:
                self_webmail_match = max(self_webmail_match, max(name_sim, local_sim))

    return {
        "lex_has_personal_salutation": has_personal_sal,
        "lex_salutation_best_name_sim": round(best_name_sim, 6),
        "lex_salutation_nickname_match": nickname_matched,
        "lex_salutation_mismatch_flag": sal_mismatch,
        "lex_has_cc_warm_intro": has_cc_warm_intro,
        "lex_autocomplete_prefix_collision_gap": round(max_collision_gap, 6),
        "lex_autocomplete_same_firstname_diff_domain": same_fn_diff_dom,
        "lex_domain_typosquat_min_dist": min_typo_dist,
        "lex_domain_is_lookalike_unseen": is_lookalike_unseen,
        "lex_self_personal_webmail_match": round(self_webmail_match, 6),
    }
