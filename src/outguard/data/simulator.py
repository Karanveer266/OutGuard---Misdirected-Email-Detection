"""Multi-tenant chronological outbound email & misdirection benchmark simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from outguard.config import PipelineConfig
from outguard.data.catalog import (
    CLEAN_FIRST_NAMES,
    CLEAN_LAST_NAMES,
    DEPARTMENTS,
    FREEMAIL_DOMAINS,
    PARTNER_ORGS,
    TENANTS,
    build_sensitive_snippet,
    mutate_domain_typo,
)
from outguard.data.labeling import synthesize_labeling_workflow
from outguard.schemas import MisdirectionType


@dataclass
class ContactPerson:
    email: str
    display_name: str
    first_name: str
    last_name: str
    nicknames: list[str]
    domain: str
    org_brand: str
    is_internal: bool
    tenant_id: str | None = None
    department: str = "External"
    is_vip: bool = False


def _make_email(first: str, last: str, domain: str, style: int) -> str:
    f, l = first.lower(), last.lower()
    if style == 0:
        return f"{f}.{l}@{domain}"
    if style == 1:
        return f"{f[0]}{l}@{domain}"
    return f"{f}_{l}@{domain}"


def generate_benchmark_dataset(
    cfg: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Simulate chronological enterprise outbound email stream + feedback workflow.

    Returns:
        events_df: Chronological outbound email table (warmup/train/val/test/drift).
        feedback_df: Delayed/noisy multi-source labeling workflow table.
        domain_to_brand: Known enterprise partner domain -> canonical org token map.
    """
    rng = np.random.default_rng(cfg.seed)
    sim = cfg.simulation

    domain_to_brand: dict[str, str] = {d: b for d, b, _ in PARTNER_ORGS}
    for _, t_dom, t_brand in TENANTS:
        domain_to_brand[t_dom] = t_brand

    # 1. Create external contacts across 24 partner enterprises
    ext_contacts: list[ContactPerson] = []
    used_emails: set[str] = set()
    for idx in range(sim.num_external_contacts):
        dom, brand, _ = PARTNER_ORGS[idx % len(PARTNER_ORGS)]
        fn, nicks = CLEAN_FIRST_NAMES[int(rng.integers(0, len(CLEAN_FIRST_NAMES)))]
        ln = CLEAN_LAST_NAMES[int(rng.integers(0, len(CLEAN_LAST_NAMES)))]
        style = int(rng.integers(0, 2))
        email = _make_email(fn, ln, dom, style)
        if email in used_emails:
            email = f"{fn.lower()}.{ln.lower()}{idx % 97}@{dom}"
        used_emails.add(email)
        ext_contacts.append(
            ContactPerson(
                email=email,
                display_name=f"{fn} {ln}",
                first_name=fn,
                last_name=ln,
                nicknames=nicks,
                domain=dom,
                org_brand=brand,
                is_internal=False,
            )
        )

    # 2. Create internal employees per tenant + working contact books + homonym pairs
    employees: list[dict[str, Any]] = []
    tenant_internals: dict[str, list[ContactPerson]] = {t[0]: [] for t in TENANTS}

    for idx in range(sim.num_employees):
        t_id, t_dom, t_brand = TENANTS[idx % len(TENANTS)]
        dept = DEPARTMENTS[idx % len(DEPARTMENTS)]
        is_vip = dept == "Executive" or (dept in ("Legal", "Finance") and idx % 5 == 0)
        fn, nicks = CLEAN_FIRST_NAMES[int(rng.integers(0, len(CLEAN_FIRST_NAMES)))]
        ln = CLEAN_LAST_NAMES[int(rng.integers(0, len(CLEAN_LAST_NAMES)))]
        email = f"{fn.lower()}.{ln.lower()}{idx // len(TENANTS)}@{t_dom}"
        person = ContactPerson(
            email=email,
            display_name=f"{fn} {ln}",
            first_name=fn,
            last_name=ln,
            nicknames=nicks,
            domain=t_dom,
            org_brand=t_brand,
            is_internal=True,
            tenant_id=t_id,
            department=dept,
            is_vip=is_vip,
        )
        tenant_internals[t_id].append(person)

        # Assign 4 primary external contacts and 4 dormant/prefix-colliding contacts
        primary_ext = list(rng.choice(ext_contacts, size=5, replace=False))
        # Ensure at least 2 prefix-sharing / same-first-name external contacts in dormant book
        primary_prefixes = {c.email[:4] for c in primary_ext}
        prefix_matches = [
            c for c in ext_contacts
            if c.email[:4] in primary_prefixes and c.domain not in {p.domain for p in primary_ext}
        ]
        if len(prefix_matches) >= 2:
            dormant_ext = list(rng.choice(prefix_matches, size=2, replace=False))
        else:
            dormant_ext = list(rng.choice(ext_contacts, size=2, replace=False))
        dormant_ext += list(rng.choice(ext_contacts, size=3, replace=False))

        personal_dom = FREEMAIL_DOMAINS[idx % len(FREEMAIL_DOMAINS)]
        personal_contact = ContactPerson(
            email=f"{fn.lower()}.{ln.lower()}@{personal_dom}",
            display_name=f"{fn} {ln}",
            first_name=fn,
            last_name=ln,
            nicknames=nicks,
            domain=personal_dom,
            org_brand="Personal",
            is_internal=False,
        )
        employees.append(
            {
                "person": person,
                "primary_ext": primary_ext,
                "dormant_ext": dormant_ext,
                "personal": personal_contact,
            }
        )

    # Subsidiary domain introduced only in post-launch drift window (Day 91+)
    subsidiary_contacts = [
        ContactPerson(
            email=f"{CLEAN_FIRST_NAMES[i][0].lower()}.{CLEAN_LAST_NAMES[i].lower()}@meridian-analytics.io",
            display_name=f"{CLEAN_FIRST_NAMES[i][0]} {CLEAN_LAST_NAMES[i]}",
            first_name=CLEAN_FIRST_NAMES[i][0],
            last_name=CLEAN_LAST_NAMES[i],
            nicknames=CLEAN_FIRST_NAMES[i][1],
            domain="meridian-analytics.io",
            org_brand="MeridianAnalytics",
            is_internal=False,
        )
        for i in range(18)
    ]

    # 3. Generate chronological stream across 5 splits
    split_counts = [
        ("warmup", sim.warmup_events, 0.0, 30.0),
        ("train", sim.train_events, 30.0, 62.0),
        ("val", sim.val_events, 62.0, 76.0),
        ("test", sim.test_events, 76.0, 90.0),
        ("drift", sim.drift_events, 90.0, 97.0),
    ]

    base_ts = 1_760_000_000.0  # Fixed UTC epoch anchor
    rows: list[dict[str, Any]] = []
    ev_counter = 0

    for split_name, n_events, day_start, day_end in split_counts:
        offsets = np.sort(rng.uniform(day_start * 86400.0, day_end * 86400.0, size=n_events))
        for offset in offsets:
            ev_counter += 1
            ts = base_ts + float(offset)
            emp = employees[int(rng.integers(0, len(employees)))]
            sender: ContactPerson = emp["person"]
            peers = [p for p in tenant_internals[sender.tenant_id] if p.email != sender.email]

            is_misdirected = (
                split_name != "warmup" and rng.random() < sim.base_misdirected_rate
            )

            # Post-launch drift injection: FinSight acquires meridian-analytics.io on Day 90
            if (
                not is_misdirected
                and split_name == "drift"
                and sender.tenant_id == "tenant_finsight"
                and rng.random() < 0.28
            ):
                sub_rec = subsidiary_contacts[int(rng.integers(0, len(subsidiary_contacts)))]
                drift_sens = "high_pii" if rng.random() < 0.35 else "moderate_commercial"
                subj, body_core, att = build_sensitive_snippet(
                    drift_sens, "FinSightCapital", rng
                )
                body = (
                    f"Hi {sub_rec.first_name},\n\nInternal only — sharing post-merger "
                    f"MeridianAnalytics integration package: {body_core}\n\nBest,\n{sender.first_name}"
                )
                rows.append(
                    _pack_row(
                        ev_counter, ts, split_name, sender,
                        [(sub_rec, "to")], False, 0, subj, body, att,
                        0, MisdirectionType.BENIGN.value, "subsidiary_acquisition_shift",
                    )
                )
                continue

            if not is_misdirected:
                row = _generate_benign_event(
                    ev_counter, ts, split_name, emp, peers, ext_contacts, rng
                )
            else:
                row = _generate_misdirected_event(
                    ev_counter, ts, split_name, emp, peers, ext_contacts, rng
                )
            rows.append(row)

    events_df = pd.DataFrame(rows)
    feedback_df = synthesize_labeling_workflow(
        events_df, noise_rate=sim.user_nudge_noise_rate, seed=cfg.seed
    )
    events_df["observed_train_label"] = feedback_df["observed_train_label"].values
    events_df["label_confidence"] = feedback_df["label_confidence"].values
    events_df["label_source"] = feedback_df["label_source"].values
    return events_df, feedback_df, domain_to_brand


def _pack_row(
    ev_id: int,
    ts: float,
    split: str,
    sender: ContactPerson,
    recips: list[tuple[ContactPerson, str]],
    is_reply: bool,
    thread_msgs: int,
    subject: str,
    body: str,
    attachments: list[str],
    true_label: int,
    m_type: str,
    eval_slice: str,
) -> dict[str, Any]:
    return {
        "event_id": f"evt_{ev_id:06d}",
        "timestamp_epoch": round(ts, 3),
        "split": split,
        "tenant_id": sender.tenant_id,
        "tenant_domain": sender.domain,
        "sender_email": sender.email,
        "sender_name": sender.display_name,
        "sender_department": sender.department,
        "sender_is_vip": bool(sender.is_vip),
        "is_reply": bool(is_reply),
        "thread_prior_messages": int(thread_msgs),
        "recipients": [
            {
                "email": r.email,
                "display_name": r.display_name,
                "domain": r.domain,
                "field": fld,
            }
            for r, fld in recips
        ],
        "subject": subject,
        "body": body,
        "attachment_names": attachments,
        "true_label": int(true_label),
        "misdirection_type": m_type,
        "eval_slice": eval_slice,
    }


def _generate_benign_event(
    ev_id: int,
    ts: float,
    split: str,
    emp: dict[str, Any],
    peers: list[ContactPerson],
    ext_contacts: list[ContactPerson],
    rng: np.random.Generator,
) -> dict[str, Any]:
    sender: ContactPerson = emp["person"]
    u = rng.random()

    # Hard benign slice 1: Recruiter / Sales cold outreach with comp/pricing
    if sender.department in ("Recruiting", "Sales") and u < 0.62:
        cand_fn, cand_nicks = CLEAN_FIRST_NAMES[int(rng.integers(0, len(CLEAN_FIRST_NAMES)))]
        cand_ln = CLEAN_LAST_NAMES[int(rng.integers(0, len(CLEAN_LAST_NAMES)))]
        dom = FREEMAIL_DOMAINS[int(rng.integers(0, len(FREEMAIL_DOMAINS)))] if sender.department == "Recruiting" else ext_contacts[int(rng.integers(0, len(ext_contacts)))].domain
        cand = ContactPerson(
            email=f"{cand_fn.lower()}.{cand_ln.lower()}{int(rng.integers(10, 999))}@{dom}",
            display_name=f"{cand_fn} {cand_ln}",
            first_name=cand_fn, last_name=cand_ln, nicknames=cand_nicks,
            domain=dom, org_brand=sender.org_brand, is_internal=False,
        )
        subj = f"{sender.org_brand} Opportunity & Compensation Overview ($185,000 Base)"
        body = f"Hi {cand.first_name},\n\nReaching out from {sender.org_brand} regarding our open role ($185,000 target package + equity). Let me know if you're open to a quick chat!"
        return _pack_row(ev_id, ts, split, sender, [(cand, "to")], False, 0, subj, body, [], 0, MisdirectionType.BENIGN.value, "recruiter_sales_cold_outreach")

    # Hard benign slice 2: VIP / Legal sensitive first-time outreach to new external counsel/partner
    if (sender.is_vip or sender.department in ("Legal", "Finance")) and u < 0.14:
        fresh = ext_contacts[int(rng.integers(0, len(ext_contacts)))]
        subj, body_core, att = build_sensitive_snippet("moderate_commercial", fresh.org_brand, rng)
        body = f"Dear {fresh.first_name},\n\n{body_core}\n\nRegards,\n{sender.display_name}"
        return _pack_row(ev_id, ts, split, sender, [(fresh, "to")], False, 0, subj, body, att, 0, MisdirectionType.BENIGN.value, "vip_first_contact_sensitive")

    # Hard benign slice 3: CC Warm Introduction ("Hi Elena, looping in Marcus...")
    if u < 0.22:
        primary: ContactPerson = emp["primary_ext"][int(rng.integers(0, len(emp["primary_ext"])))]
        intro_guest: ContactPerson = ext_contacts[int(rng.integers(0, len(ext_contacts)))]
        if intro_guest.email != primary.email:
            subj = f"Intro: {primary.first_name} <> {intro_guest.first_name} ({primary.org_brand} / {intro_guest.org_brand})"
            body = (
                f"Hi {primary.first_name},\n\nLooping in {intro_guest.first_name} from "
                f"{intro_guest.org_brand} so you two can coordinate directly on the integration timeline."
            )
            return _pack_row(
                ev_id, ts, split, sender,
                [(primary, "to"), (intro_guest, "cc")],
                False, 0, subj, body, [], 0, MisdirectionType.BENIGN.value, "cc_warm_introduction",
            )

    # Hard benign slice 4: Employee forwarding non-critical doc to own personal freemail
    if u < 0.27:
        pers: ContactPerson = emp["personal"]
        subj = f"Personal copy: {sender.org_brand} Benefits & PTO Schedule"
        body = f"Sending my updated benefits enrollment schedule to personal archive."
        return _pack_row(ev_id, ts, split, sender, [(pers, "to")], False, 0, subj, body, ["Benefits_Guide_2026.pdf"], 0, MisdirectionType.BENIGN.value, "personal_self_forward")

    # Occasional benign visit to dormant contact (establishes realistic sparse prior edge!)
    if u < 0.34:
        dorm: ContactPerson = emp["dormant_ext"][int(rng.integers(0, len(emp["dormant_ext"])))]
        subj, body_core, att = build_sensitive_snippet("routine", dorm.org_brand, rng)
        body = f"Hi {dorm.first_name},\n\n{body_core}"
        return _pack_row(ev_id, ts, split, sender, [(dorm, "to")], False, 0, subj, body, att, 0, MisdirectionType.BENIGN.value, "standard_business")

    # Standard routine internal or primary-external correspondence (including nickname & verified reply threads)
    if u < 0.54:
        peer = peers[int(rng.integers(0, len(peers)))]
        sens = "high_pii" if (sender.department in ("Finance", "HR") and rng.random() < 0.28) else "routine"
        subj, body_core, att = build_sensitive_snippet(sens, sender.org_brand, rng)
        body = f"Hi {peer.first_name},\n\n{body_core}"
        return _pack_row(ev_id, ts, split, sender, [(peer, "to")], True, int(rng.integers(2, 9)), subj, body, att, 0, MisdirectionType.BENIGN.value, "standard_business")

    # Primary external partner correspondence
    target: ContactPerson = emp["primary_ext"][int(rng.integers(0, len(emp["primary_ext"])))]
    sens_roll = rng.random()
    sens = "high_secret_or_card" if sens_roll < 0.12 else ("moderate_commercial" if sens_roll < 0.42 else "routine")
    b_label = target.org_brand if rng.random() < 0.72 else "Client"
    subj, body_core, att = build_sensitive_snippet(sens, b_label, rng)
    # Realistic ecosystem overlap: ~12% of benign external emails reference a cloud/billing vendor
    if rng.random() < 0.12:
        body_core += " (synced via KestrelCloud & DatadogOps connector)."

    use_nick = len(target.nicknames) > 0 and target.nicknames[0] != target.first_name and rng.random() < 0.32
    is_reply = rng.random() < 0.48
    thread_msgs = int(rng.integers(3, 12)) if is_reply else 0

    if use_nick:
        greet_name = target.nicknames[0]
        eval_slice = "nickname_salutation"
        body = f"Hi {greet_name},\n\n{body_core}"
    elif is_reply and rng.random() < 0.45:
        eval_slice = "routine_reply_chain"
        body = f"Thanks — {body_core}"
    else:
        eval_slice = "standard_business"
        body = f"Hi {target.first_name},\n\n{body_core}"

    recips = [(target, "to")]
    if rng.random() < 0.28:
        recips.append((peers[int(rng.integers(0, len(peers)))], "cc"))

    return _pack_row(ev_id, ts, split, sender, recips, is_reply, thread_msgs, subj, body, att, 0, MisdirectionType.BENIGN.value, eval_slice)


def _generate_misdirected_event(
    ev_id: int,
    ts: float,
    split: str,
    emp: dict[str, Any],
    peers: list[ContactPerson],
    ext_contacts: list[ContactPerson],
    rng: np.random.Generator,
) -> dict[str, Any]:
    sender: ContactPerson = emp["person"]
    intended: ContactPerson = emp["primary_ext"][int(rng.integers(0, len(emp["primary_ext"])))]
    archetype_roll = rng.random()

    sens_choices = ["high_pii", "high_secret_or_card", "moderate_commercial", "routine"]
    sens_probs = [0.34, 0.28, 0.30, 0.08]
    sens = str(rng.choice(sens_choices, p=sens_probs))

    # 1. Autocomplete / Homonym Prefix Collision (38%)
    if archetype_roll < 0.38:
        # Pick dormant/external contact sharing email prefix or first name with `intended`
        candidates = [
            c for c in emp["dormant_ext"]
            if c.domain != intended.domain and (c.email[:4] == intended.email[:4] or c.first_name == intended.first_name)
        ]
        wrong = candidates[0] if candidates else emp["dormant_ext"][0]
        if wrong.domain == intended.domain:
            wrong = next(c for c in ext_contacts if c.domain != intended.domain)

        # Realistic subtlety: 58% of emails omit explicit client brand (using recipient's or neutral shared terms)
        brand_label = intended.org_brand if rng.random() < 0.42 else wrong.org_brand
        subj, body_core, att = build_sensitive_snippet(sens, brand_label, rng)
        if brand_label == wrong.org_brand:
            # Strip brand token so body/attachment has neutral enterprise wording (no explicit org tag)
            subj = subj.replace(wrong.org_brand, "Client")
            body_core = body_core.replace(wrong.org_brand, "client")
            att = [a.replace(wrong.org_brand, "Client") for a in att]
        body = f"Hi {intended.first_name},\n\n{body_core}"
        return _pack_row(
            ev_id, ts, split, sender, [(wrong, "to")],
            False, 0, subj, body, att, 1,
            MisdirectionType.AUTOCOMPLETE_COLLISION.value, "autocomplete_prefix_collision",
        )

    # 2. Reply-All / External Thread Leak (24%)
    if archetype_roll < 0.62:
        internal_recips = [(peers[0], "to"), (peers[1], "cc")]
        leaked_ext = emp["dormant_ext"][int(rng.integers(0, len(emp["dormant_ext"])))]
        if rng.random() < 0.35:
            # Freemail personal leak on internal financial/secret thread
            leaked_ext = ContactPerson(
                email=f"contractor.{leaked_ext.last_name.lower()}@gmail.com",
                display_name=leaked_ext.display_name,
                first_name=leaked_ext.first_name, last_name=leaked_ext.last_name,
                nicknames=leaked_ext.nicknames, domain="gmail.com",
                org_brand="External", is_internal=False,
            )
        leak_sens = str(rng.choice(["high_pii", "high_secret_or_card", "moderate_commercial"], p=[0.45, 0.35, 0.20]))
        subj, body_core, att = build_sensitive_snippet(leak_sens, sender.org_brand, rng)
        marker = "Internal only — " if rng.random() < 0.48 else "Team — "
        body = f"Hi {peers[0].first_name},\n\n{marker}{body_core}"
        return _pack_row(
            ev_id, ts, split, sender, internal_recips + [(leaked_ext, "cc")],
            True, 1, subj, body, att, 1,
            MisdirectionType.EXTERNAL_THREAD_LEAK.value, "external_thread_leak",
        )

    # 3. Salutation-Recipient Identity Mismatch (16%)
    if archetype_roll < 0.78:
        wrong = next(
            c for c in ext_contacts
            if c.first_name != intended.first_name and c.domain != intended.domain
        )
        brand_label = intended.org_brand if rng.random() < 0.40 else "Client"
        subj, body_core, att = build_sensitive_snippet(sens, brand_label, rng)
        body = f"Hi {intended.first_name},\n\n{body_core}"
        return _pack_row(
            ev_id, ts, split, sender, [(wrong, "to")],
            False, 0, subj, body, att, 1,
            MisdirectionType.SALUTATION_MISMATCH.value, "salutation_mismatch",
        )

    # 4. Wrong-Tenant Attachment / Content Divergence to Frequent Contact (14%)
    if archetype_roll < 0.92:
        # Sent to a HIGH-AFFINITY known recipient, greeting matches, NO prefix collision,
        # BUT attachment & body contain ANOTHER client's confidential data!
        other_client = next(c for c in emp["primary_ext"] if c.domain != intended.domain)
        leak_sens = str(rng.choice(["high_pii", "high_secret_or_card", "moderate_commercial"], p=[0.42, 0.33, 0.25]))
        subj, body_core, att = build_sensitive_snippet(leak_sens, other_client.org_brand, rng)
        subj = f"Updated Quarterly Reconciliation Package"  # Generic subject hides mistake from sender!
        body = f"Hi {intended.first_name},\n\nPlease find the finalized package attached.\n{body_core}"
        return _pack_row(
            ev_id, ts, split, sender, [(intended, "to")],
            False, 0, subj, body, att, 1,
            MisdirectionType.WRONG_TENANT_DIVERGENCE.value, "wrong_tenant_known_contact",
        )

    # 5. Typosquat / Fat-Finger External Domain (8%)
    typo_dom = mutate_domain_typo(intended.domain, rng)
    typo_recip = ContactPerson(
        email=f"{intended.first_name.lower()}.{intended.last_name.lower()}@{typo_dom}",
        display_name=intended.display_name,
        first_name=intended.first_name, last_name=intended.last_name,
        nicknames=intended.nicknames, domain=typo_dom,
        org_brand="Unknown", is_internal=False,
    )
    brand_label = intended.org_brand if rng.random() < 0.45 else "Client"
    subj, body_core, att = build_sensitive_snippet(sens, brand_label, rng)
    body = f"Hi {intended.first_name},\n\n{body_core}"
    return _pack_row(
        ev_id, ts, split, sender, [(typo_recip, "to")],
        False, 0, subj, body, att, 1,
        MisdirectionType.DOMAIN_TYPO_LEAK.value, "typosquat_domain_leak",
    )
