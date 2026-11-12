"""Enterprise entities, name/nickname lexicons, and DLP payload builders."""

from __future__ import annotations

import numpy as np

TENANTS = [
    ("tenant_finsight", "finsight-cap.com", "FinSightCapital"),
    ("tenant_helios", "helios-bio.com", "HeliosBioLabs"),
    ("tenant_cloudscale", "cloudscale-sys.io", "CloudScaleSystems"),
    ("tenant_apex", "apex-logistics.com", "ApexLogistics"),
]

PARTNER_ORGS = [
    ("northstarbank.com", "NorthstarBank", "finance"),
    ("summitretail.com", "SummitRetail", "commerce"),
    ("kestrelcloud.io", "KestrelCloud", "cloud"),
    ("vanguardmed.org", "VanguardMed", "healthcare"),
    ("aegislegal.com", "AegisLegal", "legal"),
    ("borealenergy.com", "BorealEnergy", "energy"),
    ("crestviewvc.com", "CrestviewVC", "finance"),
    ("cobaltcyber.io", "CobaltCyber", "security"),
    ("meridianhr.com", "MeridianHR", "hr"),
    ("atlasfreight.com", "AtlasFreight", "logistics"),
    ("solarispharma.com", "SolarisPharma", "healthcare"),
    ("kirklandcounsel.com", "KirklandCounsel", "legal"),
    ("stripe-payments.com", "StripePayments", "finance"),
    ("datadog-ops.io", "DatadogOps", "cloud"),
    ("redwoodaudit.com", "RedwoodAudit", "finance"),
    ("pacificsemi.com", "PacificSemi", "tech"),
    ("veritasinsure.com", "VeritasInsure", "finance"),
    ("silverlineair.com", "SilverlineAir", "logistics"),
    ("harborhealth.org", "HarborHealth", "healthcare"),
    ("graniteequity.com", "GraniteEquity", "finance"),
    ("orionrobotics.io", "OrionRobotics", "tech"),
    ("pinnacleadvisors.com", "PinnacleAdvisors", "legal"),
    ("bluewavebio.com", "BlueWaveBio", "healthcare"),
    ("ironcladsec.io", "IroncladSec", "security"),
]

FREEMAIL_DOMAINS = ["gmail.com", "outlook.com", "yahoo.com", "icloud.com"]

FIRST_NAMES_AND_NICKNAMES: list[tuple[str, list[str]]] = [
    ("Michael", ["Mike", "Mick"]),
    ("Elizabeth", ["Liz", "Beth", "Eliza"]),
    ("Robert", ["Bob", "Rob"]),
    ("Margaret", ["Peggy", "Maggie"]),
    ("William", ["Will", "Bill"]),
    ("Alexander", ["Alex", "Sasha"]),
    (" Katherine", ["Kate", "Katie"]),
    ("Christopher", ["Chris"]),
    ("Daniel", ["Dan", "Danny"]),
    ("Jennifer", ["Jen", "Jenny"]),
    ("david", ["Dave"]),
    ("Priya", ["Priya"]),
    ("Elena", ["Elena"]),
    ("Marcus", ["Marc"]),
    ("Sarah", ["Sara"]),
    ("Kevin", ["Kev"]),
    (" Samantha", ["Sam"]),
    ("Benjamin", ["Ben"]),
    ("victoria", ["Vicky", "Tori"]),
    (" Jonathan", ["Jon"]),
    ("Natalie", ["Nat"]),
    (" Jason", ["Jay"]),
    ("Rachel", ["Rach"]),
    ("Thomas", ["Tom"]),
]

CLEAN_FIRST_NAMES: list[tuple[str, list[str]]] = [
    (fn.strip().capitalize(), [n.strip().capitalize() for n in nicks])
    for fn, nicks in FIRST_NAMES_AND_NICKNAMES
]

LAST_NAMES = [
    "Chen", "Patel", "Miller", "Wu", "Kim", "Petrov", "Garcia", "Nguyen",
    "Smith", "Johnson", "Park", "Liu", "Anderson", "Taylor", "berg", "Shah",
    "okafor", "Silva", "Cohen", "Murphy", "Reyes", "walsh", "Cho", "Kowalski",
]
CLEAN_LAST_NAMES = [ln.strip().capitalize() for ln in LAST_NAMES]

DEPARTMENTS = [
    "Finance", "Legal", "Engineering", "Sales",
    "CustomerSuccess", "Recruiting", "Executive", "HR",
]


def make_luhn_card(rng: np.random.Generator) -> str:
    """Generate a syntactically valid Luhn 16-digit payment card string."""
    digits = [4, 5, 3, 2] + [int(x) for x in rng.integers(0, 10, size=11)]
    total = 0
    for idx, d in enumerate(reversed(digits)):
        if idx % 2 == 0:
            dbl = d * 2
            total += dbl - 9 if dbl > 9 else dbl
        else:
            total += d
    check = (10 - (total % 10)) % 10
    full = digits + [check]
    s = "".join(str(x) for x in full)
    return f"{s[:4]}-{s[4:8]}-{s[8:12]}-{s[12:]}"


def mutate_domain_typo(domain: str, rng: np.random.Generator) -> str:
    """Create a realistic edit-distance-1 typosquat/fat-finger domain."""
    name, tld = domain.rsplit(".", 1)
    choice = int(rng.integers(0, 3))
    if choice == 0 and len(name) > 4:
        pos = int(rng.integers(1, len(name) - 1))
        name = name[:pos] + name[pos + 1 :]
    elif choice == 1 and len(name) > 4:
        pos = int(rng.integers(1, len(name) - 2))
        chars = list(name)
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        name = "".join(chars)
    else:
        tld = "co" if tld == "com" else "com"
    return f"{name}.{tld}"


def build_sensitive_snippet(
    level: str, org_brand: str, rng: np.random.Generator
) -> tuple[str, str, list[str]]:
    """Return (subject, body_payload, attachment_names) for requested sensitivity."""
    q = int(rng.integers(1, 5))
    if level == "high_pii":
        ssn = f"{int(rng.integers(200, 899))}-{int(rng.integers(10, 99))}-{int(rng.integers(1000, 9999))}"
        subj = f"CONFIDENTIAL: {org_brand} Q{q} Tax & Payroll Audit Schedule"
        body = (
            f"Please review the reconciled {org_brand} executive verification record "
            f"(Tax/SSN ref {ssn}) and wire ABA routing #021000021 settlement sheet before 5pm."
        )
        att = [f"{org_brand}_Q{q}_Payroll_SSN_Audit_CONFIDENTIAL.xlsx"]
        return subj, body, att

    if level == "high_secret_or_card":
        if rng.random() < 0.55:
            tok = "AKIA" + "".join(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"), size=16))
            subj = f"{org_brand} Production IAM Key Rotation & SOC2 Scope"
            body = (
                f"Attached are the {org_brand} staging/prod integration credentials "
                f"({tok}) and webhook signing configuration for deployment."
            )
            att = [f"{org_brand}_Prod_IAM_Secrets_Internal.json"]
        else:
            card = make_luhn_card(rng)
            subj = f"{org_brand} Corporate Settlement Authorization Q{q}"
            body = (
                f"Authorizing {org_brand} enterprise retainer charge on corporate card "
                f"{card} for $142,500 USD per signed SOW."
            )
            att = [f"{org_brand}_Signed_Retainer_Authorization.pdf"]
        return subj, body, att

    if level == "moderate_commercial":
        price = int(rng.integers(45, 480)) * 1000
        subj = f"{org_brand} Q{q} Enterprise MSA Redlines & Pricing Schedule"
        body = (
            f"Sharing the updated {org_brand} commercial terms (${price:,} annual commitment, "
            f"18% custom discount tier) and legal redlines for final sign-off."
        )
        att = [f"{org_brand}_MSA_Redlines_v{int(rng.integers(2, 7))}.docx"]
        return subj, body, att

    # Low / routine business communication
    topics = [
        (
            f"{org_brand} Weekly Sync & Agenda Notes",
            f"Looking forward to our {org_brand} check-in tomorrow afternoon to review milestone timelines.",
            [],
        ),
        (
            f"Updated timeline for {org_brand} onboarding",
            f"Sharing the high-level kickoff schedule and public documentation links for {org_brand}.",
            [f"{org_brand}_Onboarding_Agenda.pdf"],
        ),
    ]
    return topics[int(rng.integers(0, len(topics)))]
