"""Integration tests for FastAPI real-time scoring API, PSI drift, and A/B CUPED."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from outguard.config import ExperimentationConfig
from outguard.experimentation.ab_testing import run_online_ab_simulation
from outguard.monitoring.drift_and_oncall import compute_psi
from outguard.serving.app import create_app


def test_fastapi_realtime_scoring_and_hot_allowlist() -> None:
    root = Path(__file__).resolve().parents[1]
    app = create_app(artifacts_dir=root / "artifacts")
    client = TestClient(app)

    # 1. Healthcheck
    r_health = client.get("/healthz")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    # 2. High-severity typosquat + wrong-org PII leak -> must trigger NUDGE_USER or HOLD_BLOCK
    leak_payload = {
        "event_id": "evt_live_leak_001",
        "timestamp_epoch": 1_769_900_100.0,
        "tenant_id": "tenant_finsight",
        "tenant_domain": "finsightcap.com",
        "sender_email": "claire.bennett0@finsightcap.com",
        "sender_name": "Claire Bennett",
        "sender_department": "Finance",
        "sender_is_vip": False,
        "recipients": [
            {
                "email": "david.miller@nrothstarbank.com",
                "display_name": "David Miller",
                "domain": "nrothstarbank.com",
                "field": "to",
            }
        ],
        "is_reply": False,
        "thread_prior_messages": 0,
        "subject": "CONFIDENTIAL: SummitRetail Q4 Payroll & W-2 Audit",
        "body": "Hi Michael,\n\nAttached is SummitRetail payroll verification SSN 412-58-9134 and EIN 94-3829104.",
        "attachment_names": ["SummitRetail_Payroll_SSN_Roster_CONFIDENTIAL.xlsx"],
    }
    r_leak = client.post("/v1/score", json=leak_payload)
    assert r_leak.status_code == 200
    body = r_leak.json()
    assert body["action"] in ("NUDGE_USER", "HOLD_BLOCK")
    assert body["remediation_banner"] is not None
    assert len(body["top_risk_factors"]) >= 1

    # 3. Hot-reload allowlist on a newly acquired subsidiary domain -> suppresses alert to ALLOW
    r_allow = client.post(
        "/v1/admin/allowlist",
        json={
            "tenant_id": "tenant_finsight",
            "trusted_subsidiary_domain": "nrothstarbank.com",
        },
    )
    assert r_allow.status_code == 200

    r_after = client.post("/v1/score", json=leak_payload)
    assert r_after.status_code == 200
    assert r_after.json()["action"] == "ALLOW"
    assert r_after.json()["triggered_guardrail"] == "GUARDRAIL_SUBSIDIARY_ALLOWLIST"

    # 4. Feedback ingestion endpoint
    r_fb = client.post(
        "/v1/feedback",
        json={
            "event_id": "evt_live_leak_001",
            "sender_action": "cancelled_and_corrected",
            "notes": "User fixed recipient after banner prompt",
        },
    )
    assert r_fb.status_code == 200
    assert r_fb.json()["total_feedback_events"] == 1


def test_psi_drift_and_cuped_ab_variance_reduction() -> None:
    rng = np.random.default_rng(99)
    ref = rng.normal(0.0, 1.0, size=3000)
    same = rng.normal(0.0, 1.0, size=3000)
    shifted = rng.normal(1.2, 1.1, size=3000)

    assert compute_psi(ref, same) < 0.03
    assert compute_psi(ref, shifted) > 0.45

    ab_res = run_online_ab_simulation(
        ExperimentationConfig(duration_days=14, daily_outbound_volume=20000),
        v1_nudge_recall=0.58,
        v1_nudge_fpr=0.0025,
        v2_nudge_recall=0.97,
        v2_nudge_fpr=0.0018,
    )
    assert ab_res["cuped_variance_reduction_pct"] > 20.0
    assert ab_res["final_p_value"] < 1e-4
    assert ab_res["faulty_canary_rollback_verification"]["rollback_triggered"] is True
