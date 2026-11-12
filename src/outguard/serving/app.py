"""Production FastAPI real-time scoring, shadow telemetry, and feedback service."""

from __future__ import annotations

import pickle
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from outguard.config import PipelineConfig
from outguard.features.extractor import OutGuardFeatureExtractor
from outguard.features.state_store import PointInTimeGraphState
from outguard.models.ensemble import OutGuardHybridEnsemble
from outguard.schemas import ALL_FEATURES, PolicyAction, ScoringRequest, ScoringResponse


class FeedbackPayload(BaseModel):
    event_id: str
    sender_action: str  # e.g., "cancelled_and_corrected", "sent_anyway", "soc_confirmed_fp"
    notes: str = ""


class AllowlistUpdatePayload(BaseModel):
    tenant_id: str
    trusted_subsidiary_domain: str


class RealtimeScoringContext:
    def __init__(
        self,
        cfg: PipelineConfig,
        extractor: OutGuardFeatureExtractor,
        state: PointInTimeGraphState,
        ensemble: OutGuardHybridEnsemble,
        shadow_mode: bool = False,
    ) -> None:
        self.cfg = cfg
        self.extractor = extractor
        self.state = state
        self.ensemble = ensemble
        self.shadow_mode = shadow_mode
        self.tenant_allowlists: dict[str, set[str]] = {}
        self.latencies_ms: deque[float] = deque(maxlen=2000)
        self.feedback_log: list[dict[str, Any]] = []


def create_app(ctx: RealtimeScoringContext | None = None, artifacts_dir: str | Path = "artifacts") -> FastAPI:
    app = FastAPI(
        title="OutGuard-MED Real-Time Scoring Service",
        version="1.0.0",
        description="Low-latency outbound misdirected email detection with calibrated signal fusion & guardrails.",
    )

    state_holder: dict[str, RealtimeScoringContext] = {}
    if ctx is not None:
        state_holder["ctx"] = ctx

    def _get_ctx() -> RealtimeScoringContext:
        if "ctx" not in state_holder:
            pkl_path = Path(artifacts_dir) / "serving_runtime.pkl"
            if not pkl_path.exists():
                raise HTTPException(status_code=503, detail="Model runtime artifact not initialized.")
            with open(pkl_path, "rb") as fh:
                loaded = pickle.load(fh)
            state_holder["ctx"] = loaded
        return state_holder["ctx"]

    @app.get("/healthz")
    def healthcheck() -> dict[str, Any]:
        c = _get_ctx()
        lats = list(c.latencies_ms)
        p99 = float(np.percentile(lats, 99)) if lats else 0.0
        return {
            "status": "ok",
            "model_version": "1.0.0",
            "shadow_mode": c.shadow_mode,
            "scored_requests_window": len(lats),
            "p99_latency_ms": round(p99, 3),
        }

    @app.post("/v1/score", response_model=ScoringResponse)
    def score_outbound_email(req: ScoringRequest) -> ScoringResponse:
        t0 = time.perf_counter()
        c = _get_ctx()
        ev_dict = req.model_dump()

        # Check live hot-reloaded subsidiary domain allowlists (On-Call mitigation)
        allowed_set = c.tenant_allowlists.get(req.tenant_id, set())
        allowlist_hit = any(r.domain.lower() in allowed_set for r in req.recipients)

        # Extract strictly causal point-in-time features
        feat_dict = c.extractor.extract_single_event(ev_dict, c.state, mutate_state=False)
        feat_vec = np.array([feat_dict[k] for k in ALL_FEATURES], dtype=np.float64)

        detailed = c.ensemble.score_single_detailed(
            feat_vec, allowlisted_domain_hit=allowlist_hit
        )
        action: PolicyAction = detailed["action"]

        # Only advance communication graph if email is allowed (not blocked)
        if action == PolicyAction.ALLOW or c.shadow_mode:
            c.state.update(ev_dict)

        effective_action = PolicyAction.ALLOW if c.shadow_mode else action

        banner = None
        if effective_action in (PolicyAction.NUDGE_USER, PolicyAction.HOLD_BLOCK):
            ext_recips = [
                r.email for r in req.recipients if r.domain.lower() != req.tenant_domain.lower()
            ]
            target_str = ", ".join(ext_recips[:2]) if ext_recips else req.recipients[0].email
            reason_str = (
                detailed["top_risk_factors"][0]
                if detailed["top_risk_factors"]
                else "Unusual recipient/content combination"
            )
            prefix = "OUTBOUND HELD" if effective_action == PolicyAction.HOLD_BLOCK else "CONFIRM RECIPIENT"
            banner = f"[{prefix}] Verify '{target_str}': {reason_str}."

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        c.latencies_ms.append(elapsed_ms)

        return ScoringResponse(
            event_id=req.event_id,
            calibrated_risk_score=detailed["calibrated_risk_score"],
            raw_lgbm_prob=detailed["raw_lgbm_prob"],
            graph_specialist_prob=detailed["graph_specialist_prob"],
            lexical_specialist_prob=detailed["lexical_specialist_prob"],
            dlp_blast_radius=round(float(feat_dict["dlp_composite_blast_radius"]), 4),
            action=effective_action,
            triggered_guardrail=detailed["triggered_guardrail"],
            top_risk_factors=detailed["top_risk_factors"],
            remediation_banner=banner,
            latency_ms=round(elapsed_ms, 3),
        )

    @app.post("/v1/feedback")
    def ingest_feedback(payload: FeedbackPayload) -> dict[str, Any]:
        c = _get_ctx()
        entry = payload.model_dump()
        entry["received_at"] = time.time()
        c.feedback_log.append(entry)
        return {"status": "recorded", "total_feedback_events": len(c.feedback_log)}

    @app.post("/v1/admin/allowlist")
    def hot_reload_subsidiary_allowlist(payload: AllowlistUpdatePayload) -> dict[str, Any]:
        c = _get_ctx()
        dom = payload.trusted_subsidiary_domain.lower().strip()
        c.tenant_allowlists.setdefault(payload.tenant_id, set()).add(dom)
        return {
            "status": "updated",
            "tenant_id": payload.tenant_id,
            "allowlisted_domains": sorted(c.tenant_allowlists[payload.tenant_id]),
        }

    return app
