"""Tests for SOC threshold selection, deterministic guardrails, and exported artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from outguard.config import GuardrailConfig
from outguard.models.guardrails import FEAT_IDX, apply_guardrails_single
from outguard.models.thresholds import optimize_soc_thresholds
from outguard.schemas import ALL_FEATURES


def test_neyman_pearson_soc_threshold_budget() -> None:
    rng = np.random.default_rng(7)
    y_val = np.array([0] * 4000 + [1] * 200)
    neg_scores = rng.beta(1.0, 20.0, size=4000)
    pos_scores = rng.beta(8.0, 1.5, size=200)
    scores = np.concatenate([neg_scores, pos_scores])

    thresholds = optimize_soc_thresholds(
        y_val, scores, nudge_max_fpr=0.0025, block_max_fpr=0.0005
    )
    assert thresholds.val_nudge_fpr <= 0.0025 + 1e-9
    assert thresholds.val_block_fpr <= 0.0005 + 1e-9
    assert thresholds.block_threshold > thresholds.nudge_threshold
    assert thresholds.val_nudge_recall > 0.80


def test_deterministic_escalation_and_suppression_guardrails() -> None:
    cfg = GuardrailConfig()
    vec = np.zeros(len(ALL_FEATURES), dtype=np.float64)

    # 1. Typosquat unseen domain + sensitive payload -> escalates above nudge
    vec[FEAT_IDX["lex_domain_is_lookalike_unseen"]] = 1.0
    vec[FEAT_IDX["graph_any_first_time_external"]] = 1.0
    vec[FEAT_IDX["dlp_composite_blast_radius"]] = 0.55
    p_esc, rule_esc = apply_guardrails_single(vec, 0.25, cfg, 0.40, 0.82)
    assert rule_esc == "RULE_LOOKALIKE_TYPOSQUAT_SENSITIVE"
    assert p_esc >= 0.78

    # 2. Employee self-personal webmail without hard PII -> suppressed below nudge
    vec2 = np.zeros(len(ALL_FEATURES), dtype=np.float64)
    vec2[FEAT_IDX["lex_self_personal_webmail_match"]] = 0.96
    p_sup, rule_sup = apply_guardrails_single(vec2, 0.55, cfg, 0.40, 0.82)
    assert rule_sup == "GUARDRAIL_SELF_PERSONAL_WEBMAIL"
    assert p_sup < 0.10

    # 3. Subsidiary M&A allowlist hit -> suppressed cleanly
    p_sub, rule_sub = apply_guardrails_single(
        vec, 0.91, cfg, 0.40, 0.82, allowlisted_domains_hit=True
    )
    assert rule_sub == "GUARDRAIL_SUBSIDIARY_ALLOWLIST"
    assert p_sub < 0.40


def test_pipeline_artifacts_and_parity_reports_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = json.loads((root / "artifacts" / "metrics_summary.json").read_text())
    assert summary["online_offline_parity_passed"] is True
    assert summary["onnx_parity_passed"] is True
    assert summary["all_slice_regression_gates_passed"] is True
    assert summary["champion_h4_metrics"]["pr_auc"] > 0.95
    assert summary["champion_h4_metrics"]["recall_at_fpr_0_25pct"] > 0.92
