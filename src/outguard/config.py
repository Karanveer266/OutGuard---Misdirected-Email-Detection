"""Typed configuration management for OutGuard-MED."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class SimulationConfig(BaseModel):
    num_tenants: int = 4
    num_employees: int = 480
    num_external_contacts: int = 1400
    warmup_events: int = 8000
    train_events: int = 14000
    val_events: int = 5000
    test_events: int = 8000
    drift_events: int = 3000
    base_misdirected_rate: float = 0.032
    user_nudge_noise_rate: float = 0.015


class FeatureConfig(BaseModel):
    recency_half_life_days: float = 14.0
    min_prefix_len: int = 4
    cold_outreach_prior_smoothing: float = 5.0
    topic_svd_components: int = 12


class LGBMConfig(BaseModel):
    n_estimators: int = 240
    learning_rate: float = 0.045
    num_leaves: int = 23
    max_depth: int = 5
    min_child_samples: int = 35
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    reg_alpha: float = 0.6
    reg_lambda: float = 1.8
    scale_pos_weight: float = 4.0


class ModelConfig(BaseModel):
    lgbm: LGBMConfig = Field(default_factory=LGBMConfig)


class ThresholdConfig(BaseModel):
    block_max_fpr: float = 0.0005
    nudge_max_fpr: float = 0.0025
    bootstrap_rounds: int = 300


class GuardrailConfig(BaseModel):
    recruiter_cold_outreach_rate_min: float = 0.42
    verified_thread_min_prior_exchanges: int = 4
    self_personal_email_name_sim_min: float = 0.88
    hard_block_pii_prefix_gap_min: float = 0.55


class ExperimentationConfig(BaseModel):
    duration_days: int = 14
    daily_outbound_volume: int = 25000
    max_vip_nudge_override_rate: float = 0.35
    max_p99_latency_ms: float = 25.0
    alpha: float = 0.05


class MonitoringConfig(BaseModel):
    psi_warning_threshold: float = 0.12
    psi_critical_threshold: float = 0.25
    alert_rate_spike_multiplier: float = 1.8


class PipelineConfig(BaseModel):
    seed: int = 42
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    guardrails: GuardrailConfig = Field(default_factory=GuardrailConfig)
    experimentation: ExperimentationConfig = Field(default_factory=ExperimentationConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
        return cls.model_validate(raw)
