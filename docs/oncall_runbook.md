# Misdirected Email Detection (MED) Shared On-Call Runbook

This runbook governs on-call triage, remediation, and rollback procedures for **OutGuard-MED** real-time scoring and detection efficacy alerts.

---

## 1. Severity Definitions & SLA Targets

| Severity | Trigger Condition | Ack SLA | Mitigation SLA |
| :--- | :--- | :--- | :--- |
| **SEV-1** | Real-time scoring `p99 > 25 ms`, error rate `> 0.5%`, or VIP `HOLD_BLOCK` false-positive loop impacting executive mail flow. | 5 min | 15 min |
| **SEV-2** | Tenant nudge alert rate spikes `>= 2.2x` trailing 7-day baseline or score `PSI >= 0.25` (e.g., unannounced M&A subsidiary domain). | 15 min | 1 hour |
| **SEV-3** | Single customer-reported False Positive (FP) or missed-leak False Negative (FN) escalation from SOC / Customer Success. | 2 hours | 1 business day |

---

## 2. Playbook A: Tenant Alert Spike / Post-M&A Domain Drift (`SEV-2`)

### Symptoms
- Alert `MED_TENANT_SCORE_PSI_HIGH` or `MED_TENANT_ALERT_SPIKE` fires for a specific `tenant_id`.
- High volume of user *"Send Anyway"* overrides concentrated on a single newly observed external domain.

### Automated Triage Command
```bash
python -m outguard.cli triage-incident
```

### Investigation Checklist
1. Inspect `diagnosed_root_cause_domain` and `top_tenant_drifted_features_psi` (`graph_min_org_domain_prior_count`, `graph_any_first_time_external`).
2. Confirm with tenant admin / SOC whether the root-cause domain (e.g., `meridian-analytics.io`) is a newly acquired subsidiary, rebrand, or authorized outsourcing partner.

### Zero-Downtime Mitigation (Hot Allowlist Push)
Push the verified subsidiary domain to the real-time scoring service without model retraining:
```bash
curl -X POST http://localhost:8080/v1/admin/allowlist \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "tenant_finsight", "trusted_subsidiary_domain": "meridian-analytics.io"}'
```
- **Verified Efficacy**: Reduces `tenant_finsight` benign FPR from **8.54% -> 0.00%** immediately while preserving **98.73%** recall on genuine misdirected leaks.
- **Follow-up**: Schedule nightly Databricks/PySpark graph warm-start backfill (`jobs/spark_feature_backfill.py`).

---

## 3. Playbook B: High-Visibility VIP / Executive False Positive (`SEV-1` / `SEV-2`)

### Investigation Steps
1. Query the scoring audit record by `event_id` and inspect:
   - `calibrated_risk_score`, `graph_specialist_prob`, `lexical_specialist_prob`, `dlp_blast_radius`
   - `triggered_guardrail` and `top_risk_factors`
2. Check which common benign edge case occurred:
   - **VIP First-Time Board/M&A Outreach**: High `dlp_composite_blast_radius` + `graph_any_first_time_external = 1`, but greeting matches recipient and zero wrong-org attachment mismatch.
   - **Nickname Salutation**: Sender wrote `"Hi Peggy"` to `margaret.chen@...` using an unmapped regional diminutive -> add pair to `CLEAN_FIRST_NAMES` in `src/outguard/data/catalog.py` + unit test in `tests/test_features_and_leakage.py`.
   - **CC Warm Introduction**: Sender used non-standard intro phrasing -> extend `WARM_INTRO_RE` in `src/outguard/features/lexical.py` and re-run `pytest` + slice regression gates (`python -m outguard.cli verify`).

---

## 4. Playbook C: Canary Regression / Emergency Model Rollback (`SEV-1`)

### Automatic Rollback Triggers
During staged canary/A/B rollouts (`src/outguard/experimentation/ab_testing.py`), the deployment controller automatically aborts and reverts if any sequential guardrail trips:
- `vip_nudge_override_rate > 0.35`
- `p99_scoring_latency_ms > 25.0`

### Manual Rollback Procedure
1. Enable shadow-only mode or revert `model_bundle.json` pointer to the previous pinned artifact digest:
   ```bash
   cp artifacts/archive/model_bundle.prev.json artifacts/model_bundle.json
   systemctl reload outguard-scorer-go
   ```
2. Verify `/healthz` reports `status: "ok"` and run golden parity check:
   ```bash
   cd services/realtime-scorer-go && go test -v -run TestPythonCrossLanguageParity ./...
   ```
