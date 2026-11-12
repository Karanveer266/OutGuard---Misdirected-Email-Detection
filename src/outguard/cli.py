"""Command-line interface for OutGuard-MED."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import uvicorn

from outguard.pipeline import run_full_pipeline


def _default_root() -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "configs" / "default_pipeline.yaml").exists():
        return cwd
    return Path(__file__).resolve().parents[2]


def _cmd_run_all(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    cfg_path = Path(args.config).resolve()
    print(f"[OutGuard-MED] Executing end-to-end pipeline (config={cfg_path.name})...")
    summary = run_full_pipeline(cfg_path, root)

    ablation_path = root / "artifacts" / "ablation_results.json"
    ablation = json.loads(ablation_path.read_text(encoding="utf-8"))

    print("\n=== HYPOTHESIS ABLATION RESULTS (HELD-OUT CHRONOLOGICAL TEST SPLIT) ===")
    header = f"{'Model':<22} | {'PR-AUC':>7} | {'Rec@0.05%FPR':>12} | {'Rec@0.25%FPR':>12} | {'ECE':>7} | {'VIP FPR':>8}"
    print(header)
    print("-" * len(header))
    for m in ablation:
        print(
            f"{m['model_name']:<22} | "
            f"{m['pr_auc']:>7.4f} | "
            f"{m['recall_at_fpr_0_05pct']*100:>11.2f}% | "
            f"{m['recall_at_fpr_0_25pct']*100:>11.2f}% | "
            f"{m['ece']:>7.5f} | "
            f"{m['vip_benign_fpr']*100:>7.3f}%"
        )

    print("\n=== SYSTEM VERIFICATION GATES ===")
    print(f"  * Online/Offline Feature Parity (<1e-9): {summary['online_offline_parity_passed']}")
    print(f"  * ONNX Runtime Numerical Parity (<1e-5): {summary['onnx_parity_passed']}")
    print(f"  * Slice FP/FN Regression Gates Passed  : {summary['all_slice_regression_gates_passed']}")
    print(f"  * Online A/B Ship Recommendation       : {summary['ab_ship_recommendation']}")
    print(f"  * Post-Launch Drift Incident Triaged   : {summary['drift_incident_detected']}")
    print(f"  * Python Online p50 / p99 Latency      : {summary['python_online_latency_ms']['p50_ms']} ms / {summary['python_online_latency_ms']['p99_ms']} ms")
    print(f"  * Total End-to-End Runtime             : {summary['pipeline_runtime_sec']} s")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    req_files = [
        "metrics_summary.json",
        "ablation_results.json",
        "slice_analysis.json",
        "ab_experiment_report.json",
        "drift_incident_report.json",
        "parity_verification.json",
        "model_bundle.json",
        "med_lgbm.onnx",
    ]
    for fname in req_files:
        fpath = root / "artifacts" / fname
        if not fpath.exists():
            print(f"[FAIL] Missing artifact: {fpath}", file=sys.stderr)
            return 1

    summary = json.loads((root / "artifacts" / "metrics_summary.json").read_text(encoding="utf-8"))
    checks = [
        ("Online/Offline Parity", summary["online_offline_parity_passed"]),
        ("ONNX Runtime Parity", summary["onnx_parity_passed"]),
        ("Slice Regression Gates", summary["all_slice_regression_gates_passed"]),
        ("Drift Incident Detector", summary["drift_incident_detected"]),
    ]
    failed = False
    for label, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {label}")
        if not ok:
            failed = True
    return 1 if failed else 0


def _cmd_triage(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    rep_path = root / "artifacts" / "drift_incident_report.json"
    if not rep_path.exists():
        print("Run `outguard run-all` first to generate telemetry.", file=sys.stderr)
        return 1
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    print(f"=== MED ON-CALL INCIDENT TRIAGE: {rep['incident_id']} ===")
    print(f"Impacted Tenant         : {rep['impacted_tenant']}")
    print(f"Root-Cause Domain       : {rep['diagnosed_root_cause_domain']} ({rep['culprit_domain_event_volume']} events)")
    print(f"Tenant Score PSI        : {rep['tenant_score_psi']}")
    print(f"Top Drifted Features    : {json.dumps(rep['top_tenant_drifted_features_psi'])}")
    print(
        f"Benign FPR Mitigation   : {rep['finsight_benign_fpr_pre_mitigation']*100:.2f}% -> "
        f"{rep['finsight_benign_fpr_post_mitigation']*100:.2f}%"
    )
    print(f"Post-Mitigation Recall  : {rep['overall_drift_window_recall_post_mitigation']*100:.2f}%")
    print(f"Actionable Remediation  : {rep['recommended_remediation']}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    from outguard.serving.app import create_app

    app = create_app(artifacts_dir=root / "artifacts")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    root_default = _default_root()
    parser = argparse.ArgumentParser(prog="outguard", description="OutGuard-MED CLI")
    parser.add_argument("--root", default=str(root_default), help="Project root path")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run-all", help="Execute full reproducible pipeline")
    p_run.add_argument(
        "--config",
        default=str(root_default / "configs" / "default_pipeline.yaml"),
        help="YAML config path",
    )
    p_run.set_defaults(func=_cmd_run_all)

    p_ver = sub.add_parser("verify", help="Verify generated artifacts and safety gates")
    p_ver.set_defaults(func=_cmd_verify)

    p_tri = sub.add_parser("triage-incident", help="Inspect post-launch drift & on-call report")
    p_tri.set_defaults(func=_cmd_triage)

    p_srv = sub.add_parser("serve", help="Start FastAPI real-time scoring service")
    p_srv.add_argument("--host", default="0.0.0.0")
    p_srv.add_argument("--port", type=int, default=8080)
    p_srv.set_defaults(func=_cmd_serve)

    parsed = parser.parse_args(argv)
    return int(parsed.func(parsed))


if __name__ == "__main__":
    raise SystemExit(main())
