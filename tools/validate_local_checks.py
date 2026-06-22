from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any

from qr_common import ROOT
from summarize_suite import load_jsonl
from validate_b200_suite import EXPECTED_DEFAULT_STEPS as EXPECTED_B200_DEFAULT_STEPS


REQUIRED_FILES = [
    "manifest.jsonl",
    "run.log",
    "secret_audit.jsonl",
    "runtime_preflight.jsonl",
    "submission_validation.jsonl",
    "candidate_policy_public.jsonl",
    "candidate_implementation_status.jsonl",
    "b200_dry_run_plan.json",
    "b200_next_required_dry_run_plan.json",
]

EXPECTED_STEPS = [
    "print_spec",
    "sync_cases_check",
    "secret_audit",
    "runtime_preflight_allow_failure",
    "submission_validation",
    "candidate_policy",
    "candidate_implementation_status",
    "b200_suite_dry_run",
    "b200_next_required_dry_run",
    "pytest",
]


def _error(errors: list[dict[str, str]], check: str, message: str, file_name: str | None = None) -> None:
    row = {"check": check, "message": message}
    if file_name is not None:
        row["file"] = file_name
    errors.append(row)


def _load_jsonl(path: Path, errors: list[dict[str, str]], file_name: str) -> list[dict[str, Any]]:
    if not path.is_file():
        _error(errors, "file.exists", "required file is missing", file_name)
        return []
    try:
        rows = load_jsonl(path)
    except Exception as exc:
        _error(errors, "jsonl.parse", f"{type(exc).__name__}: {exc}", file_name)
        return []
    if not rows:
        _error(errors, "jsonl.nonempty", "JSONL file is empty", file_name)
    return rows


def _load_json(path: Path, errors: list[dict[str, str]], file_name: str) -> dict[str, Any]:
    if not path.is_file():
        _error(errors, "file.exists", "required file is missing", file_name)
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        _error(errors, "json.parse", f"{type(exc).__name__}: {exc}", file_name)
        return {}


def _is_positive(value: Any) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _resolve_tarball(suite_dir: Path, tarball: str) -> Path:
    path = Path(tarball)
    if path.is_absolute():
        return path
    for candidate in [suite_dir / path, ROOT / path]:
        if candidate.is_file():
            return candidate
    return suite_dir / path


def validate_required_files(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    for file_name in REQUIRED_FILES:
        if not (suite_dir / file_name).is_file():
            _error(errors, "file.exists", "required file is missing", file_name)


def validate_tarball(suite_dir: Path, tarball_path: Path, manifest_rows: list[dict[str, Any]], errors: list[dict[str, str]]) -> None:
    try:
        with tarfile.open(tarball_path, "r:*") as tar:
            members = {member.name: member for member in tar.getmembers()}
            for file_name in REQUIRED_FILES:
                member_name = f"{suite_dir.name}/{file_name}"
                member = members.get(member_name)
                if member is None:
                    _error(errors, "tarball.file", f"tarball is missing {member_name}", file_name)
                    continue
                if not member.isfile():
                    _error(errors, "tarball.file", f"tarball member is not a file: {member_name}", file_name)
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    _error(errors, "tarball.file", f"tarball member could not be read: {member_name}", file_name)
                    continue
                if file_name == "manifest.jsonl":
                    archived_rows = [json.loads(line) for line in extracted.read().decode("utf-8").splitlines() if line]
                    if archived_rows != manifest_rows:
                        _error(errors, "tarball.manifest", "archived manifest does not match on-disk manifest", file_name)
                else:
                    if extracted.read() != (suite_dir / file_name).read_bytes():
                        _error(errors, "tarball.file", f"tarball content does not match {file_name}", file_name)
    except Exception as exc:
        _error(errors, "tarball.open", f"{type(exc).__name__}: {exc}")


def validate_manifest(suite_dir: Path, errors: list[dict[str, str]], require_finish: bool) -> list[dict[str, Any]]:
    rows = _load_jsonl(suite_dir / "manifest.jsonl", errors, "manifest.jsonl")
    if not rows:
        return []
    if rows[0].get("event") != "local_checks_start":
        _error(errors, "manifest.start", "first manifest row must be local_checks_start", "manifest.jsonl")
    if any(row.get("event") == "local_checks_failed" for row in rows):
        _error(errors, "manifest.status", "local_checks_failed event is present", "manifest.jsonl")

    starts: dict[str, int] = {}
    finishes: dict[str, int] = {}
    for index, row in enumerate(rows):
        event = row.get("event")
        step = row.get("step")
        if event == "start":
            starts[str(step)] = index
        elif event == "finish":
            finishes[str(step)] = index
            if not _is_positive(row.get("elapsed_s")):
                _error(errors, "manifest.elapsed", f"finish row for {step} is missing positive elapsed_s", "manifest.jsonl")

    expected = list(EXPECTED_STEPS)
    if rows[0].get("skip_pytest") is True:
        expected.remove("pytest")
    for step in expected:
        if step not in starts:
            _error(errors, "manifest.steps", f"start row for {step} is missing", "manifest.jsonl")
        if step not in finishes:
            _error(errors, "manifest.steps", f"finish row for {step} is missing", "manifest.jsonl")
        if step in starts and step in finishes and finishes[step] < starts[step]:
            _error(errors, "manifest.steps", f"finish row appears before start row for {step}", "manifest.jsonl")

    if not require_finish:
        return rows

    finish = rows[-1]
    if finish.get("event") != "local_checks_finish":
        _error(errors, "manifest.finish", "final manifest row must be local_checks_finish", "manifest.jsonl")
        return rows
    if not _is_positive(finish.get("elapsed_s")):
        _error(errors, "manifest.elapsed", "local_checks_finish is missing positive elapsed_s", "manifest.jsonl")
    tarball = finish.get("tarball")
    if not isinstance(tarball, str) or not tarball:
        _error(errors, "manifest.tarball", "local_checks_finish tarball path is missing", "manifest.jsonl")
        return rows
    tarball_path = _resolve_tarball(suite_dir, tarball)
    if not tarball_path.is_file():
        _error(errors, "manifest.tarball", f"tarball does not exist: {tarball}", "manifest.jsonl")
    else:
        validate_tarball(suite_dir, tarball_path, rows, errors)
    return rows


def validate_secret_audit(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    rows = _load_jsonl(suite_dir / "secret_audit.jsonl", errors, "secret_audit.jsonl")
    if not rows:
        return
    summary = rows[-1]
    try:
        num_findings = int(summary.get("num_findings", -1))
    except (TypeError, ValueError):
        num_findings = -1
    if not summary.get("summary") or not summary.get("ok") or num_findings != 0:
        _error(errors, "secret_audit.ok", "secret audit must end with an ok zero-finding summary", "secret_audit.jsonl")


def validate_runtime_preflight(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    rows = _load_jsonl(suite_dir / "runtime_preflight.jsonl", errors, "runtime_preflight.jsonl")
    if len(rows) != 1:
        _error(errors, "runtime_preflight.count", f"expected one runtime preflight row, got {len(rows)}", "runtime_preflight.jsonl")
        return
    row = rows[0]
    if "ok" not in row:
        _error(errors, "runtime_preflight.schema", "runtime preflight row is missing ok", "runtime_preflight.jsonl")
    if not isinstance(row.get("torch"), dict):
        _error(errors, "runtime_preflight.schema", "runtime preflight row is missing torch info", "runtime_preflight.jsonl")
    if "errors" not in row:
        _error(errors, "runtime_preflight.schema", "runtime preflight row is missing errors", "runtime_preflight.jsonl")


def validate_submission(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    rows = _load_jsonl(suite_dir / "submission_validation.jsonl", errors, "submission_validation.jsonl")
    if len(rows) != 1:
        _error(errors, "submission_validation.count", f"expected one submission validation row, got {len(rows)}", "submission_validation.jsonl")
        return
    row = rows[0]
    if not row.get("ok") or row.get("errors"):
        _error(errors, "submission_validation.ok", "submission validation must pass without errors", "submission_validation.jsonl")


def validate_policy(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    rows = _load_jsonl(suite_dir / "candidate_policy_public.jsonl", errors, "candidate_policy_public.jsonl")
    if len(rows) != 12:
        _error(errors, "policy.count", f"expected 12 policy rows, got {len(rows)}", "candidate_policy_public.jsonl")
    if rows and not any(row.get("shape_collision") for row in rows):
        _error(errors, "policy.shape_collision", "policy rows must include public benchmark shape collisions", "candidate_policy_public.jsonl")


def validate_implementation_status(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    rows = _load_jsonl(
        suite_dir / "candidate_implementation_status.jsonl",
        errors,
        "candidate_implementation_status.jsonl",
    )
    cases = [row for row in rows if not row.get("summary")]
    summary = next((row for row in rows if row.get("summary")), None)
    if len(cases) != 12:
        _error(
            errors,
            "implementation_status.count",
            f"expected 12 implementation status rows, got {len(cases)}",
            "candidate_implementation_status.jsonl",
        )
    if not isinstance(summary, dict):
        _error(errors, "implementation_status.summary", "implementation status summary is missing", "candidate_implementation_status.jsonl")
    elif summary.get("num_cases") != len(cases):
        _error(errors, "implementation_status.summary", "summary num_cases does not match rows", "candidate_implementation_status.jsonl")


def validate_b200_dry_run(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    plan = _load_json(suite_dir / "b200_dry_run_plan.json", errors, "b200_dry_run_plan.json")
    if not plan:
        return
    if plan.get("dry_run") is not True:
        _error(errors, "b200_plan.dry_run", "B200 plan must be a dry run", "b200_dry_run_plan.json")
    if plan.get("will_validate_suite") is not True or plan.get("will_validate_completed_export") is not True:
        _error(errors, "b200_plan.validation", "default B200 plan must enable suite and post-export validation", "b200_dry_run_plan.json")
    names = [row.get("step") for row in plan.get("steps", []) if isinstance(row, dict)]
    for step in EXPECTED_B200_DEFAULT_STEPS:
        if step not in names:
            _error(errors, "b200_plan.steps", f"B200 dry-run plan is missing {step}", "b200_dry_run_plan.json")
    if names and names[:4] != ["print_spec", "sync_cases_check", "secret_audit", "runtime_preflight"]:
        _error(
            errors,
            "b200_plan.steps",
            "B200 dry-run plan must start with spec sync, secret audit, and runtime preflight",
            "b200_dry_run_plan.json",
        )
    if "suite_summary" in names and "suite_analysis" in names and "suite_validation" in names:
        if not names.index("suite_summary") < names.index("suite_analysis") < names.index("suite_validation"):
            _error(
                errors,
                "b200_plan.steps",
                "B200 dry-run plan must summarize and analyze before suite validation",
                "b200_dry_run_plan.json",
            )
    blockers = plan.get("validation_blockers")
    if blockers not in ([], None):
        _error(errors, "b200_plan.validation", "default B200 plan must have no validation blockers", "b200_dry_run_plan.json")

    workload = plan.get("workload")
    if not isinstance(workload, dict):
        _error(errors, "b200_plan.workload", "B200 dry-run plan must include workload metadata", "b200_dry_run_plan.json")
        return
    if not _is_positive(workload.get("num_gpu_heavy_steps")):
        _error(errors, "b200_plan.workload", "B200 dry-run plan must include GPU-heavy steps", "b200_dry_run_plan.json")
    for key in ["num_benchmark_timing_steps", "num_route_ablation_timing_steps", "num_verifier_experiment_steps"]:
        if not _is_positive(workload.get(key)):
            _error(errors, "b200_plan.workload", f"B200 workload is missing positive {key}", "b200_dry_run_plan.json")
    estimate = workload.get("runtime_estimate")
    if not isinstance(estimate, dict) or not _is_positive(estimate.get("high_minutes")):
        _error(errors, "b200_plan.workload", "B200 dry-run plan must include a positive runtime estimate", "b200_dry_run_plan.json")


def _required_kernel_priority(case_index: int, row: dict[str, Any]) -> tuple[int, int]:
    batch = int(row.get("batch") or 0)
    n = int(row.get("n") or 0)
    if batch == 640 and n == 512:
        return (0, case_index)
    if batch == 60 and n == 1024:
        return (1, case_index)
    if n in {2048, 4096}:
        return (2, case_index)
    return (3, case_index)


def _expected_next_required_target(suite_dir: Path, errors: list[dict[str, str]]) -> dict[str, Any] | None:
    rows = _load_jsonl(suite_dir / "candidate_policy_public.jsonl", errors, "candidate_policy_public.jsonl")
    candidates = [
        (case_index, row)
        for case_index, row in enumerate(rows)
        if row.get("required_cuda_kernel") and row.get("candidate_config_shape_label")
    ]
    if not candidates:
        _error(
            errors,
            "b200_next_required_plan.target",
            "candidate policy has no required CUDA kernel target",
            "candidate_policy_public.jsonl",
        )
        return None
    case_index, row = sorted(candidates, key=lambda item: _required_kernel_priority(*item))[0]
    return {
        "case_index": case_index,
        "shape_label": row.get("candidate_config_shape_label"),
        "env_prefix": row.get("candidate_config_env_prefix"),
        "benchmark_indices": row.get("candidate_config_benchmark_indices", ""),
        "correctness_indices": row.get("candidate_config_correctness_indices", ""),
        "required_cuda_kernel": row.get("required_cuda_kernel"),
        "required_repair_modes": row.get("required_repair_modes", []),
    }


def _repair_mode_axis_constraints(required_modes: Any) -> dict[str, str]:
    constraints: dict[str, str] = {}
    for raw in required_modes if isinstance(required_modes, list) else []:
        if not isinstance(raw, str) or "=" not in raw:
            continue
        key, value = [part.strip() for part in raw.split("=", 1)]
        if key == "panel_refresh_mode" and value:
            constraints["panel_refresh_modes"] = value
        elif key == "r_maintenance_mode" and value:
            constraints["r_maintenance_modes"] = value
    return constraints


def validate_b200_next_required_dry_run(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    file_name = "b200_next_required_dry_run_plan.json"
    plan = _load_json(suite_dir / file_name, errors, file_name)
    if not plan:
        return
    if plan.get("dry_run") is not True:
        _error(errors, "b200_next_required_plan.dry_run", "next-required B200 plan must be a dry run", file_name)
    if plan.get("will_validate_suite") is not True or plan.get("will_validate_completed_export") is not True:
        _error(
            errors,
            "b200_next_required_plan.validation",
            "next-required B200 plan must enable suite and post-export validation",
            file_name,
        )
    blockers = plan.get("validation_blockers")
    if blockers not in ([], None):
        _error(errors, "b200_next_required_plan.validation", "next-required B200 plan must have no validation blockers", file_name)

    expected = _expected_next_required_target(suite_dir, errors)
    target = plan.get("candidate_config_tune_policy_target")
    if not isinstance(target, dict):
        _error(errors, "b200_next_required_plan.target", "next-required plan is missing policy target", file_name)
        return
    if expected is not None:
        for key, expected_value in expected.items():
            if target.get(key) != expected_value:
                _error(
                    errors,
                    "b200_next_required_plan.target",
                    f"policy target {key}={target.get(key)!r} does not match expected {expected_value!r}",
                    file_name,
                )
    if target.get("source") != "candidate_policy":
        _error(errors, "b200_next_required_plan.target", "policy target source must be candidate_policy", file_name)
    if target.get("large_kernel_plan_mode") != "current-candidate":
        _error(errors, "b200_next_required_plan.target", "next-required plan must default to current-candidate mode", file_name)
    repair_constraints = _repair_mode_axis_constraints(target.get("required_repair_modes"))
    applied_constraints = target.get("applied_axis_constraints")
    if repair_constraints:
        if applied_constraints != repair_constraints:
            _error(
                errors,
                "b200_next_required_plan.target",
                "policy target applied_axis_constraints must match required repair modes",
                file_name,
            )

    generated = plan.get("candidate_config_tune_large_kernel_plan")
    if not isinstance(generated, dict):
        _error(errors, "b200_next_required_plan.generated_configs", "generated config preview is missing", file_name)
        return
    if generated.get("shape_label") != target.get("shape_label"):
        _error(errors, "b200_next_required_plan.generated_configs", "generated config shape does not match target", file_name)
    if generated.get("mode") != target.get("large_kernel_plan_mode"):
        _error(errors, "b200_next_required_plan.generated_configs", "generated config mode does not match target", file_name)
    configs = generated.get("configs")
    names = generated.get("config_names")
    if not isinstance(configs, list) or not configs:
        _error(errors, "b200_next_required_plan.generated_configs", "generated config preview must include config rows", file_name)
    if not isinstance(names, list) or len(names) != len(configs or []):
        _error(errors, "b200_next_required_plan.generated_configs", "generated config names must match config rows", file_name)
    if generated.get("num_configs") != len(configs or []):
        _error(errors, "b200_next_required_plan.generated_configs", "generated num_configs does not match config rows", file_name)
    path = str(generated.get("path") or "")
    if not path.endswith("candidate_config_tune_large_kernel_configs.jsonl"):
        _error(errors, "b200_next_required_plan.generated_configs", "generated config path is not the expected JSONL", file_name)
    if repair_constraints and isinstance(configs, list):
        env_prefix = str(target.get("env_prefix") or generated.get("env_prefix") or "")
        key_by_axis = {
            "panel_refresh_modes": f"{env_prefix}_PANEL_REFRESH_MODE",
            "r_maintenance_modes": f"{env_prefix}_R_MAINTENANCE_MODE",
        }
        for axis, expected_value in repair_constraints.items():
            key = key_by_axis.get(axis)
            if not key or key.startswith("_"):
                _error(
                    errors,
                    "b200_next_required_plan.generated_configs",
                    f"cannot validate generated config axis {axis}: env prefix is missing",
                    file_name,
                )
                continue
            bad = [
                str(row.get("name") or index)
                for index, row in enumerate(configs)
                if not isinstance(row, dict)
                or not isinstance(row.get("env"), dict)
                or row["env"].get(key) != expected_value
            ]
            if bad:
                _error(
                    errors,
                    "b200_next_required_plan.generated_configs",
                    f"generated configs must set {key}={expected_value}; bad configs: {', '.join(bad[:4])}",
                    file_name,
                )

    step_names = [row.get("step") for row in plan.get("steps", []) if isinstance(row, dict)]
    if "candidate_config_tune" not in step_names:
        _error(errors, "b200_next_required_plan.steps", "next-required plan must include candidate_config_tune", file_name)
    if target.get("shape_label") in {"qr512", "qr1024"}:
        if "candidate_config_accelerator_preflight" not in step_names:
            _error(errors, "b200_next_required_plan.steps", "next-required QR512/QR1024 plan must include accelerator preflight", file_name)
        elif "candidate_config_tune" in step_names and step_names.index("candidate_config_accelerator_preflight") > step_names.index(
            "candidate_config_tune"
        ):
            _error(errors, "b200_next_required_plan.steps", "accelerator preflight must run before candidate config tune", file_name)
        preflight = next(
            (row for row in plan.get("steps", []) if isinstance(row, dict) and row.get("step") == "candidate_config_accelerator_preflight"),
            {},
        )
        if "--family-cases" not in preflight.get("cmd", []):
            _error(errors, "b200_next_required_plan.steps", "accelerator preflight must include --family-cases", file_name)


def validate_run_log(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    path = suite_dir / "run.log"
    if not path.is_file():
        _error(errors, "run_log.exists", "run.log is missing", "run.log")
        return
    text = path.read_text(errors="replace")
    for snippet in ["tools/audit_secrets.py", "tools/check_b200_env.py", "tools/validate_submission.py"]:
        if snippet not in text:
            _error(errors, "run_log.coverage", f"run.log does not include {snippet}", "run.log")


def validate_local_checks(suite_dir: Path, require_finish: bool = True) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    validate_required_files(suite_dir, errors)
    validate_run_log(suite_dir, errors)
    validate_manifest(suite_dir, errors, require_finish=require_finish)
    validate_secret_audit(suite_dir, errors)
    validate_runtime_preflight(suite_dir, errors)
    validate_submission(suite_dir, errors)
    validate_policy(suite_dir, errors)
    validate_implementation_status(suite_dir, errors)
    validate_b200_dry_run(suite_dir, errors)
    validate_b200_next_required_dry_run(suite_dir, errors)
    return {
        "ok": not errors,
        "suite_dir": str(suite_dir),
        "num_errors": len(errors),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a completed local non-CUDA checks export.")
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    suite_dir = ROOT / args.suite_dir if not Path(args.suite_dir).is_absolute() else Path(args.suite_dir)
    result = validate_local_checks(suite_dir, require_finish=not args.allow_incomplete)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        status = "PASS" if result["ok"] else "FAIL"
        print(f"{status}: {suite_dir}")
        for error in result["errors"]:
            file_part = f" [{error['file']}]" if "file" in error else ""
            print(f"- {error['check']}{file_part}: {error['message']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
