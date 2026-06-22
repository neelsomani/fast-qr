from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

from qr_common import ROOT, TRACKED_RUNTIME_ENV_KEYS, file_sha256
from summarize_suite import (
    ABLATION_FILES,
    BLOCKED_QR_SWEEP_FILE,
    CANDIDATE_CONFIG_TUNE_SUMMARY,
    DEFAULT_CANDIDATE_FILE,
    GUARD_OVERHEAD_FILE,
    PAIRS,
    TAIL_POLICY_TUNE_SUMMARY,
    load_jsonl,
)


CANDIDATE_CONFIG_ACCELERATOR_PREFLIGHT_FILE = "candidate_config_accelerator_preflight.jsonl"
CANDIDATE_CONFIG_LARGE_KERNEL_PLAN_EVENT = "candidate_config_tune_large_kernel_plan"
CANDIDATE_CONFIG_ACCELERATOR_FAMILY_CASES = {
    "qr512_cuda": {"dense", "mixed", "rankdef", "clustered"},
    "qr512_blocked_cuda": {"dense", "mixed", "rankdef", "clustered"},
    "qr512_blocked_cuda_auto": {"dense", "mixed", "rankdef", "clustered"},
    "qr1024_cuda": {"dense", "mixed", "nearrank"},
    "qr1024_blocked_cuda": {"dense", "mixed", "nearrank"},
    "qr1024_blocked_cuda_auto": {"dense", "mixed", "nearrank"},
    "qr2048_blocked_cuda": {"dense", "rankdef", "mixed"},
    "qr2048_blocked_cuda_auto": {"dense", "rankdef", "mixed"},
    "qr4096_blocked_cuda": {"dense", "upper"},
    "qr4096_blocked_cuda_auto": {"dense", "upper"},
}

REQUIRED_DEFAULT_FILES = [
    "manifest.jsonl",
    "run.log",
    "secret_audit.jsonl",
    "runtime_preflight.jsonl",
    "submission_validation.jsonl",
    "candidate_public_tests.jsonl",
    "candidate_public_benchmark_correctness.jsonl",
    "candidate_dev_robustness.jsonl",
    "accelerator_preflight.jsonl",
    "candidate_policy_public.jsonl",
    "candidate_implementation_status.jsonl",
    "candidate_route_trace_public.jsonl",
    GUARD_OVERHEAD_FILE,
    "seed_sweep_margin.jsonl",
    "quantization_seed_sweep.jsonl",
    "mixed_seed_sweep.jsonl",
    "classifier_seed_sweep.jsonl",
    "candidate_tail_policy_sweep.jsonl",
    "experiments_public_benchmarks.jsonl",
    "suite_summary.json",
    "suite_summary.md",
    "suite_analysis.json",
    "suite_analysis.md",
    *[pair[0] for pair in PAIRS.values()],
    *[pair[1] for pair in PAIRS.values()],
    *ABLATION_FILES.values(),
]

BENCHMARK_FILES = [
    *[pair[0] for pair in PAIRS.values()],
    *[pair[1] for pair in PAIRS.values()],
    *ABLATION_FILES.values(),
]

POPCORN_TEST_FILES = [
    "popcorn_test/manifest.jsonl",
    "popcorn_test/submission.py",
    "popcorn_test/submission_validation.jsonl",
    "popcorn_test/popcorn.log",
]
POPCORN_LEADERBOARD_FILES = [
    "popcorn_leaderboard/manifest.jsonl",
    "popcorn_leaderboard/submission.py",
    "popcorn_leaderboard/submission_validation.jsonl",
    "popcorn_leaderboard/popcorn.log",
]

EXPECTED_EXPERIMENTS = {
    "r-projection",
    "fp16-nearby",
    "tf32-input-nearby",
    "tail-delete",
    "column-major",
    "classify",
}

EXPECTED_DEFAULT_STEPS = [
    "print_spec",
    "sync_cases_check",
    "secret_audit",
    "runtime_preflight",
    "submission_validation",
    "candidate_policy",
    "candidate_implementation_status",
    "candidate_route_trace",
    "candidate_guard_overhead",
    "pytest",
    "seed_sweep_margin",
    "quantization_seed_sweep",
    "mixed_seed_sweep",
    "classifier_seed_sweep",
    "tail_policy_sweep",
    "candidate_public_tests",
    "candidate_public_benchmark_correctness",
    "candidate_dev_robustness",
    "accelerator_preflight",
    "baseline_smoke",
    "candidate_smoke",
    "baseline_public",
    "candidate_public",
    "candidate_ablation_no_route_cache",
    "candidate_ablation_cuda_first_structured_routes",
    "candidate_ablation_no_structured_routes",
    "candidate_ablation_no_dense_tail",
    "candidate_ablation_no_data_dependent_routes",
    "candidate_ablation_no_qr512_qr1024_cuda",
    *[f"experiments_public_benchmark_{index}" for index in range(1, 12)],
    "baseline_official_style",
    "candidate_official_style",
    "suite_summary",
    "suite_analysis",
    "suite_validation",
]


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _is_positive_number(value: Any) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _manifest_has_finished_step(rows: list[dict[str, Any]], step_name: str) -> bool:
    return any(row.get("event") == "finish" and row.get("step") == step_name for row in rows)


def _error(errors: list[dict[str, str]], check: str, message: str, file_name: str | None = None) -> None:
    row = {"check": check, "message": message}
    if file_name is not None:
        row["file"] = file_name
    errors.append(row)


def _load_required_jsonl(suite_dir: Path, file_name: str, errors: list[dict[str, str]]) -> list[dict[str, Any]]:
    path = suite_dir / file_name
    if not path.is_file():
        _error(errors, "file.exists", "required JSONL file is missing", file_name)
        return []
    try:
        rows = load_jsonl(path)
    except Exception as exc:
        _error(errors, "jsonl.parse", f"{type(exc).__name__}: {exc}", file_name)
        return []
    if not rows:
        _error(errors, "jsonl.nonempty", "JSONL file is empty", file_name)
    return rows


def _suite_provenance(suite_dir: Path) -> dict[str, Any] | None:
    try:
        rows = load_jsonl(suite_dir / "manifest.jsonl")
    except Exception:
        return None
    if not rows or rows[0].get("event") != "suite_provenance":
        return None
    return rows[0]


def _expected_submission_provenance(
    suite_dir: Path,
    kind: str,
) -> dict[str, Any] | None:
    provenance = _suite_provenance(suite_dir)
    if provenance is None:
        return None
    value = provenance.get(kind)
    return value if isinstance(value, dict) else None


def _benchmark_submission_kind(file_name: str) -> str:
    return "baseline" if Path(file_name).name.startswith("baseline_geqrf_") else "submission"


def _validate_submission_row_provenance(
    row: dict[str, Any],
    file_name: str,
    check: str,
    errors: list[dict[str, str]],
    expected: dict[str, Any] | None,
    expected_kind: str = "submission",
) -> None:
    if not row.get("submission"):
        _error(errors, check, "submission is missing from row", file_name)
    elif isinstance(expected, dict) and expected.get("path") and row.get("submission") != expected.get("path"):
        _error(errors, check, f"submission path does not match suite {expected_kind} path", file_name)
    if not _is_sha256(row.get("submission_sha256")):
        _error(errors, check, "submission_sha256 is not a SHA-256 hex digest", file_name)
    elif isinstance(expected, dict) and _is_sha256(expected.get("sha256")) and row.get("submission_sha256") != expected.get("sha256"):
        _error(errors, check, f"submission_sha256 does not match suite {expected_kind} sha256", file_name)


def _validate_file_provenance_dict(
    value: Any,
    file_name: str,
    check: str,
    errors: list[dict[str, str]],
    expected: dict[str, Any] | None,
    field_name: str,
) -> None:
    if not isinstance(value, dict):
        _error(errors, check, f"{field_name} provenance is missing", file_name)
        return
    if not value.get("path"):
        _error(errors, check, f"{field_name}.path is missing", file_name)
    if not _is_sha256(value.get("sha256")):
        _error(errors, check, f"{field_name}.sha256 is not a SHA-256 hex digest", file_name)
    elif isinstance(expected, dict) and _is_sha256(expected.get("sha256")) and value.get("sha256") != expected.get("sha256"):
        _error(errors, check, f"{field_name}.sha256 does not match suite submission sha256", file_name)
    if not isinstance(value.get("bytes"), int) or value["bytes"] <= 0:
        _error(errors, check, f"{field_name}.bytes must be positive", file_name)


def _resolve_tarball_path(suite_dir: Path, tarball: str) -> Path:
    tarball_path = Path(tarball)
    if tarball_path.is_absolute():
        return tarball_path
    for candidate in [suite_dir / tarball_path, ROOT / tarball_path]:
        if candidate.is_file():
            return candidate
    return suite_dir / tarball_path


def _resolve_suite_path(suite_dir: Path, raw_path: Any) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return suite_dir / path


def _suite_relative_path(suite_dir: Path, path: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(suite_dir.resolve()))
    except ValueError:
        return None


def _candidate_config_large_kernel_plan_event(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    events = [row for row in rows if row.get("event") == CANDIDATE_CONFIG_LARGE_KERNEL_PLAN_EVENT]
    return events[-1] if events else None


def _repair_constraints_from_required_modes(required_modes: Any) -> dict[str, str]:
    constraints: dict[str, str] = {}
    if not isinstance(required_modes, list):
        return constraints
    for item in required_modes:
        raw = str(item)
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip().replace("-", "_")
        value = value.strip()
        if key == "panel_refresh_mode" and value:
            constraints["panel_refresh_modes"] = value
        elif key == "r_maintenance_mode" and value:
            constraints["r_maintenance_modes"] = value
    return constraints


def _repair_env_from_required_modes(env_prefix: str, required_modes: Any) -> dict[str, str]:
    env: dict[str, str] = {}
    constraints = _repair_constraints_from_required_modes(required_modes)
    if "panel_refresh_modes" in constraints:
        env[f"{env_prefix}_PANEL_REFRESH_MODE"] = constraints["panel_refresh_modes"]
    if "r_maintenance_modes" in constraints:
        env[f"{env_prefix}_R_MAINTENANCE_MODE"] = constraints["r_maintenance_modes"]
    return env


def validate_required_files(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    for file_name in REQUIRED_DEFAULT_FILES:
        if not (suite_dir / file_name).is_file():
            _error(errors, "file.exists", "required default-suite file is missing", file_name)


def validate_run_log(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    path = suite_dir / "run.log"
    if not path.is_file():
        _error(errors, "run_log.exists", "run.log is missing", "run.log")
        return
    text = path.read_text(errors="replace")
    if "-m pytest" not in text:
        _error(errors, "run_log.pytest", "run.log does not show a local pytest run", "run.log")


def validate_tarball_manifest(
    suite_dir: Path,
    tarball_path: Path,
    expected_rows: list[dict[str, Any]],
    errors: list[dict[str, str]],
) -> None:
    member_name = f"{suite_dir.name}/manifest.jsonl"
    try:
        with tarfile.open(tarball_path, "r:*") as tar:
            members = set(tar.getnames())
            required_files = list(REQUIRED_DEFAULT_FILES)
            if _manifest_has_finished_step(expected_rows, "tail_policy_tune"):
                required_files.append(str(TAIL_POLICY_TUNE_SUMMARY))
            if _manifest_has_finished_step(expected_rows, "candidate_config_tune"):
                required_files.append(str(CANDIDATE_CONFIG_TUNE_SUMMARY))
            plan_event = _candidate_config_large_kernel_plan_event(expected_rows)
            if plan_event is not None:
                plan_path = _resolve_suite_path(suite_dir, plan_event.get("path"))
                if plan_path is not None:
                    rel_plan_path = _suite_relative_path(suite_dir, plan_path)
                    if rel_plan_path is not None:
                        required_files.append(rel_plan_path)
            if _manifest_has_finished_step(expected_rows, "candidate_config_accelerator_preflight"):
                required_files.append(CANDIDATE_CONFIG_ACCELERATOR_PREFLIGHT_FILE)
            if _manifest_has_finished_step(expected_rows, "blocked_qr_sweep"):
                required_files.append(BLOCKED_QR_SWEEP_FILE)
            if _manifest_has_finished_step(expected_rows, "popcorn_test"):
                required_files.extend(POPCORN_TEST_FILES)
            if _manifest_has_finished_step(expected_rows, "popcorn_leaderboard"):
                required_files.extend(POPCORN_LEADERBOARD_FILES)
            for file_name in required_files:
                required_member = f"{suite_dir.name}/{file_name}"
                if required_member not in members:
                    _error(errors, "tarball.file", f"tarball is missing {required_member}", file_name)
                    continue
                if file_name == "manifest.jsonl":
                    continue
                local_path = suite_dir / file_name
                if not local_path.is_file():
                    continue
                member = tar.extractfile(required_member)
                if member is None:
                    _error(errors, "tarball.file", f"tarball member is not a file: {required_member}", file_name)
                    continue
                if member.read() != local_path.read_bytes():
                    _error(errors, "tarball.file", f"tarball content does not match {required_member}", file_name)
            extracted = tar.extractfile(member_name)
            if extracted is None:
                _error(errors, "manifest.tarball", f"tarball is missing {member_name}", "manifest.jsonl")
                return
            rows = [json.loads(line) for line in extracted.read().decode("utf-8").splitlines() if line.strip()]
    except KeyError:
        _error(errors, "manifest.tarball", f"tarball is missing {member_name}", "manifest.jsonl")
        return
    except (OSError, tarfile.TarError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _error(errors, "manifest.tarball", f"could not read manifest from tarball: {type(exc).__name__}: {exc}", "manifest.jsonl")
        return
    if not rows:
        _error(errors, "manifest.tarball", f"{member_name} inside tarball is empty", "manifest.jsonl")
    elif rows[-1].get("event") != "suite_finish":
        _error(errors, "manifest.tarball", "tarball manifest does not end with suite_finish", "manifest.jsonl")
    elif rows != expected_rows:
        _error(errors, "manifest.tarball", "tarball manifest does not match on-disk manifest", "manifest.jsonl")


def validate_manifest_steps(rows: list[dict[str, Any]], errors: list[dict[str, str]]) -> None:
    start_by_step: dict[str, list[int]] = {}
    finish_by_step: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        event = row.get("event")
        step = row.get("step")
        if event not in {"start", "finish"}:
            continue
        if not isinstance(step, str) or not step:
            _error(errors, "manifest.steps", f"{event} row is missing step name", "manifest.jsonl")
            continue
        if event == "start":
            start_by_step.setdefault(step, []).append(index)
        else:
            finish_by_step.setdefault(step, []).append(index)
            if not _is_positive_number(row.get("elapsed_s")):
                _error(errors, "manifest.steps", f"finish row for {step} is missing positive elapsed_s", "manifest.jsonl")

    for step in EXPECTED_DEFAULT_STEPS:
        starts = start_by_step.get(step, [])
        finishes = finish_by_step.get(step, [])
        if len(starts) != 1:
            _error(errors, "manifest.steps", f"expected one start row for {step}, got {len(starts)}", "manifest.jsonl")
        if len(finishes) != 1:
            _error(errors, "manifest.steps", f"expected one finish row for {step}, got {len(finishes)}", "manifest.jsonl")
        if starts and finishes and starts[0] > finishes[0]:
            _error(errors, "manifest.steps", f"finish row appears before start row for {step}", "manifest.jsonl")

    for step, starts in start_by_step.items():
        finishes = finish_by_step.get(step, [])
        if len(starts) != len(finishes):
            _error(errors, "manifest.steps", f"start/finish count mismatch for {step}", "manifest.jsonl")


def validate_manifest(suite_dir: Path, errors: list[dict[str, str]], require_finish: bool) -> None:
    rows = _load_required_jsonl(suite_dir, "manifest.jsonl", errors)
    if not rows:
        return

    first = rows[0]
    if first.get("event") != "suite_provenance":
        _error(errors, "manifest.provenance", "first manifest row must be suite_provenance", "manifest.jsonl")
        return

    repo = first.get("repo")
    if not isinstance(repo, dict):
        _error(errors, "manifest.provenance", "suite_provenance.repo is missing", "manifest.jsonl")
    else:
        for key in ["git_hash", "git_full_hash", "git_dirty", "official_upstream_commit", "git_status_porcelain"]:
            if key not in repo:
                _error(errors, "manifest.provenance", f"repo.{key} is missing", "manifest.jsonl")
        if not isinstance(repo.get("git_status_porcelain"), list):
            _error(errors, "manifest.provenance", "repo.git_status_porcelain must be a list", "manifest.jsonl")

    for key in ["submission", "baseline"]:
        value = first.get(key)
        if not isinstance(value, dict):
            _error(errors, "manifest.provenance", f"{key} provenance is missing", "manifest.jsonl")
            continue
        if not value.get("path"):
            _error(errors, "manifest.provenance", f"{key}.path is missing", "manifest.jsonl")
        if not _is_sha256(value.get("sha256")):
            _error(errors, "manifest.provenance", f"{key}.sha256 is not a SHA-256 hex digest", "manifest.jsonl")
        if not isinstance(value.get("bytes"), int) or value["bytes"] <= 0:
            _error(errors, "manifest.provenance", f"{key}.bytes must be positive", "manifest.jsonl")

    env = first.get("env")
    if not isinstance(env, dict):
        _error(errors, "manifest.provenance", "suite_provenance.env is missing", "manifest.jsonl")
    else:
        missing_env = [key for key in TRACKED_RUNTIME_ENV_KEYS if key not in env]
        if missing_env:
            _error(
                errors,
                "manifest.provenance",
                "suite_provenance.env is missing tracked keys: " + ", ".join(missing_env[:8]),
                "manifest.jsonl",
            )

    if any(row.get("event") == "suite_failed" for row in rows):
        _error(errors, "manifest.status", "suite_failed event is present", "manifest.jsonl")
    if require_finish:
        validate_manifest_steps(rows, errors)
        finish_rows = [row for row in rows if row.get("event") == "suite_finish"]
        if not finish_rows:
            _error(errors, "manifest.status", "suite_finish event is missing", "manifest.jsonl")
            return
        if rows[-1].get("event") != "suite_finish":
            _error(errors, "manifest.status", "suite_finish must be the final manifest row", "manifest.jsonl")
        finish = finish_rows[-1]
        tarball = finish.get("tarball")
        if not isinstance(tarball, str) or not tarball:
            _error(errors, "manifest.tarball", "suite_finish tarball path is missing", "manifest.jsonl")
        else:
            tarball_path = _resolve_tarball_path(suite_dir, tarball)
            if not tarball_path.is_file():
                _error(errors, "manifest.tarball", f"tarball does not exist: {tarball}", "manifest.jsonl")
            else:
                validate_tarball_manifest(suite_dir, tarball_path, rows, errors)


def validate_submission_validation(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    rows = _load_required_jsonl(suite_dir, "submission_validation.jsonl", errors)
    if not rows:
        return
    expected = _expected_submission_provenance(suite_dir, "submission")
    expected_sha256 = expected.get("sha256") if isinstance(expected, dict) else None
    if len(rows) != 1:
        _error(errors, "submission_validation.count", f"expected one validation row, got {len(rows)}", "submission_validation.jsonl")
    row = rows[0]
    if row.get("event") != "submission_validation":
        _error(errors, "submission_validation.schema", "row event must be submission_validation", "submission_validation.jsonl")
    if not row.get("ok") or not row.get("static_ok") or not row.get("import_ok"):
        _error(errors, "submission_validation.ok", "submission validation did not pass", "submission_validation.jsonl")
    if row.get("errors"):
        _error(errors, "submission_validation.errors", "submission validation recorded errors", "submission_validation.jsonl")
    for key in ["source_submission", "staged_submission"]:
        value = row.get(key)
        if not isinstance(value, dict):
            _error(errors, "submission_validation.provenance", f"{key} is missing", "submission_validation.jsonl")
            continue
        if not value.get("path") or not _is_sha256(value.get("sha256")):
            _error(errors, "submission_validation.provenance", f"{key} path or sha256 is invalid", "submission_validation.jsonl")
        elif _is_sha256(expected_sha256) and value.get("sha256") != expected_sha256:
            _error(
                errors,
                "submission_validation.provenance",
                f"{key}.sha256 does not match suite submission sha256",
                "submission_validation.jsonl",
            )
        if not isinstance(value.get("bytes"), int) or value["bytes"] <= 0:
            _error(errors, "submission_validation.provenance", f"{key}.bytes must be positive", "submission_validation.jsonl")


def validate_public_tests(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    rows = _load_required_jsonl(suite_dir, "candidate_public_tests.jsonl", errors)
    if not rows:
        return
    expected = _expected_submission_provenance(suite_dir, "submission")
    cases = [row for row in rows if not row.get("summary")]
    summary = next((row for row in rows if row.get("summary")), None)
    if len(cases) != 22:
        _error(errors, "public_tests.count", f"expected 22 public test rows, got {len(cases)}", "candidate_public_tests.jsonl")
    failed = [row for row in cases if not row.get("ok")]
    if failed:
        _error(errors, "public_tests.ok", f"{len(failed)} public test rows failed", "candidate_public_tests.jsonl")
    for row in cases:
        _validate_submission_row_provenance(
            row,
            "candidate_public_tests.jsonl",
            "public_tests.provenance",
            errors,
            expected,
        )
    if not summary or not summary.get("ok") or summary.get("num_failed") != 0 or summary.get("num_cases") != 22:
        _error(errors, "public_tests.summary", "summary must report 22 cases and zero failures", "candidate_public_tests.jsonl")


def validate_public_benchmark_correctness(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    rows = _load_required_jsonl(suite_dir, "candidate_public_benchmark_correctness.jsonl", errors)
    if not rows:
        return
    expected = _expected_submission_provenance(suite_dir, "submission")
    cases = [row for row in rows if not row.get("summary")]
    summary = next((row for row in rows if row.get("summary")), None)
    if len(cases) != 12:
        _error(
            errors,
            "benchmark_correctness.count",
            f"expected 12 public benchmark correctness rows, got {len(cases)}",
            "candidate_public_benchmark_correctness.jsonl",
        )
    failed = [row for row in cases if not row.get("ok")]
    if failed:
        _error(
            errors,
            "benchmark_correctness.ok",
            f"{len(failed)} public benchmark correctness rows failed",
            "candidate_public_benchmark_correctness.jsonl",
        )
    missing_diagnostics = [row for row in cases if not isinstance(row.get("diagnostics"), dict)]
    if missing_diagnostics:
        _error(
            errors,
            "benchmark_correctness.diagnostics",
            f"{len(missing_diagnostics)} public benchmark correctness rows are missing diagnostics",
            "candidate_public_benchmark_correctness.jsonl",
        )
    missing_layout = [
        row
        for row in cases
        if not isinstance(row.get("h_stride"), list)
        or len(row.get("h_stride", [])) != 3
        or row.get("h_layout_actual") not in ("column_major", "torch_contiguous", "other_strided")
        or not isinstance(row.get("column_major_h_actual"), bool)
    ]
    if missing_layout:
        _error(
            errors,
            "benchmark_correctness.layout",
            f"{len(missing_layout)} public benchmark correctness rows are missing actual H layout metadata",
            "candidate_public_benchmark_correctness.jsonl",
        )
    missed_margin = [row for row in cases if row.get("margin_ok") is False]
    if missed_margin:
        _error(
            errors,
            "benchmark_correctness.margin",
            f"{len(missed_margin)} public benchmark correctness rows missed the configured margin",
            "candidate_public_benchmark_correctness.jsonl",
        )
    for row in cases:
        _validate_submission_row_provenance(
            row,
            "candidate_public_benchmark_correctness.jsonl",
            "benchmark_correctness.provenance",
            errors,
            expected,
        )
    if not summary or not summary.get("ok") or summary.get("num_failed") != 0 or summary.get("num_cases") != 12:
        _error(
            errors,
            "benchmark_correctness.summary",
            "summary must report 12 cases and zero failures",
            "candidate_public_benchmark_correctness.jsonl",
        )


def validate_dev_robustness(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    rows = _load_required_jsonl(suite_dir, "candidate_dev_robustness.jsonl", errors)
    if not rows:
        return
    expected = _expected_submission_provenance(suite_dir, "submission")
    cases = [row for row in rows if not row.get("summary")]
    summary = next((row for row in rows if row.get("summary")), None)
    if len(cases) != 11:
        _error(
            errors,
            "dev_robustness.count",
            f"expected 11 dev robustness rows, got {len(cases)}",
            "candidate_dev_robustness.jsonl",
        )
    failed = [row for row in cases if not row.get("ok")]
    if failed:
        _error(errors, "dev_robustness.ok", f"{len(failed)} dev robustness rows failed", "candidate_dev_robustness.jsonl")
    missing_diagnostics = [row for row in cases if not isinstance(row.get("diagnostics"), dict)]
    if missing_diagnostics:
        _error(
            errors,
            "dev_robustness.diagnostics",
            f"{len(missing_diagnostics)} dev robustness rows are missing diagnostics",
            "candidate_dev_robustness.jsonl",
        )
    missed_margin = [row for row in cases if row.get("margin_ok") is False]
    if missed_margin:
        _error(
            errors,
            "dev_robustness.margin",
            f"{len(missed_margin)} dev robustness rows missed the configured margin",
            "candidate_dev_robustness.jsonl",
        )
    for row in cases:
        _validate_submission_row_provenance(
            row,
            "candidate_dev_robustness.jsonl",
            "dev_robustness.provenance",
            errors,
            expected,
        )
    if not summary or not summary.get("ok") or summary.get("num_failed") != 0 or summary.get("num_cases") != 11:
        _error(
            errors,
            "dev_robustness.summary",
            "summary must report 11 cases and zero failures",
            "candidate_dev_robustness.jsonl",
        )


def validate_secret_audit(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    rows = _load_required_jsonl(suite_dir, "secret_audit.jsonl", errors)
    if not rows:
        return
    findings = [row for row in rows if not row.get("summary")]
    summary = next((row for row in rows if row.get("summary")), None)
    if findings:
        _error(errors, "secret_audit.findings", f"{len(findings)} potential secret findings", "secret_audit.jsonl")
    if not summary:
        _error(errors, "secret_audit.summary", "secret audit summary is missing", "secret_audit.jsonl")
        return
    try:
        num_findings = int(summary.get("num_findings", -1))
    except (TypeError, ValueError):
        num_findings = -1
    if not summary.get("ok") or num_findings != 0:
        _error(errors, "secret_audit.ok", "secret audit summary must report zero findings", "secret_audit.jsonl")
    if not _is_positive_number(summary.get("files_scanned")):
        _error(errors, "secret_audit.coverage", "secret audit must scan at least one file", "secret_audit.jsonl")


def validate_runtime_preflight(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    rows = _load_required_jsonl(suite_dir, "runtime_preflight.jsonl", errors)
    if not rows:
        return
    if len(rows) != 1:
        _error(errors, "runtime_preflight.count", f"expected 1 runtime preflight row, got {len(rows)}", "runtime_preflight.jsonl")
    row = rows[0]
    if not row.get("ok"):
        _error(errors, "runtime_preflight.ok", "runtime preflight did not pass", "runtime_preflight.jsonl")
    torch_info = row.get("torch")
    if not isinstance(torch_info, dict):
        _error(errors, "runtime_preflight.schema", "torch runtime info is missing", "runtime_preflight.jsonl")
        return
    if torch_info.get("cuda_available") is not True:
        _error(errors, "runtime_preflight.cuda", "torch.cuda.is_available() must be true", "runtime_preflight.jsonl")
    if not _is_positive_number(torch_info.get("device_count")):
        _error(errors, "runtime_preflight.cuda", "torch must report at least one CUDA device", "runtime_preflight.jsonl")
    selected = row.get("selected_device")
    if not isinstance(selected, dict):
        _error(errors, "runtime_preflight.device", "selected CUDA device is missing", "runtime_preflight.jsonl")
        return
    name = str(selected.get("name") or "")
    if "B200" not in name:
        _error(errors, "runtime_preflight.device", "selected CUDA device name must contain B200", "runtime_preflight.jsonl")
    capability = selected.get("capability")
    try:
        major = int(capability[0]) if isinstance(capability, list) and capability else -1
    except (TypeError, ValueError):
        major = -1
    if major < 10:
        _error(errors, "runtime_preflight.capability", "selected CUDA device must have compute capability 10.x or newer", "runtime_preflight.jsonl")
    memory = selected.get("total_memory_gib")
    try:
        memory_gib = float(memory) if memory is not None else None
    except (TypeError, ValueError):
        memory_gib = None
    if memory_gib is not None and memory_gib < 150.0:
        _error(errors, "runtime_preflight.memory", "selected CUDA device must report B200-scale memory", "runtime_preflight.jsonl")


def validate_preflight(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    rows = _load_required_jsonl(suite_dir, "accelerator_preflight.jsonl", errors)
    if not rows:
        return
    summary = next((row for row in rows if row.get("summary")), None)
    if not summary or not summary.get("ok"):
        _error(errors, "preflight.summary", "accelerator preflight summary is not ok", "accelerator_preflight.jsonl")
    if summary:
        if summary.get("family_cases") is not True:
            _error(
                errors,
                "preflight.family_cases",
                "accelerator preflight must run representative family cases",
                "accelerator_preflight.jsonl",
            )
        cases_by_accelerator = summary.get("preflight_cases_by_accelerator")
        if not isinstance(cases_by_accelerator, dict):
            _error(
                errors,
                "preflight.family_cases",
                "summary must record preflight_cases_by_accelerator",
                "accelerator_preflight.jsonl",
            )
        else:
            for accelerator, expected_cases in CANDIDATE_CONFIG_ACCELERATOR_FAMILY_CASES.items():
                if accelerator not in cases_by_accelerator:
                    continue
                observed = set(cases_by_accelerator.get(accelerator) or [])
                missing = sorted(expected_cases - observed)
                if missing:
                    _error(
                        errors,
                        "preflight.family_cases",
                        f"summary for {accelerator} is missing family cases: {', '.join(missing)}",
                        "accelerator_preflight.jsonl",
                    )
    failed = [row for row in rows if not row.get("summary") and not row.get("ok")]
    if failed:
        _error(errors, "preflight.ok", f"{len(failed)} accelerator preflight rows failed", "accelerator_preflight.jsonl")
    coverage: dict[str, set[str]] = {}
    for row in rows:
        if row.get("summary"):
            continue
        accelerator = row.get("accelerator")
        if accelerator not in CANDIDATE_CONFIG_ACCELERATOR_FAMILY_CASES:
            continue
        coverage.setdefault(str(accelerator), set()).add(str(row.get("preflight_case")))
    for accelerator, observed_cases in coverage.items():
        missing = sorted(CANDIDATE_CONFIG_ACCELERATOR_FAMILY_CASES[accelerator] - observed_cases)
        if missing:
            _error(
                errors,
                "preflight.family_cases",
                f"{accelerator} is missing family cases: {', '.join(missing)}",
                "accelerator_preflight.jsonl",
            )
    missing_case = [row for row in rows if not row.get("summary") and row.get("preflight_case") is None]
    if missing_case:
        _error(
            errors,
            "preflight.family_cases",
            f"{len(missing_case)} accelerator preflight rows are missing preflight_case",
            "accelerator_preflight.jsonl",
        )


def validate_candidate_config_accelerator_preflight(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    manifest_path = suite_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        return
    try:
        manifest_rows = load_jsonl(manifest_path)
    except Exception:
        return
    if not _manifest_has_finished_step(manifest_rows, "candidate_config_accelerator_preflight"):
        return

    rows = _load_required_jsonl(suite_dir, CANDIDATE_CONFIG_ACCELERATOR_PREFLIGHT_FILE, errors)
    if not rows:
        return
    summary = next((row for row in rows if row.get("summary")), None)
    if not summary or not summary.get("ok"):
        _error(
            errors,
            "candidate_config_accelerator_preflight.summary",
            "candidate config accelerator preflight summary is not ok",
            CANDIDATE_CONFIG_ACCELERATOR_PREFLIGHT_FILE,
        )
    if summary:
        if not _is_positive_number(summary.get("num_configs")):
            _error(
                errors,
                "candidate_config_accelerator_preflight.num_configs",
                "summary must record a positive num_configs",
                CANDIDATE_CONFIG_ACCELERATOR_PREFLIGHT_FILE,
            )
        if not _is_positive_number(summary.get("num_config_accelerator_rows")):
            _error(
                errors,
                "candidate_config_accelerator_preflight.num_rows",
                "summary must record a positive config accelerator row count",
                CANDIDATE_CONFIG_ACCELERATOR_PREFLIGHT_FILE,
            )
        if summary.get("family_cases") is not True:
            _error(
                errors,
                "candidate_config_accelerator_preflight.family_cases",
                "candidate config accelerator preflight must run representative family cases",
                CANDIDATE_CONFIG_ACCELERATOR_PREFLIGHT_FILE,
            )
        cases_by_accelerator = summary.get("preflight_cases_by_accelerator")
        if not isinstance(cases_by_accelerator, dict):
            _error(
                errors,
                "candidate_config_accelerator_preflight.family_cases",
                "summary must record preflight_cases_by_accelerator",
                CANDIDATE_CONFIG_ACCELERATOR_PREFLIGHT_FILE,
            )
        else:
            for accelerator, expected_cases in CANDIDATE_CONFIG_ACCELERATOR_FAMILY_CASES.items():
                if accelerator not in cases_by_accelerator:
                    continue
                observed = set(cases_by_accelerator.get(accelerator) or [])
                missing = sorted(expected_cases - observed)
                if missing:
                    _error(
                        errors,
                        "candidate_config_accelerator_preflight.family_cases",
                        f"summary for {accelerator} is missing family cases: {', '.join(missing)}",
                        CANDIDATE_CONFIG_ACCELERATOR_PREFLIGHT_FILE,
                    )
    failed = [row for row in rows if not row.get("summary") and not row.get("ok")]
    if failed:
        _error(
            errors,
            "candidate_config_accelerator_preflight.ok",
            f"{len(failed)} candidate config accelerator preflight rows failed",
            CANDIDATE_CONFIG_ACCELERATOR_PREFLIGHT_FILE,
        )
    coverage: dict[tuple[Any, str], set[str]] = {}
    for row in rows:
        if row.get("summary"):
            continue
        accelerator = row.get("accelerator")
        if accelerator not in CANDIDATE_CONFIG_ACCELERATOR_FAMILY_CASES:
            continue
        config_key = row.get("config_index", row.get("config_name"))
        coverage.setdefault((config_key, str(accelerator)), set()).add(str(row.get("preflight_case")))
    for (config_key, accelerator), observed_cases in coverage.items():
        missing = sorted(CANDIDATE_CONFIG_ACCELERATOR_FAMILY_CASES[accelerator] - observed_cases)
        if missing:
            _error(
                errors,
                "candidate_config_accelerator_preflight.family_cases",
                f"config {config_key} for {accelerator} is missing family cases: {', '.join(missing)}",
                CANDIDATE_CONFIG_ACCELERATOR_PREFLIGHT_FILE,
            )
    missing_case = [row for row in rows if not row.get("summary") and not row.get("preflight_case")]
    if missing_case:
        _error(
            errors,
            "candidate_config_accelerator_preflight.family_cases",
            f"{len(missing_case)} candidate config accelerator preflight rows are missing preflight_case",
            CANDIDATE_CONFIG_ACCELERATOR_PREFLIGHT_FILE,
        )


def _validate_row_provenance(
    row: dict[str, Any],
    file_name: str,
    errors: list[dict[str, str]],
    expected: dict[str, Any] | None,
) -> None:
    for key in ["gpu", "driver", "cuda", "torch", "git_hash", "git_full_hash", "git_dirty", "official_upstream_commit"]:
        if key not in row:
            _error(errors, "benchmark.provenance", f"{key} is missing from benchmark row", file_name)
    _validate_submission_row_provenance(
        row,
        file_name,
        "benchmark.provenance",
        errors,
        expected,
        _benchmark_submission_kind(file_name),
    )


def validate_benchmark_file(suite_dir: Path, file_name: str, errors: list[dict[str, str]]) -> None:
    rows = _load_required_jsonl(suite_dir, file_name, errors)
    if not rows:
        return
    expected = _expected_submission_provenance(suite_dir, _benchmark_submission_kind(file_name))
    cases = [row for row in rows if isinstance(row.get("spec"), dict) and "mean_us" in row]
    summary = next((row for row in rows if "geomean_us" in row), None)
    if len(cases) != 12:
        _error(errors, "benchmark.count", f"expected 12 benchmark rows, got {len(cases)}", file_name)
    for row in cases:
        if not row.get("ok"):
            _error(errors, "benchmark.ok", "benchmark row is not ok", file_name)
        if float(row.get("mean_us", 0.0)) <= 0.0:
            _error(errors, "benchmark.mean_us", "benchmark row mean_us must be positive", file_name)
        _validate_row_provenance(row, file_name, errors, expected)
    if not summary:
        _error(errors, "benchmark.summary", "geomean summary row is missing", file_name)
    elif float(summary.get("geomean_us", 0.0)) <= 0.0 or summary.get("num_cases") != 12:
        _error(errors, "benchmark.summary", "summary must report positive geomean over 12 cases", file_name)


def validate_policy(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    rows = _load_required_jsonl(suite_dir, "candidate_policy_public.jsonl", errors)
    if not rows:
        return
    expected = _expected_submission_provenance(suite_dir, "submission")
    if len(rows) != 12:
        _error(errors, "policy.count", f"expected 12 policy rows, got {len(rows)}", "candidate_policy_public.jsonl")
    num_shape_collisions = 0
    for row in rows:
        _validate_submission_row_provenance(
            row,
            "candidate_policy_public.jsonl",
            "policy.provenance",
            errors,
            expected,
        )
        if not row.get("dispatch") or not row.get("primary"):
            _error(errors, "policy.schema", "policy row is missing dispatch or primary", "candidate_policy_public.jsonl")
        if row.get("case_metadata_passed_to_submission") is not False:
            _error(
                errors,
                "policy.interface",
                "policy row must record that case metadata is not passed to custom_kernel(data)",
                "candidate_policy_public.jsonl",
            )
        if row.get("case_metadata_available") is not False:
            _error(
                errors,
                "policy.interface",
                "policy row must record that case metadata is unavailable to the submission",
                "candidate_policy_public.jsonl",
            )
        if row.get("submission_entrypoint") != "custom_kernel(data)":
            _error(
                errors,
                "policy.interface",
                "policy row must record the official custom_kernel(data) entrypoint",
                "candidate_policy_public.jsonl",
            )
        sources = row.get("dispatch_info_sources")
        if not isinstance(sources, list) or "data.shape" not in sources:
            _error(
                errors,
                "policy.schema",
                "policy row must list data.shape as a dispatch information source",
                "candidate_policy_public.jsonl",
            )
        if not isinstance(row.get("shape_only_dispatch"), bool):
            _error(
                errors,
                "policy.schema",
                "policy row is missing boolean shape_only_dispatch",
                "candidate_policy_public.jsonl",
            )
        if not isinstance(row.get("shape_only_case_selection"), bool):
            _error(
                errors,
                "policy.schema",
                "policy row is missing boolean shape_only_case_selection",
                "candidate_policy_public.jsonl",
            )
        case_sources = row.get("case_selection_info_sources")
        if not isinstance(case_sources, list) or "data.shape" not in case_sources:
            _error(
                errors,
                "policy.schema",
                "policy row must list data.shape as a case selection information source",
                "candidate_policy_public.jsonl",
            )
        if not isinstance(row.get("uses_tensor_values_for_dispatch"), bool):
            _error(
                errors,
                "policy.schema",
                "policy row is missing boolean uses_tensor_values_for_dispatch",
                "candidate_policy_public.jsonl",
            )
        if not isinstance(row.get("uses_tensor_values_for_case_selection"), bool):
            _error(
                errors,
                "policy.schema",
                "policy row is missing boolean uses_tensor_values_for_case_selection",
                "candidate_policy_public.jsonl",
            )
        if row.get("case_info_source") not in ("data.shape", "tensor_values"):
            _error(
                errors,
                "policy.schema",
                "policy row is missing case_info_source",
                "candidate_policy_public.jsonl",
            )
        if not isinstance(row.get("classifier_needed_for_current_candidate"), bool):
            _error(
                errors,
                "policy.schema",
                "policy row is missing boolean classifier_needed_for_current_candidate",
                "candidate_policy_public.jsonl",
            )
        if not isinstance(row.get("classifier_needed_for_case_specific_path"), bool):
            _error(
                errors,
                "policy.schema",
                "policy row is missing boolean classifier_needed_for_case_specific_path",
                "candidate_policy_public.jsonl",
            )
        if not isinstance(row.get("classifier_on_current_hot_path"), bool):
            _error(
                errors,
                "policy.schema",
                "policy row is missing boolean classifier_on_current_hot_path",
                "candidate_policy_public.jsonl",
            )
        if not row.get("classifier_reason") or not row.get("classifier_decision_rule"):
            _error(
                errors,
                "policy.schema",
                "policy row is missing classifier rationale",
                "candidate_policy_public.jsonl",
            )
        if row.get("column_major_h") not in (True, False, "conditional"):
            _error(
                errors,
                "policy.schema",
                "policy row is missing column_major_h layout status",
                "candidate_policy_public.jsonl",
            )
        if row.get("h_layout") not in (
            "column_major",
            "torch.geqrf_default",
            "column_major_when_fast_path_applies_else_torch.geqrf_default",
            "column_major_when_cuda_extension_available_else_torch.geqrf_default",
        ):
            _error(
                errors,
                "policy.schema",
                "policy row has an unknown h_layout",
                "candidate_policy_public.jsonl",
            )
        if not isinstance(row.get("shape_collision"), bool):
            _error(errors, "policy.schema", "policy row is missing boolean shape_collision", "candidate_policy_public.jsonl")
        elif row["shape_collision"]:
            num_shape_collisions += 1
            collision_cases = row.get("shape_collision_cases")
            if not isinstance(collision_cases, list) or len(collision_cases) < 2:
                _error(
                    errors,
                    "policy.shape_collision",
                    "shape collision rows must list at least two colliding cases",
                    "candidate_policy_public.jsonl",
                )
            if row.get("requires_tensor_guard_for_case_specific_path") is not True:
                _error(
                    errors,
                    "policy.shape_collision",
                    "shape collision rows must record that case-specific paths need tensor guards",
                    "candidate_policy_public.jsonl",
                )
            if row.get("classifier_needed_for_case_specific_path") is not True:
                _error(
                    errors,
                    "policy.shape_collision",
                    "shape collision rows must record that case-specific paths need a classifier/tensor guard",
                    "candidate_policy_public.jsonl",
                )
            if row.get("shape_only_case_selection") is not False:
                _error(
                    errors,
                    "policy.shape_collision",
                    "shape collision rows with case-specific paths must not record shape-only case selection",
                    "candidate_policy_public.jsonl",
                )
            if row.get("uses_tensor_values_for_case_selection") is not True:
                _error(
                    errors,
                    "policy.shape_collision",
                    "shape collision rows must record tensor-value case selection",
                    "candidate_policy_public.jsonl",
                )
            if row.get("uses_tensor_values_for_dispatch") is not True:
                _error(
                    errors,
                    "policy.shape_collision",
                    "shape collision rows must record tensor-value dispatch",
                    "candidate_policy_public.jsonl",
                )
            if row.get("case_info_source") != "tensor_values":
                _error(
                    errors,
                    "policy.shape_collision",
                    "shape collision rows must record tensor_values as the case information source",
                    "candidate_policy_public.jsonl",
                )
            if row.get("classifier_on_current_hot_path") is not row.get("classifier_needed_for_current_candidate"):
                _error(
                    errors,
                    "policy.shape_collision",
                    "classifier_on_current_hot_path must match classifier_needed_for_current_candidate",
                    "candidate_policy_public.jsonl",
                )
            if row.get("classifier_needed_for_current_candidate") and row.get("uses_tensor_values_for_dispatch") is not True:
                _error(
                    errors,
                    "policy.shape_collision",
                    "classifier-active shape collision rows must record tensor-value dispatch",
                    "candidate_policy_public.jsonl",
                )
            if row.get("classifier_needed_for_current_candidate") and isinstance(sources, list) and "tensor_values" not in sources:
                _error(
                    errors,
                    "policy.shape_collision",
                    "classifier-active shape collision rows must list tensor_values as a dispatch information source",
                    "candidate_policy_public.jsonl",
                )
            if isinstance(case_sources, list) and "tensor_values" not in case_sources:
                _error(
                    errors,
                    "policy.shape_collision",
                    "shape collision rows must list tensor_values as a case selection information source",
                    "candidate_policy_public.jsonl",
                )
        elif row.get("shape_only_case_selection") is not True:
            _error(
                errors,
                "policy.shape_only",
                "non-colliding public benchmark rows should be shape-only case selection",
                "candidate_policy_public.jsonl",
            )
        elif row.get("case_info_source") != "data.shape":
            _error(
                errors,
                "policy.shape_only",
                "non-colliding public benchmark rows should record data.shape as the case information source",
                "candidate_policy_public.jsonl",
            )
        elif row.get("classifier_needed_for_current_candidate") is not False:
            _error(
                errors,
                "policy.shape_only",
                "non-colliding public benchmark rows should not require a classifier",
                "candidate_policy_public.jsonl",
            )
        elif row.get("classifier_needed_for_case_specific_path") is not False:
            _error(
                errors,
                "policy.shape_only",
                "non-colliding public benchmark rows should not require a case-specific classifier",
                "candidate_policy_public.jsonl",
            )
        elif row.get("uses_tensor_values_for_case_selection") is not False:
            _error(
                errors,
                "policy.shape_only",
                "non-colliding public benchmark rows should not use tensor-value case selection",
                "candidate_policy_public.jsonl",
            )
        if not isinstance(row.get("requires_tensor_guard_for_case_specific_path"), bool):
            _error(
                errors,
                "policy.schema",
                "policy row is missing boolean requires_tensor_guard_for_case_specific_path",
                "candidate_policy_public.jsonl",
            )
    if num_shape_collisions == 0:
        _error(
            errors,
            "policy.shape_collision",
            "public benchmark policy must include the ambiguous 512/1024 shape families",
            "candidate_policy_public.jsonl",
        )


def validate_implementation_status(
    suite_dir: Path,
    errors: list[dict[str, str]],
    require_final_kernels: bool = False,
) -> None:
    file_name = "candidate_implementation_status.jsonl"
    rows = _load_required_jsonl(suite_dir, file_name, errors)
    if not rows:
        return
    expected = _expected_submission_provenance(suite_dir, "submission")
    cases = [row for row in rows if not row.get("summary")]
    summary = next((row for row in rows if row.get("summary")), None)
    if len(cases) != 12:
        _error(errors, "implementation_status.count", f"expected 12 status rows, got {len(cases)}", file_name)
    allowed_kinds = {
        "custom_cuda_optional_fallback",
        "torch_geqrf_fallback",
        "torch_composite_experiment",
        "unknown",
    }
    allowed_readiness = {
        "partial_cuda_needs_b200_validation",
        "missing_custom_kernel",
        "experimental_not_final",
        "needs_review",
    }
    for row in cases:
        _validate_submission_row_provenance(row, file_name, "implementation_status.provenance", errors, expected)
        if not isinstance(row.get("case_index"), int):
            _error(errors, "implementation_status.schema", "status row is missing integer case_index", file_name)
        if not row.get("dispatch") or not row.get("primary"):
            _error(errors, "implementation_status.schema", "status row is missing dispatch or primary", file_name)
        if row.get("implementation_kind") not in allowed_kinds:
            _error(errors, "implementation_status.schema", "status row has unknown implementation_kind", file_name)
        if row.get("readiness") not in allowed_readiness:
            _error(errors, "implementation_status.schema", "status row has unknown readiness", file_name)
        for key in ["uses_torch_geqrf", "has_custom_cuda", "final_kernel_required"]:
            if not isinstance(row.get(key), bool):
                _error(errors, "implementation_status.schema", f"status row is missing boolean {key}", file_name)
        if not row.get("next_work"):
            _error(errors, "implementation_status.schema", "status row is missing next_work", file_name)
    if not isinstance(summary, dict):
        _error(errors, "implementation_status.summary", "status summary row is missing", file_name)
        return
    _validate_submission_row_provenance(summary, file_name, "implementation_status.provenance", errors, expected)
    if summary.get("num_cases") != len(cases):
        _error(errors, "implementation_status.summary", "summary num_cases does not match status rows", file_name)
    if not isinstance(summary.get("ready_for_final_submission"), bool):
        _error(errors, "implementation_status.summary", "summary is missing ready_for_final_submission", file_name)
    if not isinstance(summary.get("num_final_kernel_required"), int):
        _error(errors, "implementation_status.summary", "summary is missing num_final_kernel_required", file_name)
    if not isinstance(summary.get("by_implementation_kind"), dict):
        _error(errors, "implementation_status.summary", "summary is missing by_implementation_kind", file_name)
    if not isinstance(summary.get("by_readiness"), dict):
        _error(errors, "implementation_status.summary", "summary is missing by_readiness", file_name)
    if not isinstance(summary.get("next_priority_cases"), list):
        _error(errors, "implementation_status.summary", "summary is missing next_priority_cases", file_name)
    if require_final_kernels:
        remaining = [row for row in cases if row.get("final_kernel_required") is True]
        if summary.get("ready_for_final_submission") is not True:
            _error(
                errors,
                "implementation_status.final_readiness",
                "candidate is not ready for final submission: ready_for_final_submission is false",
                file_name,
            )
        if summary.get("num_final_kernel_required") != 0:
            _error(
                errors,
                "implementation_status.final_readiness",
                f"candidate still reports {summary.get('num_final_kernel_required')} benchmark cases requiring final kernels",
                file_name,
            )
        if remaining:
            preview = ", ".join(
                f"{row.get('case_index')}:{row.get('dispatch')}:{row.get('readiness')}"
                for row in remaining[:5]
            )
            _error(
                errors,
                "implementation_status.final_readiness",
                f"benchmark routes still require final kernels: {preview}",
                file_name,
            )


def validate_route_trace(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    rows = _load_required_jsonl(suite_dir, "candidate_route_trace_public.jsonl", errors)
    if not rows:
        return
    expected = _expected_submission_provenance(suite_dir, "submission")
    if len(rows) != 12:
        _error(errors, "route_trace.count", f"expected 12 route rows, got {len(rows)}", "candidate_route_trace_public.jsonl")
    for row in rows:
        _validate_submission_row_provenance(
            row,
            "candidate_route_trace_public.jsonl",
            "route_trace.provenance",
            errors,
            expected,
        )
        if not row.get("route") or not row.get("dispatch") or not row.get("spec"):
            _error(errors, "route_trace.schema", "route row is missing route, dispatch, or spec", "candidate_route_trace_public.jsonl")
        if row.get("case_metadata_available") is not False or row.get("case_metadata_passed_to_submission") is not False:
            _error(
                errors,
                "route_trace.metadata",
                "route rows must record that case metadata is not available to custom_kernel(data)",
                "candidate_route_trace_public.jsonl",
            )
        if row.get("route_decision_source") not in {"data.shape", "data.shape+tensor_values"}:
            _error(
                errors,
                "route_trace.source",
                "route row is missing a valid route_decision_source",
                "candidate_route_trace_public.jsonl",
            )
        if not isinstance(row.get("dispatch_info_sources"), list) or not row.get("dispatch_info_sources"):
            _error(
                errors,
                "route_trace.source",
                "route row must list dispatch_info_sources",
                "candidate_route_trace_public.jsonl",
            )
        if not isinstance(row.get("case_selection_info_sources"), list) or not row.get("case_selection_info_sources"):
            _error(
                errors,
                "route_trace.source",
                "route row must list case_selection_info_sources",
                "candidate_route_trace_public.jsonl",
            )
        if not isinstance(row.get("shape_only_case_selection"), bool):
            _error(
                errors,
                "route_trace.source",
                "route row must include boolean shape_only_case_selection",
                "candidate_route_trace_public.jsonl",
            )
        if not isinstance(row.get("uses_tensor_values_for_dispatch"), bool):
            _error(
                errors,
                "route_trace.source",
                "route row must include boolean uses_tensor_values_for_dispatch",
                "candidate_route_trace_public.jsonl",
            )
        if not isinstance(row.get("uses_tensor_values_for_case_selection"), bool):
            _error(
                errors,
                "route_trace.source",
                "route row must include boolean uses_tensor_values_for_case_selection",
                "candidate_route_trace_public.jsonl",
            )
        if not isinstance(row.get("requires_tensor_guard_for_case_specific_path"), bool):
            _error(
                errors,
                "route_trace.source",
                "route row must include boolean requires_tensor_guard_for_case_specific_path",
                "candidate_route_trace_public.jsonl",
            )
        if not isinstance(row.get("classifier_needed_for_case_specific_path"), bool):
            _error(
                errors,
                "route_trace.source",
                "route row must include boolean classifier_needed_for_case_specific_path",
                "candidate_route_trace_public.jsonl",
            )
        if not isinstance(row.get("classifier_on_current_hot_path"), bool):
            _error(
                errors,
                "route_trace.source",
                "route row must include boolean classifier_on_current_hot_path",
                "candidate_route_trace_public.jsonl",
            )
        collision_shape = bool(row.get("shape_collision"))
        if "shape_collision" not in row:
            collision_shape = row.get("n") in {512, 1024}
        if collision_shape and row.get("uses_tensor_values_for_case_selection") is not True:
            _error(
                errors,
                "route_trace.source",
                "colliding benchmark shapes must record tensor-value case selection",
                "candidate_route_trace_public.jsonl",
            )
        if collision_shape and row.get("shape_only_case_selection") is not False:
            _error(
                errors,
                "route_trace.source",
                "colliding benchmark shapes must not record shape-only case selection",
                "candidate_route_trace_public.jsonl",
            )
        if collision_shape and row.get("requires_tensor_guard_for_case_specific_path") is not True:
            _error(
                errors,
                "route_trace.source",
                "colliding benchmark shapes must record that case-specific paths need tensor guards",
                "candidate_route_trace_public.jsonl",
            )
        if collision_shape and row.get("classifier_needed_for_case_specific_path") is not True:
            _error(
                errors,
                "route_trace.source",
                "colliding benchmark shapes must record that case-specific paths need a classifier/tensor guard",
                "candidate_route_trace_public.jsonl",
            )
        if not collision_shape and row.get("uses_tensor_values_for_case_selection") is not False:
            _error(
                errors,
                "route_trace.source",
                "non-colliding benchmark shapes should record shape-only case selection",
                "candidate_route_trace_public.jsonl",
            )
        if not collision_shape and row.get("shape_only_case_selection") is not True:
            _error(
                errors,
                "route_trace.source",
                "non-colliding benchmark shapes must record shape-only case selection",
                "candidate_route_trace_public.jsonl",
            )
        if row.get("classifier_on_current_hot_path") is not row.get("classifier_needed_for_current_candidate"):
            _error(
                errors,
                "route_trace.source",
                "classifier_on_current_hot_path must match classifier_needed_for_current_candidate",
                "candidate_route_trace_public.jsonl",
            )
        if collision_shape and row.get("classifier_needed_for_current_candidate") and row.get("uses_tensor_values_for_dispatch") is not True:
            _error(
                errors,
                "route_trace.source",
                "classifier-active colliding benchmark shapes must record tensor-value dispatch",
                "candidate_route_trace_public.jsonl",
            )
        if not collision_shape and row.get("classifier_needed_for_current_candidate") is not False:
            _error(
                errors,
                "route_trace.source",
                "non-colliding benchmark shapes should not require a classifier",
                "candidate_route_trace_public.jsonl",
            )
        if not collision_shape and row.get("classifier_needed_for_case_specific_path") is not False:
            _error(
                errors,
                "route_trace.source",
                "non-colliding benchmark shapes should not require a case-specific classifier",
                "candidate_route_trace_public.jsonl",
            )
        if row.get("n") in {512, 1024}:
            _validate_sampled_route_plan(row, errors)


def _validate_count_dict(
    value: Any,
    errors: list[dict[str, str]],
    *,
    field_name: str,
    batch: int,
) -> None:
    if not isinstance(value, dict) or not value:
        _error(
            errors,
            "route_trace.sampled_plan",
            f"{field_name} must be a non-empty object for colliding public benchmark shapes",
            "candidate_route_trace_public.jsonl",
        )
        return
    for key, raw_count in value.items():
        if not isinstance(key, str) or not key:
            _error(errors, "route_trace.sampled_plan", f"{field_name} has an invalid key", "candidate_route_trace_public.jsonl")
        if not _is_nonnegative_int(raw_count):
            _error(
                errors,
                "route_trace.sampled_plan",
                f"{field_name}.{key} must be a non-negative integer",
                "candidate_route_trace_public.jsonl",
            )
        elif int(raw_count) > batch:
            _error(
                errors,
                "route_trace.sampled_plan",
                f"{field_name}.{key} exceeds batch size {batch}",
                "candidate_route_trace_public.jsonl",
            )


def _validate_sampled_route_plan(row: dict[str, Any], errors: list[dict[str, str]]) -> None:
    batch = int(row["batch"])
    n = int(row["n"])
    if row.get("structured_sampled_plan") is not True:
        _error(
            errors,
            "route_trace.sampled_plan",
            "colliding public benchmark route rows must record structured_sampled_plan=true",
            "candidate_route_trace_public.jsonl",
        )
    if row.get("structured_sampled_matrix_count") != batch:
        _error(
            errors,
            "route_trace.sampled_plan",
            f"structured_sampled_matrix_count must equal batch size {batch}",
            "candidate_route_trace_public.jsonl",
        )
    sampled_rows = row.get("structured_sampled_row_count")
    if not _is_positive_int(sampled_rows):
        _error(
            errors,
            "route_trace.sampled_plan",
            "structured_sampled_row_count must be positive",
            "candidate_route_trace_public.jsonl",
        )
    elif int(sampled_rows) > min(n, 32):
        _error(
            errors,
            "route_trace.sampled_plan",
            "structured_sampled_row_count must stay within the bounded sampled-classifier budget",
            "candidate_route_trace_public.jsonl",
        )
    _validate_count_dict(
        row.get("structured_candidate_counts"),
        errors,
        field_name="structured_candidate_counts",
        batch=batch,
    )
    _validate_count_dict(
        row.get("structured_exact_check_counts"),
        errors,
        field_name="structured_exact_check_counts",
        batch=batch,
    )


def validate_guard_overhead(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    rows = _load_required_jsonl(suite_dir, GUARD_OVERHEAD_FILE, errors)
    if not rows:
        return
    expected = _expected_submission_provenance(suite_dir, "submission")
    if len(rows) != 12:
        _error(errors, "guard.count", f"expected 12 guard rows, got {len(rows)}", GUARD_OVERHEAD_FILE)
    for row in rows:
        _validate_submission_row_provenance(
            row,
            GUARD_OVERHEAD_FILE,
            "guard.provenance",
            errors,
            expected,
        )
        for key in ["route", "cold_wall_us", "wall_us", "repeats", "warmup"]:
            if key not in row:
                _error(errors, "guard.schema", f"guard row is missing {key}", GUARD_OVERHEAD_FILE)
        for key in [
            "case_metadata_passed_to_submission",
            "case_selection_info_sources",
            "dispatch_info_sources",
            "uses_tensor_values_for_dispatch",
            "uses_tensor_values_for_case_selection",
            "classifier_needed_for_current_candidate",
            "route_decision_source",
        ]:
            if key not in row:
                _error(errors, "guard.schema", f"guard row is missing {key}", GUARD_OVERHEAD_FILE)
        if float(row.get("cold_wall_us", -1.0)) < 0.0 or float(row.get("wall_us", -1.0)) < 0.0:
            _error(errors, "guard.timing", "guard timing fields must be non-negative", GUARD_OVERHEAD_FILE)


def validate_seed_sweep(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    rows = _load_required_jsonl(suite_dir, "seed_sweep_margin.jsonl", errors)
    if not rows:
        return
    expected = _expected_submission_provenance(suite_dir, "submission")
    seeds = []
    for row in rows:
        if "popcorn_seed" not in row:
            _error(errors, "seed_sweep.schema", "seed sweep row is missing popcorn_seed", "seed_sweep_margin.jsonl")
        else:
            seeds.append(row.get("popcorn_seed"))
        if not row.get("ok"):
            _error(errors, "seed_sweep.ok", "seed sweep row is not ok", "seed_sweep_margin.jsonl")
        if row.get("margin_ok") is False:
            _error(errors, "seed_sweep.margin", "seed sweep row missed the configured margin", "seed_sweep_margin.jsonl")
        _validate_submission_row_provenance(
            row,
            "seed_sweep_margin.jsonl",
            "seed_sweep.provenance",
            errors,
            expected,
        )
    if None not in seeds:
        _error(errors, "seed_sweep.coverage", "seed sweep must include the unmodified public seed", "seed_sweep_margin.jsonl")
    if not any(seed is not None for seed in seeds):
        _error(errors, "seed_sweep.coverage", "seed sweep must include at least one POPCORN_SEED-mutated seed", "seed_sweep_margin.jsonl")


def validate_quantization_seed_sweep(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    file_name = "quantization_seed_sweep.jsonl"
    rows = _load_required_jsonl(suite_dir, file_name, errors)
    if not rows:
        return
    expected = _expected_submission_provenance(suite_dir, "submission")
    cases = [row for row in rows if not row.get("summary")]
    summary = next((row for row in rows if row.get("summary")), None)
    if not cases:
        _error(errors, "quantization_seed_sweep.count", "quantization seed sweep has no case rows", file_name)
    seeds = []
    experiments = set()
    shapes = set()
    for row in cases:
        for key in [
            "experiment",
            "quantization",
            "ok",
            "margin_ok",
            "factor_scaled_max",
            "orth_scaled_max",
            "worst_factor_matrix",
            "worst_orth_matrix",
            "wall_us",
        ]:
            if key not in row:
                _error(errors, "quantization_seed_sweep.schema", f"quantization row is missing {key}", file_name)
        if "popcorn_seed" not in row:
            _error(errors, "quantization_seed_sweep.schema", "quantization row is missing popcorn_seed", file_name)
        else:
            seeds.append(row.get("popcorn_seed"))
        if row.get("experiment") in {"fp16-nearby", "tf32-input-nearby"}:
            experiments.add(row.get("experiment"))
        if not row.get("ok"):
            _error(errors, "quantization_seed_sweep.ok", "quantization row is not ok", file_name)
        if row.get("margin_ok") is not True:
            _error(errors, "quantization_seed_sweep.margin", "quantization row missed the configured margin", file_name)
        if not _is_positive_number(row.get("wall_us")):
            _error(errors, "quantization_seed_sweep.timing", "wall_us must be positive", file_name)
        batch = row.get("batch")
        n = row.get("n")
        if batch is not None and n is not None:
            shapes.add((int(batch), int(n)))
        _validate_submission_row_provenance(
            row,
            file_name,
            "quantization_seed_sweep.provenance",
            errors,
            expected,
        )
    missing_experiments = {"fp16-nearby", "tf32-input-nearby"} - experiments
    if missing_experiments:
        _error(
            errors,
            "quantization_seed_sweep.coverage",
            f"missing quantization experiments: {', '.join(sorted(missing_experiments))}",
            file_name,
        )
    for shape in [(640, 512), (60, 1024), (2, 4096)]:
        if shape not in shapes:
            _error(
                errors,
                "quantization_seed_sweep.coverage",
                f"quantization sweep must cover shape batch={shape[0]}, n={shape[1]}",
                file_name,
            )
    if None not in seeds:
        _error(
            errors,
            "quantization_seed_sweep.coverage",
            "quantization sweep must include the unmodified public seed",
            file_name,
        )
    if not any(seed is not None for seed in seeds):
        _error(
            errors,
            "quantization_seed_sweep.coverage",
            "quantization sweep must include at least one POPCORN_SEED-mutated seed",
            file_name,
        )
    if not summary or not summary.get("ok") or summary.get("num_failed") != 0:
        _error(errors, "quantization_seed_sweep.summary", "summary must report zero failures", file_name)
    elif summary.get("num_margin_failed") != 0:
        _error(errors, "quantization_seed_sweep.summary", "summary must report zero margin failures", file_name)


def validate_mixed_seed_sweep(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    file_name = "mixed_seed_sweep.jsonl"
    rows = _load_required_jsonl(suite_dir, file_name, errors)
    if not rows:
        return
    expected = _expected_submission_provenance(suite_dir, "submission")
    cases = [row for row in rows if not row.get("summary")]
    summary = next((row for row in rows if row.get("summary")), None)
    if not cases:
        _error(errors, "mixed_seed_sweep.count", "mixed seed sweep has no case rows", file_name)
    seeds = []
    shapes = set()
    sources = set()
    for row in cases:
        for key in [
            "case_source",
            "route",
            "route_ok",
            "ok",
            "margin_ok",
            "factor_scaled_max",
            "orth_scaled_max",
            "worst_factor_matrix",
            "worst_orth_matrix",
            "kernel_wall_us",
        ]:
            if key not in row:
                _error(errors, "mixed_seed_sweep.schema", f"mixed seed row is missing {key}", file_name)
        if "popcorn_seed" not in row:
            _error(errors, "mixed_seed_sweep.schema", "mixed seed row is missing popcorn_seed", file_name)
        else:
            seeds.append(row.get("popcorn_seed"))
        if str(row.get("case", "dense")) != "mixed":
            _error(errors, "mixed_seed_sweep.coverage", "mixed seed row is not a mixed case", file_name)
        if not row.get("ok"):
            _error(errors, "mixed_seed_sweep.ok", "mixed seed row is not ok", file_name)
        if row.get("margin_ok") is not True:
            _error(errors, "mixed_seed_sweep.margin", "mixed seed row missed the configured margin", file_name)
        if row.get("route_ok") is not True:
            _error(errors, "mixed_seed_sweep.route", "mixed benchmark route did not match expected mixed route or CUDA bypass", file_name)
        if not _is_positive_number(row.get("kernel_wall_us")):
            _error(errors, "mixed_seed_sweep.timing", "kernel_wall_us must be positive", file_name)
        batch = row.get("batch")
        n = row.get("n")
        if batch is not None and n is not None:
            shapes.add((int(batch), int(n)))
        if row.get("case_source"):
            sources.add(str(row["case_source"]))
        _validate_submission_row_provenance(
            row,
            file_name,
            "mixed_seed_sweep.provenance",
            errors,
            expected,
        )
    for shape in [(640, 512), (60, 1024), (16, 512), (4, 1024), (2, 2048)]:
        if shape not in shapes:
            _error(
                errors,
                "mixed_seed_sweep.coverage",
                f"mixed sweep must cover shape batch={shape[0]}, n={shape[1]}",
                file_name,
            )
    if "public_benchmarks" not in sources or "public_tests" not in sources:
        _error(errors, "mixed_seed_sweep.coverage", "mixed sweep must include public benchmark and public test cases", file_name)
    if None not in seeds:
        _error(errors, "mixed_seed_sweep.coverage", "mixed sweep must include the unmodified public seed", file_name)
    if not any(seed is not None for seed in seeds):
        _error(errors, "mixed_seed_sweep.coverage", "mixed sweep must include at least one POPCORN_SEED-mutated seed", file_name)
    if not summary or not summary.get("ok") or summary.get("num_failed") != 0:
        _error(errors, "mixed_seed_sweep.summary", "summary must report zero failures", file_name)
    elif summary.get("num_margin_failed") != 0 or summary.get("num_route_mismatch") != 0:
        _error(errors, "mixed_seed_sweep.summary", "summary must report zero margin and route failures", file_name)


def validate_classifier_seed_sweep(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    file_name = "classifier_seed_sweep.jsonl"
    rows = _load_required_jsonl(suite_dir, file_name, errors)
    if not rows:
        return
    expected = _expected_submission_provenance(suite_dir, "submission")
    cases = [row for row in rows if not row.get("summary")]
    summary = next((row for row in rows if row.get("summary")), None)
    if not cases:
        _error(errors, "classifier_seed_sweep.count", "classifier seed sweep has no case rows", file_name)
    seeds = []
    shapes = set()
    for row in cases:
        for key in [
            "sampled_class",
            "expected_sampled_class",
            "classifier_ok",
            "route",
            "route_ok",
            "sampled_class_wall_us",
            "route_wall_us",
        ]:
            if key not in row:
                _error(errors, "classifier_seed_sweep.schema", f"classifier seed row is missing {key}", file_name)
        if "popcorn_seed" not in row:
            _error(errors, "classifier_seed_sweep.schema", "classifier seed row is missing popcorn_seed", file_name)
        else:
            seeds.append(row.get("popcorn_seed"))
        if row.get("classifier_ok") is not True:
            _error(errors, "classifier_seed_sweep.classifier", "sampled classifier did not match expected class", file_name)
        if row.get("route_ok") is not True:
            _error(errors, "classifier_seed_sweep.route", "candidate route did not match expected route or CUDA bypass", file_name)
        if not row.get("ok"):
            _error(errors, "classifier_seed_sweep.ok", "classifier seed sweep row is not ok", file_name)
        if not _is_positive_number(row.get("sampled_class_wall_us")):
            _error(errors, "classifier_seed_sweep.timing", "sampled_class_wall_us must be positive", file_name)
        if not _is_positive_number(row.get("route_wall_us")):
            _error(errors, "classifier_seed_sweep.timing", "route_wall_us must be positive", file_name)
        batch = row.get("batch")
        n = row.get("n")
        if batch is not None and n is not None:
            shapes.add((int(batch), int(n)))
        _validate_submission_row_provenance(
            row,
            file_name,
            "classifier_seed_sweep.provenance",
            errors,
            expected,
        )
    if (640, 512) not in shapes or (60, 1024) not in shapes:
        _error(
            errors,
            "classifier_seed_sweep.coverage",
            "classifier sweep must cover both public colliding shape families",
            file_name,
        )
    if None not in seeds:
        _error(
            errors,
            "classifier_seed_sweep.coverage",
            "classifier sweep must include the unmodified public seed",
            file_name,
        )
    if not any(seed is not None for seed in seeds):
        _error(
            errors,
            "classifier_seed_sweep.coverage",
            "classifier sweep must include at least one POPCORN_SEED-mutated seed",
            file_name,
        )
    if not summary or not summary.get("ok") or summary.get("num_failed") != 0:
        _error(errors, "classifier_seed_sweep.summary", "summary must report zero failures", file_name)
    elif summary.get("num_classifier_mismatch") != 0 or summary.get("num_route_mismatch") != 0:
        _error(
            errors,
            "classifier_seed_sweep.summary",
            "summary must report zero classifier and route mismatches",
            file_name,
        )


def validate_tail_policy_sweep(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    rows = _load_required_jsonl(suite_dir, "candidate_tail_policy_sweep.jsonl", errors)
    if not rows:
        return
    expected = _expected_submission_provenance(suite_dir, "submission")
    cases = [row for row in rows if not row.get("summary")]
    summary = next((row for row in rows if row.get("summary")), None)
    if not cases:
        _error(errors, "tail_policy.count", "tail policy sweep has no case rows", "candidate_tail_policy_sweep.jsonl")
    seeds = []
    shapes = set()
    for row in cases:
        for key in ["candidate_route", "candidate_policy_cut", "tail_cut", "cut_source", "strategy"]:
            if key not in row:
                _error(errors, "tail_policy.schema", f"tail policy row is missing {key}", "candidate_tail_policy_sweep.jsonl")
        if "popcorn_seed" not in row:
            _error(errors, "tail_policy.schema", "tail policy row is missing popcorn_seed", "candidate_tail_policy_sweep.jsonl")
        else:
            seeds.append(row.get("popcorn_seed"))
        spec = row.get("spec")
        if isinstance(spec, dict) and "batch" in spec and "n" in spec:
            shapes.add((int(spec["batch"]), int(spec["n"])))
        if not row.get("ok"):
            _error(errors, "tail_policy.ok", "tail policy row is not ok", "candidate_tail_policy_sweep.jsonl")
        if row.get("margin_ok") is False:
            _error(errors, "tail_policy.margin", "tail policy row missed the configured margin", "candidate_tail_policy_sweep.jsonl")
        if "diagnostics" not in row:
            _error(errors, "tail_policy.diagnostics", "tail policy row is missing diagnostics", "candidate_tail_policy_sweep.jsonl")
        _validate_submission_row_provenance(
            row,
            "candidate_tail_policy_sweep.jsonl",
            "tail_policy.provenance",
            errors,
            expected,
        )
    if not summary or not summary.get("ok") or summary.get("num_failed") != 0:
        _error(errors, "tail_policy.summary", "summary must report zero failures", "candidate_tail_policy_sweep.jsonl")
    if None not in seeds:
        _error(errors, "tail_policy.coverage", "tail policy sweep must include the unmodified public seed", "candidate_tail_policy_sweep.jsonl")
    if not any(seed is not None for seed in seeds):
        _error(errors, "tail_policy.coverage", "tail policy sweep must include at least one POPCORN_SEED-mutated seed", "candidate_tail_policy_sweep.jsonl")
    required_shapes = {(640, 512), (60, 1024), (8, 2048), (2, 4096)}
    missing_shapes = sorted(required_shapes - shapes)
    if missing_shapes:
        formatted = ", ".join(f"{batch}x{n}" for batch, n in missing_shapes)
        _error(
            errors,
            "tail_policy.coverage",
            f"tail policy sweep must cover dense/structured shortcut benchmark shapes; missing {formatted}",
            "candidate_tail_policy_sweep.jsonl",
        )


def validate_experiments(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    rows = _load_required_jsonl(suite_dir, "experiments_public_benchmarks.jsonl", errors)
    if not rows:
        return
    expected = _expected_submission_provenance(suite_dir, "submission")
    names = {row.get("experiment") for row in rows}
    missing = EXPECTED_EXPERIMENTS - names
    if missing:
        _error(errors, "experiments.coverage", f"missing experiment rows: {', '.join(sorted(missing))}", "experiments_public_benchmarks.jsonl")
    for row in rows:
        _validate_submission_row_provenance(
            row,
            "experiments_public_benchmarks.jsonl",
            "experiments.provenance",
            errors,
            expected,
        )
        if not row.get("experiment") or "ok" not in row or not isinstance(row.get("spec"), dict):
            _error(errors, "experiments.schema", "experiment row is missing experiment, ok, or spec", "experiments_public_benchmarks.jsonl")


def validate_blocked_qr_sweep(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    manifest_path = suite_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        return
    try:
        manifest_rows = load_jsonl(manifest_path)
    except Exception:
        return
    if not _manifest_has_finished_step(manifest_rows, "blocked_qr_sweep"):
        return

    file_name = BLOCKED_QR_SWEEP_FILE
    if not (suite_dir / file_name).is_file():
        _error(errors, "blocked_qr_sweep.exists", "blocked QR sweep output is missing after optional step ran", file_name)
        return
    rows = _load_required_jsonl(suite_dir, file_name, errors)
    if not rows:
        return
    cases = [row for row in rows if not row.get("summary")]
    summary = next((row for row in rows if row.get("summary")), None)
    if not cases:
        _error(errors, "blocked_qr_sweep.count", "blocked QR sweep has no case rows", file_name)
    for row in cases:
        for key in [
            "case_index",
            "spec",
            "panel_width",
            "update_mode",
            "precision_mode",
            "r_maintenance_mode",
            "panel_refresh_mode",
            "ok",
            "message",
            "wall_us",
            "factor_scaled_max",
            "orth_scaled_max",
        ]:
            if key not in row:
                _error(errors, "blocked_qr_sweep.schema", f"blocked QR sweep row is missing {key}", file_name)
        if not _is_positive_int(row.get("panel_width")):
            _error(errors, "blocked_qr_sweep.schema", "panel_width must be a positive integer", file_name)
        if not isinstance(row.get("ok"), bool):
            _error(errors, "blocked_qr_sweep.schema", "ok must be boolean", file_name)
        if not _is_positive_number(row.get("wall_us")):
            _error(errors, "blocked_qr_sweep.timing", "wall_us must be positive", file_name)
        if "factor_scaled_max" in row and not isinstance(row.get("factor_scaled_max"), (int, float)):
            _error(errors, "blocked_qr_sweep.diagnostics", "factor_scaled_max must be numeric", file_name)
        if "orth_scaled_max" in row and not isinstance(row.get("orth_scaled_max"), (int, float)):
            _error(errors, "blocked_qr_sweep.diagnostics", "orth_scaled_max must be numeric", file_name)

    if not isinstance(summary, dict):
        _error(errors, "blocked_qr_sweep.summary", "blocked QR sweep summary row is missing", file_name)
    else:
        num_rows = summary.get("num_rows")
        num_failed = summary.get("num_failed")
        if num_rows != len(cases):
            _error(errors, "blocked_qr_sweep.summary", "summary num_rows does not match case rows", file_name)
        if num_failed != sum(1 for row in cases if not row.get("ok")):
            _error(errors, "blocked_qr_sweep.summary", "summary num_failed does not match case rows", file_name)
        for key in ["panel_widths", "update_modes", "precision_modes", "r_maintenance_modes", "panel_refresh_modes"]:
            if not isinstance(summary.get(key), list) or not summary.get(key):
                _error(errors, "blocked_qr_sweep.summary", f"summary is missing non-empty {key}", file_name)
        if not isinstance(summary.get("ok"), bool):
            _error(errors, "blocked_qr_sweep.summary", "summary ok must be boolean", file_name)

    summary_path = suite_dir / "suite_summary.json"
    if summary_path.is_file():
        try:
            suite_summary = json.loads(summary_path.read_text())
            blocked_summary = suite_summary.get("blocked_qr_sweep")
        except Exception as exc:
            _error(errors, "summary.parse", f"{type(exc).__name__}: {exc}", "suite_summary.json")
            blocked_summary = None
        if not isinstance(blocked_summary, dict):
            _error(errors, "summary.blocked_qr_sweep", "suite_summary.json must summarize optional blocked QR sweep", "suite_summary.json")
        elif blocked_summary.get("num_rows") != len(cases):
            _error(errors, "summary.blocked_qr_sweep", "summary blocked QR row count does not match sweep file", "suite_summary.json")

    summary_md = suite_dir / "suite_summary.md"
    if summary_md.is_file() and "Blocked QR Sweep" not in summary_md.read_text():
        _error(errors, "summary.blocked_qr_sweep", "suite_summary.md must include the Blocked QR Sweep section", "suite_summary.md")

    analysis_path = suite_dir / "suite_analysis.json"
    if analysis_path.is_file():
        try:
            analysis = json.loads(analysis_path.read_text())
            blocked_analysis = analysis.get("blocked_qr_sweep")
        except Exception as exc:
            _error(errors, "analysis.parse", f"{type(exc).__name__}: {exc}", "suite_analysis.json")
            blocked_analysis = None
        if not isinstance(blocked_analysis, dict):
            _error(errors, "analysis.blocked_qr_sweep", "suite_analysis.json must include optional blocked QR sweep", "suite_analysis.json")
        elif blocked_analysis.get("num_rows") != len(cases):
            _error(errors, "analysis.blocked_qr_sweep", "analysis blocked QR row count does not match sweep file", "suite_analysis.json")

    analysis_md = suite_dir / "suite_analysis.md"
    if analysis_md.is_file() and "Blocked QR Sweep" not in analysis_md.read_text():
        _error(errors, "analysis.blocked_qr_sweep", "suite_analysis.md must include the Blocked QR Sweep section", "suite_analysis.md")


def validate_summary(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    path = suite_dir / "suite_summary.json"
    if not path.is_file():
        _error(errors, "summary.exists", "suite_summary.json is missing", "suite_summary.json")
        return
    try:
        summary = json.loads(path.read_text())
    except Exception as exc:
        _error(errors, "summary.parse", f"{type(exc).__name__}: {exc}", "suite_summary.json")
        return
    if not summary.get("ok"):
        _error(errors, "summary.ok", "suite_summary.json ok field is false", "suite_summary.json")
    names = {comparison.get("name") for comparison in summary.get("comparisons", [])}
    for expected in ["public", "official_style", "smoke"]:
        if expected not in names:
            _error(errors, "summary.coverage", f"{expected} comparison is missing", "suite_summary.json")
    if len(summary.get("ablations", [])) != len(ABLATION_FILES):
        _error(errors, "summary.coverage", "not all route ablations are summarized", "suite_summary.json")
    runtime = summary.get("runtime")
    if not isinstance(runtime, dict):
        _error(errors, "summary.runtime", "suite_summary.json must include manifest runtime summary", "suite_summary.json")
    else:
        if not _is_positive_number(runtime.get("num_steps")):
            _error(errors, "summary.runtime", "runtime summary must include completed step count", "suite_summary.json")
        if not _is_positive_number(runtime.get("total_elapsed_s")):
            _error(errors, "summary.runtime", "runtime summary must include positive total_elapsed_s", "suite_summary.json")
        if not isinstance(runtime.get("slowest_steps"), list) or not runtime["slowest_steps"]:
            _error(errors, "summary.runtime", "runtime summary must include slowest_steps", "suite_summary.json")

    summary_md = suite_dir / "suite_summary.md"
    if not summary_md.is_file():
        _error(errors, "summary.exists", "suite_summary.md is missing", "suite_summary.md")
    elif "Runtime" not in summary_md.read_text():
        _error(errors, "summary.markdown", "suite_summary.md must include the Runtime section", "suite_summary.md")


def validate_analysis(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    path = suite_dir / "suite_analysis.json"
    if not path.is_file():
        _error(errors, "analysis.exists", "suite_analysis.json is missing", "suite_analysis.json")
        return
    try:
        analysis = json.loads(path.read_text())
    except Exception as exc:
        _error(errors, "analysis.parse", f"{type(exc).__name__}: {exc}", "suite_analysis.json")
        return
    if not analysis.get("ok"):
        _error(errors, "analysis.ok", "suite_analysis.json ok field is false", "suite_analysis.json")

    comparison = analysis.get("comparison")
    if not isinstance(comparison, dict) or comparison.get("num_common_cases") != 12:
        _error(errors, "analysis.comparison", "analysis must summarize a 12-case benchmark comparison", "suite_analysis.json")

    recommendation = analysis.get("final_algorithm_recommendation")
    if not isinstance(recommendation, dict):
        _error(
            errors,
            "analysis.final_algorithm_recommendation",
            "suite_analysis.json must include final_algorithm_recommendation",
            "suite_analysis.json",
        )
    else:
        if not recommendation.get("status"):
            _error(
                errors,
                "analysis.final_algorithm_recommendation",
                "final_algorithm_recommendation is missing status",
                "suite_analysis.json",
            )
        if not recommendation.get("primary_next_step"):
            _error(
                errors,
                "analysis.final_algorithm_recommendation",
                "final_algorithm_recommendation is missing primary_next_step",
                "suite_analysis.json",
            )
        actions = recommendation.get("priority_actions")
        if not isinstance(actions, list) or not actions:
            _error(
                errors,
                "analysis.final_algorithm_recommendation",
                "final_algorithm_recommendation must include priority_actions",
                "suite_analysis.json",
            )
        if recommendation.get("classifier_required_by_api") is not False:
            _error(
                errors,
                "analysis.final_algorithm_recommendation",
                "final_algorithm_recommendation must record that the official API does not pass classifier metadata",
                "suite_analysis.json",
            )

    families = analysis.get("shape_family_priorities")
    if not isinstance(families, list) or not families:
        _error(errors, "analysis.shape_families", "shape_family_priorities must be a non-empty list", "suite_analysis.json")
    else:
        required = ["batch", "n", "num_cases", "candidate_total_us", "candidate_geomean_us", "action"]
        for row in families:
            if not isinstance(row, dict):
                _error(errors, "analysis.shape_families", "shape family row must be an object", "suite_analysis.json")
                continue
            missing = [key for key in required if key not in row]
            if missing:
                _error(
                    errors,
                    "analysis.shape_families",
                    f"shape family row is missing: {', '.join(missing)}",
                    "suite_analysis.json",
                )
            if not _is_positive_number(row.get("num_cases")):
                _error(errors, "analysis.shape_families", "shape family num_cases must be positive", "suite_analysis.json")
            if not _is_positive_number(row.get("candidate_total_us")):
                _error(errors, "analysis.shape_families", "shape family candidate_total_us must be positive", "suite_analysis.json")
            if not _is_positive_number(row.get("candidate_geomean_us")):
                _error(errors, "analysis.shape_families", "shape family candidate_geomean_us must be positive", "suite_analysis.json")
            if not row.get("action"):
                _error(errors, "analysis.shape_families", "shape family action is missing", "suite_analysis.json")

    dispatch = analysis.get("data_dependent_dispatch")
    if not isinstance(dispatch, dict):
        _error(
            errors,
            "analysis.data_dependent_dispatch",
            "suite_analysis.json must include data_dependent_dispatch decisions",
            "suite_analysis.json",
        )
    else:
        if dispatch.get("case_metadata_passed_to_submission") is not False:
            _error(
                errors,
                "analysis.data_dependent_dispatch",
                "data_dependent_dispatch must record that case metadata is not passed",
                "suite_analysis.json",
            )
        if not _is_positive_number(dispatch.get("num_shape_families")):
            _error(
                errors,
                "analysis.data_dependent_dispatch",
                "data_dependent_dispatch must include at least one colliding shape family",
                "suite_analysis.json",
            )
        for row in dispatch.get("families", []):
            if not isinstance(row, dict):
                _error(
                    errors,
                    "analysis.data_dependent_dispatch",
                    "data_dependent_dispatch family row must be an object",
                    "suite_analysis.json",
                )
                continue
            if row.get("case_metadata_passed_to_submission") is not False:
                _error(
                    errors,
                    "analysis.data_dependent_dispatch",
                    "family row must record that case metadata is not passed",
                    "suite_analysis.json",
                )
            if row.get("case_info_source") != "tensor_values":
                _error(
                    errors,
                    "analysis.data_dependent_dispatch",
                    "family row must record tensor_values as the case information source",
                    "suite_analysis.json",
                )
            if row.get("uses_tensor_values_for_case_selection") is not True:
                _error(
                    errors,
                    "analysis.data_dependent_dispatch",
                    "family row must record that tensor values are used for case selection",
                    "suite_analysis.json",
                )
            if "data.shape" not in set(row.get("route_decision_sources", [])):
                _error(
                    errors,
                    "analysis.data_dependent_dispatch",
                    "family row must include data.shape in route_decision_sources",
                    "suite_analysis.json",
                )
            if not row.get("routes"):
                _error(
                    errors,
                    "analysis.data_dependent_dispatch",
                    "family row must include the traced candidate routes",
                    "suite_analysis.json",
                )
            if row.get("classifier_needed_for_case_specific_path") is not True:
                _error(
                    errors,
                    "analysis.data_dependent_dispatch",
                    "family row must record that case-specific paths need a classifier/tensor guard",
                    "suite_analysis.json",
                )
            if not isinstance(row.get("classifier_needed_for_current_candidate"), bool):
                _error(
                    errors,
                    "analysis.data_dependent_dispatch",
                    "family row must include classifier_needed_for_current_candidate",
                    "suite_analysis.json",
                )
            if row.get("classifier_on_current_hot_path") is not row.get("classifier_needed_for_current_candidate"):
                _error(
                    errors,
                    "analysis.data_dependent_dispatch",
                    "classifier_on_current_hot_path must match classifier_needed_for_current_candidate",
                    "suite_analysis.json",
                )
            if row.get("classifier_needed_for_current_candidate") and row.get("uses_tensor_values_for_dispatch") is not True:
                _error(
                    errors,
                    "analysis.data_dependent_dispatch",
                    "classifier-active family row must record tensor-value dispatch",
                    "suite_analysis.json",
                )
            if row.get("uses_tensor_values_for_dispatch") and "tensor_values" not in set(row.get("route_decision_sources", [])):
                _error(
                    errors,
                    "analysis.data_dependent_dispatch",
                    "tensor-dispatched family row must include tensor_values in route_decision_sources",
                    "suite_analysis.json",
                )
            if not _is_positive_number(row.get("cuda_first_structured_geomean_us")):
                _error(
                    errors,
                    "analysis.data_dependent_dispatch",
                    "family row must include a positive cuda_first_structured_geomean_us",
                    "suite_analysis.json",
                )
            if not _is_positive_number(row.get("cuda_first_structured_over_default")):
                _error(
                    errors,
                    "analysis.data_dependent_dispatch",
                    "family row must include a positive cuda_first_structured_over_default ratio",
                    "suite_analysis.json",
                )
            if not _is_positive_number(row.get("default_geomean_us")):
                _error(
                    errors,
                    "analysis.data_dependent_dispatch",
                    "family row must include a positive default_geomean_us",
                    "suite_analysis.json",
                )
            if not _is_positive_number(row.get("no_structured_geomean_us")):
                _error(
                    errors,
                    "analysis.data_dependent_dispatch",
                    "family row must include a positive no_structured_geomean_us",
                    "suite_analysis.json",
                )
            if not _is_positive_number(row.get("no_structured_over_default")):
                _error(
                    errors,
                    "analysis.data_dependent_dispatch",
                    "family row must include a positive no_structured_over_default ratio",
                    "suite_analysis.json",
                )
            if not _is_positive_number(row.get("no_data_dependent_geomean_us")):
                _error(
                    errors,
                    "analysis.data_dependent_dispatch",
                    "family row must include a positive no_data_dependent_geomean_us",
                    "suite_analysis.json",
                )
            if not _is_positive_number(row.get("no_data_dependent_over_default")):
                _error(
                    errors,
                    "analysis.data_dependent_dispatch",
                    "family row must include a positive no_data_dependent_over_default ratio",
                    "suite_analysis.json",
                )
            decision = row.get("decision")
            if not decision:
                _error(
                    errors,
                    "analysis.data_dependent_dispatch",
                    "family row is missing a dispatch decision",
                    "suite_analysis.json",
                )
            route_order_decision = row.get("route_order_decision")
            if not route_order_decision:
                _error(
                    errors,
                    "analysis.data_dependent_dispatch",
                    "family row is missing a structured route-order decision",
                    "suite_analysis.json",
                )
            elif route_order_decision == "insufficient-data":
                _error(
                    errors,
                    "analysis.data_dependent_dispatch",
                    "completed suite must not leave structured route-order dispatch at insufficient-data",
                    "suite_analysis.json",
                )
            elif decision == "insufficient-data":
                _error(
                    errors,
                    "analysis.data_dependent_dispatch",
                    "completed suite must not leave data-dependent dispatch at insufficient-data",
                    "suite_analysis.json",
                )
            classifier_decision = row.get("classifier_decision")
            if not classifier_decision:
                _error(
                    errors,
                    "analysis.data_dependent_dispatch",
                    "family row is missing a classifier_decision",
                    "suite_analysis.json",
                )
            elif classifier_decision != decision:
                _error(
                    errors,
                    "analysis.data_dependent_dispatch",
                    "classifier_decision must match the primary decision for colliding shape families",
                    "suite_analysis.json",
                )
            data_decision = row.get("data_dependent_decision")
            if not data_decision:
                _error(
                    errors,
                    "analysis.data_dependent_dispatch",
                    "family row is missing a data_dependent_decision",
                    "suite_analysis.json",
                )
            elif data_decision == "insufficient-data":
                _error(
                    errors,
                    "analysis.data_dependent_dispatch",
                    "completed suite must not leave all-tensor-guard dispatch at insufficient-data",
                    "suite_analysis.json",
                )

    large_cuda = analysis.get("large_cuda_probe_ablation")
    if not isinstance(large_cuda, dict):
        _error(
            errors,
            "analysis.large_cuda_probe_ablation",
            "suite_analysis.json must include the QR512/QR1024 CUDA-probe ablation decision",
            "suite_analysis.json",
        )
    else:
        if large_cuda.get("name") != "no_qr512_qr1024_cuda":
            _error(
                errors,
                "analysis.large_cuda_probe_ablation",
                "large CUDA probe ablation must be named no_qr512_qr1024_cuda",
                "suite_analysis.json",
            )
        if not _is_positive_number(large_cuda.get("num_target_cases")):
            _error(
                errors,
                "analysis.large_cuda_probe_ablation",
                "large CUDA probe ablation must include target QR512/QR1024 cases",
                "suite_analysis.json",
            )
        if not _is_positive_number(large_cuda.get("default_geomean_us")):
            _error(
                errors,
                "analysis.large_cuda_probe_ablation",
                "large CUDA probe ablation must include positive target default_geomean_us",
                "suite_analysis.json",
            )
        if not _is_positive_number(large_cuda.get("ablation_geomean_us")):
            _error(
                errors,
                "analysis.large_cuda_probe_ablation",
                "large CUDA probe ablation must include positive target ablation_geomean_us",
                "suite_analysis.json",
            )
        if not _is_positive_number(large_cuda.get("ablation_over_default")):
            _error(
                errors,
                "analysis.large_cuda_probe_ablation",
                "large CUDA probe ablation must include a positive target ablation/default ratio",
                "suite_analysis.json",
            )
        if large_cuda.get("decision") in (None, "", "insufficient-data"):
            _error(
                errors,
                "analysis.large_cuda_probe_ablation",
                "completed suite must not leave the large CUDA probe ablation decision at insufficient-data",
                "suite_analysis.json",
            )
        target_shapes = set(large_cuda.get("target_shapes") or [])
        if {"640x512", "60x1024"} - target_shapes:
            _error(
                errors,
                "analysis.large_cuda_probe_ablation",
                "large CUDA probe ablation must cover both 640x512 and 60x1024 target shapes",
                "suite_analysis.json",
            )
        families = large_cuda.get("families")
        if not isinstance(families, list) or len(families) < 2:
            _error(
                errors,
                "analysis.large_cuda_probe_ablation",
                "large CUDA probe ablation must include per-family rows for QR512 and QR1024",
                "suite_analysis.json",
            )

    output_layouts = analysis.get("output_layouts")
    if not isinstance(output_layouts, dict):
        _error(
            errors,
            "analysis.output_layouts",
            "suite_analysis.json must include output_layouts from public benchmark correctness rows",
            "suite_analysis.json",
        )
    else:
        if output_layouts.get("num_cases") != 12:
            _error(
                errors,
                "analysis.output_layouts",
                "output_layouts must summarize 12 public benchmark cases",
                "suite_analysis.json",
            )
        if not isinstance(output_layouts.get("layout_counts"), dict):
            _error(
                errors,
                "analysis.output_layouts",
                "output_layouts must include layout_counts",
                "suite_analysis.json",
            )
        for key in ["num_column_major", "num_torch_contiguous", "num_policy_mismatch"]:
            value = output_layouts.get(key)
            if not isinstance(value, int) or value < 0:
                _error(
                    errors,
                    "analysis.output_layouts",
                    f"output_layouts must include nonnegative integer {key}",
                    "suite_analysis.json",
                )
        if output_layouts.get("num_policy_mismatch") != 0:
            _error(
                errors,
                "analysis.output_layouts",
                "output_layouts reports policy/actual H layout mismatches",
                "suite_analysis.json",
            )
        if not isinstance(output_layouts.get("shape_families"), list) or not output_layouts["shape_families"]:
            _error(
                errors,
                "analysis.output_layouts",
                "output_layouts must include shape_families",
                "suite_analysis.json",
            )

    markdown_path = suite_dir / "suite_analysis.md"
    if not markdown_path.is_file():
        _error(errors, "analysis.exists", "suite_analysis.md is missing", "suite_analysis.md")
    else:
        markdown_text = markdown_path.read_text()
        if "Final Algorithm Recommendation" not in markdown_text:
            _error(
                errors,
                "analysis.markdown",
                "suite_analysis.md must include the Final Algorithm Recommendation section",
                "suite_analysis.md",
            )
        if "Shape Family Priorities" not in markdown_text:
            _error(
                errors,
                "analysis.markdown",
                "suite_analysis.md must include the Shape Family Priorities section",
                "suite_analysis.md",
            )
        if "Data-Dependent Dispatch" not in markdown_text:
            _error(
                errors,
                "analysis.markdown",
                "suite_analysis.md must include the Data-Dependent Dispatch section",
                "suite_analysis.md",
            )
        if "Output H Layout" not in markdown_text:
            _error(
                errors,
                "analysis.markdown",
                "suite_analysis.md must include the Output H Layout section",
                "suite_analysis.md",
            )


def validate_tail_policy_tune(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    manifest_path = suite_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        return
    try:
        manifest_rows = load_jsonl(manifest_path)
    except Exception:
        return
    if not _manifest_has_finished_step(manifest_rows, "tail_policy_tune"):
        return

    file_name = str(TAIL_POLICY_TUNE_SUMMARY)
    path = suite_dir / file_name
    if not path.is_file():
        _error(errors, "tail_policy_tune.exists", "tail policy tune summary is missing after optional step ran", file_name)
        return
    try:
        raw = json.loads(path.read_text())
    except Exception as exc:
        _error(errors, "tail_policy_tune.parse", f"{type(exc).__name__}: {exc}", file_name)
        return

    if raw.get("hard_failed") is True:
        _error(errors, "tail_policy_tune.hard_failed", "tail policy tuning recorded a hard failure", file_name)

    results = raw.get("results")
    if not isinstance(results, list) or not results:
        _error(errors, "tail_policy_tune.results", "tail policy tune summary must include non-empty results", file_name)
        results = []

    num_configs = raw.get("num_configs")
    if not _is_positive_int(num_configs):
        _error(errors, "tail_policy_tune.num_configs", "num_configs must be a positive integer", file_name)
        num_configs = None
    elif num_configs != len(results):
        _error(
            errors,
            "tail_policy_tune.num_configs",
            f"num_configs={num_configs} does not match {len(results)} result rows",
            file_name,
        )

    num_failed_configs = 0
    num_benchmarked_configs = 0
    for index, row in enumerate(results):
        if not isinstance(row, dict):
            _error(errors, "tail_policy_tune.results", f"result {index} must be an object", file_name)
            continue
        if not row.get("name"):
            _error(errors, "tail_policy_tune.results", f"result {index} is missing a config name", file_name)
        correctness = row.get("correctness")
        if not isinstance(correctness, dict):
            _error(errors, "tail_policy_tune.correctness", f"result {index} is missing correctness summary", file_name)
        else:
            failed = correctness.get("num_failed")
            if not _is_nonnegative_int(failed):
                _error(
                    errors,
                    "tail_policy_tune.correctness",
                    f"result {index} correctness.num_failed must be a nonnegative integer",
                    file_name,
                )
            elif failed > 0:
                num_failed_configs += 1

        benchmark = row.get("benchmark")
        if benchmark is None:
            continue
        if not isinstance(benchmark, dict):
            _error(errors, "tail_policy_tune.benchmark", f"result {index} benchmark must be null or an object", file_name)
            continue
        if not _is_positive_number(benchmark.get("geomean_us")):
            _error(errors, "tail_policy_tune.benchmark", f"result {index} benchmark.geomean_us must be positive", file_name)
        else:
            num_benchmarked_configs += 1

    best = raw.get("best")
    if num_benchmarked_configs > 0:
        if not isinstance(best, dict):
            _error(errors, "tail_policy_tune.best", "benchmarkable tune results must include a best config", file_name)
        else:
            best_benchmark = best.get("benchmark")
            if not isinstance(best_benchmark, dict) or not _is_positive_number(best_benchmark.get("geomean_us")):
                _error(errors, "tail_policy_tune.best", "best config must include positive benchmark.geomean_us", file_name)
    elif best is not None and not isinstance(best, dict):
        _error(errors, "tail_policy_tune.best", "best must be null or an object", file_name)

    summary_path = suite_dir / "suite_summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text())
            tune = summary.get("tail_policy_tune")
        except Exception as exc:
            _error(errors, "summary.parse", f"{type(exc).__name__}: {exc}", "suite_summary.json")
            tune = None
        if not isinstance(tune, dict):
            _error(errors, "summary.tail_policy_tune", "suite_summary.json must summarize optional tail-policy tuning", "suite_summary.json")
        else:
            if num_configs is not None and tune.get("num_configs") != num_configs:
                _error(errors, "summary.tail_policy_tune", "summary num_configs does not match tune summary", "suite_summary.json")
            if tune.get("num_failed_configs") != num_failed_configs:
                _error(
                    errors,
                    "summary.tail_policy_tune",
                    "summary num_failed_configs does not match tune summary",
                    "suite_summary.json",
                )
            if tune.get("num_benchmarked_configs") != num_benchmarked_configs:
                _error(
                    errors,
                    "summary.tail_policy_tune",
                    "summary num_benchmarked_configs does not match tune summary",
                    "suite_summary.json",
                )

    summary_md = suite_dir / "suite_summary.md"
    if summary_md.is_file() and "Tail Policy Tune" not in summary_md.read_text():
        _error(errors, "summary.tail_policy_tune", "suite_summary.md must include the Tail Policy Tune section", "suite_summary.md")

    analysis_path = suite_dir / "suite_analysis.json"
    if analysis_path.is_file():
        try:
            analysis = json.loads(analysis_path.read_text())
            tune = analysis.get("tail_policy_tune")
        except Exception as exc:
            _error(errors, "analysis.parse", f"{type(exc).__name__}: {exc}", "suite_analysis.json")
            tune = None
        if not isinstance(tune, dict):
            _error(errors, "analysis.tail_policy_tune", "suite_analysis.json must include optional tail-policy tuning", "suite_analysis.json")
        else:
            if num_configs is not None and tune.get("num_configs") != num_configs:
                _error(errors, "analysis.tail_policy_tune", "analysis num_configs does not match tune summary", "suite_analysis.json")
            if tune.get("num_failed_configs") != num_failed_configs:
                _error(
                    errors,
                    "analysis.tail_policy_tune",
                    "analysis num_failed_configs does not match tune summary",
                    "suite_analysis.json",
                )
            if tune.get("num_benchmarked_configs") != num_benchmarked_configs:
                _error(
                    errors,
                    "analysis.tail_policy_tune",
                    "analysis num_benchmarked_configs does not match tune summary",
                    "suite_analysis.json",
                )

    analysis_md = suite_dir / "suite_analysis.md"
    if analysis_md.is_file() and "Tail Policy Tune" not in analysis_md.read_text():
        _error(errors, "analysis.tail_policy_tune", "suite_analysis.md must include the Tail Policy Tune section", "suite_analysis.md")


def validate_candidate_config_generated_plan(
    suite_dir: Path,
    manifest_rows: list[dict[str, Any]],
    tune_results: list[dict[str, Any]],
    errors: list[dict[str, str]],
) -> None:
    event = _candidate_config_large_kernel_plan_event(manifest_rows)
    if event is None:
        return

    plan_path = _resolve_suite_path(suite_dir, event.get("path"))
    if plan_path is None:
        _error(
            errors,
            "candidate_config_tune.generated_configs",
            "generated candidate config plan event is missing a path",
            "manifest.jsonl",
        )
        return

    rel_plan_path = _suite_relative_path(suite_dir, plan_path)
    file_name = rel_plan_path or str(plan_path)
    if rel_plan_path is None:
        _error(
            errors,
            "candidate_config_tune.generated_configs",
            "generated candidate config plan path must be inside the suite directory",
            "manifest.jsonl",
        )
    if not plan_path.is_file():
        _error(
            errors,
            "candidate_config_tune.generated_configs",
            "generated candidate config JSONL is missing",
            file_name,
        )
        return

    try:
        generated_rows = load_jsonl(plan_path)
    except Exception as exc:
        _error(
            errors,
            "candidate_config_tune.generated_configs",
            f"could not parse generated candidate config JSONL: {type(exc).__name__}: {exc}",
            file_name,
        )
        return
    if not generated_rows:
        _error(
            errors,
            "candidate_config_tune.generated_configs",
            "generated candidate config JSONL is empty",
            file_name,
        )
        return

    generated_names: list[str] = []
    generated_by_name: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(generated_rows):
        if not isinstance(row, dict):
            _error(errors, "candidate_config_tune.generated_configs", f"generated row {index} must be an object", file_name)
            continue
        name = row.get("name")
        if not isinstance(name, str) or not name:
            _error(errors, "candidate_config_tune.generated_configs", f"generated row {index} is missing a name", file_name)
            continue
        env = row.get("env")
        if not isinstance(env, dict):
            _error(errors, "candidate_config_tune.generated_configs", f"generated row {name} env must be an object", file_name)
            continue
        generated_names.append(name)
        generated_by_name[name] = row

    if _is_positive_int(event.get("num_configs")) and int(event["num_configs"]) != len(generated_rows):
        _error(
            errors,
            "candidate_config_tune.generated_configs",
            f"manifest num_configs={event['num_configs']} does not match {len(generated_rows)} generated rows",
            "manifest.jsonl",
        )
    elif not _is_positive_int(event.get("num_configs")):
        _error(
            errors,
            "candidate_config_tune.generated_configs",
            "manifest generated plan event must record positive num_configs",
            "manifest.jsonl",
        )

    manifest_names = event.get("config_names")
    if not isinstance(manifest_names, list) or not all(isinstance(name, str) for name in manifest_names):
        _error(
            errors,
            "candidate_config_tune.generated_configs",
            "manifest generated plan event must record config_names",
            "manifest.jsonl",
        )
    elif manifest_names != generated_names:
        _error(
            errors,
            "candidate_config_tune.generated_configs",
            "manifest config_names do not match generated JSONL row order",
            "manifest.jsonl",
        )

    results_by_name = {
        str(row.get("name")): row
        for row in tune_results
        if isinstance(row, dict) and isinstance(row.get("name"), str)
    }
    missing_tuned_names = [name for name in generated_names if name not in results_by_name]
    if missing_tuned_names:
        _error(
            errors,
            "candidate_config_tune.generated_configs",
            "tune summary is missing generated configs: " + ", ".join(missing_tuned_names[:5]),
            str(CANDIDATE_CONFIG_TUNE_SUMMARY),
        )

    for name, generated in generated_by_name.items():
        result = results_by_name.get(name)
        if result is None:
            continue
        result_env = result.get("env")
        generated_env = generated.get("env")
        if not isinstance(result_env, dict) or not isinstance(generated_env, dict):
            continue
        mismatches = [
            key
            for key, value in generated_env.items()
            if str(result_env.get(key)) != str(value)
        ]
        if mismatches:
            _error(
                errors,
                "candidate_config_tune.generated_configs",
                f"tune result {name} does not preserve generated env keys: {', '.join(mismatches[:5])}",
                str(CANDIDATE_CONFIG_TUNE_SUMMARY),
            )

    policy_target = event.get("policy_target")
    if not isinstance(policy_target, dict):
        return
    required_modes = policy_target.get("required_repair_modes")
    constraints = _repair_constraints_from_required_modes(required_modes)
    if not constraints:
        return

    applied = policy_target.get("applied_axis_constraints")
    if applied != constraints:
        _error(
            errors,
            "candidate_config_tune.generated_configs",
            f"policy target applied_axis_constraints must be {constraints}, got {applied}",
            "manifest.jsonl",
        )

    env_prefix = policy_target.get("env_prefix")
    if not isinstance(env_prefix, str) or not env_prefix:
        shape_label = str(policy_target.get("shape_label") or event.get("shape_label") or "").upper()
        env_prefix = "FAST_QR_" + "".join(char if char.isalnum() else "_" for char in shape_label).strip("_")
    required_env = _repair_env_from_required_modes(env_prefix, required_modes)
    for name, generated in generated_by_name.items():
        env = generated.get("env")
        if not isinstance(env, dict):
            continue
        missing_or_bad = [
            f"{key}={value}"
            for key, value in required_env.items()
            if str(env.get(key)) != str(value)
        ]
        if missing_or_bad:
            _error(
                errors,
                "candidate_config_tune.generated_configs",
                f"generated config {name} is missing required repair env: {', '.join(missing_or_bad)}",
                file_name,
            )


def validate_candidate_config_tune(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    manifest_path = suite_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        return
    try:
        manifest_rows = load_jsonl(manifest_path)
    except Exception:
        return
    if not _manifest_has_finished_step(manifest_rows, "candidate_config_tune"):
        return

    file_name = str(CANDIDATE_CONFIG_TUNE_SUMMARY)
    path = suite_dir / file_name
    if not path.is_file():
        _error(errors, "candidate_config_tune.exists", "candidate config tune summary is missing after optional step ran", file_name)
        return
    try:
        raw = json.loads(path.read_text())
    except Exception as exc:
        _error(errors, "candidate_config_tune.parse", f"{type(exc).__name__}: {exc}", file_name)
        return

    if raw.get("hard_failed") is True:
        _error(errors, "candidate_config_tune.hard_failed", "candidate config tuning recorded a hard failure", file_name)

    results = raw.get("results")
    if not isinstance(results, list) or not results:
        _error(errors, "candidate_config_tune.results", "candidate config tune summary must include non-empty results", file_name)
        results = []

    num_configs = raw.get("num_configs")
    if not _is_positive_int(num_configs):
        _error(errors, "candidate_config_tune.num_configs", "num_configs must be a positive integer", file_name)
        num_configs = None
    elif num_configs != len(results):
        _error(
            errors,
            "candidate_config_tune.num_configs",
            f"num_configs={num_configs} does not match {len(results)} result rows",
            file_name,
        )

    num_failed_configs = 0
    num_benchmarked_configs = 0
    for index, row in enumerate(results):
        if not isinstance(row, dict):
            _error(errors, "candidate_config_tune.results", f"result {index} must be an object", file_name)
            continue
        if not row.get("name"):
            _error(errors, "candidate_config_tune.results", f"result {index} is missing a config name", file_name)
        env = row.get("env")
        if not isinstance(env, dict):
            _error(errors, "candidate_config_tune.env", f"result {index} env must be an object", file_name)
        correctness = row.get("correctness")
        if not isinstance(correctness, dict):
            _error(errors, "candidate_config_tune.correctness", f"result {index} is missing correctness summary", file_name)
        else:
            failed = correctness.get("num_failed")
            if not _is_nonnegative_int(failed):
                _error(
                    errors,
                    "candidate_config_tune.correctness",
                    f"result {index} correctness.num_failed must be a nonnegative integer",
                    file_name,
                )
            elif failed > 0:
                num_failed_configs += 1

        benchmark = row.get("benchmark")
        if benchmark is None:
            continue
        if not isinstance(benchmark, dict):
            _error(errors, "candidate_config_tune.benchmark", f"result {index} benchmark must be null or an object", file_name)
            continue
        if not _is_positive_number(benchmark.get("geomean_us")):
            _error(errors, "candidate_config_tune.benchmark", f"result {index} benchmark.geomean_us must be positive", file_name)
        else:
            num_benchmarked_configs += 1

    best = raw.get("best")
    if num_benchmarked_configs > 0:
        if not isinstance(best, dict):
            _error(errors, "candidate_config_tune.best", "benchmarkable tune results must include a best config", file_name)
        else:
            best_benchmark = best.get("benchmark")
            if not isinstance(best_benchmark, dict) or not _is_positive_number(best_benchmark.get("geomean_us")):
                _error(errors, "candidate_config_tune.best", "best config must include positive benchmark.geomean_us", file_name)
    elif best is not None and not isinstance(best, dict):
        _error(errors, "candidate_config_tune.best", "best must be null or an object", file_name)

    validate_candidate_config_generated_plan(suite_dir, manifest_rows, results, errors)

    summary_path = suite_dir / "suite_summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text())
            tune = summary.get("candidate_config_tune")
        except Exception as exc:
            _error(errors, "summary.parse", f"{type(exc).__name__}: {exc}", "suite_summary.json")
            tune = None
        if not isinstance(tune, dict):
            _error(errors, "summary.candidate_config_tune", "suite_summary.json must summarize optional candidate config tuning", "suite_summary.json")
        else:
            if num_configs is not None and tune.get("num_configs") != num_configs:
                _error(errors, "summary.candidate_config_tune", "summary num_configs does not match tune summary", "suite_summary.json")
            if tune.get("num_failed_configs") != num_failed_configs:
                _error(
                    errors,
                    "summary.candidate_config_tune",
                    "summary num_failed_configs does not match tune summary",
                    "suite_summary.json",
                )
            if tune.get("num_benchmarked_configs") != num_benchmarked_configs:
                _error(
                    errors,
                    "summary.candidate_config_tune",
                    "summary num_benchmarked_configs does not match tune summary",
                    "suite_summary.json",
                )

    summary_md = suite_dir / "suite_summary.md"
    if summary_md.is_file() and "Candidate Config Tune" not in summary_md.read_text():
        _error(errors, "summary.candidate_config_tune", "suite_summary.md must include the Candidate Config Tune section", "suite_summary.md")

    analysis_path = suite_dir / "suite_analysis.json"
    if analysis_path.is_file():
        try:
            analysis = json.loads(analysis_path.read_text())
            tune = analysis.get("candidate_config_tune")
        except Exception as exc:
            _error(errors, "analysis.parse", f"{type(exc).__name__}: {exc}", "suite_analysis.json")
            tune = None
        if not isinstance(tune, dict):
            _error(errors, "analysis.candidate_config_tune", "suite_analysis.json must include optional candidate config tuning", "suite_analysis.json")
        else:
            if num_configs is not None and tune.get("num_configs") != num_configs:
                _error(errors, "analysis.candidate_config_tune", "analysis num_configs does not match tune summary", "suite_analysis.json")
            if tune.get("num_failed_configs") != num_failed_configs:
                _error(
                    errors,
                    "analysis.candidate_config_tune",
                    "analysis num_failed_configs does not match tune summary",
                    "suite_analysis.json",
                )
            if tune.get("num_benchmarked_configs") != num_benchmarked_configs:
                _error(
                    errors,
                    "analysis.candidate_config_tune",
                    "analysis num_benchmarked_configs does not match tune summary",
                    "suite_analysis.json",
                )

    analysis_md = suite_dir / "suite_analysis.md"
    if analysis_md.is_file() and "Candidate Config Tune" not in analysis_md.read_text():
        _error(errors, "analysis.candidate_config_tune", "suite_analysis.md must include the Candidate Config Tune section", "suite_analysis.md")


def validate_popcorn_submit(
    suite_dir: Path,
    errors: list[dict[str, str]],
    step_name: str,
    artifact_dir: str,
    mode: str,
    required_files: list[str],
) -> None:
    manifest_path = suite_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        return
    try:
        suite_manifest_rows = load_jsonl(manifest_path)
    except Exception:
        return
    if not _manifest_has_finished_step(suite_manifest_rows, step_name):
        return
    expected = _expected_submission_provenance(suite_dir, "submission")
    check_prefix = step_name

    for file_name in required_files:
        if not (suite_dir / file_name).is_file():
            _error(errors, f"{check_prefix}.exists", f"optional Popcorn {mode} artifact is missing", file_name)

    submit_manifest_name = f"{artifact_dir}/manifest.jsonl"
    submit_manifest_path = suite_dir / submit_manifest_name
    if not submit_manifest_path.is_file():
        return
    try:
        rows = load_jsonl(submit_manifest_path)
    except Exception as exc:
        _error(errors, f"{check_prefix}.manifest", f"{type(exc).__name__}: {exc}", submit_manifest_name)
        return
    if not rows:
        _error(errors, f"{check_prefix}.manifest", "Popcorn manifest is empty", submit_manifest_name)
        return

    start = rows[0]
    if start.get("event") != "submit_start":
        _error(errors, f"{check_prefix}.manifest", "Popcorn manifest must start with submit_start", submit_manifest_name)
    if start.get("dry_run") is not False:
        _error(errors, f"{check_prefix}.manifest", f"suite Popcorn {mode} must not be a dry run", submit_manifest_name)
    args = start.get("args")
    if not isinstance(args, dict):
        _error(errors, f"{check_prefix}.manifest", "Popcorn submit_start row is missing args", submit_manifest_name)
    else:
        if args.get("mode") != mode:
            _error(errors, f"{check_prefix}.manifest", f"suite Popcorn step must run mode={mode}", submit_manifest_name)
        if args.get("gpu") != "B200":
            _error(errors, f"{check_prefix}.manifest", "suite Popcorn step must target gpu=B200", submit_manifest_name)
        if args.get("leaderboard") != "qr_v2":
            _error(errors, f"{check_prefix}.manifest", "suite Popcorn step must target leaderboard=qr_v2", submit_manifest_name)

    for key in ["source_submission", "staged_submission"]:
        _validate_file_provenance_dict(
            start.get(key),
            submit_manifest_name,
            f"{check_prefix}.provenance",
            errors,
            expected,
            key,
        )

    validation = start.get("validation")
    if not isinstance(validation, dict) or validation.get("ok") is not True:
        _error(errors, f"{check_prefix}.validation", "Popcorn step must record passing one-file submission validation", submit_manifest_name)
    elif validation.get("errors"):
        _error(errors, f"{check_prefix}.validation", "Popcorn manifest validation row recorded errors", submit_manifest_name)
    elif isinstance(validation, dict):
        for key in ["source_submission", "staged_submission"]:
            _validate_file_provenance_dict(
                validation.get(key),
                submit_manifest_name,
                f"{check_prefix}.provenance",
                errors,
                expected,
                f"validation.{key}",
            )

    planned_modes = [row.get("mode") for row in rows if row.get("event") == "mode_plan"]
    if planned_modes != [mode]:
        _error(errors, f"{check_prefix}.manifest", f"Popcorn manifest must plan exactly one {mode} mode submission", submit_manifest_name)

    finishes = [row for row in rows if row.get("event") == "mode_finish"]
    if len(finishes) != 1:
        _error(errors, f"{check_prefix}.manifest", f"expected one mode_finish row, got {len(finishes)}", submit_manifest_name)
    else:
        finish = finishes[0]
        if finish.get("mode") != mode:
            _error(errors, f"{check_prefix}.manifest", f"mode_finish row must be for {mode} mode", submit_manifest_name)
        if finish.get("exit_code") != 0 or finish.get("error") not in (None, ""):
            _error(errors, f"{check_prefix}.exit_code", f"Popcorn {mode} mode did not finish cleanly", submit_manifest_name)
        if not _is_positive_number(finish.get("elapsed_s")):
            _error(errors, f"{check_prefix}.manifest", "mode_finish row is missing positive elapsed_s", submit_manifest_name)

    final = rows[-1]
    if final.get("event") != "submit_finish":
        _error(errors, f"{check_prefix}.manifest", "Popcorn manifest must end with submit_finish", submit_manifest_name)
    else:
        if final.get("ok") is not True or final.get("dry_run") is not False:
            _error(errors, f"{check_prefix}.manifest", "Popcorn submit_finish must report ok=true and dry_run=false", submit_manifest_name)

    validation_file = f"{artifact_dir}/submission_validation.jsonl"
    validation_rows = _load_required_jsonl(suite_dir, validation_file, errors)
    if validation_rows:
        validation_row = validation_rows[-1]
        if validation_row.get("ok") is not True or validation_row.get("errors"):
            _error(
                errors,
                f"{check_prefix}.validation",
                "nested Popcorn submission validation must pass without errors",
                validation_file,
            )
        for key in ["source_submission", "staged_submission"]:
            _validate_file_provenance_dict(
                validation_row.get(key),
                validation_file,
                f"{check_prefix}.provenance",
                errors,
                expected,
                key,
            )

    staged_file = f"{artifact_dir}/submission.py"
    staged = suite_dir / staged_file
    if staged.is_file() and staged.stat().st_size <= 0:
        _error(errors, f"{check_prefix}.submission", "nested Popcorn submission.py is empty", staged_file)
    elif staged.is_file() and isinstance(expected, dict) and _is_sha256(expected.get("sha256")):
        try:
            staged_sha256 = file_sha256(staged)
        except OSError as exc:
            _error(errors, f"{check_prefix}.provenance", f"could not hash nested submission.py: {exc}", staged_file)
        else:
            if staged_sha256 != expected.get("sha256"):
                _error(
                    errors,
                    f"{check_prefix}.provenance",
                    "nested submission.py sha256 does not match suite submission sha256",
                    staged_file,
                )

    log_file = f"{artifact_dir}/popcorn.log"
    log_path = suite_dir / log_file
    if log_path.is_file():
        text = log_path.read_text(errors="replace")
        if "popcorn submit" not in text or f"--mode {mode}" not in text:
            _error(errors, f"{check_prefix}.log", f"popcorn.log must contain the Popcorn {mode} submission command", log_file)


def validate_popcorn_test(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    validate_popcorn_submit(suite_dir, errors, "popcorn_test", "popcorn_test", "test", POPCORN_TEST_FILES)


def validate_popcorn_leaderboard(suite_dir: Path, errors: list[dict[str, str]]) -> None:
    validate_popcorn_submit(
        suite_dir,
        errors,
        "popcorn_leaderboard",
        "popcorn_leaderboard",
        "leaderboard",
        POPCORN_LEADERBOARD_FILES,
    )


def validate_suite(
    suite_dir: Path,
    require_finish: bool = True,
    require_final_kernels: bool = False,
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    validate_required_files(suite_dir, errors)
    validate_run_log(suite_dir, errors)
    validate_manifest(suite_dir, errors, require_finish=require_finish)
    validate_submission_validation(suite_dir, errors)
    validate_public_tests(suite_dir, errors)
    validate_public_benchmark_correctness(suite_dir, errors)
    validate_dev_robustness(suite_dir, errors)
    validate_secret_audit(suite_dir, errors)
    validate_runtime_preflight(suite_dir, errors)
    validate_preflight(suite_dir, errors)
    validate_candidate_config_accelerator_preflight(suite_dir, errors)
    validate_policy(suite_dir, errors)
    validate_implementation_status(suite_dir, errors, require_final_kernels=require_final_kernels)
    validate_route_trace(suite_dir, errors)
    validate_guard_overhead(suite_dir, errors)
    validate_seed_sweep(suite_dir, errors)
    validate_quantization_seed_sweep(suite_dir, errors)
    validate_mixed_seed_sweep(suite_dir, errors)
    validate_classifier_seed_sweep(suite_dir, errors)
    validate_tail_policy_sweep(suite_dir, errors)
    validate_experiments(suite_dir, errors)
    validate_blocked_qr_sweep(suite_dir, errors)
    validate_summary(suite_dir, errors)
    validate_analysis(suite_dir, errors)
    validate_tail_policy_tune(suite_dir, errors)
    validate_candidate_config_tune(suite_dir, errors)
    validate_popcorn_test(suite_dir, errors)
    validate_popcorn_leaderboard(suite_dir, errors)
    for file_name in BENCHMARK_FILES:
        validate_benchmark_file(suite_dir, file_name, errors)
    return {
        "ok": not errors,
        "suite_dir": str(suite_dir),
        "num_errors": len(errors),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate that a completed default B200 suite export is complete and trustworthy.",
        allow_abbrev=False,
    )
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--allow-incomplete", action="store_true", help="Do not require a suite_finish manifest row.")
    parser.add_argument(
        "--require-final-kernels",
        action="store_true",
        help="Fail if candidate_implementation_status.jsonl still reports non-final benchmark routes.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    suite_dir = ROOT / args.suite_dir if not Path(args.suite_dir).is_absolute() else Path(args.suite_dir)
    result = validate_suite(
        suite_dir,
        require_finish=not args.allow_incomplete,
        require_final_kernels=args.require_final_kernels,
    )
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
    sys.exit(main())
