from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

from qr_common import ROOT, repo_provenance
from validate_local_checks import validate_local_checks


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def append_manifest(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def make_tarball(suite_dir: Path) -> Path:
    tar_path = suite_dir.with_suffix(".tgz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(suite_dir, arcname=suite_dir.name)
    return tar_path


def validate_completed_export(suite_dir: Path) -> None:
    require_valid_local_export(suite_dir)
    print("local export validation: PASS")


def next_required_b200_summary(suite_dir: Path) -> str:
    path = suite_dir / "b200_next_required_dry_run_plan.json"
    if not path.is_file():
        return "next-required B200 target: unavailable (missing b200_next_required_dry_run_plan.json)"

    try:
        plan = json.loads(path.read_text())
    except Exception as exc:
        return f"next-required B200 target: unavailable ({type(exc).__name__}: {exc})"

    target = plan.get("candidate_config_tune_policy_target")
    generated = plan.get("candidate_config_tune_large_kernel_plan")
    if not isinstance(target, dict) or not isinstance(generated, dict):
        return "next-required B200 target: unavailable (plan is missing target or generated config preview)"

    shape = target.get("shape_label", "unknown")
    kernel = target.get("required_cuda_kernel", "unknown")
    benchmark_indices = target.get("benchmark_indices", "")
    correctness_indices = target.get("correctness_indices", "")
    mode = target.get("large_kernel_plan_mode") or generated.get("mode", "unknown")
    max_configs = generated.get("max_configs", generated.get("num_configs", "unknown"))
    num_configs = generated.get("num_configs", "unknown")
    plan_suite_dir = plan.get("suite_dir")
    b200_suite_name = (
        Path(plan_suite_dir).name
        if isinstance(plan_suite_dir, str) and plan_suite_dir
        else f"{suite_dir.name}_b200_next_required"
    )

    command = [
        "python",
        "tools/run_b200_suite.py",
        "--suite-name",
        str(b200_suite_name),
        "--candidate-config-tune-next-required",
        "--candidate-config-tune-large-kernel-plan-max-configs",
        str(max_configs),
    ]
    if mode and mode != "current-candidate":
        command.extend(["--candidate-config-tune-large-kernel-plan-mode", str(mode)])

    return "\n".join(
        [
            f"next-required B200 target: {shape} / {kernel}",
            f"benchmark indices: {benchmark_indices}",
            f"correctness indices: {correctness_indices}",
            f"generated configs: {num_configs} ({mode})",
            f"next B200 command: {shlex.join(command)}",
        ]
    )


def require_valid_local_export(suite_dir: Path) -> None:
    result = validate_local_checks(suite_dir, require_finish=True)
    if result["ok"]:
        return
    preview = "; ".join(f"{error['check']}: {error['message']}" for error in result["errors"][:5])
    raise RuntimeError(f"local export validation failed with {result['num_errors']} errors: {preview}")


def step_record(step: tuple[str, list[str]] | tuple[str, list[str], str]) -> dict:
    if len(step) == 2:
        name, cmd = step
        stdout_file = None
    else:
        name, cmd, stdout_file = step
    return {"step": name, "cmd": cmd, "stdout_file": stdout_file}


def run_command(cmd: list[str], log_path: Path, stdout_file: Path | None = None) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    started = time.perf_counter()
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(cmd)}\n")
        log.write(completed.stdout)
        log.write(f"\nexit_code={completed.returncode}; elapsed_s={time.perf_counter() - started:.3f}\n")
    if stdout_file is not None:
        stdout_file.write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout, end="")
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit code {completed.returncode}: {' '.join(cmd)}")


def build_steps(python: str, suite_dir: Path, skip_pytest: bool) -> list[tuple[str, list[str]] | tuple[str, list[str], str]]:
    steps: list[tuple[str, list[str]] | tuple[str, list[str], str]] = [
        ("print_spec", [python, "tools/print_spec.py"]),
        ("sync_cases_check", [python, "tools/sync_cases_from_task_yml.py", "--check"]),
        (
            "secret_audit",
            [
                python,
                "tools/audit_secrets.py",
                "--json",
                "--out",
                str(suite_dir / "secret_audit.jsonl"),
            ],
        ),
        (
            "runtime_preflight_allow_failure",
            [
                python,
                "tools/check_b200_env.py",
                "--json",
                "--allow-failure",
                "--out",
                str(suite_dir / "runtime_preflight.jsonl"),
            ],
        ),
        (
            "submission_validation",
            [
                python,
                "tools/validate_submission.py",
                "--submission",
                "submissions/candidate.py",
                "--stage-dir",
                str(suite_dir / "submission_stage"),
                "--json",
                "--out",
                str(suite_dir / "submission_validation.jsonl"),
            ],
        ),
        (
            "candidate_policy",
            [
                python,
                "tools/candidate_policy.py",
                "--submission",
                "submissions/candidate.py",
                "--cases",
                "cases/public_benchmarks.txt",
                "--json",
                "--record-env",
                "--out",
                str(suite_dir / "candidate_policy_public.jsonl"),
            ],
        ),
        (
            "candidate_implementation_status",
            [
                python,
                "tools/implementation_status.py",
                "--submission",
                "submissions/candidate.py",
                "--cases",
                "cases/public_benchmarks.txt",
                "--json",
                "--record-env",
                "--out",
                str(suite_dir / "candidate_implementation_status.jsonl"),
            ],
        ),
        (
            "b200_suite_dry_run",
            [
                python,
                "tools/run_b200_suite.py",
                "--dry-run",
                "--dry-run-json",
                "--suite-name",
                suite_dir.name + "_b200",
            ],
            "b200_dry_run_plan.json",
        ),
        (
            "b200_next_required_dry_run",
            [
                python,
                "tools/run_b200_suite.py",
                "--dry-run",
                "--dry-run-json",
                "--suite-name",
                suite_dir.name + "_b200_next_required",
                "--candidate-config-tune-next-required",
                "--candidate-config-tune-large-kernel-plan-max-configs",
                "8",
            ],
            "b200_next_required_dry_run_plan.json",
        ),
    ]
    if not skip_pytest:
        steps.append(("pytest", [python, "-m", "pytest", "-q"]))
    return steps


def print_dry_run(suite_dir: Path, steps: list[tuple[str, list[str]] | tuple[str, list[str], str]], as_json: bool) -> None:
    plan = {
        "dry_run": True,
        "suite_dir": str(suite_dir),
        "num_steps": len(steps),
        "steps": [step_record(step) for step in steps],
    }
    if as_json:
        print(json.dumps(plan, sort_keys=True))
        return
    print(f"DRY RUN: {suite_dir}")
    for index, step in enumerate(plan["steps"], start=1):
        stdout_file = f" > {step['stdout_file']}" if step["stdout_file"] else ""
        print(f"{index:02d}. {step['step']}: {' '.join(step['cmd'])}{stdout_file}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run local non-CUDA checks for the QR B200 lab without attempting benchmark timing.",
        allow_abbrev=False,
    )
    parser.add_argument("--suite-name", default=None)
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-json", action="store_true")
    parser.add_argument(
        "--print-next-command",
        metavar="SUITE_DIR",
        default=None,
        help="Print the next-required B200 target/command from an existing local-check export and exit.",
    )
    args = parser.parse_args()

    if args.print_next_command:
        suite_dir = ROOT / args.print_next_command if not Path(args.print_next_command).is_absolute() else Path(args.print_next_command)
        try:
            require_valid_local_export(suite_dir)
        except RuntimeError as exc:
            print(f"cannot print next B200 command: {exc}", file=sys.stderr)
            return 1
        print(next_required_b200_summary(suite_dir))
        return 0

    suite_name = args.suite_name or f"local_checks_{timestamp()}"
    suite_dir = ROOT / "results" / suite_name
    steps = build_steps(sys.executable, suite_dir, args.skip_pytest)

    if args.dry_run:
        print_dry_run(suite_dir, steps, as_json=args.dry_run_json)
        return 0

    if suite_dir.exists():
        parser.error(
            f"result directory already exists: {suite_dir}; "
            "choose a different --suite-name or remove the existing directory intentionally"
        )

    suite_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = suite_dir / "manifest.jsonl"
    log_path = suite_dir / "run.log"
    suite_started = time.perf_counter()
    append_manifest(
        manifest_path,
        {
            "event": "local_checks_start",
            "time": datetime.now().isoformat(),
            "repo": repo_provenance(),
            "skip_pytest": args.skip_pytest,
        },
    )

    try:
        for step in steps:
            if len(step) == 2:
                step_name, cmd = step
                stdout_file = None
            else:
                step_name, cmd, stdout_name = step
                stdout_file = suite_dir / stdout_name
            started = time.perf_counter()
            append_manifest(manifest_path, {"event": "start", "step": step_name, "time": datetime.now().isoformat()})
            run_command(cmd, log_path, stdout_file)
            append_manifest(
                manifest_path,
                {
                    "event": "finish",
                    "step": step_name,
                    "elapsed_s": time.perf_counter() - started,
                    "time": datetime.now().isoformat(),
                },
            )
        tar_path = suite_dir.with_suffix(".tgz")
        append_manifest(
            manifest_path,
            {
                "event": "local_checks_finish",
                "elapsed_s": time.perf_counter() - suite_started,
                "tarball": str(tar_path),
                "time": datetime.now().isoformat(),
            },
        )
        tar_path = make_tarball(suite_dir)
        validate_completed_export(suite_dir)
        print(f"\nDONE\nresults: {suite_dir}\ntarball: {tar_path}\n{next_required_b200_summary(suite_dir)}")
        return 0
    except Exception as exc:
        append_manifest(
            manifest_path,
            {
                "event": "local_checks_failed",
                "error": str(exc),
                "elapsed_s": time.perf_counter() - suite_started,
                "time": datetime.now().isoformat(),
            },
        )
        tar_path = make_tarball(suite_dir)
        print(f"\nFAILED: {exc}\npartial results: {suite_dir}\npartial tarball: {tar_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
