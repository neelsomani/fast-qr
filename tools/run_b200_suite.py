from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

from qr_common import ROOT


BENCHMARK_EXPERIMENT_INDEXES = [3, 7, 9, 10, 4, 8, 11, 5, 6]


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_command(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    start = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(cmd)}\n")
        log.flush()
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log.write(completed.stdout)
        log.write(f"\nexit_code={completed.returncode}; elapsed_s={time.perf_counter() - start:.3f}\n")
    print(completed.stdout, end="")
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit code {completed.returncode}: {' '.join(cmd)}")


def append_manifest(manifest_path: Path, row: dict) -> None:
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def make_tarball(suite_dir: Path) -> Path:
    tar_path = suite_dir.with_suffix(".tgz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(suite_dir, arcname=suite_dir.name)
    return tar_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the first-pass B200 QR lab suite and export JSONL results.",
        allow_abbrev=False,
    )
    parser.add_argument("--submission", default="submissions/candidate.py")
    parser.add_argument("--baseline", default="submissions/baseline_geqrf.py")
    parser.add_argument("--suite-name", default=None)
    parser.add_argument("--smoke-repeats", type=int, default=1)
    parser.add_argument("--baseline-repeats", type=int, default=3)
    parser.add_argument("--official-repeats", type=int, default=1000)
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-baseline-public", action="store_true")
    parser.add_argument("--skip-experiments", action="store_true")
    parser.add_argument("--skip-official-style", action="store_true")
    parser.add_argument(
        "--include-local-official-eval",
        action="store_true",
        help="Also run frozen official/eval.py in test mode. This can add substantial time.",
    )
    parser.add_argument("--popcorn-seed", type=int, default=None)
    args = parser.parse_args()

    name = args.suite_name or f"b200_suite_{timestamp()}"
    suite_dir = ROOT / "results" / name
    suite_dir.mkdir(parents=True, exist_ok=False)
    log_path = suite_dir / "run.log"
    manifest_path = suite_dir / "manifest.jsonl"

    env = os.environ.copy()
    python = sys.executable
    popcorn_seed_args = []
    if args.popcorn_seed is not None:
        popcorn_seed_args = ["--popcorn-seed", str(args.popcorn_seed)]

    steps: list[tuple[str, list[str]]] = [
        ("print_spec", [python, "tools/print_spec.py"]),
        ("sync_cases_check", [python, "tools/sync_cases_from_task_yml.py", "--check"]),
    ]
    if not args.skip_pytest:
        steps.append(("pytest", [python, "-m", "pytest"]))
    if args.include_local_official_eval:
        steps.append(
            (
                "local_official_eval_test",
                [
                    python,
                    "tools/run_official_eval.py",
                    "--submission",
                    args.submission,
                    "--cases",
                    "cases/public_tests.txt",
                    "--mode",
                    "test",
                    *popcorn_seed_args,
                ],
            )
        )
    if not args.skip_smoke:
        steps.append(
            (
                "baseline_smoke",
                [
                    python,
                    "tools/bench_local.py",
                    "--submission",
                    args.baseline,
                    "--cases",
                    "cases/public_benchmarks.txt",
                    "--repeats",
                    str(args.smoke_repeats),
                    "--recheck",
                    "--record-env",
                    "--out",
                    str(suite_dir / "baseline_geqrf_smoke.jsonl"),
                    *popcorn_seed_args,
                ],
            )
        )
    if not args.skip_baseline_public:
        steps.append(
            (
                "baseline_public",
                [
                    python,
                    "tools/bench_local.py",
                    "--submission",
                    args.baseline,
                    "--cases",
                    "cases/public_benchmarks.txt",
                    "--repeats",
                    str(args.baseline_repeats),
                    "--recheck",
                    "--record-env",
                    "--out",
                    str(suite_dir / "baseline_geqrf_public.jsonl"),
                    *popcorn_seed_args,
                ],
            )
        )
    if not args.skip_experiments:
        for index in BENCHMARK_EXPERIMENT_INDEXES:
            steps.append(
                (
                    f"experiments_public_benchmark_{index}",
                    [
                        python,
                        "tools/experiments.py",
                        "--submission",
                        args.submission,
                        "--cases",
                        "cases/public_benchmarks.txt",
                        "--index",
                        str(index),
                        "--experiments",
                        "all",
                        "--out",
                        str(suite_dir / "experiments_public_benchmarks.jsonl"),
                        *popcorn_seed_args,
                    ],
                )
            )
    if not args.skip_official_style:
        steps.append(
            (
                "baseline_official_style",
                [
                    python,
                    "tools/bench_local.py",
                    "--submission",
                    args.baseline,
                    "--cases",
                    "cases/public_benchmarks.txt",
                    "--repeats",
                    str(args.official_repeats),
                    "--official-stopping",
                    "--leaderboard-warmup",
                    "--recheck",
                    "--record-env",
                    "--out",
                    str(suite_dir / "baseline_geqrf_official_style.jsonl"),
                    *popcorn_seed_args,
                ],
            )
        )

    suite_started = time.perf_counter()
    try:
        for step_name, cmd in steps:
            step_started = time.perf_counter()
            append_manifest(manifest_path, {"event": "start", "step": step_name, "time": datetime.now().isoformat()})
            run_command(cmd, log_path, env)
            append_manifest(
                manifest_path,
                {
                    "event": "finish",
                    "step": step_name,
                    "elapsed_s": time.perf_counter() - step_started,
                    "time": datetime.now().isoformat(),
                },
            )
        tar_path = make_tarball(suite_dir)
        append_manifest(
            manifest_path,
            {
                "event": "suite_finish",
                "elapsed_s": time.perf_counter() - suite_started,
                "tarball": str(tar_path),
                "time": datetime.now().isoformat(),
            },
        )
        print(f"\nDONE\nresults: {suite_dir}\ntarball: {tar_path}")
        return 0
    except Exception as exc:
        append_manifest(
            manifest_path,
            {
                "event": "suite_failed",
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
