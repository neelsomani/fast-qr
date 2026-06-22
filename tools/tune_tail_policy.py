from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from qr_common import ROOT, file_provenance, repo_provenance
from summarize_suite import load_jsonl


DEFAULT_CONFIGS: list[dict[str, Any]] = [
    {"name": "default", "env": {}},
    {"name": "dense512_cut16", "env": {"FAST_QR_DENSE_TAIL_CUT_512": "16"}},
    {"name": "dense512_cut24", "env": {"FAST_QR_DENSE_TAIL_CUT_512": "24"}},
    {"name": "dense512_cut32", "env": {"FAST_QR_DENSE_TAIL_CUT_512": "32"}},
    {"name": "dense1024_cut64", "env": {"FAST_QR_DENSE_TAIL_CUT_1024": "64"}},
    {"name": "dense1024_cut96", "env": {"FAST_QR_DENSE_TAIL_CUT_1024": "96"}},
    {"name": "mixed1024_cut0", "env": {"FAST_QR_MIXED_DENSE_TAIL_CUT_1024": "0"}},
    {"name": "mixed1024_cut8", "env": {"FAST_QR_MIXED_DENSE_TAIL_CUT_1024": "8"}},
    {"name": "mixed1024_cut12", "env": {"FAST_QR_MIXED_DENSE_TAIL_CUT_1024": "12"}},
    {"name": "dense2048_cut64", "env": {"FAST_QR_DENSE_TAIL_CUT_2048": "64"}},
    {"name": "dense2048_cut96", "env": {"FAST_QR_DENSE_TAIL_CUT_2048": "96"}},
    {"name": "dense4096_cut128", "env": {"FAST_QR_DENSE_TAIL_CUT_4096": "128"}},
    {"name": "dense4096_cut256", "env": {"FAST_QR_DENSE_TAIL_CUT_4096": "256"}},
]


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_env_assignments(raw: str) -> dict[str, str]:
    env: dict[str, str] = {}
    if not raw:
        return env
    for piece in raw.split(","):
        item = piece.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"expected KEY=VALUE assignment, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"empty env key in {item!r}")
        env[key] = value.strip()
    return env


def parse_inline_config(raw: str) -> dict[str, Any]:
    if ":" not in raw:
        raise ValueError(f"config must look like name:KEY=VALUE, got {raw!r}")
    name, assignments = raw.split(":", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"config name is empty in {raw!r}")
    return {"name": name, "env": parse_env_assignments(assignments)}


def load_config_rows(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    configs: list[dict[str, Any]] = []
    for row in rows:
        name = row.get("name")
        env = row.get("env")
        if not isinstance(name, str) or not name:
            raise ValueError(f"config row in {path} is missing string name")
        if not isinstance(env, dict):
            raise ValueError(f"config row {name!r} in {path} is missing env object")
        configs.append({"name": name, "env": {str(key): str(value) for key, value in env.items()}})
    return configs


def load_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs = list(DEFAULT_CONFIGS)
    if args.config_jsonl:
        path = ROOT / args.config_jsonl if not Path(args.config_jsonl).is_absolute() else Path(args.config_jsonl)
        configs = load_config_rows(path)
    if args.config:
        configs.extend(parse_inline_config(raw) for raw in args.config)

    seen = set()
    out = []
    for config in configs:
        name = config["name"]
        if name in seen:
            raise ValueError(f"duplicate config name: {name}")
        seen.add(name)
        out.append(config)
    return out


def merged_env(base: dict[str, str], overrides: dict[str, str]) -> dict[str, str]:
    out = base.copy()
    out.update(overrides)
    return out


def config_prefix(index: int, name: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in name)
    return f"{index:03d}_{safe}"


def command_plan(args: argparse.Namespace, out_dir: Path, config: dict[str, Any], index: int) -> list[dict[str, Any]]:
    prefix = config_prefix(index, config["name"])
    python = sys.executable
    correctness_out = out_dir / f"{prefix}_tail_policy_sweep.jsonl"
    benchmark_out = out_dir / f"{prefix}_benchmark.jsonl"

    steps: list[dict[str, Any]] = []
    if not args.skip_correctness:
        steps.append(
            {
                "step": "correctness",
                "out": str(correctness_out),
                "cmd": [
                    python,
                    "tools/tail_policy_sweep.py",
                    "--submission",
                    args.submission,
                    "--cases",
                    args.cases,
                    "--indices",
                    args.correctness_indices,
                    "--popcorn-seeds",
                    args.popcorn_seeds,
                    "--tail-cuts",
                    "candidate",
                    "--diagnose",
                    "--max-factor-scaled",
                    f"{args.max_factor_scaled:g}",
                    "--max-orth-scaled",
                    f"{args.max_orth_scaled:g}",
                    "--record-env",
                    "--out",
                    str(correctness_out),
                ],
            }
        )
    if not args.skip_benchmark:
        cmd = [
            python,
            "tools/bench_local.py",
            "--submission",
            args.submission,
            "--cases",
            args.cases,
            "--repeats",
            str(args.repeats),
            "--recheck",
            "--record-env",
            "--out",
            str(benchmark_out),
        ]
        if args.official_stopping:
            cmd.extend(["--official-stopping", "--leaderboard-warmup", "--max-time-ns", f"{args.max_time_ns:g}"])
        steps.append({"step": "benchmark", "out": str(benchmark_out), "cmd": cmd})
    return steps


def run_step(step: dict[str, Any], env: dict[str, str], log_path: Path) -> int:
    start = time.perf_counter()
    cmd = step["cmd"]
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
    return completed.returncode


def summarize_correctness(path: Path) -> dict[str, Any] | None:
    rows = load_jsonl(path)
    if not rows:
        return None
    cases = [row for row in rows if not row.get("summary")]
    diagnostics = [row.get("diagnostics") for row in cases if isinstance(row.get("diagnostics"), dict)]
    factor_values = [float(row["factor_scaled_max"]) for row in diagnostics]
    orth_values = [float(row["orth_scaled_max"]) for row in diagnostics]
    return {
        "num_rows": len(cases),
        "num_failed": sum(1 for row in cases if not row.get("ok") or row.get("margin_ok") is False),
        "max_factor_scaled": max(factor_values) if factor_values else None,
        "max_orth_scaled": max(orth_values) if orth_values else None,
    }


def summarize_benchmark(path: Path) -> dict[str, Any] | None:
    rows = load_jsonl(path)
    if not rows:
        return None
    summary = next((row for row in rows if "geomean_us" in row), None)
    cases = [row for row in rows if isinstance(row.get("spec"), dict) and "mean_us" in row]
    if not summary:
        return {"num_cases": len(cases), "geomean_us": None}
    return {
        "num_cases": len(cases),
        "geomean_us": float(summary["geomean_us"]),
    }


def should_skip_benchmark_after_correctness(correctness_failed: bool, benchmark_failed_configs: bool) -> bool:
    return correctness_failed and not benchmark_failed_configs


def summarize_run(out_dir: Path, configs: list[dict[str, Any]]) -> dict[str, Any]:
    results = []
    for index, config in enumerate(configs):
        prefix = config_prefix(index, config["name"])
        correctness = summarize_correctness(out_dir / f"{prefix}_tail_policy_sweep.jsonl")
        benchmark = summarize_benchmark(out_dir / f"{prefix}_benchmark.jsonl")
        results.append(
            {
                "name": config["name"],
                "env": config["env"],
                "correctness": correctness,
                "benchmark": benchmark,
            }
        )
    ranked = sorted(
        [row for row in results if row.get("benchmark") and row["benchmark"].get("geomean_us")],
        key=lambda row: row["benchmark"]["geomean_us"],
    )
    return {
        "ok": True,
        "out_dir": str(out_dir),
        "num_configs": len(results),
        "results": results,
        "best": ranked[0] if ranked else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a B200 tail-policy autotune grid using correctness gates before benchmarks.",
        allow_abbrev=False,
    )
    parser.add_argument("--submission", default="submissions/candidate.py")
    parser.add_argument("--cases", default="cases/public_benchmarks.txt")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--config-jsonl", default=None)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help='Add one config as "name:KEY=VALUE,KEY2=VALUE2". Can be repeated.',
    )
    parser.add_argument("--correctness-indices", default="3,4,5,6,7,8,9,10,11")
    parser.add_argument("--popcorn-seeds", default="public,0,1,2,3")
    parser.add_argument("--max-factor-scaled", type=float, default=18.0)
    parser.add_argument("--max-orth-scaled", type=float, default=80.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--official-stopping", action="store_true")
    parser.add_argument("--max-time-ns", type=float, default=30e9)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument(
        "--continue-on-fail",
        action="store_true",
        help="Deprecated compatibility flag. The tune grid continues after failed configs by default.",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop the tune grid after the first failed step.")
    parser.add_argument(
        "--benchmark-failed-configs",
        action="store_true",
        help="Run benchmark steps even when the preceding correctness gate failed.",
    )
    parser.add_argument(
        "--allow-failed-configs",
        action="store_true",
        help="Exit zero when configs fail correctness gates but the tune run itself completed.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = ROOT / "results" / f"tail_policy_tune_{timestamp()}" if args.out_dir is None else Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    log_path = out_dir / "run.log"
    summary_path = out_dir / "summary.json"

    configs = load_configs(args)
    env = os.environ.copy()
    submission = ROOT / args.submission if not Path(args.submission).is_absolute() else Path(args.submission)

    append_jsonl(
        manifest_path,
        {
            "event": "tune_start",
            "time": datetime.now().isoformat(),
            "dry_run": args.dry_run,
            "repo": repo_provenance(),
            "submission": file_provenance(submission),
            "args": vars(args),
        },
    )

    failed = False
    hard_failed = False
    for index, config in enumerate(configs):
        steps = command_plan(args, out_dir, config, index)
        append_jsonl(
            manifest_path,
            {
                "event": "config_plan",
                "index": index,
                "name": config["name"],
                "env": config["env"],
                "steps": steps,
            },
        )
        if args.dry_run:
            continue

        config_failed = False
        correctness_failed = False
        for step in steps:
            if step["step"] == "benchmark" and should_skip_benchmark_after_correctness(
                correctness_failed,
                args.benchmark_failed_configs,
            ):
                append_jsonl(
                    manifest_path,
                    {
                        "event": "step_skipped",
                        "config": config["name"],
                        "step": step["step"],
                        "out": step["out"],
                        "reason": "correctness_failed",
                    },
                )
                continue

            append_jsonl(manifest_path, {"event": "step_start", "config": config["name"], **step})
            code = run_step(step, merged_env(env, config["env"]), log_path)
            append_jsonl(manifest_path, {"event": "step_finish", "config": config["name"], "step": step["step"], "exit_code": code})
            if code != 0:
                config_failed = True
                failed = True
                expected_config_failure = False
                if step["step"] == "correctness":
                    correctness_failed = True
                    expected_config_failure = summarize_correctness(Path(step["out"])) is not None
                if not expected_config_failure:
                    hard_failed = True
                if hard_failed or args.fail_fast:
                    break
        if config_failed and args.fail_fast:
            break
        if hard_failed:
            break

    summary = summarize_run(out_dir, configs)
    summary["dry_run"] = args.dry_run
    summary["ok"] = not failed
    summary["hard_failed"] = hard_failed
    summary["allow_failed_configs"] = args.allow_failed_configs
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    append_jsonl(manifest_path, {"event": "tune_finish", "ok": not failed, "summary": str(summary_path), "time": datetime.now().isoformat()})

    print(json.dumps(summary, sort_keys=True))
    if hard_failed:
        return 1
    if failed and not args.allow_failed_configs:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
