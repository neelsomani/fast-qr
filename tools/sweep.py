from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from bench_local import run_leaderboard_warmup, run_one
from diagnose import diagnose
from qr_common import (
    ROOT,
    append_jsonl,
    apply_popcorn_seed,
    clone_data,
    ensure_official_on_path,
    environment_info,
    load_cases,
    load_submission,
    require_cuda,
)


ensure_official_on_path()
from reference import check_implementation, generate_input  # noqa: E402


def load_config(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    stripped = value.lstrip()
    if stripped.startswith("{"):
        return json.loads(value)
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text())
    return json.loads(value)


def diagnostic_row(custom_kernel, spec: dict[str, Any]) -> dict[str, Any]:
    data = generate_input(**spec)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    output = custom_kernel(clone_data(data))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    good, message = check_implementation(data, output)
    row: dict[str, Any] = {"diagnostic_passed": good, "diagnostic_message": message}
    if good:
        h, tau = output
        row.update(diagnose(data, h, tau))
    return row


def parse_indices(value: str | None) -> set[int]:
    if not value:
        return set()
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def should_diagnose_case(index: int, diagnose: bool, diagnose_every: int, diagnose_cases: set[int]) -> bool:
    if index in diagnose_cases:
        return True
    if not diagnose:
        return False
    if diagnose_every <= 1:
        return True
    return index % diagnose_every == 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a QR v2 sweep and append result-tracking JSONL.",
        allow_abbrev=False,
    )
    parser.add_argument("--submission", default="submissions/candidate.py")
    parser.add_argument("--cases", default="cases/public_benchmarks.txt")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--popcorn-seed", type=int, default=None)
    parser.add_argument("--recheck", action="store_true")
    parser.add_argument("--official-stopping", action="store_true")
    parser.add_argument("--max-time-ns", type=float, default=10e9)
    parser.add_argument("--leaderboard-warmup", action="store_true")
    parser.add_argument("--out", default="results/runs.jsonl")
    parser.add_argument("--label", default=None)
    parser.add_argument("--config-json", default=None, help="Inline JSON object or path to JSON.")
    parser.add_argument("--diagnose", action="store_true", help="Attach expensive residual diagnostics.")
    parser.add_argument("--diagnose-every", type=int, default=1, help="With --diagnose, diagnose every Nth case.")
    parser.add_argument("--diagnose-cases", default="", help="Comma-separated zero-based case indexes to diagnose.")
    args = parser.parse_args()

    try:
        require_cuda(torch)
    except RuntimeError as exc:
        parser.error(str(exc))
    submission_path = ROOT / args.submission if not Path(args.submission).is_absolute() else Path(args.submission)
    cases_path = ROOT / args.cases if not Path(args.cases).is_absolute() else Path(args.cases)
    custom_kernel = load_submission(submission_path)
    config = load_config(args.config_json)
    env = environment_info(torch)
    diagnose_cases = parse_indices(args.diagnose_cases)
    cases = apply_popcorn_seed(load_cases(cases_path), args.popcorn_seed)
    if args.leaderboard_warmup:
        run_leaderboard_warmup(custom_kernel, cases)

    rows = []
    for index, spec in enumerate(cases):
        result = run_one(
            custom_kernel,
            spec,
            args.repeats,
            args.recheck,
            official_stopping=args.official_stopping,
            max_time_ns=args.max_time_ns,
        )
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **env,
            "submission": str(submission_path.relative_to(ROOT) if submission_path.is_relative_to(ROOT) else submission_path),
            "label": args.label,
            "case": spec,
            "config": config,
            "benchmark_mode": {
                "official_stopping": args.official_stopping,
                "max_time_ns": args.max_time_ns,
                "leaderboard_warmup": args.leaderboard_warmup,
                "popcorn_seed": args.popcorn_seed,
                "recheck": args.recheck,
                "repeats": args.repeats,
            },
            "passed": bool(result.get("ok")),
        }
        for key in ("count", "runs", "mean_us", "std_us", "err_us", "best_us", "worst_us", "error"):
            if key in result:
                row[key] = result[key]
        if result.get("ok") and should_diagnose_case(index, args.diagnose, args.diagnose_every, diagnose_cases):
            row.update(diagnostic_row(custom_kernel, spec))
        print(json.dumps(row, sort_keys=True), flush=True)
        rows.append(row)
        if not result.get("ok"):
            break

    append_jsonl(ROOT / args.out if not Path(args.out).is_absolute() else args.out, rows)
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
