from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

from qr_common import (
    ROOT,
    append_jsonl,
    apply_popcorn_seed,
    batch_count,
    clone_data,
    ensure_official_on_path,
    environment_info,
    file_provenance,
    load_cases,
    load_submission,
    require_cuda,
    relative_path,
)


ensure_official_on_path()
from reference import check_implementation, generate_input  # noqa: E402
from utils import clear_l2_cache  # noqa: E402


def stats(durations_ns: list[float]) -> dict[str, float | int]:
    mean = sum(durations_ns) / len(durations_ns)
    variance = sum((value - mean) ** 2 for value in durations_ns)
    std = math.sqrt(variance / max(1, len(durations_ns) - 1))
    err = std / math.sqrt(len(durations_ns)) if durations_ns else 0.0
    return {
        "runs": len(durations_ns),
        "mean_us": mean / 1000.0,
        "std_us": std / 1000.0,
        "err_us": err / 1000.0,
        "best_us": min(durations_ns) / 1000.0,
        "worst_us": max(durations_ns) / 1000.0,
    }


def should_stop_official_style(durations_ns: list[float], benchmark_started_ns: int, max_time_ns: float) -> bool:
    total_bm_duration = time.perf_counter_ns() - benchmark_started_ns
    if len(durations_ns) <= 2 or total_bm_duration <= 1e8:
        return False
    current = stats(durations_ns)
    mean_us = float(current["mean_us"])
    err_us = float(current["err_us"])
    return (
        err_us / mean_us < 0.001
        or mean_us * 1000.0 * int(current["runs"]) > max_time_ns
        or total_bm_duration > 120e9
    )


def run_one(
    custom_kernel,
    spec: dict,
    repeats: int,
    recheck: bool,
    official_stopping: bool = False,
    max_time_ns: float = 10e9,
) -> dict:
    count = batch_count(spec)
    args = dict(spec)
    data_list = []
    for _ in range(count):
        if "seed" in args:
            args["seed"] += 42
        data_list.append(generate_input(**args))

    check_copy = clone_data(data_list)

    outputs = [custom_kernel(clone_data(data)) for data in data_list]
    torch.cuda.synchronize()
    for reference_data, output in zip(check_copy, outputs):
        good, message = check_implementation(reference_data, output)
        if not good:
            return {"ok": False, "error": message, "spec": spec, "count": count}

    durations_ns_per_input: list[float] = []
    benchmark_started_ns = time.perf_counter_ns()
    for _ in range(repeats):
        torch.cuda.synchronize()
        clear_l2_cache()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        outputs = [custom_kernel(data) for data in data_list]
        end.record()

        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        durations_ns_per_input.append(elapsed_ms * 1e6 / len(data_list))

        if recheck:
            for reference_data, output in zip(check_copy, outputs):
                good, message = check_implementation(reference_data, output)
                if not good:
                    return {"ok": False, "error": message, "spec": spec, "count": count}

        if official_stopping and should_stop_official_style(durations_ns_per_input, benchmark_started_ns, max_time_ns):
            break

    out = {"ok": True, "spec": spec, "count": count}
    out.update(stats(durations_ns_per_input))
    return out


def run_leaderboard_warmup(custom_kernel, cases: list[dict]) -> None:
    for spec in cases:
        result = run_one(
            custom_kernel,
            spec,
            repeats=1000,
            recheck=False,
            official_stopping=True,
            max_time_ns=5e8,
        )
        if not result["ok"]:
            raise RuntimeError(f"leaderboard warmup failed for {spec}: {result['error']}")


def parse_indices(raw: str, count: int) -> list[int]:
    if not raw:
        return list(range(count))
    indexes = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        index = int(token)
        if index < 0 or index >= count:
            raise ValueError(f"case index {index} out of range for {count} cases")
        indexes.append(index)
    if not indexes:
        raise ValueError("no case indexes selected")
    return indexes


def summarize(results: list[dict]) -> dict:
    valid = [result for result in results if result.get("ok")]
    geomean = math.exp(sum(math.log(result["mean_us"]) for result in valid) / len(valid))
    return {"geomean_us": geomean, "num_cases": len(valid)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Faithful local QR v2 benchmark harness.",
        allow_abbrev=False,
    )
    parser.add_argument("--submission", required=True, help="Path to a submission.py-compatible file.")
    parser.add_argument("--cases", default="cases/public_benchmarks.txt", help="Case file to run.")
    parser.add_argument("--indices", default="", help="Comma-separated case indexes. Defaults to all cases.")
    parser.add_argument("--repeats", type=int, default=5, help="Timed repeats per case.")
    parser.add_argument(
        "--popcorn-seed",
        type=int,
        default=None,
        help="Apply official POPCORN_SEED combination to all case seeds.",
    )
    parser.add_argument(
        "--official-stopping",
        action="store_true",
        help="Use the official adaptive stopping criteria; --repeats becomes max repeats.",
    )
    parser.add_argument(
        "--max-time-ns",
        type=float,
        default=10e9,
        help="Official-style per-case mean*runs stopping threshold in nanoseconds.",
    )
    parser.add_argument(
        "--leaderboard-warmup",
        action="store_true",
        help="Run official leaderboard-style warmup benchmarks before reporting results.",
    )
    parser.add_argument("--recheck", action="store_true", help="Recheck outputs from timed iterations.")
    parser.add_argument("--out", default=None, help="Append raw JSONL results to this path.")
    parser.add_argument(
        "--record-env",
        action="store_true",
        help="Include torch/CUDA/GPU metadata in each JSONL row.",
    )
    args = parser.parse_args()

    try:
        require_cuda(torch)
    except RuntimeError as exc:
        parser.error(str(exc))
    submission_path = ROOT / args.submission if not Path(args.submission).is_absolute() else Path(args.submission)
    custom_kernel = load_submission(submission_path)
    all_cases = load_cases(ROOT / args.cases if not Path(args.cases).is_absolute() else args.cases)
    try:
        indexes = parse_indices(args.indices, len(all_cases))
    except ValueError as exc:
        parser.error(str(exc))
    selected_cases = apply_popcorn_seed([all_cases[index] for index in indexes], args.popcorn_seed)
    cases = list(zip(indexes, selected_cases))
    if args.leaderboard_warmup:
        run_leaderboard_warmup(custom_kernel, [spec for _, spec in cases])

    env = environment_info(torch) if args.record_env else {}
    if args.record_env:
        env["submission"] = relative_path(submission_path)
        env["submission_sha256"] = file_provenance(submission_path)["sha256"]
    results: list[dict] = []
    for case_index, spec in cases:
        result = run_one(
            custom_kernel,
            spec,
            args.repeats,
            args.recheck,
            official_stopping=args.official_stopping,
            max_time_ns=args.max_time_ns,
        )
        result["case_index"] = case_index
        if env:
            result = {**env, **result}
        print(json.dumps(result, sort_keys=True), flush=True)
        results.append(result)
        if not result["ok"]:
            break

    if all(result["ok"] for result in results):
        summary = summarize(results)
        print(json.dumps(summary, sort_keys=True), flush=True)
        results.append(summary)

    if args.out:
        append_jsonl(ROOT / args.out if not Path(args.out).is_absolute() else args.out, results)
    return 0 if all(result.get("ok", True) for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
