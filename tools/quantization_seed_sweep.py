from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

from experiments import quantize_tf32_like, repair_r
from qr_common import (
    ROOT,
    append_jsonl,
    apply_popcorn_seed,
    ensure_official_on_path,
    environment_info,
    file_provenance,
    format_case,
    load_cases,
    parse_case,
    parse_popcorn_seed_tokens,
    validate_factor_structure,
)


ensure_official_on_path()
from reference import check_implementation, generate_input  # noqa: E402
from diagnose import diagnose  # noqa: E402


DEFAULT_INDICES = "3,4,5,6,7,8,9,10,11"
DEFAULT_EXPERIMENTS = ("fp16-nearby", "tf32-input-nearby")


def parse_int_list(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_experiments(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_EXPERIMENTS)
    if value.strip() == "all":
        return list(DEFAULT_EXPERIMENTS)
    experiments = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(experiments) - set(DEFAULT_EXPERIMENTS))
    if unknown:
        raise ValueError(f"unknown quantization experiment(s): {', '.join(unknown)}")
    return experiments


def selected_cases(args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    if args.case:
        return [(0, parse_case(args.case))]

    cases_path = ROOT / args.cases if not Path(args.cases).is_absolute() else Path(args.cases)
    cases = load_cases(cases_path)
    indices = parse_int_list(args.indices)
    if not indices:
        indices = list(range(len(cases)))
    selected = []
    for index in indices:
        if index < 0 or index >= len(cases):
            raise IndexError(f"case index {index} is outside 0..{len(cases) - 1}")
        selected.append((index, cases[index]))
    return selected


def _synchronize(data: torch.Tensor) -> None:
    if data.is_cuda:
        torch.cuda.synchronize(data.device)


def _timed(data: torch.Tensor, fn: Callable[[], Any]) -> tuple[Any, float, float | None]:
    _synchronize(data)
    cuda_us = None
    if data.is_cuda:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_event = end_event = None

    started_ns = time.perf_counter_ns()
    value = fn()
    if data.is_cuda and end_event is not None:
        end_event.record()
    _synchronize(data)
    wall_us = (time.perf_counter_ns() - started_ns) / 1000.0

    if data.is_cuda and start_event is not None and end_event is not None:
        cuda_us = float(start_event.elapsed_time(end_event) * 1000.0)
    return value, wall_us, cuda_us


@torch.no_grad()
def _fp16_nearby(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    aq = data.to(torch.float16).to(torch.float32)
    h, tau = torch.geqrf(aq)
    h2 = repair_r(data, h, tau)
    return h2, tau, {
        "quantization": "fp16-input",
        "meaning": "FP32 geqrf on FP16-rounded input followed by R repair against original FP32 A",
    }


@torch.no_grad()
def _tf32_input_nearby(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    old_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        aq = quantize_tf32_like(data)
        h, tau = torch.geqrf(aq)
        h2 = repair_r(data, h, tau)
    finally:
        torch.set_float32_matmul_precision(old_precision)
    return h2, tau, {
        "quantization": "tf32-input",
        "meaning": "FP32 geqrf on TF32-rounded input followed by R repair; this is not TF32 QR arithmetic",
    }


def _experiment_fn(name: str) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]:
    if name == "fp16-nearby":
        return _fp16_nearby
    if name == "tf32-input-nearby":
        return _tf32_input_nearby
    raise ValueError(f"unknown quantization experiment {name!r}")


@torch.no_grad()
def run_case(
    spec: dict[str, Any],
    experiment: str,
    max_factor_scaled: float | None = None,
    max_orth_scaled: float | None = None,
) -> dict[str, Any]:
    data = generate_input(**spec)
    fn = _experiment_fn(experiment)
    (h, tau, extra), wall_us, cuda_us = _timed(data, lambda: fn(data))

    good, message = check_implementation(data, (h, tau))
    row: dict[str, Any] = {
        "experiment": experiment,
        "ok": good,
        "message": message,
        "spec": spec,
        "case_text": format_case(spec),
        "batch": int(data.shape[0]),
        "n": int(data.shape[-1]),
        "case": spec.get("case", "dense"),
        "cond": spec.get("cond"),
        "seed": spec.get("seed"),
        "device": str(data.device),
        "wall_us": wall_us,
        "cuda_us": cuda_us,
    }
    row.update(extra)

    h_checked, tau_checked, structure_error = validate_factor_structure((h, tau), data)
    if structure_error:
        row["diagnostics_error"] = structure_error
        if max_factor_scaled is not None or max_orth_scaled is not None:
            row["margin_ok"] = False
        return row

    diagnostics = diagnose(data, h_checked, tau_checked)
    row["diagnostics"] = diagnostics
    row.update(diagnostics)

    margin_ok = True
    if max_factor_scaled is not None:
        factor_margin_ok = diagnostics["factor_scaled_max"] <= max_factor_scaled
        row["max_factor_scaled_limit"] = max_factor_scaled
        row["factor_margin_ok"] = factor_margin_ok
        margin_ok = margin_ok and factor_margin_ok
    if max_orth_scaled is not None:
        orth_margin_ok = diagnostics["orth_scaled_max"] <= max_orth_scaled
        row["max_orth_scaled_limit"] = max_orth_scaled
        row["orth_margin_ok"] = orth_margin_ok
        margin_ok = margin_ok and orth_margin_ok
    if max_factor_scaled is not None or max_orth_scaled is not None:
        row["margin_ok"] = margin_ok
    return row


def _max_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    if not values:
        return None
    return max(values)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cases = [row for row in rows if not row.get("summary")]
    seeds = {row.get("popcorn_seed") for row in cases}
    experiments = sorted({str(row.get("experiment")) for row in cases if row.get("experiment")})
    num_failed = sum(1 for row in cases if not row.get("ok"))
    num_margin_failed = sum(1 for row in cases if row.get("margin_ok") is False)
    return {
        "summary": True,
        "ok": num_failed == 0 and num_margin_failed == 0,
        "num_rows": len(cases),
        "num_failed": num_failed,
        "num_margin_failed": num_margin_failed,
        "num_passed": sum(1 for row in cases if row.get("ok") and row.get("margin_ok", True)),
        "experiments": experiments,
        "num_public_seed_rows": sum(1 for row in cases if row.get("popcorn_seed") is None),
        "num_popcorn_seed_rows": sum(1 for row in cases if row.get("popcorn_seed") is not None),
        "popcorn_seeds": sorted("public" if seed is None else str(seed) for seed in seeds),
        "max_factor_scaled": _max_numeric(cases, "factor_scaled_max"),
        "max_orth_scaled": _max_numeric(cases, "orth_scaled_max"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep FP16/TF32 nearby-QR verifier experiments across QR cases and POPCORN_SEED values.",
        allow_abbrev=False,
    )
    parser.add_argument("--submission", default="submissions/candidate.py")
    parser.add_argument("--cases", default="cases/public_benchmarks.txt")
    parser.add_argument("--case", default=None)
    parser.add_argument("--indices", default=DEFAULT_INDICES)
    parser.add_argument("--experiments", default=",".join(DEFAULT_EXPERIMENTS))
    parser.add_argument(
        "--popcorn-seeds",
        default="public,1,2,3",
        help="Comma-separated POPCORN_SEED values; use 'public' for the unmodified public seed.",
    )
    parser.add_argument("--max-factor-scaled", type=float, default=18.0)
    parser.add_argument("--max-orth-scaled", type=float, default=80.0)
    parser.add_argument("--allow-failure", action="store_true")
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--json", action="store_true", help="Accepted for consistency; output is always JSONL.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--record-env", action="store_true", help="Include repo, torch/CUDA, and submission provenance.")
    args = parser.parse_args()

    submission = ROOT / args.submission if not Path(args.submission).is_absolute() else Path(args.submission)
    env = environment_info(torch) if args.record_env else {}
    if args.record_env:
        provenance = file_provenance(submission)
        env["submission"] = provenance["path"]
        env["submission_sha256"] = provenance["sha256"]

    rows: list[dict[str, Any]] = []
    failed = False
    experiments = parse_experiments(args.experiments)
    seeds = parse_popcorn_seed_tokens(args.popcorn_seeds, default=[None]) or [None]
    for popcorn_seed in seeds:
        for case_index, base_spec in selected_cases(args):
            spec = apply_popcorn_seed([base_spec], popcorn_seed)[0]
            for experiment in experiments:
                try:
                    row = run_case(
                        spec,
                        experiment,
                        max_factor_scaled=args.max_factor_scaled,
                        max_orth_scaled=args.max_orth_scaled,
                    )
                except Exception as exc:
                    row = {
                        "experiment": experiment,
                        "ok": False,
                        "message": f"{type(exc).__name__}: {exc}",
                        "spec": spec,
                        "case_text": format_case(spec),
                    }
                    if args.max_factor_scaled is not None or args.max_orth_scaled is not None:
                        row["margin_ok"] = False
                if env:
                    row = {**env, **row}
                row.update({"case_index": case_index, "popcorn_seed": popcorn_seed})
                print(json.dumps(row, sort_keys=True), flush=True)
                rows.append(row)
                row_failed = (not row.get("ok")) or row.get("margin_ok") is False
                failed = failed or row_failed
                if row_failed and args.stop_on_fail:
                    if args.out:
                        append_jsonl(ROOT / args.out if not Path(args.out).is_absolute() else args.out, rows)
                    return 1

    summary = summarize(rows)
    print(json.dumps(summary, sort_keys=True), flush=True)
    rows.append(summary)

    if args.out:
        append_jsonl(ROOT / args.out if not Path(args.out).is_absolute() else args.out, rows)
    return 0 if summary["ok"] or args.allow_failure else 1


if __name__ == "__main__":
    sys.exit(main())
