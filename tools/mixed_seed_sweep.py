from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

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
from trace_candidate_routes import load_candidate_module, plan_diagnostics, plan_group_counts


ensure_official_on_path()
from reference import check_implementation, generate_input  # noqa: E402
from diagnose import diagnose  # noqa: E402


DEFAULT_BENCHMARK_INDICES = "7,8"
DEFAULT_TEST_INDICES = "19,20,21"


def parse_int_list(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _selected_from_file(path: str | Path, indices: str, source: str) -> list[tuple[str, int, dict[str, Any]]]:
    cases_path = ROOT / path if not Path(path).is_absolute() else Path(path)
    cases = load_cases(cases_path)
    selected_indices = parse_int_list(indices)
    if not selected_indices:
        selected_indices = list(range(len(cases)))
    selected = []
    for index in selected_indices:
        if index < 0 or index >= len(cases):
            raise IndexError(f"{source} case index {index} is outside 0..{len(cases) - 1}")
        spec = cases[index]
        if str(spec.get("case", "dense")) != "mixed":
            raise ValueError(f"{source} case index {index} is not a mixed case: {format_case(spec)}")
        selected.append((source, index, spec))
    return selected


def selected_cases(args: argparse.Namespace) -> list[tuple[str, int, dict[str, Any]]]:
    if args.case:
        return [("inline", 0, parse_case(args.case))]
    selected = []
    selected.extend(_selected_from_file(args.benchmark_cases, args.benchmark_indices, "public_benchmarks"))
    selected.extend(_selected_from_file(args.test_cases, args.test_indices, "public_tests"))
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


def expected_mixed_route(spec: dict[str, Any]) -> str | None:
    n = int(spec["n"])
    if n == 512:
        return "qr512_mixed_fast"
    if n == 1024:
        return "qr1024_mixed_fast"
    return None


def allowed_mixed_routes(spec: dict[str, Any]) -> set[str] | None:
    n = int(spec["n"])
    if n == 512:
        return {"qr512_mixed_fast", "qr512_cuda_fast", "qr512_blocked_cuda_auto_fast"}
    if n == 1024:
        return {"qr1024_mixed_fast", "qr1024_cuda_fast", "qr1024_blocked_cuda_auto_fast"}
    return None


def actual_route_plan(candidate, data: torch.Tensor) -> tuple[str, dict[str, Any] | None]:
    if hasattr(candidate, "_route_plan_for_data"):
        route, plan = candidate._route_plan_for_data(data)
        return str(route), plan
    if hasattr(candidate, "_route_for_data"):
        return str(candidate._route_for_data(data)), None
    return "custom_kernel", None


def sampled_class(candidate, data: torch.Tensor) -> str | None:
    n = int(data.shape[-1])
    if n == 512 and hasattr(candidate, "classify_512_sampled"):
        return str(candidate.classify_512_sampled(data))
    if n == 1024 and hasattr(candidate, "classify_1024_sampled"):
        return str(candidate.classify_1024_sampled(data))
    if hasattr(candidate, "_classify_sampled"):
        return str(candidate._classify_sampled(data))
    return None


def mixed_plan_metadata(candidate, data: torch.Tensor, spec: dict[str, Any], plan: dict[str, Any] | None) -> dict[str, Any]:
    n = int(data.shape[-1])
    if plan is None and n in (512, 1024) and hasattr(candidate, "_mixed_structured_plan"):
        plan = candidate._mixed_structured_plan(data, cond=int(spec.get("cond", 2)))
    if plan is None:
        return {}
    return {
        "structured_group_counts": plan_group_counts(plan),
        **plan_diagnostics(plan),
    }


@torch.no_grad()
def run_case(
    candidate,
    spec: dict[str, Any],
    max_factor_scaled: float | None = None,
    max_orth_scaled: float | None = None,
) -> dict[str, Any]:
    data = generate_input(**spec)
    (route, plan), route_wall_us, route_cuda_us = _timed(data, lambda: actual_route_plan(candidate, data))
    cls, class_wall_us, class_cuda_us = _timed(data, lambda: sampled_class(candidate, data))
    output, kernel_wall_us, kernel_cuda_us = _timed(data, lambda: candidate.custom_kernel(data.clone()))

    good, message = check_implementation(data, output)
    expected_route = expected_mixed_route(spec)
    allowed_routes = allowed_mixed_routes(spec)
    route_cuda_bypass = route in {"qr512_cuda_fast", "qr1024_cuda_fast"}
    route_ok = allowed_routes is None or route in allowed_routes
    row: dict[str, Any] = {
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
        "route": route,
        "expected_route": expected_route,
        "allowed_routes": sorted(allowed_routes) if allowed_routes is not None else None,
        "route_ok": bool(route_ok),
        "route_cuda_bypass": bool(route_cuda_bypass),
        "sampled_class": cls,
        "sampled_class_ok": cls in (None, "mixed"),
        "case_metadata_available": False,
        "case_metadata_passed_to_submission": False,
        "case_info_source": "tensor_values" if int(data.shape[-1]) in (512, 1024) else "data.shape",
        "route_wall_us": route_wall_us,
        "route_cuda_us": route_cuda_us,
        "sampled_class_wall_us": class_wall_us,
        "sampled_class_cuda_us": class_cuda_us,
        "kernel_wall_us": kernel_wall_us,
        "kernel_cuda_us": kernel_cuda_us,
    }
    row.update(mixed_plan_metadata(candidate, data, spec, plan))

    h, tau, structure_error = validate_factor_structure(output, data)
    if structure_error:
        row["diagnostics_error"] = structure_error
        if max_factor_scaled is not None or max_orth_scaled is not None:
            row["margin_ok"] = False
        return row

    diagnostics = diagnose(data, h, tau)
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


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cases = [row for row in rows if not row.get("summary")]
    seeds = {row.get("popcorn_seed") for row in cases}
    shapes = sorted(
        {
            f"{int(row['batch'])}x{int(row['n'])}"
            for row in cases
            if row.get("batch") is not None and row.get("n") is not None
        }
    )
    factor_values = [float(row["factor_scaled_max"]) for row in cases if isinstance(row.get("factor_scaled_max"), (int, float))]
    orth_values = [float(row["orth_scaled_max"]) for row in cases if isinstance(row.get("orth_scaled_max"), (int, float))]
    num_failed = sum(1 for row in cases if not row.get("ok"))
    num_margin_failed = sum(1 for row in cases if row.get("margin_ok") is False)
    num_route_mismatch = sum(1 for row in cases if row.get("route_ok") is False)
    return {
        "summary": True,
        "ok": num_failed == 0 and num_margin_failed == 0 and num_route_mismatch == 0,
        "num_rows": len(cases),
        "num_failed": num_failed,
        "num_margin_failed": num_margin_failed,
        "num_route_mismatch": num_route_mismatch,
        "num_public_seed_rows": sum(1 for row in cases if row.get("popcorn_seed") is None),
        "num_popcorn_seed_rows": sum(1 for row in cases if row.get("popcorn_seed") is not None),
        "popcorn_seeds": sorted("public" if seed is None else str(seed) for seed in seeds),
        "case_sources": sorted({str(row.get("case_source")) for row in cases if row.get("case_source")}),
        "shapes": shapes,
        "max_factor_scaled": max(factor_values) if factor_values else None,
        "max_orth_scaled": max(orth_values) if orth_values else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run mixed-batch QR correctness sweeps across public and POPCORN-mutated seeds.",
        allow_abbrev=False,
    )
    parser.add_argument("--submission", default="submissions/candidate.py")
    parser.add_argument("--benchmark-cases", default="cases/public_benchmarks.txt")
    parser.add_argument("--benchmark-indices", default=DEFAULT_BENCHMARK_INDICES)
    parser.add_argument("--test-cases", default="cases/public_tests.txt")
    parser.add_argument("--test-indices", default=DEFAULT_TEST_INDICES)
    parser.add_argument("--case", default=None)
    parser.add_argument("--popcorn-seeds", default="public,1,2,3")
    parser.add_argument("--max-factor-scaled", type=float, default=18.0)
    parser.add_argument("--max-orth-scaled", type=float, default=80.0)
    parser.add_argument("--allow-failure", action="store_true")
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--json", action="store_true", help="Accepted for consistency; output is always JSONL.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--record-env", action="store_true", help="Include repo, torch/CUDA, and submission provenance.")
    args = parser.parse_args()

    submission = ROOT / args.submission if not Path(args.submission).is_absolute() else Path(args.submission)
    candidate = load_candidate_module(submission)
    env = environment_info(torch) if args.record_env else {}
    if args.record_env:
        provenance = file_provenance(submission)
        env["submission"] = provenance["path"]
        env["submission_sha256"] = provenance["sha256"]

    rows: list[dict[str, Any]] = []
    failed = False
    seeds = parse_popcorn_seed_tokens(args.popcorn_seeds, default=[None]) or [None]
    for popcorn_seed in seeds:
        for case_source, case_index, base_spec in selected_cases(args):
            spec = apply_popcorn_seed([base_spec], popcorn_seed)[0]
            try:
                row = run_case(
                    candidate,
                    spec,
                    max_factor_scaled=args.max_factor_scaled,
                    max_orth_scaled=args.max_orth_scaled,
                )
            except Exception as exc:
                row = {
                    "ok": False,
                    "message": f"{type(exc).__name__}: {exc}",
                    "spec": spec,
                    "case_text": format_case(spec),
                }
                if args.max_factor_scaled is not None or args.max_orth_scaled is not None:
                    row["margin_ok"] = False
            if env:
                row = {**env, **row}
            row.update({"case_source": case_source, "case_index": case_index, "popcorn_seed": popcorn_seed})
            print(json.dumps(row, sort_keys=True), flush=True)
            rows.append(row)
            row_failed = (not row.get("ok")) or row.get("margin_ok") is False or row.get("route_ok") is False
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
