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
)
from route_expectations import allowed_family_routes, expected_case_route
from trace_candidate_routes import load_candidate_module, plan_group_counts


ensure_official_on_path()
from reference import generate_input  # noqa: E402


DEFAULT_INDICES = "3,4,7,8,9,10,11"


def parse_int_list(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


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


def expected_sampled_class(spec: dict[str, Any]) -> str | None:
    n = int(spec["n"])
    case = str(spec.get("case", "dense"))
    if n == 512:
        if case in {"dense", "mixed", "rankdef", "clustered"}:
            return case
    if n == 1024:
        if case in {"dense", "mixed"}:
            return case
        if case == "nearrank":
            return "nearrank"
    return None


def expected_route(spec: dict[str, Any]) -> str | None:
    return expected_case_route(spec)


def allowed_routes(spec: dict[str, Any]) -> set[str] | None:
    return allowed_family_routes(spec)


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


def sampled_class(candidate, data: torch.Tensor) -> str:
    n = int(data.shape[-1])
    if n == 512 and hasattr(candidate, "classify_512_sampled"):
        return str(candidate.classify_512_sampled(data))
    if n == 1024 and hasattr(candidate, "classify_1024_sampled"):
        return str(candidate.classify_1024_sampled(data))
    if hasattr(candidate, "_classify_sampled"):
        return str(candidate._classify_sampled(data))
    return "unavailable"


def route_plan(candidate, data: torch.Tensor) -> tuple[str, dict[str, Any] | None]:
    if hasattr(candidate, "_route_plan_for_data"):
        route, plan = candidate._route_plan_for_data(data)
        return str(route), plan
    if hasattr(candidate, "_route_for_data"):
        return str(candidate._route_for_data(data)), None
    return "custom_kernel", None


def maybe_plan(candidate, data: torch.Tensor, spec: dict[str, Any], plan: dict[str, Any] | None, include_plan: bool) -> dict[str, Any] | None:
    if not include_plan:
        return plan
    if plan is not None:
        return plan
    if not hasattr(candidate, "_mixed_structured_plan"):
        return None
    if int(data.shape[-1]) not in (512, 1024):
        return None
    if str(spec.get("case", "dense")) != "mixed":
        return None
    return candidate._mixed_structured_plan(data, cond=int(spec.get("cond", 2)))


def plan_diagnostics(plan: dict[str, Any] | None) -> dict[str, Any]:
    if plan is None:
        return {}
    out: dict[str, Any] = {
        "structured_group_counts": plan_group_counts(plan),
    }
    for source_key, output_key in [
        ("candidate_counts", "structured_candidate_counts"),
        ("exact_check_counts", "structured_exact_check_counts"),
    ]:
        value = plan.get(source_key)
        if isinstance(value, dict):
            out[output_key] = {str(key): int(count) for key, count in value.items()}
    for key in ["sampled_plan", "sampled_matrix_count", "sampled_row_count"]:
        if key in plan:
            out[f"structured_{key}"] = plan[key]
    return out


@torch.no_grad()
def run_case(candidate, spec: dict[str, Any], include_plan: bool) -> dict[str, Any]:
    data = generate_input(**spec)
    expected_class = expected_sampled_class(spec)
    expected_candidate_route = expected_route(spec)

    cls, class_wall_us, class_cuda_us = _timed(data, lambda: sampled_class(candidate, data))
    (route, plan), route_wall_us, route_cuda_us = _timed(data, lambda: route_plan(candidate, data))
    plan = maybe_plan(candidate, data, spec, plan, include_plan)

    candidate_allowed_routes = allowed_routes(spec)
    cuda_bypass = route in {"qr512_cuda_fast", "qr1024_cuda_fast"}
    classifier_ok = expected_class is None or cls == expected_class
    route_ok = candidate_allowed_routes is None or route in candidate_allowed_routes

    row: dict[str, Any] = {
        "ok": bool(classifier_ok and route_ok),
        "spec": spec,
        "case_text": format_case(spec),
        "batch": int(data.shape[0]),
        "n": int(data.shape[-1]),
        "case": spec.get("case", "dense"),
        "device": str(data.device),
        "sampled_class": cls,
        "expected_sampled_class": expected_class,
        "classifier_ok": bool(classifier_ok),
        "route": route,
        "expected_route": expected_candidate_route,
        "allowed_routes": sorted(candidate_allowed_routes) if candidate_allowed_routes is not None else None,
        "route_ok": bool(route_ok),
        "route_cuda_bypass": bool(cuda_bypass),
        "sampled_class_wall_us": class_wall_us,
        "sampled_class_cuda_us": class_cuda_us,
        "route_wall_us": route_wall_us,
        "route_cuda_us": route_cuda_us,
        "case_metadata_available": False,
        "case_metadata_passed_to_submission": False,
        "case_info_source": "tensor_values" if int(data.shape[-1]) in (512, 1024) else "data.shape",
        "uses_tensor_values_for_case_selection": int(data.shape[-1]) in (512, 1024),
    }
    row.update(plan_diagnostics(plan))
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cases = [row for row in rows if not row.get("summary")]
    seeds = {row.get("popcorn_seed") for row in cases}
    return {
        "summary": True,
        "ok": all(row.get("ok") for row in cases),
        "num_rows": len(cases),
        "num_failed": sum(1 for row in cases if not row.get("ok")),
        "num_classifier_mismatch": sum(1 for row in cases if row.get("classifier_ok") is False),
        "num_route_mismatch": sum(1 for row in cases if row.get("route_ok") is False),
        "num_route_cuda_bypass": sum(1 for row in cases if row.get("route_cuda_bypass") is True),
        "num_public_seed_rows": sum(1 for row in cases if row.get("popcorn_seed") is None),
        "num_popcorn_seed_rows": sum(1 for row in cases if row.get("popcorn_seed") is not None),
        "popcorn_seeds": sorted("public" if seed is None else str(seed) for seed in seeds),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep sampled classifier decisions across colliding QR benchmark cases and POPCORN_SEED values.",
        allow_abbrev=False,
    )
    parser.add_argument("--submission", default="submissions/candidate.py")
    parser.add_argument("--cases", default="cases/public_benchmarks.txt")
    parser.add_argument("--case", default=None)
    parser.add_argument("--indices", default=DEFAULT_INDICES)
    parser.add_argument("--popcorn-seeds", default="public,1,2,3")
    parser.add_argument("--include-plan", action="store_true", help="Include structured mixed-plan counters for mixed rows.")
    parser.add_argument("--allow-mismatch", action="store_true")
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
    seeds = parse_popcorn_seed_tokens(args.popcorn_seeds, default=[None]) or [None]
    for popcorn_seed in seeds:
        for case_index, base_spec in selected_cases(args):
            spec = apply_popcorn_seed([base_spec], popcorn_seed)[0]
            try:
                row = run_case(candidate, spec, include_plan=args.include_plan)
            except Exception as exc:
                row = {
                    "ok": False,
                    "message": f"{type(exc).__name__}: {exc}",
                    "spec": spec,
                    "case_text": format_case(spec),
                }
            if env:
                row = {**env, **row}
            row.update({"case_index": case_index, "popcorn_seed": popcorn_seed})
            print(json.dumps(row, sort_keys=True), flush=True)
            rows.append(row)

    summary = summarize(rows)
    print(json.dumps(summary, sort_keys=True), flush=True)
    rows.append(summary)

    if args.out:
        append_jsonl(ROOT / args.out if not Path(args.out).is_absolute() else args.out, rows)
    return 0 if summary["ok"] or args.allow_mismatch else 1


if __name__ == "__main__":
    sys.exit(main())
