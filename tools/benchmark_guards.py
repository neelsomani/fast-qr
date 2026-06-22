from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

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
)
from trace_candidate_routes import (
    load_candidate_module,
    route_uses_case_classifier,
    route_uses_tensor_guard,
    shape_requires_tensor_route,
)


ensure_official_on_path()
from reference import generate_input  # noqa: E402


@torch.no_grad()
def route_decision(candidate, data: torch.Tensor) -> str:
    if hasattr(candidate, "_route_for_data"):
        return str(candidate._route_for_data(data))

    batch, n, _ = data.shape

    if candidate._should_try_identity_q(data) and candidate._is_exact_upper_or_diagonal(data):
        return "identity_q"

    if batch == 20 and n == 32:
        return "qr32_fast"
    if batch == 40 and n == 176:
        return "qr176_fast"
    if batch == 40 and n == 352:
        return "qr352_fast"

    if n == 512:
        rank = candidate._rankdef_effective_cols(n)
        if bool(candidate._batch_tail_columns_are_exact_zero(data, rank).all().item()):
            return "qr512_rankdef_fast"
        clustered_cols = candidate._clustered_effective_cols(n)
        if bool(candidate._batch_tail_columns_are_tiny_relative(data, clustered_cols).all().item()):
            return "qr512_clustered_fast"
        if candidate._has_structured_mixed_subset(data, cond=2):
            return "qr512_mixed_fast"

        cls = candidate.classify_512_sampled(data)
        if cls == "rankdef":
            return "qr512_rankdef_fast"
        if cls == "clustered":
            return "qr512_clustered_fast"
        if cls == "mixed":
            return "qr512_mixed_fast"
        return "qr512_dense_fast"

    if n == 1024:
        rank = candidate._rankdef_effective_cols(n)
        if bool(candidate._batch_tail_columns_are_exact_zero(data, rank).all().item()):
            return "qr1024_rankdef_fast"
        clustered_cols = candidate._clustered_effective_cols(n)
        if bool(candidate._batch_tail_columns_are_tiny_relative(data, clustered_cols).all().item()):
            return "qr1024_clustered_fast"
        if candidate._tail_matches_head_columns(data, rank):
            return "qr1024_nearrank_fast"
        if candidate._has_structured_mixed_subset(data, cond=2):
            return "qr1024_mixed_fast"

        cls = candidate.classify_1024_sampled(data)
        if cls == "rankdef":
            return "qr1024_rankdef_fast"
        if cls == "clustered":
            return "qr1024_clustered_fast"
        if cls == "mixed":
            return "qr1024_mixed_fast"
        if cls == "nearrank":
            return "qr1024_nearrank_fast"
        return "qr1024_dense_fast"

    if n == 2048:
        rank = candidate._rankdef_effective_cols(n)
        if bool(candidate._batch_tail_columns_are_exact_zero(data, rank).all().item()):
            return "qr2048_rankdef_fast"
        if candidate._has_structured_mixed_subset(data, cond=2):
            return "qr2048_mixed_fast"
        return _dense_tail_route(candidate, data, "qr2048_dense_fast")
    if n == 4096:
        return _dense_tail_route(candidate, data, "qr4096_dense_fast")

    return "torch.geqrf"


def _dense_tail_route(candidate, data: torch.Tensor, route_name: str) -> str:
    n = data.shape[-1]
    cut = candidate._dense_tail_cut(n)
    if cut <= 0:
        return "torch.geqrf"
    rank = n - cut
    threshold = candidate._dense_tail_threshold(n)
    if threshold > 0.0 and candidate._tail_columns_are_tiny_relative(data, rank, threshold):
        return route_name
    return "torch.geqrf"


def _synchronize(data: torch.Tensor) -> None:
    if data.is_cuda:
        torch.cuda.synchronize(data.device)


def _clear_route_cache(candidate) -> None:
    cache = getattr(candidate, "_ROUTE_CACHE", None)
    if cache is not None:
        cache.clear()


@torch.no_grad()
def measure_one_route_decision(candidate, data: torch.Tensor) -> dict[str, Any]:
    _synchronize(data)
    cuda_us = None
    if data.is_cuda:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_event = end_event = None

    start_ns = time.perf_counter_ns()
    route = route_decision(candidate, data)
    if data.is_cuda and end_event is not None:
        end_event.record()
    _synchronize(data)
    elapsed_ns = time.perf_counter_ns() - start_ns

    if data.is_cuda and start_event is not None and end_event is not None:
        cuda_us = float(start_event.elapsed_time(end_event) * 1000.0)

    return {
        "route": route,
        "wall_us": elapsed_ns / 1000.0,
        "cuda_us": cuda_us,
    }


@torch.no_grad()
def measure_route_decision(candidate, data: torch.Tensor, repeats: int, warmup: int) -> dict[str, Any]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    _clear_route_cache(candidate)
    cold = measure_one_route_decision(candidate, data)

    route = None
    for _ in range(warmup):
        route = route_decision(candidate, data)
    _synchronize(data)

    cuda_us = None
    if data.is_cuda:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_event = end_event = None

    start_ns = time.perf_counter_ns()
    for _ in range(repeats):
        route = route_decision(candidate, data)
    if data.is_cuda and end_event is not None:
        end_event.record()
    _synchronize(data)
    elapsed_ns = time.perf_counter_ns() - start_ns

    if data.is_cuda and start_event is not None and end_event is not None:
        cuda_us = float(start_event.elapsed_time(end_event) * 1000.0 / repeats)

    return {
        "route": route or cold["route"],
        "repeats": repeats,
        "warmup": warmup,
        "cold_wall_us": cold["wall_us"],
        "cold_cuda_us": cold["cuda_us"],
        "wall_us": elapsed_ns / 1000.0 / repeats,
        "cuda_us": cuda_us,
    }


def run_case(candidate, spec: dict[str, Any], repeats: int, warmup: int) -> dict[str, Any]:
    data = generate_input(**spec)
    batch = int(data.shape[0])
    n = int(data.shape[-1])
    measurement = measure_route_decision(candidate, data, repeats=repeats, warmup=warmup)
    route = str(measurement["route"])
    uses_case_tensor_values = shape_requires_tensor_route(batch, n)
    uses_case_classifier = route_uses_case_classifier(route)
    uses_tensor_guard = route_uses_tensor_guard(candidate, batch, n, route)
    dispatch_info_sources = ["data.shape"]
    if uses_tensor_guard:
        dispatch_info_sources.append("tensor_values")
    case_selection_info_sources = ["data.shape"]
    if uses_case_tensor_values:
        case_selection_info_sources.append("tensor_values")
    row: dict[str, Any] = {
        "spec": format_case(spec),
        "batch": batch,
        "n": n,
        "case": spec.get("case", "dense"),
        "device": str(data.device),
        "case_metadata_available": False,
        "case_metadata_passed_to_submission": False,
        "case_info_source": "tensor_values" if uses_case_tensor_values else "data.shape",
        "case_selection_info_sources": case_selection_info_sources,
        "shape_collision": uses_case_tensor_values,
        "shape_only_case_selection": not uses_case_tensor_values,
        "shape_only_dispatch": not uses_tensor_guard,
        "uses_tensor_values_for_dispatch": uses_tensor_guard,
        "uses_tensor_values_for_case_selection": uses_case_tensor_values,
        "requires_tensor_guard_for_case_specific_path": uses_case_tensor_values,
        "classifier_needed_for_case_specific_path": uses_case_tensor_values,
        "classifier_needed_for_current_candidate": uses_case_classifier,
        "classifier_on_current_hot_path": uses_case_classifier,
        "dispatch_info_sources": dispatch_info_sources,
        "route_decision_source": "+".join(dispatch_info_sources),
    }
    row.update(measurement)
    return row


def parse_indices(raw: str, count: int) -> list[int]:
    if not raw:
        return list(range(count))
    out = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        idx = int(piece)
        if idx < 0 or idx >= count:
            raise IndexError(f"case index {idx} is outside 0..{count - 1}")
        out.append(idx)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure candidate route/classifier guard overhead without running QR.",
        allow_abbrev=False,
    )
    parser.add_argument("--submission", default="submissions/candidate.py")
    parser.add_argument("--cases", default="cases/public_benchmarks.txt")
    parser.add_argument("--indices", default="")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--popcorn-seed", type=int, default=None)
    parser.add_argument("--json", action="store_true")
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
    cases_path = ROOT / args.cases if not Path(args.cases).is_absolute() else Path(args.cases)
    cases = apply_popcorn_seed(load_cases(cases_path), args.popcorn_seed)
    selected = parse_indices(args.indices, len(cases))

    rows = []
    for index in selected:
        row = run_case(candidate, cases[index], repeats=args.repeats, warmup=args.warmup)
        if env:
            row = {**env, **row}
        row["case_index"] = index
        rows.append(row)
        if args.json:
            print(json.dumps(row, sort_keys=True), flush=True)
        else:
            cuda = "n/a" if row["cuda_us"] is None else f"{row['cuda_us']:.3f}"
            cold_cuda = "n/a" if row["cold_cuda_us"] is None else f"{row['cold_cuda_us']:.3f}"
            print(
                f"{index}: {row['spec']} -> {row['route']} "
                f"cold_wall_us={row['cold_wall_us']:.3f} cold_cuda_us={cold_cuda} "
                f"hot_wall_us={row['wall_us']:.3f} hot_cuda_us={cuda}",
                flush=True,
            )

    if args.out:
        append_jsonl(ROOT / args.out if not Path(args.out).is_absolute() else args.out, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
