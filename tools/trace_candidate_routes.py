from __future__ import annotations

import argparse
import importlib.util
import json
import sys
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


ensure_official_on_path()
from reference import generate_input  # noqa: E402


def load_candidate_module(path: str | Path):
    ensure_official_on_path()
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    submission_dir = str(path.parent)
    if submission_dir in sys.path:
        sys.path.remove(submission_dir)
    sys.path.insert(1, submission_dir)

    module_name = f"route_trace_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load candidate from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def count_mask(mask: torch.Tensor) -> int:
    return int(mask.to(torch.int64).sum().item())


def plan_group_counts(plan: dict[str, Any]) -> dict[str, int]:
    return {
        "rankdef": int(plan["rankdef_idx"].numel()),
        "clustered": int(plan["clustered_idx"].numel()),
        "scaled_nearrank": int(plan["scaled_nearrank_idx"].numel()),
        "tiny_dense_tail": int(plan["tiny_dense_idx"].numel()),
        "fallback": int(plan["fallback_idx"].numel()),
    }


def plan_diagnostics(plan: dict[str, Any] | None) -> dict[str, Any]:
    if plan is None:
        return {}

    out: dict[str, Any] = {}
    candidate_counts = plan.get("candidate_counts")
    if isinstance(candidate_counts, dict):
        out["structured_candidate_counts"] = {
            str(key): int(value) for key, value in candidate_counts.items()
        }
    exact_check_counts = plan.get("exact_check_counts")
    if isinstance(exact_check_counts, dict):
        out["structured_exact_check_counts"] = {
            str(key): int(value) for key, value in exact_check_counts.items()
        }
    if "sampled_plan" in plan:
        out["structured_sampled_plan"] = bool(plan.get("sampled_plan"))
    if "trusted_sampled_guards" in plan:
        out["structured_trusted_sampled_guards"] = bool(plan.get("trusted_sampled_guards"))
    if "sampled_matrix_count" in plan:
        out["structured_sampled_matrix_count"] = int(plan["sampled_matrix_count"])
    if "sampled_row_count" in plan:
        out["structured_sampled_row_count"] = int(plan["sampled_row_count"])
    return out


def structured_group_counts(candidate, data: torch.Tensor, cond: int = 2) -> dict[str, int]:
    if hasattr(candidate, "_mixed_structured_plan"):
        return plan_group_counts(candidate._mixed_structured_plan(data, cond=cond))

    batch, n, _ = data.shape
    remaining = torch.ones((batch,), device=data.device, dtype=torch.bool)
    rank = int(candidate._rankdef_effective_cols(n))
    clustered_cols = int(candidate._clustered_effective_cols(n))

    rankdef = candidate._batch_tail_columns_are_exact_zero(data, rank) & remaining
    remaining = remaining & ~rankdef

    clustered = candidate._batch_tail_columns_are_tiny_relative(data, clustered_cols) & remaining
    remaining = remaining & ~clustered

    scaled_nearrank = candidate._batch_tail_matches_scaled_head_columns(data, rank, cond) & remaining
    remaining = remaining & ~scaled_nearrank

    tiny_dense = torch.zeros((batch,), device=data.device, dtype=torch.bool)
    mixed_cut = int(candidate._mixed_dense_tail_cut(n))
    if mixed_cut > 0:
        tiny_dense = (
            candidate._batch_tail_columns_are_tiny_relative(
                data,
                n - mixed_cut,
                float(candidate._mixed_dense_tail_threshold(n)),
            )
            & remaining
        )
        remaining = remaining & ~tiny_dense

    return {
        "rankdef": count_mask(rankdef),
        "clustered": count_mask(clustered),
        "scaled_nearrank": count_mask(scaled_nearrank),
        "tiny_dense_tail": count_mask(tiny_dense),
        "fallback": count_mask(remaining),
    }


def actual_route_plan(candidate, data: torch.Tensor) -> tuple[str, dict[str, Any] | None]:
    if hasattr(candidate, "_route_plan_for_data"):
        route, plan = candidate._route_plan_for_data(data)
        return str(route), plan
    if hasattr(candidate, "_route_for_data"):
        return str(candidate._route_for_data(data)), None
    return "unknown", None


def dispatch_for_shape(batch: int, n: int) -> str:
    if batch == 20 and n == 32:
        return "qr32_fast"
    if batch == 40 and n == 176:
        return "qr176_fast"
    if batch == 40 and n == 352:
        return "qr352_fast"
    if n == 512:
        return "qr512_fast"
    if n == 1024:
        return "qr1024_fast"
    if n == 2048:
        return "qr2048_fast"
    if n == 4096:
        return "qr4096_fast"
    return "fallback"


def shape_requires_tensor_route(batch: int, n: int) -> bool:
    return n in (512, 1024)


CASE_CLASSIFIER_ROUTES = {
    "qr512_rankdef_fast",
    "qr512_clustered_fast",
    "qr512_mixed_fast",
    "qr1024_rankdef_fast",
    "qr1024_clustered_fast",
    "qr1024_nearrank_fast",
    "qr1024_mixed_fast",
    "qr2048_rankdef_fast",
    "qr2048_mixed_fast",
}


def route_uses_case_classifier(route: str) -> bool:
    return route in CASE_CLASSIFIER_ROUTES


def route_uses_sampled_classifier(candidate, data: torch.Tensor, route: str) -> bool:
    batch, n, _ = data.shape
    if not bool(candidate._structured_routes_enabled()):
        return False
    if route_uses_case_classifier(route):
        return True
    if not shape_requires_tensor_route(batch, n):
        return False

    structured_first = structured_before_cuda(candidate, n)
    if structured_first:
        return True

    if n == 512:
        cuda_preempts_classifier = (
            bool(candidate._qr512_blocked_cuda_route_enabled(data))
            or bool(candidate._qr512_cuda_route_enabled(data))
        )
    elif n == 1024:
        cuda_preempts_classifier = (
            bool(candidate._qr1024_blocked_cuda_route_enabled(data))
            or bool(candidate._qr1024_cuda_route_enabled(data))
        )
    else:
        cuda_preempts_classifier = False

    return not cuda_preempts_classifier


def route_uses_dense_tail_guard(candidate, batch: int, n: int, route: str | None = None) -> bool:
    if route in CASE_CLASSIFIER_ROUTES or route in {
        "qr512_cuda_fast",
        "qr512_blocked_cuda_fast",
        "qr512_blocked_cuda_auto_fast",
        "qr1024_cuda_fast",
        "qr1024_blocked_cuda_fast",
        "qr1024_blocked_cuda_auto_fast",
        "qr2048_blocked_cuda_fast",
        "qr2048_blocked_cuda_auto_fast",
        "qr4096_blocked_cuda_fast",
        "qr4096_blocked_cuda_auto_fast",
    }:
        return False
    if n in (512, 1024, 2048, 4096):
        return bool(candidate._dense_tail_routes_enabled()) and int(candidate._dense_tail_cut(n)) > 0
    return False


def route_uses_tensor_guard(candidate, batch: int, n: int, route: str | None = None) -> bool:
    if route is not None and route_uses_case_classifier(route):
        return True
    if route is None and shape_requires_tensor_route(batch, n) and bool(candidate._structured_routes_enabled()):
        return True
    return route_uses_dense_tail_guard(candidate, batch, n, route)


def structured_before_cuda(candidate, n: int) -> bool:
    if hasattr(candidate, "_structured_before_cuda"):
        return bool(candidate._structured_before_cuda(n))
    return False


def dense_tail_row(candidate, data: torch.Tensor) -> dict[str, Any]:
    n = data.shape[-1]
    cut = int(candidate._dense_tail_cut(n))
    force = bool(candidate._dense_tail_force(n)) if hasattr(candidate, "_dense_tail_force") else False
    if cut <= 0:
        return {"dense_tail_cut": 0, "dense_tail_force": force, "dense_tail_applies": False}
    rank = n - cut
    threshold = float(candidate._dense_tail_threshold(n))
    return {
        "dense_tail_cut": cut,
        "dense_tail_rank": rank,
        "dense_tail_threshold": threshold,
        "dense_tail_force": force,
        "dense_tail_applies": bool(candidate._tail_columns_are_tiny_relative(data, rank, threshold)),
    }


def trace_route(candidate, spec: dict[str, Any]) -> dict[str, Any]:
    data = generate_input(**spec)
    batch, n, _ = data.shape
    case = str(spec.get("case", "dense"))
    route, plan = actual_route_plan(candidate, data)
    uses_case_tensor_values = shape_requires_tensor_route(batch, n) or route_uses_case_classifier(route)
    uses_case_classifier = route_uses_sampled_classifier(candidate, data, route)
    uses_tensor_guard = uses_case_classifier or route_uses_dense_tail_guard(candidate, batch, n, route)
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
        "case": case,
        "device": str(data.device),
        "dispatch": dispatch_for_shape(batch, n),
        "route": route,
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

    if batch == 20 and n == 32:
        pass
    elif batch == 40 and n == 176:
        pass
    elif batch == 40 and n == 352:
        pass
    elif n == 512:
        rank = int(candidate._rankdef_effective_cols(n))
        clustered_cols = int(candidate._clustered_effective_cols(n))
        rankdef_all = bool(candidate._batch_tail_columns_are_exact_zero(data, rank).all().item())
        clustered_all = bool(candidate._batch_tail_columns_are_tiny_relative(data, clustered_cols).all().item())
        mixed_subset = bool(candidate._has_structured_mixed_subset(data, cond=2))
        sampled = candidate.classify_512_sampled(data)
        diagnostic_plan = plan if plan is not None else candidate._mixed_structured_plan(data, cond=2)
        row.update(
            {
                "rankdef_cols": rank,
                "clustered_cols": clustered_cols,
                "sampled_class": sampled,
                "structured_before_cuda": structured_before_cuda(candidate, n),
                "cuda_route_bypasses_classifier": not uses_case_classifier,
                "rankdef_all": rankdef_all,
                "clustered_all": clustered_all,
                "structured_mixed_subset": mixed_subset,
                "structured_group_counts": plan_group_counts(diagnostic_plan),
                **plan_diagnostics(diagnostic_plan),
                **dense_tail_row(candidate, data),
            }
        )
    elif n == 1024:
        rank = int(candidate._rankdef_effective_cols(n))
        clustered_cols = int(candidate._clustered_effective_cols(n))
        rankdef_all = bool(candidate._batch_tail_columns_are_exact_zero(data, rank).all().item())
        clustered_all = bool(candidate._batch_tail_columns_are_tiny_relative(data, clustered_cols).all().item())
        near_all = bool(candidate._tail_matches_head_columns(data, rank))
        mixed_subset = bool(candidate._has_structured_mixed_subset(data, cond=2))
        sampled = candidate.classify_1024_sampled(data)
        diagnostic_plan = plan if plan is not None else candidate._mixed_structured_plan(data, cond=2)
        row.update(
            {
                "nearrank_cols": rank,
                "rankdef_cols": rank,
                "clustered_cols": clustered_cols,
                "sampled_class": sampled,
                "structured_before_cuda": structured_before_cuda(candidate, n),
                "cuda_route_bypasses_classifier": not uses_case_classifier,
                "rankdef_all": rankdef_all,
                "clustered_all": clustered_all,
                "nearrank_all": near_all,
                "structured_mixed_subset": mixed_subset,
                "structured_group_counts": plan_group_counts(diagnostic_plan),
                **plan_diagnostics(diagnostic_plan),
                **dense_tail_row(candidate, data),
            }
        )
    elif n == 2048:
        rank = int(candidate._rankdef_effective_cols(n))
        rankdef_all = bool(candidate._batch_tail_columns_are_exact_zero(data, rank).all().item())
        mixed_subset = bool(candidate._has_structured_mixed_subset(data, cond=2))
        diagnostic_plan = plan if plan is not None else candidate._mixed_structured_plan(data, cond=2)
        row.update(
            {
                "rankdef_cols": rank,
                "rankdef_all": rankdef_all,
                "structured_mixed_subset": mixed_subset,
                "structured_group_counts": plan_group_counts(diagnostic_plan),
                **plan_diagnostics(diagnostic_plan),
                **dense_tail_row(candidate, data),
            }
        )
    elif n == 4096:
        row.update(dense_tail_row(candidate, data))

    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Trace candidate route decisions on generated QR benchmark inputs.",
        allow_abbrev=False,
    )
    parser.add_argument("--submission", default="submissions/candidate.py")
    parser.add_argument("--cases", default="cases/public_benchmarks.txt")
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

    rows = []
    for index, spec in enumerate(cases):
        row = trace_route(candidate, spec)
        if env:
            row = {**env, **row}
        row["case_index"] = index
        rows.append(row)
        if args.json:
            print(json.dumps(row, sort_keys=True), flush=True)
        else:
            print(f"{index}: {row['spec']} -> {row['route']}", flush=True)

    if args.out:
        append_jsonl(ROOT / args.out if not Path(args.out).is_absolute() else args.out, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
