from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from qr_common import ROOT, format_case, parse_case
from summarize_suite import (
    ABLATION_FILES,
    GUARD_OVERHEAD_FILE,
    PAIRS,
    fmt_count,
    fmt_env_dict,
    fmt_key_list,
    fmt_ratio,
    fmt_speedup,
    fmt_us,
    load_jsonl,
    summarize_ablation,
    summarize_blocked_qr_sweep,
    summarize_candidate_config_tune,
    summarize_guard_overhead,
    summarize_pair,
    summarize_tail_policy_tune,
)


COMPARISON_PRIORITY = ["official_style", "public", "smoke"]
LARGE_CUDA_PROBE_ABLATION = "no_qr512_qr1024_cuda"
LARGE_CUDA_PROBE_TARGET_SHAPES = {(640, 512), (60, 1024)}


def choose_comparison(suite_dir: Path) -> dict[str, Any] | None:
    for name in COMPARISON_PRIORITY:
        baseline_file, candidate_file = PAIRS[name]
        comparison = summarize_pair(name, suite_dir, baseline_file, candidate_file)
        if comparison is not None and comparison["num_common_cases"] > 0:
            return comparison
    return None


def _rows_by_spec(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for row in rows:
        spec = row.get("spec")
        if isinstance(spec, str):
            out[spec] = row
    return out


def _case_text(row: dict[str, Any]) -> str:
    case_text = row.get("case_text")
    if isinstance(case_text, str):
        return case_text
    spec = row.get("spec")
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        return format_case(spec)
    return ""


def _safe_log(value: float | None) -> float | None:
    if value is None or value <= 0.0:
        return None
    return math.log(value)


def rank_cases(comparison: dict[str, Any], suite_dir: Path) -> list[dict[str, Any]]:
    route_by_spec = _rows_by_spec(load_jsonl(suite_dir / "candidate_route_trace_public.jsonl"))
    policy_by_spec = _rows_by_spec(load_jsonl(suite_dir / "candidate_policy_public.jsonl"))
    ranked = []
    for row in comparison["cases"]:
        speedup = row["speedup"]
        log_speedup = _safe_log(speedup)
        spec = row["spec"]
        route_row = route_by_spec.get(spec, {})
        policy_row = policy_by_spec.get(spec, {})
        ranked.append(
            {
                **row,
                "log_speedup": log_speedup,
                "route": route_row.get("route"),
                "dispatch": route_row.get("dispatch") or policy_row.get("dispatch"),
                "primary": policy_row.get("primary"),
                "batch": policy_row.get("batch"),
                "n": policy_row.get("n"),
                "case": policy_row.get("case"),
                "case_info_source": policy_row.get("case_info_source"),
                "classifier_needed_for_current_candidate": policy_row.get(
                    "classifier_needed_for_current_candidate"
                ),
                "classifier_needed_for_case_specific_path": policy_row.get(
                    "classifier_needed_for_case_specific_path"
                ),
                "classifier_on_current_hot_path": policy_row.get("classifier_on_current_hot_path"),
                "classifier_reason": policy_row.get("classifier_reason"),
                "classifier_decision_rule": policy_row.get("classifier_decision_rule"),
                "shape_collision": policy_row.get("shape_collision"),
                "shape_collision_cases": policy_row.get("shape_collision_cases"),
                "requires_tensor_guard_for_case_specific_path": policy_row.get(
                    "requires_tensor_guard_for_case_specific_path"
                ),
            }
        )
    return ranked


def _case_shape(row: dict[str, Any]) -> tuple[int | None, int | None]:
    batch = row.get("batch")
    n = row.get("n")
    if batch is not None and n is not None:
        return int(batch), int(n)
    try:
        spec = parse_case(str(row.get("spec") or ""))
        return int(spec["batch"]), int(spec["n"])
    except Exception:
        return None, None


def _policy_shape(row: dict[str, Any]) -> tuple[int, int] | None:
    batch = row.get("batch")
    n = row.get("n")
    if batch is not None and n is not None:
        return int(batch), int(n)
    try:
        spec = parse_case(str(row.get("spec") or ""))
        return int(spec["batch"]), int(spec["n"])
    except Exception:
        return None


def _family_action(family: dict[str, Any]) -> str:
    primaries = set(family["primaries"])
    routes = set(family["routes"])
    dispatches = set(family["dispatches"])
    if "torch.geqrf" in primaries or "torch.geqrf" in routes:
        return "replace torch.geqrf fallback"
    if family["shape_collision"]:
        return "keep tensor guard cheap while tuning case paths"
    if any(primary and "dense_tail_projection" in primary for primary in primaries):
        return "validate/tune tail policy before hardcoding"
    if any(dispatch and dispatch.startswith("qr") for dispatch in dispatches):
        return "tune specialized kernel"
    return "inspect route"


def analyze_shape_families(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int | None, int | None], list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        grouped[_case_shape(row)].append(row)

    families = []
    for (batch, n), group in grouped.items():
        candidate_values = [float(row["candidate_mean_us"]) for row in group if row.get("candidate_mean_us")]
        baseline_values = [float(row["baseline_mean_us"]) for row in group if row.get("baseline_mean_us")]
        candidate_geo = math.exp(sum(math.log(value) for value in candidate_values) / len(candidate_values)) if candidate_values else None
        baseline_geo = math.exp(sum(math.log(value) for value in baseline_values) / len(baseline_values)) if baseline_values else None
        speedup = None
        if baseline_geo is not None and candidate_geo is not None and candidate_geo > 0.0:
            speedup = baseline_geo / candidate_geo

        routes = sorted({str(row.get("route") or "") for row in group if row.get("route")})
        dispatches = sorted({str(row.get("dispatch") or "") for row in group if row.get("dispatch")})
        primaries = sorted({str(row.get("primary") or "") for row in group if row.get("primary")})
        collision_cases: list[str] = []
        for row in group:
            for label in row.get("shape_collision_cases") or []:
                if label not in collision_cases:
                    collision_cases.append(str(label))
        case_labels: list[str] = []
        for row in group:
            label = str(row.get("case") or "dense")
            if label not in case_labels:
                case_labels.append(label)

        family = {
            "batch": batch,
            "n": n,
            "num_cases": len(group),
            "cases": [row["spec"] for row in group],
            "case_labels": case_labels,
            "candidate_total_us": sum(candidate_values) if candidate_values else None,
            "candidate_geomean_us": candidate_geo,
            "baseline_geomean_us": baseline_geo,
            "speedup": speedup,
            "routes": routes,
            "dispatches": dispatches,
            "primaries": primaries,
            "shape_collision": any(bool(row.get("shape_collision")) for row in group),
            "shape_collision_cases": collision_cases,
            "requires_tensor_guard": any(bool(row.get("requires_tensor_guard_for_case_specific_path")) for row in group),
            "classifier_needed": any(bool(row.get("classifier_needed_for_current_candidate")) for row in group),
            "classifier_needed_for_case_specific_path": any(
                bool(row.get("classifier_needed_for_case_specific_path")) for row in group
            ),
            "classifier_on_current_hot_path": any(bool(row.get("classifier_on_current_hot_path")) for row in group),
        }
        family["action"] = _family_action(family)
        families.append(family)

    return sorted(
        families,
        key=lambda row: (
            row["candidate_total_us"] if row["candidate_total_us"] is not None else -1.0,
            row["num_cases"],
        ),
        reverse=True,
    )


def analyze_ablations(suite_dir: Path) -> list[dict[str, Any]]:
    decisions = []
    for name, file_name in ABLATION_FILES.items():
        ablation = summarize_ablation(name, suite_dir, file_name)
        if ablation is None:
            continue
        ratio = ablation["ablation_over_default"]
        if ratio is None:
            decision = "insufficient-data"
        elif ratio > 1.01:
            decision = "keep-default"
        elif ratio < 0.99:
            decision = "investigate-ablation"
        else:
            decision = "neutral-within-noise"

        per_case = sorted(
            ablation["cases"],
            key=lambda row: row["ablation_over_default"] if row["ablation_over_default"] is not None else 1.0,
        )
        decisions.append(
            {
                "name": name,
                "decision": decision,
                "ablation_over_default": ratio,
                "default_geomean_us": ablation["default_geomean_us"],
                "ablation_geomean_us": ablation["ablation_geomean_us"],
                "fastest_ablation_cases": per_case[:3],
                "slowest_ablation_cases": list(reversed(per_case[-3:])),
            }
        )
    return decisions


def _large_cuda_probe_decision(ratio: float | None) -> str:
    if ratio is None:
        return "insufficient-data"
    if ratio > 1.01:
        return "keep-qr512-qr1024-cuda"
    if ratio < 0.99:
        return "disable-qr512-qr1024-cuda"
    return "neutral-measure-again"


def analyze_large_cuda_probe_ablation(suite_dir: Path) -> dict[str, Any] | None:
    file_name = ABLATION_FILES.get(LARGE_CUDA_PROBE_ABLATION)
    if file_name is None:
        return None
    ablation = summarize_ablation(LARGE_CUDA_PROBE_ABLATION, suite_dir, file_name)
    if ablation is None:
        return None

    target_rows = []
    family_rows: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in ablation["cases"]:
        shape = _case_shape(row)
        if shape not in LARGE_CUDA_PROBE_TARGET_SHAPES:
            continue
        target_rows.append(row)
        family_rows[shape].append(row)

    target_default = _geomean_from_rows(target_rows, "default_mean_us")
    target_ablation = _geomean_from_rows(target_rows, "ablation_mean_us")
    target_ratio = None
    if target_default is not None and target_ablation is not None and target_default > 0.0:
        target_ratio = target_ablation / target_default

    families = []
    for (batch, n), rows in sorted(family_rows.items(), key=lambda item: (item[0][1], item[0][0])):
        default_geo = _geomean_from_rows(rows, "default_mean_us")
        ablation_geo = _geomean_from_rows(rows, "ablation_mean_us")
        ratio = None
        if default_geo is not None and ablation_geo is not None and default_geo > 0.0:
            ratio = ablation_geo / default_geo
        families.append(
            {
                "batch": batch,
                "n": n,
                "num_cases": len(rows),
                "default_geomean_us": default_geo,
                "ablation_geomean_us": ablation_geo,
                "ablation_over_default": ratio,
                "decision": _large_cuda_probe_decision(ratio),
                "cases": rows,
            }
        )

    return {
        "name": LARGE_CUDA_PROBE_ABLATION,
        "ablation_file": file_name,
        "target_shapes": [f"{batch}x{n}" for batch, n in sorted(LARGE_CUDA_PROBE_TARGET_SHAPES, key=lambda item: (item[1], item[0]))],
        "num_target_cases": len(target_rows),
        "default_geomean_us": target_default,
        "ablation_geomean_us": target_ablation,
        "ablation_over_default": target_ratio,
        "all_case_ablation_over_default": ablation.get("ablation_over_default"),
        "decision": _large_cuda_probe_decision(target_ratio),
        "families": families,
    }


def analyze_guard_overhead(suite_dir: Path, cases: list[dict[str, Any]]) -> dict[str, Any] | None:
    guard = summarize_guard_overhead(suite_dir)
    if guard is None:
        return None
    candidate_by_spec = {row["spec"]: row for row in cases}
    rows = []
    for guard_row in guard["cases"]:
        spec = guard_row["spec"]
        candidate = candidate_by_spec.get(spec)
        candidate_mean = None if candidate is None else candidate.get("candidate_mean_us")
        hot_pct = None
        cold_pct = None
        if candidate_mean and candidate_mean > 0.0:
            hot_pct = 100.0 * guard_row["wall_us"] / candidate_mean
            if guard_row["cold_wall_us"] is not None:
                cold_pct = 100.0 * guard_row["cold_wall_us"] / candidate_mean
        rows.append(
            {
                **guard_row,
                "candidate_mean_us": candidate_mean,
                "hot_wall_pct_of_candidate": hot_pct,
                "cold_wall_pct_of_candidate": cold_pct,
            }
        )
    hot_values = [row["hot_wall_pct_of_candidate"] for row in rows if row["hot_wall_pct_of_candidate"] is not None]
    cold_values = [row["cold_wall_pct_of_candidate"] for row in rows if row["cold_wall_pct_of_candidate"] is not None]
    case_selection_rows = [row for row in rows if row.get("uses_tensor_values_for_case_selection")]
    case_selection_hot_values = [
        row["hot_wall_pct_of_candidate"]
        for row in case_selection_rows
        if row["hot_wall_pct_of_candidate"] is not None
    ]
    non_case_selection_rows = [
        row
        for row in rows
        if row.get("uses_tensor_values_for_dispatch") and not row.get("uses_tensor_values_for_case_selection")
    ]
    return {
        **guard,
        "cases": rows,
        "hot_wall_pct_max": max(hot_values) if hot_values else None,
        "cold_wall_pct_max": max(cold_values) if cold_values else None,
        "case_selection_num_cases": len(case_selection_rows),
        "case_selection_hot_wall_pct_max": max(case_selection_hot_values) if case_selection_hot_values else None,
        "case_selection_hot_wall_us_max": max((row["wall_us"] for row in case_selection_rows), default=None),
        "non_case_selection_tensor_guard_num_cases": len(non_case_selection_rows),
        "largest_hot_pct_cases": sorted(
            [row for row in rows if row["hot_wall_pct_of_candidate"] is not None],
            key=lambda row: row["hot_wall_pct_of_candidate"],
            reverse=True,
        )[:5],
        "largest_case_selection_hot_pct_cases": sorted(
            [row for row in case_selection_rows if row["hot_wall_pct_of_candidate"] is not None],
            key=lambda row: row["hot_wall_pct_of_candidate"],
            reverse=True,
        )[:5],
    }


def _geomean_from_rows(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None and float(row[key]) > 0.0]
    if not values:
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _route_ablation_decision(
    ratio: float | None,
    hot_guard_pct_max: float | None,
    *,
    keep_label: str,
    disable_label: str,
) -> str:
    if ratio is None:
        return "insufficient-data"

    # ratio = ablated-route runtime / default runtime.
    if ratio > 1.01:
        gain_pct = (ratio - 1.0) * 100.0
        if hot_guard_pct_max is not None and hot_guard_pct_max > max(1.0, gain_pct * 0.5):
            return "keep-but-optimize-guards"
        return keep_label
    if ratio < 0.99:
        return disable_label
    return "neutral-measure-again"


def _classifier_dispatch_decision(ratio: float | None, hot_guard_pct_max: float | None) -> str:
    return _route_ablation_decision(
        ratio,
        hot_guard_pct_max,
        keep_label="keep-structured-routes",
        disable_label="disable-structured-routes",
    )


def _data_dependent_dispatch_decision(ratio: float | None, hot_guard_pct_max: float | None) -> str:
    return _route_ablation_decision(
        ratio,
        hot_guard_pct_max,
        keep_label="keep-data-dependent-routes",
        disable_label="disable-data-dependent-routes",
    )


def _structured_route_order_decision(ratio: float | None, hot_guard_pct_max: float | None) -> str:
    return _route_ablation_decision(
        ratio,
        hot_guard_pct_max,
        keep_label="keep-structured-before-cuda",
        disable_label="prefer-cuda-first-structured-routes",
    )


def analyze_data_dependent_dispatch(
    suite_dir: Path,
    cases: list[dict[str, Any]],
    guard_overhead: dict[str, Any] | None,
) -> dict[str, Any] | None:
    policy_rows = load_jsonl(suite_dir / "candidate_policy_public.jsonl")
    route_by_spec = _rows_by_spec(load_jsonl(suite_dir / "candidate_route_trace_public.jsonl"))
    colliding = [
        row
        for row in policy_rows
        if row.get("shape_collision") and row.get("requires_tensor_guard_for_case_specific_path")
    ]
    if not colliding:
        return {
            "case_metadata_passed_to_submission": False,
            "num_shape_families": 0,
            "families": [],
            "message": "no shape-colliding public benchmark families require tensor guards",
        }

    structured_ablation = summarize_ablation(
        "no_structured_routes",
        suite_dir,
        ABLATION_FILES["no_structured_routes"],
    )
    structured_ablation_by_spec = {}
    if structured_ablation is not None:
        structured_ablation_by_spec = {row["spec"]: row for row in structured_ablation["cases"]}

    cuda_first_ablation = summarize_ablation(
        "cuda_first_structured_routes",
        suite_dir,
        ABLATION_FILES["cuda_first_structured_routes"],
    )
    cuda_first_ablation_by_spec = {}
    if cuda_first_ablation is not None:
        cuda_first_ablation_by_spec = {row["spec"]: row for row in cuda_first_ablation["cases"]}

    data_ablation = summarize_ablation(
        "no_data_dependent_routes",
        suite_dir,
        ABLATION_FILES["no_data_dependent_routes"],
    )
    data_ablation_by_spec = {}
    if data_ablation is not None:
        data_ablation_by_spec = {row["spec"]: row for row in data_ablation["cases"]}

    candidate_by_spec = {row["spec"]: row for row in cases}
    guard_by_spec = {}
    if guard_overhead is not None:
        guard_by_spec = {row["spec"]: row for row in guard_overhead.get("cases", [])}

    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in colliding:
        shape = _policy_shape(row)
        if shape is not None:
            groups[shape].append(row)

    families = []
    for (batch, n), group in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        case_rows = []
        for policy in group:
            spec = str(policy["spec"])
            structured_ablation_row = structured_ablation_by_spec.get(spec)
            cuda_first_ablation_row = cuda_first_ablation_by_spec.get(spec)
            data_ablation_row = data_ablation_by_spec.get(spec)
            candidate_row = candidate_by_spec.get(spec, {})
            guard_row = guard_by_spec.get(spec, {})
            route_row = route_by_spec.get(spec, {})
            default_mean = None
            no_structured_mean = None
            no_structured_ratio = None
            cuda_first_mean = None
            cuda_first_ratio = None
            no_data_mean = None
            no_data_ratio = None
            if structured_ablation_row is not None:
                default_mean = structured_ablation_row.get("default_mean_us")
                no_structured_mean = structured_ablation_row.get("ablation_mean_us")
                no_structured_ratio = structured_ablation_row.get("ablation_over_default")
            if cuda_first_ablation_row is not None:
                default_mean = default_mean or cuda_first_ablation_row.get("default_mean_us")
                cuda_first_mean = cuda_first_ablation_row.get("ablation_mean_us")
                cuda_first_ratio = cuda_first_ablation_row.get("ablation_over_default")
            if data_ablation_row is not None:
                default_mean = default_mean or data_ablation_row.get("default_mean_us")
                no_data_mean = data_ablation_row.get("ablation_mean_us")
                no_data_ratio = data_ablation_row.get("ablation_over_default")
            if default_mean is None and candidate_row:
                default_mean = candidate_row.get("candidate_mean_us")

            case_rows.append(
                {
                    "spec": spec,
                    "case": policy.get("case"),
                    "primary": policy.get("primary"),
                    "route": route_row.get("route"),
                    "sampled_class": route_row.get("sampled_class"),
                    "structured_group_counts": route_row.get("structured_group_counts"),
                    "structured_candidate_counts": route_row.get("structured_candidate_counts"),
                    "structured_exact_check_counts": route_row.get("structured_exact_check_counts"),
                    "structured_sampled_plan": route_row.get("structured_sampled_plan"),
                    "structured_sampled_matrix_count": route_row.get("structured_sampled_matrix_count"),
                    "structured_sampled_row_count": route_row.get("structured_sampled_row_count"),
                    "case_selection_info_sources": route_row.get("case_selection_info_sources"),
                    "shape_only_case_selection": route_row.get("shape_only_case_selection"),
                    "dispatch_info_sources": route_row.get("dispatch_info_sources"),
                    "route_decision_source": route_row.get("route_decision_source"),
                    "uses_tensor_values_for_dispatch": route_row.get("uses_tensor_values_for_dispatch"),
                    "uses_tensor_values_for_case_selection": route_row.get("uses_tensor_values_for_case_selection"),
                    "requires_tensor_guard_for_case_specific_path": policy.get(
                        "requires_tensor_guard_for_case_specific_path"
                    ),
                    "classifier_needed_for_case_specific_path": policy.get(
                        "classifier_needed_for_case_specific_path"
                    ),
                    "classifier_needed_for_current_candidate": policy.get(
                        "classifier_needed_for_current_candidate"
                    ),
                    "classifier_on_current_hot_path": policy.get("classifier_on_current_hot_path"),
                    "default_mean_us": default_mean,
                    "no_structured_mean_us": no_structured_mean,
                    "no_structured_over_default": no_structured_ratio,
                    "cuda_first_structured_mean_us": cuda_first_mean,
                    "cuda_first_structured_over_default": cuda_first_ratio,
                    "no_data_dependent_mean_us": no_data_mean,
                    "no_data_dependent_over_default": no_data_ratio,
                    "guard_hot_pct_of_candidate": guard_row.get("hot_wall_pct_of_candidate"),
                    "guard_cold_pct_of_candidate": guard_row.get("cold_wall_pct_of_candidate"),
                }
            )

        default_geo = _geomean_from_rows(case_rows, "default_mean_us")
        no_structured_geo = _geomean_from_rows(case_rows, "no_structured_mean_us")
        cuda_first_geo = _geomean_from_rows(case_rows, "cuda_first_structured_mean_us")
        no_data_geo = _geomean_from_rows(case_rows, "no_data_dependent_mean_us")
        no_structured_ratio = None
        if default_geo is not None and no_structured_geo is not None and default_geo > 0.0:
            no_structured_ratio = no_structured_geo / default_geo
        cuda_first_ratio = None
        if default_geo is not None and cuda_first_geo is not None and default_geo > 0.0:
            cuda_first_ratio = cuda_first_geo / default_geo
        no_data_ratio = None
        if default_geo is not None and no_data_geo is not None and default_geo > 0.0:
            no_data_ratio = no_data_geo / default_geo
        hot_values = [
            float(row["guard_hot_pct_of_candidate"])
            for row in case_rows
            if row.get("guard_hot_pct_of_candidate") is not None
        ]
        cold_values = [
            float(row["guard_cold_pct_of_candidate"])
            for row in case_rows
            if row.get("guard_cold_pct_of_candidate") is not None
        ]
        candidate_count_values = [
            int(value)
            for row in case_rows
            if isinstance(row.get("structured_candidate_counts"), dict)
            for value in row["structured_candidate_counts"].values()
        ]
        exact_check_count_values = [
            int(value)
            for row in case_rows
            if isinstance(row.get("structured_exact_check_counts"), dict)
            for value in row["structured_exact_check_counts"].values()
        ]
        sampled_row_counts = sorted(
            {
                int(row["structured_sampled_row_count"])
                for row in case_rows
                if row.get("structured_sampled_row_count") is not None
            }
        )
        hot_max = max(hot_values) if hot_values else None
        cold_max = max(cold_values) if cold_values else None

        families.append(
            {
                "batch": batch,
                "n": n,
                "num_cases": len(case_rows),
                "shape_collision_cases": group[0].get("shape_collision_cases", []),
                "case_metadata_passed_to_submission": False,
                "case_info_source": "tensor_values",
                "route_decision_sources": sorted(
                    {
                        str(source)
                        for row in case_rows
                        for source in (row.get("dispatch_info_sources") or [])
                    }
                ),
                "uses_tensor_values_for_dispatch": any(
                    bool(row.get("uses_tensor_values_for_dispatch")) for row in case_rows
                ),
                "uses_tensor_values_for_case_selection": any(
                    bool(row.get("uses_tensor_values_for_case_selection")) for row in case_rows
                ),
                "routes": sorted({str(row["route"]) for row in case_rows if row.get("route")}),
                "requires_tensor_guard_for_case_specific_path": True,
                "classifier_needed_for_case_specific_path": any(
                    bool(row.get("classifier_needed_for_case_specific_path")) for row in case_rows
                ),
                "classifier_needed_for_current_candidate": any(
                    bool(row.get("classifier_needed_for_current_candidate")) for row in case_rows
                ),
                "classifier_on_current_hot_path": any(
                    bool(row.get("classifier_on_current_hot_path")) for row in case_rows
                ),
                "default_geomean_us": default_geo,
                "cuda_first_structured_geomean_us": cuda_first_geo,
                "cuda_first_structured_over_default": cuda_first_ratio,
                "no_structured_geomean_us": no_structured_geo,
                "no_structured_over_default": no_structured_ratio,
                "no_data_dependent_geomean_us": no_data_geo,
                "no_data_dependent_over_default": no_data_ratio,
                "hot_guard_pct_max": hot_max,
                "cold_guard_pct_max": cold_max,
                "structured_candidate_count_max": max(candidate_count_values) if candidate_count_values else None,
                "structured_exact_check_count_max": max(exact_check_count_values) if exact_check_count_values else None,
                "structured_sampled_row_counts": sampled_row_counts,
                "decision": _classifier_dispatch_decision(no_structured_ratio, hot_max),
                "classifier_decision": _classifier_dispatch_decision(no_structured_ratio, hot_max),
                "route_order_decision": _structured_route_order_decision(cuda_first_ratio, hot_max),
                "data_dependent_decision": _data_dependent_dispatch_decision(no_data_ratio, hot_max),
                "cases": case_rows,
            }
        )

    return {
        "case_metadata_passed_to_submission": False,
        "classifier_ablation": "no_structured_routes",
        "route_order_ablation": "cuda_first_structured_routes",
        "all_tensor_guard_ablation": "no_data_dependent_routes",
        "num_shape_families": len(families),
        "families": families,
    }


def analyze_experiments(suite_dir: Path) -> dict[str, Any] | None:
    rows = load_jsonl(suite_dir / "experiments_public_benchmarks.jsonl")
    if not rows:
        return None

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        experiment = row.get("experiment")
        if isinstance(experiment, str):
            grouped[experiment].append(row)

    experiments = {}
    for name, group in grouped.items():
        total = len(group)
        passed = sum(1 for row in group if row.get("ok"))
        experiments[name] = {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total else None,
        }

    tail_best_by_spec = {}
    for row in grouped.get("tail-delete", []):
        if not row.get("ok"):
            continue
        spec = json.dumps(row.get("spec", {}), sort_keys=True)
        current = tail_best_by_spec.get(spec)
        tail_cut = int(row.get("tail_cut", 0))
        if current is None or tail_cut > current["tail_cut"]:
            tail_best_by_spec[spec] = {"spec": row.get("spec"), "tail_cut": tail_cut, "reflectors_kept": row.get("reflectors_kept")}

    return {
        "experiments": experiments,
        "tail_delete_best_by_spec": list(tail_best_by_spec.values()),
    }


def analyze_classifier_seed_sweep(suite_dir: Path) -> dict[str, Any] | None:
    rows = load_jsonl(suite_dir / "classifier_seed_sweep.jsonl")
    if not rows:
        return None

    cases = [row for row in rows if not row.get("summary")]
    summary = next((row for row in rows if row.get("summary")), {})
    shapes = sorted(
        {
            f"{int(row['batch'])}x{int(row['n'])}"
            for row in cases
            if row.get("batch") is not None and row.get("n") is not None
        }
    )
    mismatches = [
        row
        for row in cases
        if row.get("classifier_ok") is False or row.get("route_ok") is False or not row.get("ok")
    ]
    return {
        "ok": bool(summary.get("ok", not mismatches)),
        "num_rows": int(summary.get("num_rows", len(cases)) or 0),
        "num_failed": int(summary.get("num_failed", len(mismatches)) or 0),
        "num_classifier_mismatch": int(summary.get("num_classifier_mismatch", 0) or 0),
        "num_route_mismatch": int(summary.get("num_route_mismatch", 0) or 0),
        "num_route_cuda_bypass": int(summary.get("num_route_cuda_bypass", 0) or 0),
        "num_public_seed_rows": int(summary.get("num_public_seed_rows", 0) or 0),
        "num_popcorn_seed_rows": int(summary.get("num_popcorn_seed_rows", 0) or 0),
        "popcorn_seeds": summary.get("popcorn_seeds", []),
        "shapes": shapes,
        "mismatches": mismatches[:10],
    }


def analyze_correctness_margins(suite_dir: Path, file_name: str) -> dict[str, Any] | None:
    rows = load_jsonl(suite_dir / file_name)
    cases = [row for row in rows if not row.get("summary")]
    if not cases:
        return None

    diagnostic_rows = [row for row in cases if isinstance(row.get("diagnostics"), dict)]
    factor_values = [float(row["diagnostics"]["factor_scaled_max"]) for row in diagnostic_rows]
    orth_values = [float(row["diagnostics"]["orth_scaled_max"]) for row in diagnostic_rows]
    worst_factor = sorted(
        diagnostic_rows,
        key=lambda row: float(row["diagnostics"]["factor_scaled_max"]),
        reverse=True,
    )[:5]
    worst_orth = sorted(
        diagnostic_rows,
        key=lambda row: float(row["diagnostics"]["orth_scaled_max"]),
        reverse=True,
    )[:5]

    return {
        "file": file_name,
        "num_cases": len(cases),
        "num_failed": sum(1 for row in cases if not row.get("ok")),
        "num_margin_failed": sum(1 for row in cases if row.get("margin_ok") is False),
        "num_with_diagnostics": len(diagnostic_rows),
        "max_factor_scaled": max(factor_values) if factor_values else None,
        "max_orth_scaled": max(orth_values) if orth_values else None,
        "worst_factor_cases": worst_factor,
        "worst_orth_cases": worst_orth,
    }


def _layout_policy_result(policy_layout: str | None, actual_layout: str | None) -> str:
    if actual_layout not in {"column_major", "torch_contiguous", "other_strided"}:
        return "missing-actual-layout"
    if policy_layout == "column_major":
        return "matches" if actual_layout == "column_major" else "mismatch"
    if policy_layout == "torch.geqrf_default":
        return "matches" if actual_layout == "torch_contiguous" else "mismatch"
    if policy_layout in {
        "column_major_when_fast_path_applies_else_torch.geqrf_default",
        "column_major_when_cuda_extension_available_else_torch.geqrf_default",
    }:
        return "conditional-accepted" if actual_layout in {"column_major", "torch_contiguous"} else "mismatch"
    return "unknown-policy-layout"


def analyze_output_layouts(suite_dir: Path) -> dict[str, Any] | None:
    correctness_rows = [row for row in load_jsonl(suite_dir / "candidate_public_benchmark_correctness.jsonl") if not row.get("summary")]
    if not correctness_rows:
        return None

    policy_by_spec = _rows_by_spec(load_jsonl(suite_dir / "candidate_policy_public.jsonl"))
    route_by_spec = _rows_by_spec(load_jsonl(suite_dir / "candidate_route_trace_public.jsonl"))

    layout_counts: dict[str, int] = defaultdict(int)
    cases = []
    grouped: dict[tuple[int | None, int | None], list[dict[str, Any]]] = defaultdict(list)
    for row in correctness_rows:
        spec_text = _case_text(row)
        policy = policy_by_spec.get(spec_text, {})
        route = route_by_spec.get(spec_text, {})
        spec = row.get("spec")
        parsed = spec if isinstance(spec, dict) else parse_case(spec_text)
        batch = int(parsed["batch"]) if "batch" in parsed else None
        n = int(parsed["n"]) if "n" in parsed else None
        actual_layout = row.get("h_layout_actual")
        layout_counts[str(actual_layout)] += 1
        case_row = {
            "spec": spec_text,
            "batch": batch,
            "n": n,
            "case": str(parsed.get("case", "dense")),
            "dispatch": route.get("dispatch") or policy.get("dispatch"),
            "route": route.get("route"),
            "primary": policy.get("primary"),
            "policy_h_layout": policy.get("h_layout"),
            "policy_column_major_h": policy.get("column_major_h"),
            "h_layout_actual": actual_layout,
            "column_major_h_actual": row.get("column_major_h_actual"),
            "h_stride": row.get("h_stride"),
            "h_is_contiguous": row.get("h_is_contiguous"),
            "policy_layout_result": _layout_policy_result(policy.get("h_layout"), actual_layout),
        }
        cases.append(case_row)
        grouped[(batch, n)].append(case_row)

    shape_families = []
    for (batch, n), group in sorted(grouped.items(), key=lambda item: (item[0][1] or -1, item[0][0] or -1)):
        family_counts: dict[str, int] = defaultdict(int)
        for row in group:
            family_counts[str(row.get("h_layout_actual"))] += 1
        shape_families.append(
            {
                "batch": batch,
                "n": n,
                "num_cases": len(group),
                "layout_counts": dict(sorted(family_counts.items())),
                "num_column_major": sum(1 for row in group if row.get("column_major_h_actual") is True),
                "num_torch_contiguous": sum(1 for row in group if row.get("h_layout_actual") == "torch_contiguous"),
                "num_policy_mismatch": sum(1 for row in group if row.get("policy_layout_result") == "mismatch"),
                "cases": [row["spec"] for row in group],
            }
        )

    return {
        "file": "candidate_public_benchmark_correctness.jsonl",
        "num_cases": len(cases),
        "layout_counts": dict(sorted(layout_counts.items())),
        "num_column_major": sum(1 for row in cases if row.get("column_major_h_actual") is True),
        "num_torch_contiguous": sum(1 for row in cases if row.get("h_layout_actual") == "torch_contiguous"),
        "num_policy_mismatch": sum(1 for row in cases if row.get("policy_layout_result") == "mismatch"),
        "shape_families": shape_families,
        "cases": cases,
    }


def _shape_label(row: dict[str, Any]) -> str:
    batch = row.get("batch")
    n = row.get("n")
    if batch is None or n is None:
        return "unknown"
    return f"{batch}x{n}"


def _correctness_blockers(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    blockers = []
    for label, key in [
        ("public benchmark correctness", "benchmark_correctness"),
        ("dev robustness", "dev_robustness"),
        ("quantization seed sweep", "quantization_seed_sweep"),
        ("mixed seed sweep", "mixed_seed_sweep"),
        ("tail policy sweep", "tail_policy_sweep"),
    ]:
        row = analysis.get(key)
        if not row:
            continue
        failed = int(row.get("num_failed") or 0)
        margin_failed = int(row.get("num_margin_failed") or 0)
        if failed or margin_failed:
            blockers.append(
                {
                    "area": label,
                    "failed_cases": failed,
                    "margin_failed_cases": margin_failed,
                    "max_factor_scaled": row.get("max_factor_scaled"),
                    "max_orth_scaled": row.get("max_orth_scaled"),
                }
            )
    return blockers


def build_final_algorithm_recommendation(analysis: dict[str, Any]) -> dict[str, Any]:
    blockers = _correctness_blockers(analysis)
    actions: list[dict[str, Any]] = []

    if blockers:
        first = blockers[0]
        actions.append(
            {
                "area": "correctness",
                "action": f"fix {first['area']} failures before treating timings as algorithm decisions",
                "evidence": {
                    "failed_cases": first["failed_cases"],
                    "margin_failed_cases": first["margin_failed_cases"],
                    "max_factor_scaled": first["max_factor_scaled"],
                    "max_orth_scaled": first["max_orth_scaled"],
                },
            }
        )

    for family in analysis.get("shape_family_priorities", [])[:4]:
        actions.append(
            {
                "area": "shape-family",
                "shape": _shape_label(family),
                "action": family.get("action") or "inspect route",
                "evidence": {
                    "num_cases": family.get("num_cases"),
                    "case_labels": family.get("case_labels", []),
                    "candidate_total_us": family.get("candidate_total_us"),
                    "candidate_geomean_us": family.get("candidate_geomean_us"),
                    "speedup": family.get("speedup"),
                    "classifier_needed": family.get("classifier_needed"),
                    "shape_collision_cases": family.get("shape_collision_cases", []),
                },
            }
        )

    dispatch_actions = []
    dispatch = analysis.get("data_dependent_dispatch")
    if dispatch:
        for family in dispatch.get("families", []):
            decision = family.get("decision") or "insufficient-data"
            action = {
                "area": "data-dependent-dispatch",
                "shape": _shape_label(family),
                "action": decision,
                "evidence": {
                    "case_metadata_passed_to_submission": family.get("case_metadata_passed_to_submission"),
                    "case_info_source": family.get("case_info_source"),
                    "cases": family.get("shape_collision_cases", []),
                    "no_structured_over_default": family.get("no_structured_over_default"),
                    "no_data_dependent_over_default": family.get("no_data_dependent_over_default"),
                    "hot_guard_pct_max": family.get("hot_guard_pct_max"),
                    "classifier_needed_for_case_specific_path": family.get(
                        "classifier_needed_for_case_specific_path"
                    ),
                    "classifier_on_current_hot_path": family.get("classifier_on_current_hot_path"),
                    "routes": family.get("routes", []),
                },
            }
            dispatch_actions.append(action)
            actions.append(action)

    classifier_sweep = analysis.get("classifier_seed_sweep")
    if classifier_sweep and not classifier_sweep.get("ok"):
        actions.append(
            {
                "area": "classifier-seed-sweep",
                "action": "fix sampled classifier mismatches before relying on case-specific paths",
                "evidence": {
                    "num_classifier_mismatch": classifier_sweep.get("num_classifier_mismatch"),
                    "num_route_mismatch": classifier_sweep.get("num_route_mismatch"),
                    "shapes": classifier_sweep.get("shapes"),
                },
            }
        )

    layouts = analysis.get("output_layouts")
    if layouts:
        if layouts.get("num_policy_mismatch"):
            layout_action = "fix H layout policy mismatches"
        elif layouts.get("num_column_major", 0) == 0:
            layout_action = "move fast paths toward column-major H where benchmark timing justifies it"
        else:
            layout_action = "keep column-major H on accepted fast paths"
        actions.append(
            {
                "area": "output-layout",
                "action": layout_action,
                "evidence": {
                    "layout_counts": layouts.get("layout_counts", {}),
                    "num_column_major": layouts.get("num_column_major"),
                    "num_torch_contiguous": layouts.get("num_torch_contiguous"),
                    "num_policy_mismatch": layouts.get("num_policy_mismatch"),
                },
            }
        )

    tune = analysis.get("tail_policy_tune")
    if tune and tune.get("best_name"):
        actions.append(
            {
                "area": "tail-policy",
                "action": f"promote or retest tail policy config `{tune['best_name']}`",
                "evidence": {
                    "best_geomean_us": tune.get("best_geomean_us"),
                    "num_failed_configs": tune.get("num_failed_configs"),
                    "num_benchmarked_configs": tune.get("num_benchmarked_configs"),
                },
            }
        )

    config_tune = analysis.get("candidate_config_tune")
    if config_tune and config_tune.get("best_name"):
        actions.append(
            {
                "area": "candidate-config",
                "action": f"promote or retest candidate config `{config_tune['best_name']}`",
                "evidence": {
                    "objective": config_tune.get("objective"),
                    "best_geomean_us": config_tune.get("best_geomean_us"),
                    "num_failed_configs": config_tune.get("num_failed_configs"),
                    "num_benchmarked_configs": config_tune.get("num_benchmarked_configs"),
                },
            }
        )
        route_order = config_tune.get("route_order") if isinstance(config_tune.get("route_order"), dict) else None
        if route_order and route_order.get("decision") not in {None, "insufficient-data"}:
            actions.append(
                {
                    "area": "route-order",
                    "action": f"{route_order.get('decision')} for classifier/structured-before-CUDA dispatch",
                    "evidence": {
                        "structured_over_cuda": route_order.get("structured_over_cuda"),
                        "num_compared_pairs": route_order.get("num_compared_pairs"),
                        "best_cuda_first_name": route_order.get("best_cuda_first_name"),
                        "best_structured_first_name": route_order.get("best_structured_first_name"),
                    },
                }
            )

    blocked_qr = analysis.get("blocked_qr_sweep")
    if blocked_qr:
        passing_low_precision = blocked_qr.get("passing_low_precision_configs") or []
        if passing_low_precision:
            actions.append(
                {
                    "area": "blocked-qr-low-precision",
                    "action": "port prefix panel refresh plus panel-prefix R maintenance into QR512/QR1024 CUDA blocked updates",
                    "evidence": {
                        "passing_low_precision_configs": passing_low_precision,
                        "panel_widths": blocked_qr.get("panel_widths"),
                        "update_modes": blocked_qr.get("update_modes"),
                    },
                }
            )
        elif blocked_qr.get("num_rows"):
            actions.append(
                {
                    "area": "blocked-qr-low-precision",
                    "action": "keep blocked QR trailing updates FP32 until low-precision refresh/R-maintenance sweep passes",
                    "evidence": {
                        "num_rows": blocked_qr.get("num_rows"),
                        "num_failed": blocked_qr.get("num_failed"),
                        "precision_modes": blocked_qr.get("precision_modes"),
                    },
                }
            )

    large_cuda = analysis.get("large_cuda_probe_ablation")
    if large_cuda and large_cuda.get("decision") not in {None, "insufficient-data"}:
        actions.append(
            {
                "area": "large-cuda-probe",
                "action": large_cuda.get("decision"),
                "evidence": {
                    "target_shapes": large_cuda.get("target_shapes"),
                    "target_cases": large_cuda.get("num_target_cases"),
                    "target_ablation_over_default": large_cuda.get("ablation_over_default"),
                    "all_case_ablation_over_default": large_cuda.get("all_case_ablation_over_default"),
                },
            }
        )

    for row in analysis.get("ablation_decisions", []):
        if row.get("decision") == "investigate-ablation":
            actions.append(
                {
                    "area": "ablation",
                    "action": f"investigate `{row.get('name')}` because the ablation beat default timing",
                    "evidence": {
                        "ablation_over_default": row.get("ablation_over_default"),
                        "default_geomean_us": row.get("default_geomean_us"),
                        "ablation_geomean_us": row.get("ablation_geomean_us"),
                    },
                }
            )

    comparison = analysis.get("comparison", {})
    top_family = analysis.get("shape_family_priorities", [{}])[0] if analysis.get("shape_family_priorities") else {}
    if blockers:
        status = "correctness-blocked"
        primary = actions[0]["action"]
    elif top_family:
        status = "ready-for-next-kernel-decision"
        primary = f"prioritize {_shape_label(top_family)}: {top_family.get('action') or 'inspect route'}"
    else:
        status = "insufficient-data"
        primary = "collect a complete 12-case B200 suite before choosing the final algorithm"

    return {
        "status": status,
        "primary_next_step": primary,
        "comparison_name": comparison.get("name"),
        "geomean_speedup": comparison.get("geomean_speedup"),
        "correctness_blockers": blockers,
        "classifier_required_by_api": False,
        "classifier_required_for_case_specific_paths": bool(dispatch_actions),
        "dispatch_decisions": dispatch_actions,
        "priority_actions": actions[:10],
    }


def analyze_suite(suite_dir: Path) -> dict[str, Any]:
    comparison = choose_comparison(suite_dir)
    if comparison is None:
        return {
            "ok": False,
            "suite_dir": str(suite_dir),
            "message": "no benchmark comparison files were found",
        }

    cases = rank_cases(comparison, suite_dir)
    weakest = sorted(cases, key=lambda row: row["speedup"] if row["speedup"] is not None else 0.0)[:5]
    slowest = sorted(cases, key=lambda row: row["candidate_mean_us"], reverse=True)[:5]
    absolute_wins = sorted(
        cases,
        key=lambda row: row["baseline_mean_us"] - row["candidate_mean_us"],
        reverse=True,
    )[:5]

    guard_overhead = analyze_guard_overhead(suite_dir, cases)

    analysis = {
        "ok": True,
        "suite_dir": str(suite_dir),
        "comparison": {
            "name": comparison["name"],
            "baseline_geomean_us": comparison["baseline_geomean_us"],
            "candidate_geomean_us": comparison["candidate_geomean_us"],
            "geomean_speedup": comparison["geomean_speedup"],
            "num_common_cases": comparison["num_common_cases"],
        },
        "weakest_speedup_cases": weakest,
        "slowest_candidate_cases": slowest,
        "largest_absolute_wins": absolute_wins,
        "shape_family_priorities": analyze_shape_families(cases),
        "ablation_decisions": analyze_ablations(suite_dir),
        "large_cuda_probe_ablation": analyze_large_cuda_probe_ablation(suite_dir),
        "guard_overhead": guard_overhead,
        "classifier_seed_sweep": analyze_classifier_seed_sweep(suite_dir),
        "data_dependent_dispatch": analyze_data_dependent_dispatch(suite_dir, cases, guard_overhead),
        "benchmark_correctness": analyze_correctness_margins(
            suite_dir,
            "candidate_public_benchmark_correctness.jsonl",
        ),
        "output_layouts": analyze_output_layouts(suite_dir),
        "quantization_seed_sweep": analyze_correctness_margins(suite_dir, "quantization_seed_sweep.jsonl"),
        "mixed_seed_sweep": analyze_correctness_margins(suite_dir, "mixed_seed_sweep.jsonl"),
        "blocked_qr_sweep": summarize_blocked_qr_sweep(suite_dir),
        "tail_policy_sweep": analyze_correctness_margins(suite_dir, "candidate_tail_policy_sweep.jsonl"),
        "tail_policy_tune": summarize_tail_policy_tune(suite_dir),
        "candidate_config_tune": summarize_candidate_config_tune(suite_dir),
        "dev_robustness": analyze_correctness_margins(suite_dir, "candidate_dev_robustness.jsonl"),
        "experiments": analyze_experiments(suite_dir),
    }
    analysis["final_algorithm_recommendation"] = build_final_algorithm_recommendation(analysis)
    return analysis


def render_markdown(analysis: dict[str, Any]) -> str:
    if not analysis.get("ok"):
        return f"# B200 Result Analysis\n\n{analysis.get('message', 'analysis failed')}\n"

    comparison = analysis["comparison"]
    lines = [
        "# B200 Result Analysis",
        "",
        f"Suite: `{analysis['suite_dir']}`",
        "",
        "## Geomean",
        "",
        f"- comparison: {comparison['name']}",
        f"- cases: {comparison['num_common_cases']}",
        f"- baseline geomean us: {fmt_us(comparison['baseline_geomean_us'])}",
        f"- candidate geomean us: {fmt_us(comparison['candidate_geomean_us'])}",
        f"- geomean speedup: {fmt_speedup(comparison['geomean_speedup'])}",
    ]

    recommendation = analysis.get("final_algorithm_recommendation")
    if recommendation:
        lines.extend(
            [
                "",
                "## Final Algorithm Recommendation",
                "",
                f"- status: {recommendation.get('status')}",
                f"- primary next step: {recommendation.get('primary_next_step')}",
                f"- classifier required by API: {recommendation.get('classifier_required_by_api')}",
                f"- classifier required for case-specific paths: {recommendation.get('classifier_required_for_case_specific_paths')}",
                "",
                "| priority | area | shape | action | evidence |",
                "| ---: | --- | --- | --- | --- |",
            ]
        )
        for index, row in enumerate(recommendation.get("priority_actions", []), start=1):
            evidence = row.get("evidence", {})
            evidence_text = ", ".join(f"{key}={value}" for key, value in evidence.items() if value not in (None, [], {}))
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(index),
                        str(row.get("area") or ""),
                        str(row.get("shape") or ""),
                        str(row.get("action") or ""),
                        evidence_text,
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
        "## Weakest Speedups",
        "",
        "| case | route | candidate us | baseline us | speedup |",
        "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in analysis["weakest_speedup_cases"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["spec"],
                    str(row.get("route") or row.get("dispatch") or ""),
                    fmt_us(row["candidate_mean_us"]),
                    fmt_us(row["baseline_mean_us"]),
                    fmt_speedup(row["speedup"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Slowest Candidate Cases",
            "",
            "| case | route | candidate us | speedup |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in analysis["slowest_candidate_cases"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["spec"],
                    str(row.get("route") or row.get("dispatch") or ""),
                    fmt_us(row["candidate_mean_us"]),
                    fmt_speedup(row["speedup"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Shape Family Priorities",
            "",
            "| shape | cases | candidate total us | candidate geomean us | speedup | guard | classifier | action |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for row in analysis["shape_family_priorities"]:
        shape = f"{row['batch']}x{row['n']}" if row["batch"] is not None and row["n"] is not None else "unknown"
        if row.get("shape_collision_cases"):
            guard = "collision: " + ",".join(row["shape_collision_cases"])
        elif row.get("requires_tensor_guard"):
            guard = "required"
        else:
            guard = "shape-only"
        if row.get("classifier_on_current_hot_path"):
            classifier = "hot path"
        elif row.get("classifier_needed_for_case_specific_path"):
            classifier = "case-specific only"
        else:
            classifier = "not needed"
        lines.append(
            "| "
            + " | ".join(
                [
                    shape,
                    str(row["num_cases"]),
                    fmt_us(row["candidate_total_us"]),
                    fmt_us(row["candidate_geomean_us"]),
                    fmt_speedup(row["speedup"]),
                    guard,
                    classifier,
                    row["action"],
                ]
            )
            + " |"
        )

    lines.extend(["", "## Ablation Decisions", "", "| ablation | decision | ablation/default |", "| --- | --- | ---: |"])
    for row in analysis["ablation_decisions"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["name"],
                    row["decision"],
                    fmt_speedup(row["ablation_over_default"]),
                ]
            )
            + " |"
        )

    large_cuda = analysis.get("large_cuda_probe_ablation")
    if large_cuda:
        lines.extend(
            [
                "",
                "## Large CUDA Probe Ablation",
                "",
                f"- ablation: `{large_cuda.get('name')}`",
                f"- target shapes: {', '.join(large_cuda.get('target_shapes') or [])}",
                f"- decision: {large_cuda.get('decision')}",
                f"- target ablation/default: {fmt_speedup(large_cuda.get('ablation_over_default'))}",
                f"- all-case ablation/default: {fmt_speedup(large_cuda.get('all_case_ablation_over_default'))}",
                "",
                "| shape | cases | default us | ablation us | ablation/default | decision |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in large_cuda.get("families", []):
            shape = f"{row['batch']}x{row['n']}"
            lines.append(
                "| "
                + " | ".join(
                    [
                        shape,
                        str(row.get("num_cases")),
                        fmt_us(row.get("default_geomean_us")),
                        fmt_us(row.get("ablation_geomean_us")),
                        fmt_speedup(row.get("ablation_over_default")),
                        str(row.get("decision") or ""),
                    ]
                )
                + " |"
            )

    guard = analysis.get("guard_overhead")
    if guard:
        lines.extend(
            [
                "",
                "## Guard Overhead",
                "",
                f"- hot cached max percent of candidate runtime: {guard['hot_wall_pct_max']:.3f}%"
                if guard["hot_wall_pct_max"] is not None
                else "- hot cached max percent of candidate runtime: n/a",
                f"- cold max percent of candidate runtime: {guard['cold_wall_pct_max']:.3f}%"
                if guard["cold_wall_pct_max"] is not None
                else "- cold max percent of candidate runtime: n/a",
                f"- case-selection tensor-guard cases: {guard.get('case_selection_num_cases', 0)}",
                f"- case-selection hot max percent of candidate runtime: {guard['case_selection_hot_wall_pct_max']:.3f}%"
                if guard.get("case_selection_hot_wall_pct_max") is not None
                else "- case-selection hot max percent of candidate runtime: n/a",
                f"- non-case-selection tensor-guard cases: {guard.get('non_case_selection_tensor_guard_num_cases', 0)}",
                "",
                "| case | route | hot wall us | hot % | cold wall us | cold % |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in guard["largest_hot_pct_cases"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["spec"],
                        row["route"],
                        fmt_us(row["wall_us"]),
                        "n/a" if row["hot_wall_pct_of_candidate"] is None else f"{row['hot_wall_pct_of_candidate']:.3f}",
                        fmt_us(row["cold_wall_us"]),
                        "n/a" if row["cold_wall_pct_of_candidate"] is None else f"{row['cold_wall_pct_of_candidate']:.3f}",
                    ]
                )
                + " |"
            )

    dispatch = analysis.get("data_dependent_dispatch")
    if dispatch:
        lines.extend(
            [
                "",
                "## Data-Dependent Dispatch",
                "",
                f"- case metadata passed to submission: {dispatch.get('case_metadata_passed_to_submission')}",
                f"- shape families requiring tensor guards: {dispatch.get('num_shape_families')}",
                "",
                "| shape | cases | routes | source | classifier hot path | max exact checks | sampled rows | cuda-first/default | no-structured/default | no-data/default | hot guard max % | decision |",
                "| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in dispatch.get("families", []):
            shape = f"{row['batch']}x{row['n']}"
            case_list = ",".join(str(label) for label in row.get("shape_collision_cases", []))
            hot_pct = row.get("hot_guard_pct_max")
            routes = ",".join(str(route) for route in row.get("routes", []))
            sources = ",".join(str(source) for source in row.get("route_decision_sources", []))
            sampled_rows = ",".join(str(value) for value in row.get("structured_sampled_row_counts", [])) or "n/a"
            max_exact = row.get("structured_exact_check_count_max")
            lines.append(
                "| "
                + " | ".join(
                    [
                        shape,
                        case_list,
                        routes,
                        sources,
                        str(row.get("classifier_on_current_hot_path")),
                        "n/a" if max_exact is None else str(max_exact),
                        sampled_rows,
                        fmt_speedup(row.get("cuda_first_structured_over_default")),
                        fmt_speedup(row.get("no_structured_over_default")),
                        fmt_speedup(row.get("no_data_dependent_over_default")),
                        "n/a" if hot_pct is None else f"{hot_pct:.3f}",
                        str(row.get("route_order_decision") or row.get("decision") or ""),
                    ]
                )
                + " |"
            )

    classifier_sweep = analysis.get("classifier_seed_sweep")
    if classifier_sweep:
        lines.extend(
            [
                "",
                "## Classifier Seed Sweep",
                "",
                f"- ok: {classifier_sweep.get('ok')}",
                f"- rows: {classifier_sweep.get('num_rows')}",
                f"- shapes: {', '.join(classifier_sweep.get('shapes') or [])}",
                f"- POPCORN seeds: {', '.join(str(seed) for seed in classifier_sweep.get('popcorn_seeds') or [])}",
                f"- classifier mismatches: {classifier_sweep.get('num_classifier_mismatch')}",
                f"- route mismatches: {classifier_sweep.get('num_route_mismatch')}",
                f"- CUDA route bypass rows: {classifier_sweep.get('num_route_cuda_bypass')}",
            ]
        )

    blocked = analysis.get("blocked_qr_sweep")
    if blocked:
        lines.extend(
            [
                "",
                "## Blocked QR Sweep",
                "",
                f"- ok: {blocked.get('ok')}",
                f"- rows: {blocked.get('num_rows')}",
                f"- failed rows: {blocked.get('num_failed')}",
                f"- panel widths: {', '.join(str(value) for value in blocked.get('panel_widths') or [])}",
                f"- update modes: {', '.join(str(value) for value in blocked.get('update_modes') or [])}",
                f"- precision modes: {', '.join(str(value) for value in blocked.get('precision_modes') or [])}",
                f"- passing low-precision configs: {blocked.get('num_passing_low_precision_configs')}",
                "",
                "| precision | R maintenance | panel refresh | rows | failed | max factor | max orth |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in blocked.get("by_config", []):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("precision_mode") or ""),
                        str(row.get("r_maintenance_mode") or ""),
                        str(row.get("panel_refresh_mode") or ""),
                        str(row.get("num_rows")),
                        str(row.get("num_failed")),
                        fmt_us(row.get("max_factor_scaled")),
                        fmt_us(row.get("max_orth_scaled")),
                    ]
                )
                + " |"
            )

    output_layouts = analysis.get("output_layouts")
    if output_layouts:
        lines.extend(
            [
                "",
                "## Output H Layout",
                "",
                f"- cases: {output_layouts['num_cases']}",
                f"- column-major cases: {output_layouts['num_column_major']}",
                f"- torch-contiguous cases: {output_layouts['num_torch_contiguous']}",
                f"- policy layout mismatches: {output_layouts['num_policy_mismatch']}",
                "",
                "| shape | cases | layout counts | column-major | torch-contiguous | policy mismatches |",
                "| --- | ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for row in output_layouts.get("shape_families", []):
            shape = f"{row['batch']}x{row['n']}" if row["batch"] is not None and row["n"] is not None else "unknown"
            counts = ", ".join(f"{key}={value}" for key, value in row.get("layout_counts", {}).items())
            lines.append(
                "| "
                + " | ".join(
                    [
                        shape,
                        str(row["num_cases"]),
                        counts,
                        str(row["num_column_major"]),
                        str(row["num_torch_contiguous"]),
                        str(row["num_policy_mismatch"]),
                    ]
                )
                + " |"
            )

    for heading, key in [
        ("Public Benchmark Correctness", "benchmark_correctness"),
        ("Quantization Seed Sweep", "quantization_seed_sweep"),
        ("Mixed Seed Sweep", "mixed_seed_sweep"),
        ("Tail Policy Sweep", "tail_policy_sweep"),
        ("Dev Robustness", "dev_robustness"),
    ]:
        correctness = analysis.get(key)
        if not correctness:
            continue
        lines.extend(
            [
                "",
                f"## {heading}",
                "",
                f"- cases: {correctness['num_cases']}",
                f"- failed cases: {correctness['num_failed']}",
                f"- margin failures: {correctness['num_margin_failed']}",
                f"- max factor scaled: {fmt_us(correctness['max_factor_scaled'])}",
                f"- max orth scaled: {fmt_us(correctness['max_orth_scaled'])}",
                "",
                "| case | factor scaled | orth scaled |",
                "| --- | ---: | ---: |",
            ]
        )
        for row in correctness["worst_factor_cases"]:
            diagnostics = row["diagnostics"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.get("case_text") or str(row.get("spec") or ""),
                        fmt_us(diagnostics["factor_scaled_max"]),
                        fmt_us(diagnostics["orth_scaled_max"]),
                    ]
                )
                + " |"
            )

    tune = analysis.get("tail_policy_tune")
    if tune:
        lines.extend(
            [
                "",
                "## Tail Policy Tune",
                "",
                f"- configs: {tune.get('num_configs')}",
                f"- configs with correctness failures: {tune.get('num_failed_configs')}",
                f"- benchmarked configs: {tune.get('num_benchmarked_configs')}",
                f"- configs with resource metrics: {tune.get('num_configs_with_resource_metrics')}",
                f"- best config: {tune.get('best_name') or 'n/a'}",
                f"- best geomean us: {fmt_us(tune.get('best_geomean_us'))}",
                "",
                "| config | correctness failures | benchmark geomean us | regs/thread | smem bytes | est occupancy |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in tune.get("results", [])[:12]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("name") or ""),
                        str(row.get("correctness_num_failed")),
                        fmt_us(row.get("benchmark_geomean_us")),
                        fmt_count(row.get("resource_max_registers_per_thread")),
                        fmt_count(row.get("resource_max_smem_bytes")),
                        fmt_ratio(row.get("resource_min_estimated_occupancy")),
                    ]
                )
                + " |"
            )

    tune = analysis.get("candidate_config_tune")
    if tune:
        lines.extend(
            [
                "",
                "## Candidate Config Tune",
                "",
                f"- objective: {tune.get('objective') or 'minimize_geomean_us'}",
                f"- configs: {tune.get('num_configs')}",
                f"- configs with correctness failures: {tune.get('num_failed_configs')}",
                f"- benchmarked configs: {tune.get('num_benchmarked_configs')}",
                f"- configs with inert env keys: {tune.get('num_configs_with_inert_env', 0)}",
                f"- configs with CUDA-route-bypassed env keys: {tune.get('num_configs_with_cuda_route_bypassed_env', 0)}",
                f"- best config: {tune.get('best_name') or 'n/a'}",
                f"- best geomean us: {fmt_us(tune.get('best_geomean_us'))}",
            ]
        )
        route_order = tune.get("route_order")
        if route_order:
            lines.extend(
                [
                    f"- route-order decision: {route_order.get('decision')}",
                    f"- route-order structured/cuda: {fmt_ratio(route_order.get('structured_over_cuda'))}",
                    f"- route-order compared pairs: {route_order.get('num_compared_pairs')}",
                ]
            )
        lines.extend(
            [
                "",
                "| config | env | correctness failures | benchmark geomean us | inert env keys | CUDA-bypassed env keys |",
                "| --- | --- | ---: | ---: | --- | --- |",
            ]
        )
        for row in tune.get("results", [])[:12]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("name") or ""),
                        fmt_env_dict(row.get("env")),
                        str(row.get("correctness_num_failed")),
                        fmt_us(row.get("benchmark_geomean_us")),
                        fmt_key_list(row.get("inert_env_keys")),
                        fmt_key_list(row.get("cuda_route_bypassed_env_keys")),
                    ]
                )
                + " |"
            )
        if route_order and route_order.get("pairs"):
            lines.extend(
                [
                    "",
                    "| route-order label | shared env | cuda-first config | cuda-first us | structured-first config | structured-first us | structured/cuda | decision |",
                    "| --- | --- | --- | ---: | --- | ---: | ---: | --- |",
                ]
            )
            for row in route_order.get("pairs", []):
                shared_env = ", ".join(f"{key}={value}" for key, value in row.get("comparison_env", {}).items())
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(row.get("route_order_label") or ""),
                            shared_env,
                            str(row.get("cuda_first_name") or ""),
                            fmt_us(row.get("cuda_first_geomean_us")),
                            str(row.get("structured_first_name") or ""),
                            fmt_us(row.get("structured_first_geomean_us")),
                            fmt_ratio(row.get("structured_over_cuda")),
                            str(row.get("decision") or ""),
                        ]
                    )
                    + " |"
                )

    experiments = analysis.get("experiments")
    if experiments:
        lines.extend(["", "## Experiments", "", "| experiment | passed | total | pass rate |", "| --- | ---: | ---: | ---: |"])
        for name, row in sorted(experiments["experiments"].items()):
            rate = row["pass_rate"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        str(row["passed"]),
                        str(row["total"]),
                        "n/a" if rate is None else f"{100.0 * rate:.1f}%",
                    ]
                )
                + " |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze a B200 suite and highlight algorithm decisions.",
        allow_abbrev=False,
    )
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--markdown-out", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    suite_dir = ROOT / args.suite_dir if not Path(args.suite_dir).is_absolute() else Path(args.suite_dir)
    analysis = analyze_suite(suite_dir)
    markdown = render_markdown(analysis)

    if args.json_out:
        out = ROOT / args.json_out if not Path(args.json_out).is_absolute() else Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    if args.markdown_out:
        out = ROOT / args.markdown_out if not Path(args.markdown_out).is_absolute() else Path(args.markdown_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown)

    if args.json:
        print(json.dumps(analysis, sort_keys=True))
    else:
        print(markdown)
    return 0 if analysis["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
