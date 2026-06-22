from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch

from qr_common import ROOT, append_jsonl, environment_info, file_provenance, format_case, load_cases
from spec_utils import benchmark_shape_collisions


def load_candidate_module(path: str | Path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()

    official = str(ROOT / "official")
    if official in sys.path:
        sys.path.remove(official)
    sys.path.insert(0, official)

    module_name = f"candidate_policy_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load candidate from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _dense_tail_policy(candidate, n: int) -> dict[str, Any]:
    cut = int(candidate._dense_tail_cut(n))
    force = bool(candidate._dense_tail_force(n)) if hasattr(candidate, "_dense_tail_force") else False
    if cut <= 0:
        return {"cut": 0, "rank": n, "threshold": 0.0, "force": force}
    return {
        "cut": cut,
        "rank": n - cut,
        "threshold": float(candidate._dense_tail_threshold(n)),
        "force": force,
    }


def _mixed_tail_policy(candidate, n: int) -> dict[str, Any]:
    cut = int(candidate._mixed_dense_tail_cut(n))
    if cut <= 0:
        return {"cut": 0, "rank": n, "threshold": 0.0}
    return {
        "cut": cut,
        "rank": n - cut,
        "threshold": float(candidate._mixed_dense_tail_threshold(n)),
    }


def _set_layout_flags(row: dict[str, Any]) -> None:
    primary = str(row.get("primary", ""))
    if primary == "torch.geqrf":
        row["column_major_h"] = False
        row["h_layout"] = "torch.geqrf_default"
        return
    if primary == "inline_cuda_compact_householder_or_fallback":
        row["column_major_h"] = "conditional"
        row["h_layout"] = "column_major_when_cuda_extension_available_else_torch.geqrf_default"
        return
    if "fallback" in primary:
        row["column_major_h"] = "conditional"
        row["h_layout"] = "column_major_when_fast_path_applies_else_torch.geqrf_default"
        return
    row["column_major_h"] = True
    row["h_layout"] = "column_major"


def _set_route_source_flags(row: dict[str, Any]) -> None:
    uses_route_tensor_values = bool(row.get("classifier_on_current_hot_path"))
    dense_tail = row.get("dense_tail")
    cuda_first_bypass = bool(row.get("cuda_route_bypasses_classifier"))
    if (
        row.get("primary") == "dense_tail_projection_or_fallback"
        and isinstance(dense_tail, dict)
        and not cuda_first_bypass
    ):
        uses_route_tensor_values = uses_route_tensor_values or int(dense_tail.get("cut") or 0) > 0

    row["uses_tensor_values_for_dispatch"] = uses_route_tensor_values
    row["shape_only_dispatch"] = not uses_route_tensor_values
    sources = ["data.shape"]
    if uses_route_tensor_values:
        sources.append("tensor_values")
    row["dispatch_info_sources"] = sources


def _structured_before_cuda(candidate, n: int) -> bool:
    if hasattr(candidate, "_structured_before_cuda"):
        return bool(candidate._structured_before_cuda(n))
    return False


def policy_for_spec(candidate, spec: dict[str, Any]) -> dict[str, Any]:
    batch = int(spec["batch"])
    n = int(spec["n"])
    case = str(spec.get("case", "dense"))
    row: dict[str, Any] = {
        "spec": format_case(spec),
        "batch": batch,
        "n": n,
        "case": case,
        "submission_entrypoint": "custom_kernel(data)",
        "case_metadata_available": False,
        "case_metadata_passed_to_submission": False,
        "case_info_source": "data.shape",
        "case_selection_info_sources": ["data.shape"],
        "dispatch_info_sources": ["data.shape"],
        "shape_only_case_selection": True,
        "shape_only_dispatch": True,
        "uses_tensor_values_for_dispatch": False,
        "uses_tensor_values_for_case_selection": False,
        "classifier_needed_for_current_candidate": False,
        "classifier_needed_for_case_specific_path": False,
        "classifier_on_current_hot_path": False,
        "classifier_reason": "shape uniquely identifies this public benchmark row",
        "classifier_decision_rule": "not_applicable_shape_unique",
        "fallback": "torch.geqrf",
    }

    if batch == 20 and n == 32:
        row.update(
            {
                "dispatch": "qr32_fast",
                "primary": "inline_cuda_compact_householder_or_fallback",
                "cuda_kernel": "geqrf32_kernel",
            }
        )
    elif batch == 40 and n == 176:
        row.update(
            {
                "dispatch": "qr176_fast",
                "primary": "inline_cuda_compact_householder_or_fallback",
                "cuda_kernel": "geqrf176_kernel",
            }
        )
    elif batch == 40 and n == 352:
        row.update(
            {
                "dispatch": "qr352_fast",
                "primary": "inline_cuda_compact_householder_or_fallback",
                "cuda_kernel": "geqrf352_kernel",
            }
        )
    elif n == 512:
        rank = int(candidate._rankdef_effective_cols(n))
        clustered_cols = int(candidate._clustered_effective_cols(n))
        row.update(
            {
                "dispatch": "qr512_fast",
                "cuda_route": "qr512_cuda_fast",
                "cuda_kernel": "geqrf512_kernel",
                "blocked_cuda_route": "qr512_blocked_cuda_auto_fast",
                "blocked_cuda_base_route": "qr512_blocked_cuda_fast",
                "blocked_cuda_kernel": "geqrf512_blocked_kernel",
                "blocked_cuda_enable_env": "FAST_QR_ENABLE_QR512_BLOCKED_CUDA",
                "blocked_cuda_precision_env": "FAST_QR_QR512_BLOCKED_PRECISION_MODE",
                "cuda_primary": "inline_cuda_compact_householder",
                "cuda_disable_env": "FAST_QR_DISABLE_QR512_CUDA",
                "cuda_route_bypasses_classifier": not _structured_before_cuda(candidate, n),
                "structured_before_cuda": _structured_before_cuda(candidate, n),
                "structured_before_cuda_env": "FAST_QR_QR512_STRUCTURED_BEFORE_CUDA",
                "global_structured_before_cuda_env": "FAST_QR_STRUCTURED_ROUTES_BEFORE_CUDA",
                "required_cuda_kernel": "qr512_blocked_householder_r_maintenance",
                "required_cuda_reason": (
                    "The n=512 family dominates the geomean at benchmark batch=640; the current one-CTA QR512 "
                    "CUDA path is a correctness/perf probe, not the final blocked compact-Householder path"
                ),
                "required_repair_modes": ["panel_refresh_mode=prefix", "r_maintenance_mode=panel-prefix"],
                "candidate_config_shape_label": "qr512",
                "candidate_config_env_prefix": "FAST_QR_QR512",
                "candidate_config_benchmark_indices": "3,7,9,10",
                "candidate_config_correctness_indices": "3,6,7,8,9,10,11,19",
                "rankdef_cols": rank,
                "clustered_cols": clustered_cols,
                "dense_tail": _dense_tail_policy(candidate, n),
                "mixed_dense_tail": _mixed_tail_policy(candidate, n),
            }
        )
        if case == "rankdef":
            row["primary"] = "embedded_rectangular_geqrf"
            row["active_cols"] = rank
        elif case == "clustered":
            row["primary"] = "embedded_rectangular_geqrf"
            row["active_cols"] = clustered_cols
        elif case == "mixed":
            row["primary"] = "per_matrix_mixed_structured_fast"
            row["per_matrix_groups"] = [
                f"rankdef_cols={rank}",
                f"clustered_cols={clustered_cols}",
                f"scaled_nearrank_cols={rank}",
                "fallback=torch.geqrf",
            ]
        else:
            row["primary"] = "dense_tail_projection_or_fallback"
            row["active_cols"] = n - int(row["dense_tail"]["cut"])
    elif n == 1024:
        rank = int(candidate._rankdef_effective_cols(n))
        row.update(
            {
                "dispatch": "qr1024_fast",
                "cuda_route": "qr1024_cuda_fast",
                "cuda_kernel": "geqrf1024_kernel",
                "blocked_cuda_route": "qr1024_blocked_cuda_auto_fast",
                "blocked_cuda_base_route": "qr1024_blocked_cuda_fast",
                "blocked_cuda_kernel": "geqrf1024_blocked_kernel",
                "blocked_cuda_enable_env": "FAST_QR_ENABLE_QR1024_BLOCKED_CUDA",
                "blocked_cuda_precision_env": "FAST_QR_QR1024_BLOCKED_PRECISION_MODE",
                "cuda_primary": "inline_cuda_compact_householder",
                "cuda_disable_env": "FAST_QR_DISABLE_QR1024_CUDA",
                "cuda_route_bypasses_classifier": not _structured_before_cuda(candidate, n),
                "structured_before_cuda": _structured_before_cuda(candidate, n),
                "structured_before_cuda_env": "FAST_QR_QR1024_STRUCTURED_BEFORE_CUDA",
                "global_structured_before_cuda_env": "FAST_QR_STRUCTURED_ROUTES_BEFORE_CUDA",
                "required_cuda_kernel": "qr1024_blocked_householder_r_maintenance",
                "required_cuda_reason": (
                    "The n=1024 family is a high-impact ambiguous-shape family at benchmark batch=60; the "
                    "current one-CTA QR1024 CUDA path is a correctness/perf probe, not the final blocked "
                    "compact-Householder path"
                ),
                "required_repair_modes": ["panel_refresh_mode=prefix", "r_maintenance_mode=panel-prefix"],
                "candidate_config_shape_label": "qr1024",
                "candidate_config_env_prefix": "FAST_QR_QR1024",
                "candidate_config_benchmark_indices": "4,8,11",
                "candidate_config_correctness_indices": "4,12,13,14,15,20",
                "nearrank_cols": rank,
                "dense_tail": _dense_tail_policy(candidate, n),
                "mixed_dense_tail": _mixed_tail_policy(candidate, n),
            }
        )
        if case == "nearrank":
            row["primary"] = "nearrank_tail_projection"
            row["active_cols"] = rank
        elif case == "rankdef":
            row["primary"] = "embedded_rectangular_geqrf"
            row["active_cols"] = rank
        elif case == "clustered":
            row["primary"] = "embedded_rectangular_geqrf"
            row["active_cols"] = int(candidate._clustered_effective_cols(n))
        elif case == "mixed":
            row["primary"] = "per_matrix_mixed_structured_fast"
            groups = [
                f"rankdef_cols={rank}",
                f"clustered_cols={int(candidate._clustered_effective_cols(n))}",
                f"scaled_nearrank_cols={rank}",
            ]
            mixed_cut = int(row["mixed_dense_tail"]["cut"])
            if mixed_cut > 0:
                groups.append(f"tiny_dense_tail_cut={mixed_cut}")
            groups.append("fallback=torch.geqrf")
            row["per_matrix_groups"] = groups
        else:
            row["primary"] = "dense_tail_projection_or_fallback"
            row["active_cols"] = n - int(row["dense_tail"]["cut"])
    elif batch == 8 and n == 2048:
        row.update(
            {
                "dispatch": "qr2048_fast",
                "primary": "dense_tail_projection_or_fallback",
                "blocked_cuda_route": "qr2048_blocked_cuda_auto_fast",
                "blocked_cuda_base_route": "qr2048_blocked_cuda_fast",
                "blocked_cuda_kernel": "geqrf2048_blocked_kernel",
                "blocked_cuda_enable_env": "FAST_QR_ENABLE_QR2048_BLOCKED_CUDA",
                "blocked_cuda_precision_env": "FAST_QR_QR2048_BLOCKED_PRECISION_MODE",
                "cuda_primary": "inline_cuda_compact_householder",
                "dense_tail": _dense_tail_policy(candidate, n),
                "required_cuda_kernel": "qr2048_multi_cta_blocked_householder",
                "required_cuda_reason": "batch=8 needs intra-matrix parallelism; one-CTA QR is not a viable final path",
                "candidate_config_shape_label": "qr2048",
                "candidate_config_env_prefix": "FAST_QR_QR2048",
                "candidate_config_benchmark_indices": "5",
                "candidate_config_correctness_indices": "16,21",
            }
        )
        row["active_cols"] = n - int(row["dense_tail"]["cut"])
    elif batch == 2 and n == 4096:
        row.update(
            {
                "dispatch": "qr4096_fast",
                "primary": "dense_tail_projection_or_fallback",
                "blocked_cuda_route": "qr4096_blocked_cuda_auto_fast",
                "blocked_cuda_base_route": "qr4096_blocked_cuda_fast",
                "blocked_cuda_kernel": "geqrf4096_blocked_kernel",
                "blocked_cuda_enable_env": "FAST_QR_ENABLE_QR4096_BLOCKED_CUDA",
                "blocked_cuda_precision_env": "FAST_QR_QR4096_BLOCKED_PRECISION_MODE",
                "cuda_primary": "inline_cuda_compact_householder",
                "dense_tail": _dense_tail_policy(candidate, n),
                "required_cuda_kernel": "qr4096_multi_cta_blocked_householder",
                "required_cuda_reason": "batch=2 needs heavy intra-matrix parallelism; one-CTA QR is not a viable final path",
                "candidate_config_shape_label": "qr4096",
                "candidate_config_env_prefix": "FAST_QR_QR4096",
                "candidate_config_benchmark_indices": "6",
                "candidate_config_correctness_indices": "5,18",
            }
        )
        row["active_cols"] = n - int(row["dense_tail"]["cut"])
    else:
        row.update({"dispatch": "fallback", "primary": "torch.geqrf"})

    _set_layout_flags(row)
    return row


def policy_rows(candidate_path: str | Path, cases_path: str | Path) -> list[dict[str, Any]]:
    candidate = load_candidate_module(candidate_path)
    path = Path(cases_path)
    if not path.is_absolute():
        path = ROOT / path
    specs = load_cases(path)
    collision_by_shape = {
        (int(row["batch"]), int(row["n"])): row
        for row in benchmark_shape_collisions(specs)
    }
    rows = []
    for spec in specs:
        row = policy_for_spec(candidate, spec)
        collision = collision_by_shape.get((int(spec["batch"]), int(spec["n"])))
        row["case_metadata_passed_to_submission"] = False
        row["shape_collision"] = collision is not None
        if collision is not None:
            row["shape_collision_cases"] = collision["cases"]
            row["shape_collision_indexes"] = collision["indexes"]
            row["requires_tensor_guard_for_case_specific_path"] = True
            row["case_info_source"] = "tensor_values"
            row["case_selection_info_sources"] = ["data.shape", "tensor_values"]
            row["shape_only_case_selection"] = False
            row["uses_tensor_values_for_case_selection"] = True
            row["classifier_needed_for_case_specific_path"] = True
            row["classifier_on_current_hot_path"] = not bool(row.get("cuda_route_bypasses_classifier"))
            row["classifier_needed_for_current_candidate"] = bool(row["classifier_on_current_hot_path"])
            if row["classifier_on_current_hot_path"]:
                row["classifier_reason"] = (
                    "official custom_kernel receives only data and this shape maps to multiple public benchmark cases"
                )
                row["classifier_decision_rule"] = (
                    "keep only if B200 timing beats FAST_QR_DISABLE_STRUCTURED_ROUTES=1 after guard overhead"
                )
            else:
                row["classifier_reason"] = (
                    "case metadata is not passed, but the current CUDA-first route uses shape-only dispatch before "
                    "case-specific structured paths"
                )
                row["classifier_decision_rule"] = (
                    "promote a case-specific classifier only if FAST_QR_QR{n}_STRUCTURED_BEFORE_CUDA=1 beats "
                    "the CUDA-first route after guard overhead"
                ).format(n=int(spec["n"]))
        else:
            row["requires_tensor_guard_for_case_specific_path"] = False
        _set_route_source_flags(row)
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Print current candidate dispatch/cutoff policy.")
    parser.add_argument("--submission", default="submissions/candidate.py")
    parser.add_argument("--cases", default="cases/public_benchmarks.txt")
    parser.add_argument("--json", action="store_true", help="Emit JSON lines instead of a table.")
    parser.add_argument("--out", default=None, help="Also write policy rows as JSONL to this path.")
    parser.add_argument("--record-env", action="store_true", help="Include repo, torch/CUDA, and submission provenance.")
    args = parser.parse_args()

    rows = policy_rows(args.submission, args.cases)
    if args.record_env:
        submission = ROOT / args.submission if not Path(args.submission).is_absolute() else Path(args.submission)
        provenance = file_provenance(submission)
        env = environment_info(torch)
        env["submission"] = provenance["path"]
        env["submission_sha256"] = provenance["sha256"]
        rows = [{**env, **row} for row in rows]
    if args.out:
        out = Path(args.out)
        append_jsonl(out if out.is_absolute() else ROOT / out, rows)

    if args.json:
        for row in rows:
            print(json.dumps(row, sort_keys=True))
        return 0

    for index, row in enumerate(rows):
        extra = []
        if "active_cols" in row:
            extra.append(f"active_cols={row['active_cols']}")
        if row.get("primary") == "dense_tail_projection_or_fallback" and row.get("dense_tail", {}).get("cut"):
            extra.append(f"dense_cut={row['dense_tail']['cut']}")
        if row.get("primary") == "per_matrix_mixed_structured_fast" and row.get("per_matrix_groups"):
            extra.append("groups=" + "|".join(row["per_matrix_groups"]))
        if row.get("primary") == "per_matrix_mixed_structured_fast" and row.get("mixed_dense_tail", {}).get("cut"):
            extra.append(f"mixed_cut={row['mixed_dense_tail']['cut']}")
        if row.get("shape_collision"):
            extra.append("shape_collision=" + "|".join(row["shape_collision_cases"]))
        details = f" ({', '.join(extra)})" if extra else ""
        print(f"{index}: {row['spec']} -> {row['dispatch']} / {row['primary']}{details}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
