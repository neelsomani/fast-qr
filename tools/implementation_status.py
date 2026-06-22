from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from candidate_policy import policy_rows
from qr_common import ROOT, append_jsonl, environment_info, file_provenance, relative_path


TORCH_COMPOSITE_PRIMARIES = {
    "dense_tail_projection_or_fallback",
    "embedded_rectangular_geqrf",
    "embedded_rectangular_geqrf_or_fallback",
    "nearrank_tail_projection",
    "per_matrix_mixed_structured_fast",
}


def _required_kernel_next_work(row: dict[str, Any], fallback: str) -> str:
    required = row.get("required_cuda_kernel")
    if not required:
        return fallback
    label = row.get("candidate_config_shape_label", "large-shape")
    modes = row.get("required_repair_modes") or []
    mode_text = f" with {', '.join(str(mode) for mode in modes)}" if modes else ""
    return (
        f"B200-validate and tune current tile-parallel blocked repair configs{mode_text}; "
        f"then promote the best {required} config for {label}; keep optional fallback only until B200 correctness is proven"
    )


def implementation_status_for_policy(row: dict[str, Any], case_index: int) -> dict[str, Any]:
    primary = str(row.get("primary") or "")
    dispatch = str(row.get("dispatch") or "")
    n = int(row.get("n") or 0)
    batch = int(row.get("batch") or 0)

    if primary == "inline_cuda_compact_householder_or_fallback" or row.get("cuda_primary") == "inline_cuda_compact_householder":
        implementation_kind = "custom_cuda_optional_fallback"
        readiness = "partial_cuda_needs_b200_validation"
        uses_torch_geqrf = True
        has_custom_cuda = True
        final_kernel_required = True
        next_work = _required_kernel_next_work(
            row,
            "verify inline CUDA path on B200, then remove or justify fallback for timed benchmark",
        )
    elif primary == "torch.geqrf":
        implementation_kind = "torch_geqrf_fallback"
        readiness = "missing_custom_kernel"
        uses_torch_geqrf = True
        has_custom_cuda = False
        final_kernel_required = True
        next_work = f"write compact Householder CUDA path for batch={batch}, n={n}"
    elif primary in TORCH_COMPOSITE_PRIMARIES or "fallback" in primary:
        implementation_kind = "torch_composite_experiment"
        readiness = "experimental_not_final"
        uses_torch_geqrf = True
        has_custom_cuda = False
        final_kernel_required = True
        next_work = _required_kernel_next_work(
            row,
            "replace PyTorch geqrf/projection shortcut with block-local custom QR/R maintenance",
        )
    else:
        implementation_kind = "unknown"
        readiness = "needs_review"
        uses_torch_geqrf = True
        has_custom_cuda = False
        final_kernel_required = True
        next_work = "classify implementation path before trusting timing"

    priority = "normal"
    if batch == 640 and n == 512:
        priority = "highest"
    elif batch == 60 and n == 1024:
        priority = "high"
    elif n in {176, 352, 2048, 4096}:
        priority = "medium"

    return {
        "case_index": case_index,
        "spec": row.get("spec"),
        "batch": batch,
        "n": n,
        "case": row.get("case", "dense"),
        "dispatch": dispatch,
        "primary": primary,
        "implementation_kind": implementation_kind,
        "readiness": readiness,
        "priority": priority,
        "uses_torch_geqrf": uses_torch_geqrf,
        "has_custom_cuda": has_custom_cuda,
        "final_kernel_required": final_kernel_required,
        "column_major_h": row.get("column_major_h"),
        "h_layout": row.get("h_layout"),
        "shape_collision": row.get("shape_collision"),
        "uses_tensor_values_for_dispatch": row.get("uses_tensor_values_for_dispatch"),
        "classifier_needed_for_current_candidate": row.get("classifier_needed_for_current_candidate"),
        "required_cuda_kernel": row.get("required_cuda_kernel"),
        "required_cuda_reason": row.get("required_cuda_reason"),
        "required_repair_modes": row.get("required_repair_modes", []),
        "candidate_config_shape_label": row.get("candidate_config_shape_label"),
        "candidate_config_env_prefix": row.get("candidate_config_env_prefix"),
        "candidate_config_benchmark_indices": row.get("candidate_config_benchmark_indices"),
        "candidate_config_correctness_indices": row.get("candidate_config_correctness_indices"),
        "next_work": next_work,
    }


def readiness_rows(submission: str | Path, cases: str | Path) -> list[dict[str, Any]]:
    rows = policy_rows(submission, cases)
    return [implementation_status_for_policy(row, index) for index, row in enumerate(rows)]


def summarize_readiness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, int] = {}
    by_readiness: dict[str, int] = {}
    for row in rows:
        by_kind[str(row["implementation_kind"])] = by_kind.get(str(row["implementation_kind"]), 0) + 1
        by_readiness[str(row["readiness"])] = by_readiness.get(str(row["readiness"]), 0) + 1

    missing = [row for row in rows if row.get("final_kernel_required")]
    highest = [row for row in missing if row.get("priority") == "highest"]
    high = [row for row in missing if row.get("priority") == "high"]
    next_targets = highest or high or missing
    return {
        "summary": True,
        "ok": True,
        "ready_for_final_submission": not missing,
        "num_cases": len(rows),
        "num_final_kernel_required": len(missing),
        "num_custom_cuda_partial": by_kind.get("custom_cuda_optional_fallback", 0),
        "num_torch_composite_experiment": by_kind.get("torch_composite_experiment", 0),
        "num_torch_geqrf_fallback": by_kind.get("torch_geqrf_fallback", 0),
        "by_implementation_kind": dict(sorted(by_kind.items())),
        "by_readiness": dict(sorted(by_readiness.items())),
        "next_priority_cases": [
            {
                "case_index": row["case_index"],
                "spec": row["spec"],
                "dispatch": row["dispatch"],
                "readiness": row["readiness"],
                "next_work": row["next_work"],
            }
            for row in next_targets[:4]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report which public benchmark routes are real custom kernels, experiments, or fallbacks.",
        allow_abbrev=False,
    )
    parser.add_argument("--submission", default="submissions/candidate.py")
    parser.add_argument("--cases", default="cases/public_benchmarks.txt")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--record-env", action="store_true")
    args = parser.parse_args()

    submission_path = ROOT / args.submission if not Path(args.submission).is_absolute() else Path(args.submission)
    cases_path = ROOT / args.cases if not Path(args.cases).is_absolute() else Path(args.cases)
    rows = readiness_rows(submission_path, cases_path)
    summary = summarize_readiness(rows)

    if args.record_env:
        import torch

        env = environment_info(torch)
        env["submission"] = relative_path(submission_path)
        env["submission_sha256"] = file_provenance(submission_path)["sha256"]
        rows = [{**env, **row} for row in rows]
        summary = {**env, **summary}

    output_rows = [*rows, summary]
    if args.json:
        for row in output_rows:
            print(json.dumps(row, sort_keys=True))
    else:
        for row in rows:
            print(
                f"{row['case_index']:2d} {row['dispatch']:24s} "
                f"{row['implementation_kind']:30s} {row['readiness']}"
            )
        print(json.dumps(summary, sort_keys=True))

    if args.out:
        append_jsonl(ROOT / args.out if not Path(args.out).is_absolute() else args.out, output_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
