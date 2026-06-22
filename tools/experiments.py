from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
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
    load_submission,
    parse_case,
    validate_factor_structure,
)


ensure_official_on_path()
from reference import check_implementation, generate_input  # noqa: E402
from diagnose import diagnose  # noqa: E402


DEFAULT_EXPERIMENTS = [
    "r-projection",
    "fp16-nearby",
    "tf32-input-nearby",
    "tail-delete",
    "column-major",
]


@torch.no_grad()
def repair_r(a: torch.Tensor, h: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    q = torch.linalg.householder_product(h, tau)
    r_star = torch.triu(q.double().transpose(-1, -2) @ a.double()).to(torch.float32)
    return torch.tril(h, diagonal=-1) + r_star


@torch.no_grad()
def quantize_tf32_like(a: torch.Tensor) -> torch.Tensor:
    mantissa, exponent = torch.frexp(a)
    mantissa = torch.round(mantissa * 2048.0) / 2048.0
    return torch.ldexp(mantissa, exponent)


def check_row(name: str, spec: dict[str, Any], h: torch.Tensor, tau: torch.Tensor, data: torch.Tensor, extra=None) -> dict:
    good, message = check_implementation(data, (h, tau))
    row = {
        "experiment": name,
        "ok": good,
        "spec": spec,
        "message": message,
    }
    if extra:
        row.update(extra)
    return row


def diagnostic_summary(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "factor_scaled_max",
        "tri_scaled_max",
        "orth_scaled_max",
        "reconstruction_scaled_max",
        "worst_factor_matrix",
        "worst_tri_matrix",
        "worst_orth_matrix",
        "worst_reconstruction_matrix",
    )
    return {f"{key}_{prefix}": values[key] for key in keys}


@torch.no_grad()
def experiment_r_projection(custom_kernel, data: torch.Tensor, spec: dict[str, Any]) -> list[dict]:
    try:
        output = custom_kernel(data.clone())
    except Exception as exc:
        return [
            {
                "experiment": "r-projection",
                "ok": False,
                "spec": spec,
                "message": f"candidate raised {type(exc).__name__}: {exc}",
            }
        ]

    before_good, before_message = check_implementation(data, output)
    h, tau, structure_error = validate_factor_structure(output, data)
    if structure_error:
        return [
            {
                "experiment": "r-projection",
                "ok": False,
                "spec": spec,
                "before_ok": before_good,
                "before_message": before_message,
                "message": f"cannot repair R: {structure_error}",
            }
        ]

    before = diagnose(data, h, tau)
    try:
        h2 = repair_r(data, h, tau)
        after = diagnose(data, h2, tau)
    except Exception as exc:
        return [
            {
                "experiment": "r-projection",
                "ok": False,
                "spec": spec,
                "before_ok": before_good,
                "before_message": before_message,
                **diagnostic_summary("before", before),
                "message": f"R projection raised {type(exc).__name__}: {exc}",
            }
        ]
    extra = {
        "before_ok": before_good,
        "before_message": before_message,
        **diagnostic_summary("before", before),
        **diagnostic_summary("after", after),
    }
    return [check_row("r-projection", spec, h2, tau, data, extra)]


@torch.no_grad()
def experiment_fp16_nearby(data: torch.Tensor, spec: dict[str, Any]) -> list[dict]:
    aq = data.to(torch.float16).to(torch.float32)
    h, tau = torch.geqrf(aq)
    h2 = repair_r(data, h, tau)
    return [check_row("fp16-nearby", spec, h2, tau, data)]


@torch.no_grad()
def experiment_tf32_input_nearby(data: torch.Tensor, spec: dict[str, Any]) -> list[dict]:
    old_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        aq = quantize_tf32_like(data)
        h, tau = torch.geqrf(aq)
        h2 = repair_r(data, h, tau)
    finally:
        torch.set_float32_matmul_precision(old_precision)
    return [
        check_row(
            "tf32-input-nearby",
            spec,
            h2,
            tau,
            data,
            {"meaning": "FP32 geqrf on TF32-rounded input, not TF32 QR arithmetic"},
        )
    ]


@torch.no_grad()
def experiment_tail_delete(data: torch.Tensor, spec: dict[str, Any], cuts: list[int]) -> list[dict]:
    h, tau = torch.geqrf(data)
    n = data.shape[-1]
    rows = []
    for cut in cuts:
        if cut >= n:
            continue
        keep = max(0, n - cut)
        tau2 = tau.clone()
        if keep < n:
            tau2[:, keep:] = 0.0
        h2 = repair_r(data, h, tau2)
        rows.append(check_row("tail-delete", spec, h2, tau2, data, {"tail_cut": cut, "reflectors_kept": keep}))
    return rows


@torch.no_grad()
def experiment_column_major(data: torch.Tensor, spec: dict[str, Any]) -> list[dict]:
    h, tau = torch.geqrf(data)
    h_col = torch.empty_strided(
        h.shape,
        stride=(h.shape[1] * h.shape[2], 1, h.shape[1]),
        device=h.device,
        dtype=h.dtype,
    )
    h_col.copy_(h)
    return [
        check_row(
            "column-major",
            spec,
            h_col,
            tau,
            data,
            {"h_stride": tuple(int(value) for value in h_col.stride())},
        )
    ]


@torch.no_grad()
def experiment_identity_q(data: torch.Tensor, spec: dict[str, Any]) -> list[dict]:
    h = data.clone()
    tau = torch.zeros(data.shape[0], data.shape[1], device=data.device, dtype=data.dtype)
    return [check_row("identity-q", spec, h, tau, data)]


@torch.no_grad()
def classify_features(data: torch.Tensor) -> dict[str, Any]:
    start_time = time.perf_counter_ns()
    if data.is_cuda:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    abs_data = data.abs()
    col_norm = torch.linalg.vector_norm(data, dim=-2)
    row_norm = torch.linalg.vector_norm(data, dim=-1)
    max_col = col_norm.max(dim=-1).values.clamp_min(1e-30)
    min_col = col_norm.min(dim=-1).values
    tail = col_norm[:, col_norm.shape[-1] // 2 :].mean(dim=-1)
    head = col_norm[:, : col_norm.shape[-1] // 2].mean(dim=-1).clamp_min(1e-30)
    zero_tail_cols = (col_norm < (max_col[:, None] * 1.0e-7)).float().mean(dim=-1)
    row_dynamic = row_norm.max(dim=-1).values / row_norm.min(dim=-1).values.clamp_min(1e-30)
    diag_strength = abs_data.diagonal(dim1=-2, dim2=-1).mean(dim=-1) / abs_data.mean(dim=(-2, -1)).clamp_min(1e-30)

    labels: list[str] = []
    for idx in range(data.shape[0]):
        tail_ratio = float((tail[idx] / head[idx]).item())
        zero_fraction = float(zero_tail_cols[idx].item())
        dynamic = float(row_dynamic[idx].item())
        diag_ratio = float(diag_strength[idx].item())
        if zero_fraction > 0.20:
            labels.append("rankdef-like")
        elif tail_ratio < 1.0e-4:
            labels.append("clustered-like")
        elif dynamic > 1.0e4:
            labels.append("rowscale-like")
        elif diag_ratio > 8.0:
            labels.append("band-or-upper-like")
        elif tail_ratio < 0.05:
            labels.append("nearrank-like")
        else:
            labels.append("dense-like")

    if data.is_cuda:
        end_event.record()
        torch.cuda.synchronize()
        elapsed_us = start_event.elapsed_time(end_event) * 1000.0
    else:
        elapsed_us = (time.perf_counter_ns() - start_time) / 1000.0

    return {
        "elapsed_us": elapsed_us,
        "label_counts": dict(Counter(labels)),
        "min_col_ratio_min": float((min_col / max_col).min().item()),
        "tail_to_head_mean": float((tail / head).mean().item()),
        "row_dynamic_max": float(row_dynamic.max().item()),
    }


@torch.no_grad()
def experiment_classify(data: torch.Tensor, spec: dict[str, Any]) -> list[dict]:
    return [{"experiment": "classify", "ok": True, "spec": spec, **classify_features(data)}]


def selected_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.case:
        return [parse_case(args.case)]

    cases_path = ROOT / args.cases if not Path(args.cases).is_absolute() else Path(args.cases)
    cases = load_cases(cases_path)
    if args.all:
        return cases
    if args.index < 0 or args.index >= len(cases):
        raise IndexError(f"--index must be in [0, {len(cases) - 1}] for {cases_path}")
    return [cases[args.index]]


def parse_experiments(value: str) -> list[str]:
    if value == "all":
        return [*DEFAULT_EXPERIMENTS, "classify"]
    if value == "all-with-controls":
        return [*DEFAULT_EXPERIMENTS, "identity-q", "classify"]
    experiments = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {*DEFAULT_EXPERIMENTS, "identity-q", "classify"}
    unknown = [item for item in experiments if item not in allowed]
    if unknown:
        raise ValueError(f"unknown experiments: {', '.join(unknown)}")
    return experiments


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run cheap QR v2 verifier experiments.",
        allow_abbrev=False,
    )
    parser.add_argument("--submission", default="submissions/candidate.py")
    parser.add_argument("--cases", default="cases/public_tests.txt")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--case", default=None)
    parser.add_argument("--all", action="store_true", help="Run every case in --cases.")
    parser.add_argument("--popcorn-seed", type=int, default=None)
    parser.add_argument("--experiments", default=",".join(DEFAULT_EXPERIMENTS))
    parser.add_argument("--tail-cuts", default="0,8,16,32,64,128")
    parser.add_argument("--out", default=None, help="Append JSONL rows to this path.")
    parser.add_argument("--record-env", action="store_true", help="Include repo, torch/CUDA, and submission provenance.")
    args = parser.parse_args()

    submission = ROOT / args.submission if not Path(args.submission).is_absolute() else Path(args.submission)
    custom_kernel = load_submission(submission)
    env = environment_info(torch) if args.record_env else {}
    if args.record_env:
        provenance = file_provenance(submission)
        env["submission"] = provenance["path"]
        env["submission_sha256"] = provenance["sha256"]
    experiments = parse_experiments(args.experiments)
    tail_cuts = [int(value) for value in args.tail_cuts.split(",") if value.strip()]

    rows: list[dict] = []
    for spec in apply_popcorn_seed(selected_specs(args), args.popcorn_seed):
        print(f"case: {format_case(spec)}", file=sys.stderr, flush=True)
        data = generate_input(**spec)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if "r-projection" in experiments:
            rows.extend(experiment_r_projection(custom_kernel, data, spec))
        if "fp16-nearby" in experiments:
            rows.extend(experiment_fp16_nearby(data, spec))
        if "tf32-input-nearby" in experiments:
            rows.extend(experiment_tf32_input_nearby(data, spec))
        if "tail-delete" in experiments:
            rows.extend(experiment_tail_delete(data, spec, tail_cuts))
        if "column-major" in experiments:
            rows.extend(experiment_column_major(data, spec))
        if "identity-q" in experiments:
            rows.extend(experiment_identity_q(data, spec))
        if "classify" in experiments:
            rows.extend(experiment_classify(data, spec))

    if env:
        rows = [{**env, **row} for row in rows]

    for row in rows:
        print(json.dumps(row, sort_keys=True), flush=True)

    if args.out:
        append_jsonl(ROOT / args.out if not Path(args.out).is_absolute() else args.out, rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())
