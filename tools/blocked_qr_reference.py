from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from qr_common import ROOT, append_jsonl, apply_popcorn_seed, ensure_official_on_path, format_case, load_cases, parse_case


ensure_official_on_path()
from reference import check_implementation, generate_input  # noqa: E402
from diagnose import diagnose  # noqa: E402

UPDATE_MODES = ("reflectors", "block-full", "compact-wy")
PRECISION_MODES = ("fp32", "tf32-input", "fp16-input")
R_MAINTENANCE_MODES = ("none", "panel-prefix")
PANEL_REFRESH_MODES = ("none", "prefix")


def allocate_column_major_h(batch: int, n: int, data: torch.Tensor) -> torch.Tensor:
    return torch.empty_strided(
        (batch, n, n),
        stride=(n * n, 1, n),
        device=data.device,
        dtype=torch.float32,
    )


def _quantize_tf32_like(values: torch.Tensor) -> torch.Tensor:
    mantissa, exponent = torch.frexp(values)
    mantissa = torch.round(mantissa * 2048.0) / 2048.0
    return torch.ldexp(mantissa, exponent)


def _round_update_operand(values: torch.Tensor, precision_mode: str) -> torch.Tensor:
    if precision_mode == "fp32":
        return values
    if precision_mode == "tf32-input":
        return _quantize_tf32_like(values)
    if precision_mode == "fp16-input":
        return values.to(torch.float16).to(torch.float32)
    raise ValueError(f"precision_mode must be one of {PRECISION_MODES}, got {precision_mode!r}")


def _update_matmul(left: torch.Tensor, right: torch.Tensor, precision_mode: str) -> torch.Tensor:
    return _round_update_operand(left, precision_mode) @ _round_update_operand(right, precision_mode)


def _apply_panel_reflectors(
    trailing: torch.Tensor,
    panel_h: torch.Tensor,
    panel_tau: torch.Tensor,
    precision_mode: str = "fp32",
) -> None:
    batch, _, width = panel_h.shape
    for j in range(width):
        reflector = panel_h[:, j:, j].clone()
        reflector[:, 0] = 1.0
        target = trailing[:, j:, :]
        dot = _update_matmul(reflector.unsqueeze(1), target, precision_mode).squeeze(1)
        tau_j = _round_update_operand(panel_tau[:, j].reshape(batch, 1, 1), precision_mode)
        reflector = _round_update_operand(reflector, precision_mode)
        dot = _round_update_operand(dot, precision_mode)
        target -= tau_j * reflector.unsqueeze(-1) * dot.unsqueeze(1)


def _panel_left_transform(panel_h: torch.Tensor, panel_tau: torch.Tensor) -> torch.Tensor:
    batch, rows, width = panel_h.shape
    transform = torch.eye(rows, device=panel_h.device, dtype=panel_h.dtype).expand(batch, rows, rows).clone()
    _apply_panel_reflectors(transform, panel_h, panel_tau)
    return transform


def _apply_panel_block_full(
    trailing: torch.Tensor,
    panel_h: torch.Tensor,
    panel_tau: torch.Tensor,
    precision_mode: str,
) -> None:
    transform = _panel_left_transform(panel_h, panel_tau)
    trailing.copy_(_update_matmul(transform, trailing, precision_mode))


def _panel_v_matrix(panel_h: torch.Tensor) -> torch.Tensor:
    batch, rows, width = panel_h.shape
    v = torch.zeros((batch, rows, width), device=panel_h.device, dtype=panel_h.dtype)
    for j in range(width):
        v[:, j:, j] = panel_h[:, j:, j]
        v[:, j, j] = 1.0
    return v


def _form_compact_wy_t(v: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    batch, _, width = v.shape
    t = torch.zeros((batch, width, width), device=v.device, dtype=v.dtype)
    for j in range(width):
        t[:, j, j] = tau[:, j]
        if j == 0:
            continue
        vj = v[:, j:, j]
        previous = v[:, j:, :j]
        y = -tau[:, j].reshape(batch, 1) * (previous.transpose(1, 2) @ vj.unsqueeze(-1)).squeeze(-1)
        t[:, :j, j] = (t[:, :j, :j] @ y.unsqueeze(-1)).squeeze(-1)
    return t


def _apply_panel_compact_wy(
    trailing: torch.Tensor,
    panel_h: torch.Tensor,
    panel_tau: torch.Tensor,
    precision_mode: str = "fp32",
) -> None:
    v = _panel_v_matrix(panel_h)
    t = _form_compact_wy_t(v, panel_tau)
    projection = _update_matmul(v.transpose(1, 2), trailing, precision_mode)
    middle = _update_matmul(t.transpose(1, 2), projection, precision_mode)
    trailing -= _update_matmul(v, middle, precision_mode)


def _apply_trailing_update(
    trailing: torch.Tensor,
    panel_h: torch.Tensor,
    panel_tau: torch.Tensor,
    update_mode: str,
    precision_mode: str,
) -> None:
    if update_mode == "reflectors":
        _apply_panel_reflectors(trailing, panel_h, panel_tau, precision_mode)
    elif update_mode == "block-full":
        _apply_panel_block_full(trailing, panel_h, panel_tau, precision_mode)
    elif update_mode == "compact-wy":
        _apply_panel_compact_wy(trailing, panel_h, panel_tau, precision_mode)
    else:
        raise ValueError(f"update_mode must be one of {UPDATE_MODES}, got {update_mode!r}")


def _apply_prefix_reflectors(
    target: torch.Tensor,
    h: torch.Tensor,
    tau: torch.Tensor,
    prefix_cols: int,
) -> None:
    batch = h.shape[0]
    for j in range(prefix_cols):
        reflector = h[:, j:, j].clone()
        reflector[:, 0] = 1.0
        subtarget = target[:, j:, :]
        dot = (reflector.unsqueeze(1) @ subtarget).squeeze(1)
        subtarget -= tau[:, j].reshape(batch, 1, 1) * reflector.unsqueeze(-1) * dot.unsqueeze(1)


def _repair_panel_r_from_original(
    work: torch.Tensor,
    tau: torch.Tensor,
    original: torch.Tensor,
    panel_start: int,
    width: int,
) -> None:
    n = work.shape[-1]
    prefix_cols = panel_start + width
    projected = original[:, :, panel_start:].clone()
    _apply_prefix_reflectors(projected, work, tau, prefix_cols)

    for local_col in range(width):
        col = panel_start + local_col
        work[:, panel_start : col + 1, col].copy_(projected[:, panel_start : col + 1, local_col])

    trailing_start = panel_start + width
    if trailing_start < n:
        work[:, panel_start:trailing_start, trailing_start:].copy_(projected[:, panel_start:trailing_start, width:])


def _refresh_panel_from_original(
    work: torch.Tensor,
    tau: torch.Tensor,
    original: torch.Tensor,
    panel_start: int,
    width: int,
) -> None:
    if panel_start == 0:
        return
    refreshed = original[:, :, panel_start : panel_start + width].clone()
    _apply_prefix_reflectors(refreshed, work, tau, panel_start)
    work[:, panel_start:, panel_start : panel_start + width].copy_(refreshed[:, panel_start:, :])


@torch.no_grad()
def blocked_geqrf_reference(
    data: torch.Tensor,
    *,
    panel_width: int = 32,
    column_major_h: bool = True,
    update_mode: str = "reflectors",
    precision_mode: str = "fp32",
    r_maintenance_mode: str = "none",
    panel_refresh_mode: str = "none",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Panel-blocked compact Householder QR reference for CUDA kernel design.

    This intentionally uses torch.geqrf for each panel, then applies the panel
    transform to the trailing matrix. The default `reflectors` mode applies the
    reflectors one by one. The `block-full` mode materializes the full panel
    transform once and applies it as a block. The `compact-wy` mode forms the
    triangular block reflector and applies `I - V T.T V.T`, which is the closest
    PyTorch reference to the intended CUDA block update. `precision_mode` keeps
    panel factorization in FP32 and only rounds operands used by the trailing
    update, matching the mixed-precision direction we need to test for CUDA.
    `r_maintenance_mode="panel-prefix"` repairs finalized panel rows by applying
    stored prefix reflectors to the original FP32 input, which prototypes
    block-local R maintenance without a full materialized `Q`.
    `panel_refresh_mode="prefix"` recomputes each active panel from the original
    FP32 input before factorization, preventing low-precision trailing updates
    from poisoning later Householder vectors.
    """
    if panel_width <= 0:
        raise ValueError("panel_width must be positive")
    if update_mode not in UPDATE_MODES:
        raise ValueError(f"update_mode must be one of {UPDATE_MODES}, got {update_mode!r}")
    if precision_mode not in PRECISION_MODES:
        raise ValueError(f"precision_mode must be one of {PRECISION_MODES}, got {precision_mode!r}")
    if r_maintenance_mode not in R_MAINTENANCE_MODES:
        raise ValueError(
            f"r_maintenance_mode must be one of {R_MAINTENANCE_MODES}, got {r_maintenance_mode!r}"
        )
    if panel_refresh_mode not in PANEL_REFRESH_MODES:
        raise ValueError(f"panel_refresh_mode must be one of {PANEL_REFRESH_MODES}, got {panel_refresh_mode!r}")
    if data.dtype != torch.float32 or data.ndim != 3 or data.shape[-1] != data.shape[-2]:
        raise ValueError("data must be a float32 tensor with shape (batch, n, n)")

    batch, n, _ = data.shape
    work = data.clone()
    original = data.clone()
    tau = torch.empty((batch, n), device=data.device, dtype=torch.float32)

    for k in range(0, n, panel_width):
        width = min(panel_width, n - k)
        if panel_refresh_mode == "prefix":
            _refresh_panel_from_original(work, tau, original, k, width)
        panel = work[:, k:, k : k + width].contiguous()
        panel_h, panel_tau = torch.geqrf(panel)

        work[:, k:, k : k + width].copy_(panel_h)
        tau[:, k : k + width].copy_(panel_tau)

        trailing_start = k + width
        if trailing_start < n:
            trailing = work[:, k:, trailing_start:]
            _apply_trailing_update(trailing, panel_h, panel_tau, update_mode, precision_mode)
        if r_maintenance_mode == "panel-prefix":
            _repair_panel_r_from_original(work, tau, original, k, width)

    if not column_major_h:
        return work.contiguous(), tau

    h = allocate_column_major_h(batch, n, data)
    h.copy_(work)
    return h, tau


def selected_specs(args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    if args.case:
        return [(0, parse_case(args.case))]

    cases_path = ROOT / args.cases if not Path(args.cases).is_absolute() else Path(args.cases)
    cases = load_cases(cases_path)
    if args.indices:
        selected = []
        for raw in args.indices.split(","):
            raw = raw.strip()
            if not raw:
                continue
            index = int(raw)
            if index < 0 or index >= len(cases):
                raise IndexError(f"case index {index} is outside 0..{len(cases) - 1}")
            selected.append((index, cases[index]))
        return selected
    return [(index, spec) for index, spec in enumerate(cases)]


def run_case(
    spec: dict[str, Any],
    panel_width: int,
    column_major_h: bool,
    update_mode: str = "reflectors",
    precision_mode: str = "fp32",
    r_maintenance_mode: str = "none",
    panel_refresh_mode: str = "none",
    diagnose_output: bool = False,
) -> dict[str, Any]:
    data = generate_input(**spec)
    start = time.perf_counter_ns()
    h, tau = blocked_geqrf_reference(
        data,
        panel_width=panel_width,
        column_major_h=column_major_h,
        update_mode=update_mode,
        precision_mode=precision_mode,
        r_maintenance_mode=r_maintenance_mode,
        panel_refresh_mode=panel_refresh_mode,
    )
    elapsed_us = (time.perf_counter_ns() - start) / 1000.0
    good, message = check_implementation(data, (h, tau))
    batch, n, _ = data.shape
    row = {
        "ok": bool(good),
        "message": message,
        "spec": spec,
        "case_text": format_case(spec),
        "batch": int(batch),
        "n": int(n),
        "panel_width": int(panel_width),
        "update_mode": update_mode,
        "precision_mode": precision_mode,
        "r_maintenance_mode": r_maintenance_mode,
        "panel_refresh_mode": panel_refresh_mode,
        "column_major_h": bool(column_major_h),
        "h_stride": list(h.stride()),
        "tau_stride": list(tau.stride()),
        "wall_us": elapsed_us,
    }
    if diagnose_output:
        row.update(diagnose(data, h, tau))
    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a PyTorch panel-blocked compact-Householder QR reference.",
        allow_abbrev=False,
    )
    parser.add_argument("--cases", default="cases/public_tests.txt")
    parser.add_argument("--case", default=None)
    parser.add_argument("--indices", default="0", help="Comma-separated case indexes. Defaults to first public test.")
    parser.add_argument("--panel-width", type=int, default=32)
    parser.add_argument("--update-mode", choices=UPDATE_MODES, default="reflectors")
    parser.add_argument("--precision-mode", choices=PRECISION_MODES, default="fp32")
    parser.add_argument("--r-maintenance-mode", choices=R_MAINTENANCE_MODES, default="none")
    parser.add_argument("--panel-refresh-mode", choices=PANEL_REFRESH_MODES, default="none")
    parser.add_argument("--row-major-h", action="store_true", help="Return contiguous H instead of column-major-strided H.")
    parser.add_argument("--diagnose", action="store_true", help="Include detailed residual diagnostics.")
    parser.add_argument("--popcorn-seed", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = []
    selected = selected_specs(args)
    specs = apply_popcorn_seed([spec for _, spec in selected], args.popcorn_seed)
    for (index, _), spec in zip(selected, specs):
        row = run_case(
            spec,
            panel_width=args.panel_width,
            column_major_h=not args.row_major_h,
            update_mode=args.update_mode,
            precision_mode=args.precision_mode,
            r_maintenance_mode=args.r_maintenance_mode,
            panel_refresh_mode=args.panel_refresh_mode,
            diagnose_output=args.diagnose,
        )
        row["case_index"] = index
        rows.append(row)
        if args.json:
            print(json.dumps(row, sort_keys=True), flush=True)
        else:
            status = "PASS" if row["ok"] else "FAIL"
            print(
                f"{index}: {status} {row['case_text']} panel={args.panel_width} "
                f"update={args.update_mode} precision={args.precision_mode} "
                f"r={args.r_maintenance_mode} refresh={args.panel_refresh_mode} wall_us={row['wall_us']:.1f}"
            )
            if not row["ok"]:
                print(row["message"], file=sys.stderr)

    if args.out:
        append_jsonl(ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out), rows)
    return 0 if all(row["ok"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
