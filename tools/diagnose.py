from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from qr_common import ROOT, apply_popcorn_seed, ensure_official_on_path, load_cases, load_submission, parse_case


ensure_official_on_path()
from reference import generate_input  # noqa: E402


def _matrix_l1(value: torch.Tensor) -> torch.Tensor:
    return torch.linalg.matrix_norm(value.double(), ord=1, dim=(-2, -1))


@torch.no_grad()
def diagnose(a: torch.Tensor, h: torch.Tensor, tau: torch.Tensor, include_vectors: bool = False) -> dict[str, Any]:
    n = a.shape[-1]
    eps = torch.finfo(torch.float32).eps

    q = torch.linalg.householder_product(h, tau)
    r = torch.triu(h)

    a64 = a.double()
    q64 = q.double()
    r64 = r.double()

    projected = q64.transpose(-1, -2) @ a64
    factor_residual = _matrix_l1(r64 - projected)
    scale = _matrix_l1(a64).clamp_min(1e-30)
    factor_scaled = factor_residual / (eps * n * scale)

    lower = torch.tril(projected, diagonal=-1)
    tri_residual = _matrix_l1(lower)
    tri_scaled = tri_residual / (eps * n * scale)

    eye = torch.eye(n, device=a.device, dtype=torch.float64).expand_as(projected)
    qtq = q64.transpose(-1, -2) @ q64
    orth_residual = _matrix_l1(qtq - eye)
    orth_scaled = orth_residual / (eps * n)

    recon = q64 @ r64
    recon_residual = _matrix_l1(recon - a64)
    recon_scaled = recon_residual / (eps * n * scale)

    result: dict[str, Any] = {
        "factor_residual_max": float(factor_residual.max().item()),
        "factor_scaled_max": float(factor_scaled.max().item()),
        "tri_residual_max": float(tri_residual.max().item()),
        "tri_scaled_max": float(tri_scaled.max().item()),
        "orth_residual_max": float(orth_residual.max().item()),
        "orth_scaled_max": float(orth_scaled.max().item()),
        "reconstruction_residual_max": float(recon_residual.max().item()),
        "reconstruction_scaled_max": float(recon_scaled.max().item()),
        "worst_factor_matrix": int(factor_scaled.argmax().item()),
        "worst_tri_matrix": int(tri_scaled.argmax().item()),
        "worst_orth_matrix": int(orth_scaled.argmax().item()),
        "worst_reconstruction_matrix": int(recon_scaled.argmax().item()),
    }
    if include_vectors:
        result.update(
            {
                "factor_scaled": factor_scaled.detach().cpu().tolist(),
                "tri_scaled": tri_scaled.detach().cpu().tolist(),
                "orth_scaled": orth_scaled.detach().cpu().tolist(),
                "reconstruction_scaled": recon_scaled.detach().cpu().tolist(),
            }
        )
    return result


def selected_spec(args: argparse.Namespace) -> dict:
    if args.case:
        return parse_case(args.case)
    cases_path = ROOT / args.cases if not Path(args.cases).is_absolute() else Path(args.cases)
    cases = load_cases(cases_path)
    if args.index < 0 or args.index >= len(cases):
        raise IndexError(f"--index must be in [0, {len(cases) - 1}] for {cases_path}")
    return cases[args.index]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report QR residual diagnostics for one case.",
        allow_abbrev=False,
    )
    parser.add_argument("--submission", default="submissions/candidate.py")
    parser.add_argument("--cases", default="cases/public_tests.txt")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--case", default=None)
    parser.add_argument("--popcorn-seed", type=int, default=None)
    parser.add_argument("--per-matrix", action="store_true", help="Include per-matrix scaled residual arrays.")
    args = parser.parse_args()

    custom_kernel = load_submission(ROOT / args.submission if not Path(args.submission).is_absolute() else args.submission)
    spec = apply_popcorn_seed([selected_spec(args)], args.popcorn_seed)[0]
    data = generate_input(**spec)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    h, tau = custom_kernel(data.clone())
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(json.dumps(diagnose(data, h, tau, include_vectors=args.per_matrix), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
