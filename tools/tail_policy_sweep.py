from __future__ import annotations

import argparse
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
    parse_case,
    parse_popcorn_seed_tokens,
    validate_factor_structure,
)
from trace_candidate_routes import load_candidate_module


ensure_official_on_path()
from reference import check_implementation, generate_input  # noqa: E402
from diagnose import diagnose  # noqa: E402


CutToken = int | str


def parse_cut_tokens(value: str) -> list[CutToken]:
    tokens: list[CutToken] = []
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        if token == "candidate":
            tokens.append(token)
            continue
        cut = int(token)
        if cut < 0:
            raise ValueError("tail cuts must be non-negative")
        tokens.append(cut)
    if not tokens:
        raise ValueError("at least one tail cut must be provided")
    return tokens


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


def candidate_route(candidate, data: torch.Tensor) -> str:
    if hasattr(candidate, "_route_for_data"):
        return str(candidate._route_for_data(data))
    return "custom_kernel"


def candidate_policy_cut(candidate, data: torch.Tensor, route: str) -> int:
    n = int(data.shape[-1])
    if route in {
        "qr512_dense_fast",
        "qr1024_dense_fast",
        "qr2048_fast",
        "qr2048_dense_fast",
        "qr4096_fast",
        "qr4096_dense_fast",
    }:
        return int(candidate._dense_tail_cut(n)) if hasattr(candidate, "_dense_tail_cut") else 0
    if route in {"qr512_mixed_fast", "qr1024_mixed_fast"}:
        return int(candidate._mixed_dense_tail_cut(n)) if hasattr(candidate, "_mixed_dense_tail_cut") else 0
    if route in {"qr512_rankdef_fast", "qr1024_rankdef_fast", "qr1024_nearrank_fast"}:
        return n - int(candidate._rankdef_effective_cols(n)) if hasattr(candidate, "_rankdef_effective_cols") else 0
    if route in {"qr512_clustered_fast", "qr1024_clustered_fast"}:
        return n - int(candidate._clustered_effective_cols(n)) if hasattr(candidate, "_clustered_effective_cols") else 0
    return 0


@torch.no_grad()
def output_for_cut(candidate, data: torch.Tensor, cut: int):
    n = int(data.shape[-1])
    if cut == 0:
        return torch.geqrf(data)
    if cut >= n:
        raise ValueError(f"tail cut {cut} must be smaller than n={n}")
    if not hasattr(candidate, "_embedded_geqrf_with_tail_projection"):
        raise AttributeError("candidate does not expose _embedded_geqrf_with_tail_projection")
    return candidate._embedded_geqrf_with_tail_projection(data, n - cut)


@torch.no_grad()
def run_policy_cut(
    candidate,
    data: torch.Tensor,
    spec: dict[str, Any],
    token: CutToken,
    diagnose_output: bool,
    max_factor_scaled: float | None = None,
    max_orth_scaled: float | None = None,
) -> dict[str, Any]:
    route = candidate_route(candidate, data)
    policy_cut = candidate_policy_cut(candidate, data, route)
    if token == "candidate":
        cut = policy_cut
        cut_source = "candidate"
        strategy = "candidate_custom_kernel"
        output = candidate.custom_kernel(data.clone())
    else:
        cut = int(token)
        cut_source = "override"
        strategy = "embedded_tail_projection"
        output = output_for_cut(candidate, data.clone(), cut)

    good, message = check_implementation(data, output)
    row: dict[str, Any] = {
        "ok": good,
        "message": message,
        "spec": spec,
        "case_text": format_case(spec),
        "candidate_route": route,
        "candidate_policy_cut": policy_cut,
        "tail_cut": cut,
        "reflectors_kept": max(0, int(data.shape[-1]) - cut),
        "cut_source": cut_source,
        "strategy": strategy,
    }

    needs_diagnostics = diagnose_output or max_factor_scaled is not None or max_orth_scaled is not None
    if needs_diagnostics:
        h, tau, structure_error = validate_factor_structure(output, data)
        if structure_error:
            row["diagnostics_error"] = structure_error
        else:
            diagnostics = diagnose(data, h, tau)
            row["diagnostics"] = diagnostics
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep candidate tail-cut policies across QR cases and POPCORN_SEED values.",
        allow_abbrev=False,
    )
    parser.add_argument("--submission", default="submissions/candidate.py")
    parser.add_argument("--cases", default="cases/public_benchmarks.txt")
    parser.add_argument("--case", default=None)
    parser.add_argument("--indices", default="", help="Comma-separated case indexes. Defaults to all cases.")
    parser.add_argument(
        "--popcorn-seeds",
        default="public",
        help="Comma-separated POPCORN_SEED values; use 'public' for the unmodified public seed.",
    )
    parser.add_argument("--tail-cuts", default="candidate", help='Comma-separated cuts, e.g. "candidate,0,8,16,32".')
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--max-factor-scaled", type=float, default=None)
    parser.add_argument("--max-orth-scaled", type=float, default=None)
    parser.add_argument("--stop-on-fail", action="store_true")
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
    cuts = parse_cut_tokens(args.tail_cuts)
    seeds = parse_popcorn_seed_tokens(args.popcorn_seeds, default=[None]) or [None]
    cases = selected_cases(args)

    rows = []
    failed = False
    for popcorn_seed in seeds:
        for case_index, base_spec in cases:
            spec = apply_popcorn_seed([base_spec], popcorn_seed)[0]
            data = generate_input(**spec)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            for token in cuts:
                try:
                    row = run_policy_cut(
                        candidate,
                        data,
                        spec,
                        token,
                        args.diagnose,
                        max_factor_scaled=args.max_factor_scaled,
                        max_orth_scaled=args.max_orth_scaled,
                    )
                except Exception as exc:
                    route = candidate_route(candidate, data)
                    row = {
                        "ok": False,
                        "message": f"{type(exc).__name__}: {exc}",
                        "spec": spec,
                        "case_text": format_case(spec),
                        "candidate_route": route,
                        "candidate_policy_cut": candidate_policy_cut(candidate, data, route),
                        "tail_cut": token,
                        "cut_source": "candidate" if token == "candidate" else "override",
                    }
                if env:
                    row = {**env, **row}
                row.update({"case_index": case_index, "popcorn_seed": popcorn_seed})
                print(json.dumps(row, sort_keys=True), flush=True)
                rows.append(row)
                row_failed = (not row.get("ok")) or row.get("margin_ok") is False
                failed = failed or row_failed
                if row_failed and args.stop_on_fail:
                    if args.out:
                        append_jsonl(ROOT / args.out if not Path(args.out).is_absolute() else args.out, rows)
                    return 1

    summary = {
        "summary": True,
        "ok": not failed,
        "num_rows": len(rows),
        "num_failed": sum(1 for row in rows if not row.get("ok") or row.get("margin_ok") is False),
        "num_passed": sum(1 for row in rows if row.get("ok") and row.get("margin_ok", True)),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    rows.append(summary)

    if args.out:
        append_jsonl(ROOT / args.out if not Path(args.out).is_absolute() else args.out, rows)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
