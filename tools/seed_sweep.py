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
    load_cases,
    load_submission,
    parse_case,
    parse_popcorn_seed_tokens,
    validate_factor_structure,
)


ensure_official_on_path()
from reference import check_implementation, generate_input  # noqa: E402
from diagnose import diagnose  # noqa: E402


def parse_int_list(value: str | None, default: list[int] | None = None) -> list[int]:
    if not value:
        return default or []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def selected_cases(args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    if args.case:
        return [(0, parse_case(args.case))]

    cases_path = ROOT / args.cases if not Path(args.cases).is_absolute() else Path(args.cases)
    cases = load_cases(cases_path)
    indexes = parse_int_list(args.indices)
    if not indexes:
        indexes = list(range(len(cases)))
    return [(index, cases[index]) for index in indexes]


@torch.no_grad()
def run_case(
    custom_kernel,
    spec: dict[str, Any],
    diagnose_output: bool,
    max_factor_scaled: float | None = None,
    max_orth_scaled: float | None = None,
) -> dict[str, Any]:
    data = generate_input(**spec)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    output = custom_kernel(data.clone())
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    good, message = check_implementation(data, output)
    row: dict[str, Any] = {"ok": good, "message": message}
    needs_diagnostics = diagnose_output or max_factor_scaled is not None or max_orth_scaled is not None
    if needs_diagnostics:
        h, tau, structure_error = validate_factor_structure(output, data)
        if structure_error:
            row["diagnostics_error"] = structure_error
        else:
            row.update(diagnose(data, h, tau))
            margin_ok = True
            if max_factor_scaled is not None:
                row["max_factor_scaled_limit"] = max_factor_scaled
                factor_margin_ok = row["factor_scaled_max"] <= max_factor_scaled
                row["factor_margin_ok"] = factor_margin_ok
                margin_ok = margin_ok and factor_margin_ok
            if max_orth_scaled is not None:
                row["max_orth_scaled_limit"] = max_orth_scaled
                orth_margin_ok = row["orth_scaled_max"] <= max_orth_scaled
                row["orth_margin_ok"] = orth_margin_ok
                margin_ok = margin_ok and orth_margin_ok
            if max_factor_scaled is not None or max_orth_scaled is not None:
                row["margin_ok"] = margin_ok
    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run POPCORN_SEED robustness sweeps for QR submissions.",
        allow_abbrev=False,
    )
    parser.add_argument("--submission", default="submissions/candidate.py")
    parser.add_argument("--cases", default="cases/public_tests.txt")
    parser.add_argument("--case", default=None)
    parser.add_argument("--indices", default="", help="Comma-separated case indexes. Defaults to all cases.")
    parser.add_argument(
        "--popcorn-seeds",
        default="public,0,1,2,3,4,5",
        help="Comma-separated POPCORN_SEED values; use 'public' for the unmodified public seed.",
    )
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--max-factor-scaled", type=float, default=None)
    parser.add_argument("--max-orth-scaled", type=float, default=None)
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--record-env", action="store_true", help="Include repo, torch/CUDA, and submission provenance.")
    args = parser.parse_args()

    submission = ROOT / args.submission if not Path(args.submission).is_absolute() else Path(args.submission)
    custom_kernel = load_submission(submission)
    env = environment_info(torch) if args.record_env else {}
    if args.record_env:
        provenance = file_provenance(submission)
        env["submission"] = provenance["path"]
        env["submission_sha256"] = provenance["sha256"]
    seeds = parse_popcorn_seed_tokens(args.popcorn_seeds, default=[None])
    base_cases = selected_cases(args)

    rows = []
    for popcorn_seed in seeds:
        for index, base_spec in base_cases:
            spec = apply_popcorn_seed([base_spec], popcorn_seed)[0]
            row = run_case(
                custom_kernel,
                spec,
                args.diagnose,
                max_factor_scaled=args.max_factor_scaled,
                max_orth_scaled=args.max_orth_scaled,
            )
            if env:
                row = {**env, **row}
            row.update({"case_index": index, "popcorn_seed": popcorn_seed, "spec": spec})
            print(json.dumps(row, sort_keys=True), flush=True)
            rows.append(row)
            failed_margin = row.get("margin_ok") is False
            if (not row["ok"] or failed_margin) and args.stop_on_fail:
                if args.out:
                    append_jsonl(ROOT / args.out if not Path(args.out).is_absolute() else args.out, rows)
                return 1

    if args.out:
        append_jsonl(ROOT / args.out if not Path(args.out).is_absolute() else args.out, rows)
    return 0 if all(row["ok"] and row.get("margin_ok", True) for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
