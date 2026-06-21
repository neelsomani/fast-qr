from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from qr_common import (
    ROOT,
    apply_popcorn_seed,
    ensure_official_on_path,
    format_case,
    load_cases,
    load_submission,
    parse_case,
    validate_factor_structure,
)


ensure_official_on_path()
from reference import check_implementation, generate_input  # noqa: E402


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
        description="Run one QR v2 correctness case.",
        allow_abbrev=False,
    )
    parser.add_argument("--submission", default="submissions/candidate.py")
    parser.add_argument("--cases", default="cases/public_tests.txt")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--case", default=None, help='Inline spec, e.g. "batch: 20; n: 32; cond: 1; seed: 1"')
    parser.add_argument("--popcorn-seed", type=int, default=None)
    parser.add_argument("--diagnose", action="store_true", help="Print detailed residual diagnostics.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    custom_kernel = load_submission(ROOT / args.submission if not Path(args.submission).is_absolute() else args.submission)
    spec = apply_popcorn_seed([selected_spec(args)], args.popcorn_seed)[0]
    data = generate_input(**spec)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    output = custom_kernel(data.clone())
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    good, message = check_implementation(data, output)
    result = {"ok": good, "spec": spec, "message": message}

    if args.diagnose:
        from diagnose import diagnose

        h, tau, structure_error = validate_factor_structure(output, data)
        if structure_error:
            result["diagnostics_error"] = structure_error
        else:
            result["diagnostics"] = diagnose(data, h, tau)

    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        status = "PASS" if good else "FAIL"
        print(f"{status}: {format_case(spec)}")
        print(message)
        if args.diagnose and "diagnostics" in result:
            print(json.dumps(result["diagnostics"], indent=2, sort_keys=True))
        if args.diagnose and "diagnostics_error" in result:
            print(f"diagnostics_error: {result['diagnostics_error']}")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
