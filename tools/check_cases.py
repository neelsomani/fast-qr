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
    load_submission,
    validate_factor_structure,
)


ensure_official_on_path()
from reference import check_implementation, generate_input  # noqa: E402


def parse_indices(raw: str, count: int) -> list[int]:
    if not raw.strip():
        return list(range(count))
    indexes = [int(value.strip()) for value in raw.split(",") if value.strip()]
    for index in indexes:
        if index < 0 or index >= count:
            raise IndexError(f"case index {index} is outside [0, {count - 1}]")
    return indexes


def output_layout_metadata(output, data: torch.Tensor) -> dict[str, Any]:
    row: dict[str, Any] = {}
    if not isinstance(output, tuple) or len(output) != 2:
        return {"output_layout_error": "output must be a tuple `(H, tau)`"}

    h = output[0]
    if not isinstance(h, torch.Tensor):
        return {"output_layout_error": "H is not a tensor"}

    batch, n, _ = data.shape
    stride = [int(value) for value in h.stride()]
    shape = [int(value) for value in h.shape]
    expected_column_major = [n * n, 1, n]
    column_major = shape == [batch, n, n] and stride == expected_column_major
    row.update(
        {
            "h_shape": shape,
            "h_stride": stride,
            "h_is_contiguous": bool(h.is_contiguous()),
            "column_major_h_actual": column_major,
            "h_layout_actual": "column_major"
            if column_major
            else ("torch_contiguous" if h.is_contiguous() else "other_strided"),
        }
    )
    return row


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
    row: dict[str, Any] = {
        "ok": good,
        "spec": spec,
        "message": message,
    }
    row.update(output_layout_metadata(output, data))

    needs_diagnostics = diagnose_output or max_factor_scaled is not None or max_orth_scaled is not None
    if needs_diagnostics:
        from diagnose import diagnose

        h, tau, structure_error = validate_factor_structure(output, data)
        if structure_error:
            row["diagnostics_error"] = structure_error
        else:
            diagnostics = diagnose(data, h, tau)
            row["diagnostics"] = diagnostics
            margin_ok = True
            if max_factor_scaled is not None:
                row["max_factor_scaled_limit"] = max_factor_scaled
                factor_margin_ok = diagnostics["factor_scaled_max"] <= max_factor_scaled
                row["factor_margin_ok"] = factor_margin_ok
                margin_ok = margin_ok and factor_margin_ok
            if max_orth_scaled is not None:
                row["max_orth_scaled_limit"] = max_orth_scaled
                orth_margin_ok = diagnostics["orth_scaled_max"] <= max_orth_scaled
                row["orth_margin_ok"] = orth_margin_ok
                margin_ok = margin_ok and orth_margin_ok
            if max_factor_scaled is not None or max_orth_scaled is not None:
                row["margin_ok"] = margin_ok

    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run multiple QR v2 correctness cases.",
        allow_abbrev=False,
    )
    parser.add_argument("--submission", default="submissions/candidate.py")
    parser.add_argument("--cases", default="cases/public_tests.txt")
    parser.add_argument("--indices", default="", help="Comma-separated case indexes. Defaults to all cases.")
    parser.add_argument("--popcorn-seed", type=int, default=None)
    parser.add_argument("--diagnose", action="store_true", help="Include detailed residual diagnostics.")
    parser.add_argument("--max-factor-scaled", type=float, default=None)
    parser.add_argument("--max-orth-scaled", type=float, default=None)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON rows.")
    parser.add_argument("--out", default=None, help="Append JSONL rows to this path.")
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--record-env", action="store_true", help="Include repo, torch/CUDA, and submission provenance.")
    args = parser.parse_args()

    submission = ROOT / args.submission if not Path(args.submission).is_absolute() else Path(args.submission)
    cases_path = ROOT / args.cases if not Path(args.cases).is_absolute() else Path(args.cases)
    all_cases = load_cases(cases_path)
    indexes = parse_indices(args.indices, len(all_cases))
    cases = apply_popcorn_seed([all_cases[index] for index in indexes], args.popcorn_seed)
    custom_kernel = load_submission(submission)
    env = environment_info(torch) if args.record_env else {}
    if args.record_env:
        provenance = file_provenance(submission)
        env["submission"] = provenance["path"]
        env["submission_sha256"] = provenance["sha256"]

    rows = []
    passed = True
    for case_index, spec in zip(indexes, cases):
        row = run_case(
            custom_kernel,
            spec,
            args.diagnose,
            max_factor_scaled=args.max_factor_scaled,
            max_orth_scaled=args.max_orth_scaled,
        )
        if env:
            row = {**env, **row}
        row["case_index"] = case_index
        row["case_text"] = format_case(spec)
        rows.append(row)

        if args.json:
            print(json.dumps(row, sort_keys=True), flush=True)
        else:
            status = "PASS" if row["ok"] else "FAIL"
            print(f"{status}: {case_index}: {row['case_text']}", flush=True)
            if not row["ok"]:
                print(row["message"], flush=True)

        row_passed = row["ok"] and row.get("margin_ok", True)
        if not row_passed:
            passed = False
            if args.stop_on_fail:
                break

    summary = {
        "ok": passed,
        "num_cases": len(rows),
        "num_passed": sum(1 for row in rows if row["ok"] and row.get("margin_ok", True)),
        "num_failed": sum(1 for row in rows if not (row["ok"] and row.get("margin_ok", True))),
        "summary": True,
    }
    rows.append(summary)
    if args.json:
        print(json.dumps(summary, sort_keys=True), flush=True)
    else:
        print(json.dumps(summary, sort_keys=True), flush=True)

    if args.out:
        append_jsonl(ROOT / args.out if not Path(args.out).is_absolute() else args.out, rows)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
