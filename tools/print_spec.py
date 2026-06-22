from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qr_common import ROOT, format_case
from spec_utils import (
    benchmark_shape_collisions,
    custom_kernel_interface,
    evaluator_benchmark_contract,
    format_custom_kernel_interface,
    ranking_by,
    reference_tolerance_factors,
    specs_from_task_yml,
    task_name,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print the frozen official qr_v2 spec summary.",
        allow_abbrev=False,
    )
    parser.add_argument("--task-yml", default="official/task.yml")
    parser.add_argument("--reference", default="official/reference.py")
    parser.add_argument("--submission", default="official/submission.py")
    parser.add_argument("--eval", default="official/eval.py")
    args = parser.parse_args()

    task_yml = ROOT / args.task_yml if not Path(args.task_yml).is_absolute() else Path(args.task_yml)
    reference = ROOT / args.reference if not Path(args.reference).is_absolute() else Path(args.reference)
    submission = ROOT / args.submission if not Path(args.submission).is_absolute() else Path(args.submission)
    eval_py = ROOT / args.eval if not Path(args.eval).is_absolute() else Path(args.eval)
    tests = specs_from_task_yml("tests", task_yml)
    benchmarks = specs_from_task_yml("benchmarks", task_yml)
    tolerances = reference_tolerance_factors(reference)
    interface = custom_kernel_interface(submission)
    collisions = benchmark_shape_collisions(benchmarks)
    benchmark_contract = evaluator_benchmark_contract(eval_py)
    positional_args = list(interface.get("positional_args") or [])
    case_metadata_passed = (
        len(positional_args) > 1
        or bool(interface.get("vararg"))
        or bool(interface.get("kwarg"))
    )

    print(f"name: {task_name(task_yml)}")
    print(f"submission_interface: {format_custom_kernel_interface(interface)}")
    print(f"case_metadata_passed_to_submission: {str(case_metadata_passed).lower()}")
    print(f"submission_info_sources: data.shape, data.dtype, data.device, tensor_values")
    print(f"tests: {len(tests)}")
    print(f"benchmarks: {len(benchmarks)}")
    print(f"ranking_by: {ranking_by(task_yml)}")
    print(f"factor_rtol_factor: {tolerances['factor_rtol_factor']:g}")
    print(f"orth_rtol_factor: {tolerances['orth_rtol_factor']:g}")
    print(
        "benchmark_imports_submission_in_worker: "
        f"{str(benchmark_contract['submission_imported_inside_benchmark_worker']).lower()}"
    )
    print(
        "benchmark_calls_custom_kernel_before_timing: "
        f"{str(benchmark_contract['custom_kernel_called_before_timing']).lower()}"
    )
    print(
        "benchmark_calls_custom_kernel_inside_timed_loop: "
        f"{str(benchmark_contract['custom_kernel_called_inside_timed_loop']).lower()}"
    )
    print(
        "benchmark_rechecks_timed_outputs_when_requested: "
        f"{str(benchmark_contract['timed_outputs_rechecked_when_requested']).lower()}"
    )
    print(
        "benchmark_clears_l2_inside_timed_loop: "
        f"{str(benchmark_contract['l2_cache_cleared_inside_timed_loop']).lower()}"
    )
    print(f"benchmark_shape_collisions: {len(collisions)}")
    print(f"shape_only_case_selection_sufficient: {str(not collisions).lower()}")
    if collisions:
        print("ambiguous benchmark shapes:")
        for row in collisions:
            cases = ", ".join(row["cases"])
            indexes = ", ".join(str(index) for index in row["indexes"])
            print(f"- batch: {row['batch']}; n: {row['n']} -> cases: {cases}; benchmark_indexes: {indexes}")
    print("benchmark cases:")
    for index, spec in enumerate(benchmarks):
        print(f"{index}: {format_case(spec)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
