from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qr_common import ROOT, format_case
from spec_utils import ranking_by, reference_tolerance_factors, specs_from_task_yml, task_name


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print the frozen official qr_v2 spec summary.",
        allow_abbrev=False,
    )
    parser.add_argument("--task-yml", default="official/task.yml")
    parser.add_argument("--reference", default="official/reference.py")
    args = parser.parse_args()

    task_yml = ROOT / args.task_yml if not Path(args.task_yml).is_absolute() else Path(args.task_yml)
    reference = ROOT / args.reference if not Path(args.reference).is_absolute() else Path(args.reference)
    tests = specs_from_task_yml("tests", task_yml)
    benchmarks = specs_from_task_yml("benchmarks", task_yml)
    tolerances = reference_tolerance_factors(reference)

    print(f"name: {task_name(task_yml)}")
    print(f"tests: {len(tests)}")
    print(f"benchmarks: {len(benchmarks)}")
    print(f"ranking_by: {ranking_by(task_yml)}")
    print(f"factor_rtol_factor: {tolerances['factor_rtol_factor']:g}")
    print(f"orth_rtol_factor: {tolerances['orth_rtol_factor']:g}")
    print("benchmark cases:")
    for index, spec in enumerate(benchmarks):
        print(f"{index}: {format_case(spec)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
