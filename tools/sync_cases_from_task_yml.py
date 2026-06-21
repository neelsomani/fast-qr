from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qr_common import ROOT
from spec_utils import render_case_file, specs_from_task_yml


def write_or_check(path: Path, content: str, check: bool) -> bool:
    if check:
        current = path.read_text() if path.exists() else ""
        if current != content:
            print(f"out of sync: {path}", file=sys.stderr)
            return False
        print(f"in sync: {path}")
        return True

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"wrote: {path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate public case files from official/task.yml.",
        allow_abbrev=False,
    )
    parser.add_argument("--task-yml", default="official/task.yml")
    parser.add_argument("--tests-out", default="cases/public_tests.txt")
    parser.add_argument("--benchmarks-out", default="cases/public_benchmarks.txt")
    parser.add_argument("--check", action="store_true", help="Fail if generated output differs.")
    args = parser.parse_args()

    task_yml = ROOT / args.task_yml if not Path(args.task_yml).is_absolute() else Path(args.task_yml)
    tests_out = ROOT / args.tests_out if not Path(args.tests_out).is_absolute() else Path(args.tests_out)
    benchmarks_out = (
        ROOT / args.benchmarks_out if not Path(args.benchmarks_out).is_absolute() else Path(args.benchmarks_out)
    )

    ok = True
    ok &= write_or_check(tests_out, render_case_file(specs_from_task_yml("tests", task_yml)), args.check)
    ok &= write_or_check(
        benchmarks_out,
        render_case_file(specs_from_task_yml("benchmarks", task_yml)),
        args.check,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
