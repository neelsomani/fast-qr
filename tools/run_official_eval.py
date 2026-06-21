from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from qr_common import ROOT


def write_official_case_file(src: Path, dst: Path) -> Path:
    lines = []
    for line in src.read_text().splitlines():
        stripped = line.split("#", 1)[0].strip()
        if stripped:
            lines.append(stripped)
    dst.write_text("\n".join(lines) + ("\n" if lines else ""))
    return dst


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run frozen official/eval.py with a local submission and fake POPCORN_FD.",
        allow_abbrev=False,
    )
    parser.add_argument("--submission", default="submissions/candidate.py")
    parser.add_argument("--cases", default="cases/public_tests.txt")
    parser.add_argument("--mode", choices=["test", "benchmark", "leaderboard"], default="test")
    parser.add_argument("--popcorn-seed", type=int, default=None)
    args = parser.parse_args()

    submission = ROOT / args.submission if not Path(args.submission).is_absolute() else Path(args.submission)
    cases = ROOT / args.cases if not Path(args.cases).is_absolute() else Path(args.cases)

    with tempfile.TemporaryDirectory(prefix="qr_v2_official_") as tmp:
        tmp_path = Path(tmp)
        for name in ["eval.py", "reference.py", "task.py", "utils.py"]:
            shutil.copy2(ROOT / "official" / name, tmp_path / name)
        shutil.copy2(submission, tmp_path / "submission.py")

        official_cases = write_official_case_file(cases, tmp_path / "cases.txt")
        output_path = tmp_path / "popcorn_output.txt"
        with output_path.open("w") as output:
            env = os.environ.copy()
            env["POPCORN_FD"] = str(output.fileno())
            if args.popcorn_seed is not None:
                env["POPCORN_SEED"] = str(args.popcorn_seed)
            completed = subprocess.run(
                [sys.executable, str(tmp_path / "eval.py"), args.mode, str(official_cases)],
                cwd=tmp_path,
                env=env,
                pass_fds=(output.fileno(),),
                check=False,
            )

        print(output_path.read_text(), end="")
        return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
