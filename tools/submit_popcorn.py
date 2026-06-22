from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from qr_common import ROOT, file_provenance, repo_provenance
from validate_submission import validate_submission


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def selected_modes(mode: str) -> list[str]:
    if mode == "both":
        return ["test", "leaderboard"]
    return [mode]


def build_command(
    submission_path: Path,
    leaderboard: str,
    gpu: str,
    mode: str,
    popcorn_bin: str = "popcorn",
    extra_args: list[str] | None = None,
) -> list[str]:
    cmd = [
        popcorn_bin,
        "submit",
        "--leaderboard",
        leaderboard,
        "--gpu",
        gpu,
        "--mode",
        mode,
        str(submission_path),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def stage_submission(source: Path, out_dir: Path) -> Path:
    staged = out_dir / "submission.py"
    shutil.copy2(source, staged)
    return staged


def run_command(cmd: list[str], log_path: Path, timeout_s: float | None = None) -> int:
    start = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(cmd)}\n")
        log.flush()
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        log.write(completed.stdout)
        log.write(f"\nexit_code={completed.returncode}; elapsed_s={time.perf_counter() - start:.3f}\n")
    print(completed.stdout, end="")
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage and submit a QR v2 candidate to Popcorn test/leaderboard modes.",
        allow_abbrev=False,
    )
    parser.add_argument("--submission", default="submissions/candidate.py")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--leaderboard", default="qr_v2")
    parser.add_argument("--gpu", default="B200")
    parser.add_argument("--mode", choices=["test", "leaderboard", "both"], default="test")
    parser.add_argument("--popcorn-bin", default="popcorn")
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--extra-arg", action="append", default=[], help="Extra argument appended to popcorn submit.")
    parser.add_argument("--skip-validation", action="store_true", help="Skip one-file submission validation before staging.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source = ROOT / args.submission if not Path(args.submission).is_absolute() else Path(args.submission)
    source = source.resolve()
    if not source.is_file():
        parser.error(f"submission file does not exist: {source}")

    out_dir = ROOT / "results" / f"popcorn_submit_{timestamp()}" if args.out_dir is None else Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    validation = None
    if not args.skip_validation:
        validation = validate_submission(source, stage_dir=out_dir / "validation_stage")
        append_jsonl(out_dir / "submission_validation.jsonl", validation)
        if not validation["ok"]:
            print(json.dumps(validation, sort_keys=True))
            return 2

    staged = stage_submission(source, out_dir)
    manifest_path = out_dir / "manifest.jsonl"
    log_path = out_dir / "popcorn.log"
    modes = selected_modes(args.mode)

    append_jsonl(
        manifest_path,
        {
            "event": "submit_start",
            "time": datetime.now().isoformat(),
            "dry_run": args.dry_run,
            "repo": repo_provenance(),
            "source_submission": file_provenance(source),
            "staged_submission": file_provenance(staged),
            "validation": validation,
            "args": vars(args),
        },
    )

    failed = False
    for mode in modes:
        cmd = build_command(
            staged,
            leaderboard=args.leaderboard,
            gpu=args.gpu,
            mode=mode,
            popcorn_bin=args.popcorn_bin,
            extra_args=args.extra_arg,
        )
        append_jsonl(
            manifest_path,
            {
                "event": "mode_plan",
                "mode": mode,
                "cmd": cmd,
                "time": datetime.now().isoformat(),
            },
        )
        if args.dry_run:
            continue
        started = time.perf_counter()
        try:
            code = run_command(cmd, log_path, timeout_s=args.timeout_s)
            error = None
        except subprocess.TimeoutExpired as exc:
            code = 124
            error = f"TimeoutExpired: {exc}"
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\nTIMEOUT: {error}\n")
        elapsed_s = time.perf_counter() - started
        append_jsonl(
            manifest_path,
            {
                "event": "mode_finish",
                "mode": mode,
                "exit_code": code,
                "error": error,
                "elapsed_s": elapsed_s,
                "time": datetime.now().isoformat(),
            },
        )
        failed = failed or code != 0
        if failed:
            break

    append_jsonl(
        manifest_path,
        {
            "event": "submit_finish",
            "ok": not failed,
            "dry_run": args.dry_run,
            "out_dir": str(out_dir),
            "time": datetime.now().isoformat(),
        },
    )

    print(
        json.dumps(
            {
                "ok": not failed,
                "dry_run": args.dry_run,
                "out_dir": str(out_dir),
                "submission": str(staged),
                "modes": modes,
            },
            sort_keys=True,
        )
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
