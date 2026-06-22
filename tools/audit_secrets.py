from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from qr_common import ROOT, append_jsonl


MAX_TEXT_BYTES = 2 * 1024 * 1024
SKIP_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "results",
}
SKIP_SUFFIXES = {
    ".a",
    ".cubin",
    ".dylib",
    ".fatbin",
    ".o",
    ".png",
    ".pyc",
    ".sass",
    ".so",
}


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    severity: str = "high"


RULES = [
    Rule("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    Rule("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{36,255}\b")),
    Rule("openai_api_key", re.compile(r"\bsk-[A-Za-z0-9_-]{32,}\b")),
    Rule("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
    Rule("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    Rule("private_key_block", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")),
]


def iter_repo_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"git ls-files failed: {completed.stderr.strip()}")
    files = []
    for raw in completed.stdout.splitlines():
        if not raw:
            continue
        path = ROOT / raw
        if should_skip(path):
            continue
        files.append(path)
    return files


def should_skip(path: Path) -> bool:
    try:
        rel = path.relative_to(ROOT)
    except ValueError:
        return True
    if any(part in SKIP_PARTS for part in rel.parts):
        return True
    if path.suffix in SKIP_SUFFIXES:
        return True
    return not path.is_file()


def read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) > MAX_TEXT_BYTES:
        return None
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def preview(value: str) -> str:
    if len(value) <= 12:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def scan_text(path: Path, text: str, rules: Iterable[Rule] = RULES) -> list[dict]:
    findings = []
    try:
        rel = str(path.relative_to(ROOT))
    except ValueError:
        rel = str(path)
    for line_no, line in enumerate(text.splitlines(), start=1):
        for rule in rules:
            for match in rule.pattern.finditer(line):
                findings.append(
                    {
                        "ok": False,
                        "file": rel,
                        "line": line_no,
                        "rule": rule.name,
                        "severity": rule.severity,
                        "match_preview": preview(match.group(0)),
                    }
                )
    return findings


def scan_repo() -> list[dict]:
    rows = []
    files = iter_repo_files()
    scanned = 0
    skipped = 0
    for path in files:
        text = read_text(path)
        if text is None:
            skipped += 1
            continue
        scanned += 1
        rows.extend(scan_text(path, text))
    rows.append(
        {
            "summary": True,
            "ok": not rows,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "files_considered": len(files),
            "files_scanned": scanned,
            "files_skipped": skipped,
            "num_findings": len(rows),
            "rules": [rule.name for rule in RULES],
        }
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan repo files for high-signal secret patterns.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", default=None, help="Append JSONL rows to this file.")
    parser.add_argument("--allow-findings", action="store_true", help="Exit 0 even if findings are present.")
    args = parser.parse_args()

    rows = scan_repo()
    if args.out:
        out = Path(args.out)
        append_jsonl(out if out.is_absolute() else ROOT / out, rows)

    if args.json:
        for row in rows:
            print(json.dumps(row, sort_keys=True))
    else:
        summary = rows[-1]
        status = "PASS" if summary["ok"] else "FAIL"
        print(f"secret audit: {status}")
        print(
            f"files_scanned={summary['files_scanned']} "
            f"files_skipped={summary['files_skipped']} "
            f"findings={summary['num_findings']}"
        )
        for row in rows[:-1]:
            print(f"{row['file']}:{row['line']}: {row['rule']} {row['match_preview']}", file=sys.stderr)

    return 0 if rows[-1]["ok"] or args.allow_findings else 1


if __name__ == "__main__":
    sys.exit(main())
