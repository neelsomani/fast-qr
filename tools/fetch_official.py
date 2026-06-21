from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

from qr_common import ROOT


REFERENCE_REPO = "https://github.com/gpu-mode/reference-kernels.git"
RAW_ROOT = "https://raw.githubusercontent.com/gpu-mode/reference-kernels"


def resolve_sha(explicit_sha: str | None) -> str:
    if explicit_sha:
        return explicit_sha.strip()
    output = subprocess.check_output(["git", "ls-remote", REFERENCE_REPO, "HEAD"], text=True)
    return output.split()[0]


def fetch(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response:
        path.write_bytes(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch official qr_v2 files from a single pinned reference-kernels commit.",
        allow_abbrev=False,
    )
    parser.add_argument("--sha", default=None, help="reference-kernels commit SHA. Defaults to current HEAD.")
    args = parser.parse_args()

    sha = resolve_sha(args.sha)
    official = ROOT / "official"
    (official / "UPSTREAM_COMMIT").write_text(sha + "\n")
    (official / "FETCHED_AT").write_text(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") + "\n")

    qr_base = f"{RAW_ROOT}/{sha}/problems/linalg/qr_v2"
    for name in ["submission.py", "reference.py", "task.py", "task.yml", "eval.py"]:
        fetch(f"{qr_base}/{name}", official / name)
    fetch(f"{RAW_ROOT}/{sha}/problems/pmpp_v2/utils.py", official / "utils.py")
    print(f"fetched official files from {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
