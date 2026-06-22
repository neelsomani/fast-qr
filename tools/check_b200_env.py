from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qr_common import ROOT, append_jsonl


def nvidia_smi_query() -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return {"available": False, "error": "nvidia-smi not found"}
    except subprocess.TimeoutExpired:
        return {"available": False, "error": "nvidia-smi timed out"}

    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            rows.append({"name": parts[0], "driver": parts[1], "memory_total_mib": parts[2]})
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip(),
        "gpus": rows,
    }


def torch_cuda_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {
            "import_ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "cuda_available": False,
            "device_count": 0,
            "devices": [],
        }

    info: dict[str, Any] = {
        "import_ok": True,
        "torch": getattr(torch, "__version__", None),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if hasattr(torch, "cuda") else 0,
        "devices": [],
    }
    if info["cuda_available"]:
        for index in range(info["device_count"]):
            try:
                capability = torch.cuda.get_device_capability(index)
                name = torch.cuda.get_device_name(index)
                props = torch.cuda.get_device_properties(index)
                total_memory = int(getattr(props, "total_memory", 0))
            except Exception as exc:
                info["devices"].append({"index": index, "error": f"{type(exc).__name__}: {exc}"})
                continue
            info["devices"].append(
                {
                    "index": index,
                    "name": name,
                    "capability": list(capability),
                    "total_memory_bytes": total_memory,
                    "total_memory_gib": round(total_memory / (1024**3), 3) if total_memory else None,
                }
            )
    return info


def evaluate(info: dict[str, Any], require_name: str, min_major: int, min_memory_gib: float) -> tuple[bool, list[str]]:
    errors = []
    torch_info = info["torch"]
    if not torch_info.get("import_ok"):
        errors.append("torch import failed")
    if not torch_info.get("cuda_available"):
        errors.append("torch.cuda.is_available() is false")
    if int(torch_info.get("device_count") or 0) <= 0:
        errors.append("torch sees no CUDA devices")

    devices = torch_info.get("devices") or []
    if not devices:
        errors.append("no torch CUDA device records")
        return False, errors

    selected = devices[0]
    name = str(selected.get("name") or "")
    capability = selected.get("capability") or []
    memory_gib = selected.get("total_memory_gib")

    if require_name and require_name.lower() not in name.lower():
        errors.append(f"selected CUDA device name {name!r} does not contain {require_name!r}")
    if not capability or int(capability[0]) < min_major:
        errors.append(f"selected CUDA device capability {capability!r} is below major {min_major}")
    if min_memory_gib > 0 and (memory_gib is None or float(memory_gib) < min_memory_gib):
        errors.append(f"selected CUDA device memory {memory_gib!r} GiB is below {min_memory_gib:g} GiB")

    return not errors, errors


def build_row(args: argparse.Namespace) -> dict[str, Any]:
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "require_name": args.require_name,
        "min_compute_major": args.min_compute_major,
        "min_memory_gib": args.min_memory_gib,
        "nvidia_smi": nvidia_smi_query(),
        "torch": torch_cuda_info(),
    }
    ok, errors = evaluate(row, args.require_name, args.min_compute_major, args.min_memory_gib)
    row["ok"] = ok
    row["errors"] = errors
    row["selected_device"] = (row["torch"].get("devices") or [None])[0]
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the current runtime is suitable for B200 QR timing.")
    parser.add_argument("--require-name", default="B200", help="Substring expected in the selected CUDA device name.")
    parser.add_argument("--min-compute-major", type=int, default=10)
    parser.add_argument("--min-memory-gib", type=float, default=150.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", default=None, help="Append the preflight row to this JSONL file.")
    parser.add_argument("--allow-failure", action="store_true", help="Record failures but exit 0.")
    args = parser.parse_args()

    row = build_row(args)
    if args.out:
        out = Path(args.out)
        append_jsonl(out if out.is_absolute() else ROOT / out, [row])

    if args.json:
        print(json.dumps(row, sort_keys=True))
    else:
        device = row.get("selected_device") or {}
        status = "PASS" if row["ok"] else "FAIL"
        print(f"runtime preflight: {status}")
        print(f"torch: {row['torch'].get('torch')} cuda={row['torch'].get('torch_cuda')}")
        print(f"selected_device: {device.get('name')} capability={device.get('capability')}")
        if row["errors"]:
            for error in row["errors"]:
                print(f"error: {error}", file=sys.stderr)

    return 0 if row["ok"] or args.allow_failure else 1


if __name__ == "__main__":
    sys.exit(main())
