from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_DIR = ROOT / "official"
BENCHMARK_INPUT_BYTES_TARGET = 256 * 1024 * 1024
MAX_ITERATIONS_PER_BENCHMARK = 50


def ensure_official_on_path() -> None:
    path = str(OFFICIAL_DIR)
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def parse_case(line: str) -> dict[str, int | str]:
    raw = line.split("#", 1)[0].strip()
    if not raw:
        return {}

    out: dict[str, int | str] = {}
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"invalid case field {part!r} in line {line!r}")
        key, value = [piece.strip() for piece in part.split(":", 1)]
        if not key:
            raise ValueError(f"empty case key in line {line!r}")
        try:
            out[key] = int(value)
        except ValueError:
            out[key] = value
    return out


def format_case(spec: dict[str, Any]) -> str:
    preferred = ["batch", "n", "cond", "seed", "case"]
    keys = [key for key in preferred if key in spec]
    keys.extend(sorted(key for key in spec if key not in preferred))
    return "; ".join(f"{key}: {spec[key]}" for key in keys)


def load_cases(path: str | Path) -> list[dict[str, int | str]]:
    cases: list[dict[str, int | str]] = []
    for line in Path(path).read_text().splitlines():
        spec = parse_case(line)
        if spec:
            cases.append(spec)
    return cases


def combine_seed(a: int, b: int) -> int:
    return int(a + (a + b) * (a + b + 1) // 2)


def apply_popcorn_seed(cases: list[dict[str, Any]], seed: int | None) -> list[dict[str, Any]]:
    if seed is None:
        return [dict(case) for case in cases]
    out = []
    for case in cases:
        updated = dict(case)
        if "seed" in updated:
            updated["seed"] = combine_seed(int(updated["seed"]), int(seed))
        out.append(updated)
    return out


def clone_data(value: Any) -> Any:
    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(clone_data(item) for item in value)
    if isinstance(value, list):
        return [clone_data(item) for item in value]
    if isinstance(value, dict):
        return {key: clone_data(item) for key, item in value.items()}
    return value


def validate_factor_structure(output: Any, data: Any):
    import torch

    if not isinstance(output, tuple) or len(output) != 2:
        return None, None, "output must be a tuple `(H, tau)`"

    h, tau = output
    batch, n, _ = data.shape

    if not isinstance(h, torch.Tensor) or not isinstance(tau, torch.Tensor):
        return None, None, "H and tau must be tensors"
    if h.shape != (batch, n, n):
        return None, None, f"H shape mismatch: got {tuple(h.shape)}"
    if tau.shape != (batch, n):
        return None, None, f"tau shape mismatch: got {tuple(tau.shape)}"
    if h.dtype != torch.float32 or tau.dtype != torch.float32:
        return None, None, f"dtype mismatch: H={h.dtype}, tau={tau.dtype}"
    if h.device != data.device or tau.device != data.device:
        return None, None, f"device mismatch: H={h.device}, tau={tau.device}, data={data.device}"
    if not torch.isfinite(h).all().item() or not torch.isfinite(tau).all().item():
        return None, None, "H or tau contains NaN/Inf"

    try:
        q = torch.linalg.householder_product(h, tau)
        if not torch.isfinite(q).all().item():
            return None, None, "materialized Q contains NaN/Inf"
    except Exception as exc:
        return None, None, f"householder_product failed: {type(exc).__name__}: {exc}"

    return h, tau, None


def batch_count(spec: dict[str, Any]) -> int:
    batch = int(spec["batch"])
    n = int(spec["n"])
    bytes_per_input = batch * n * n * 4
    if bytes_per_input <= 0:
        return 1
    return max(
        1,
        min(MAX_ITERATIONS_PER_BENCHMARK, BENCHMARK_INPUT_BYTES_TARGET // bytes_per_input),
    )


def load_submission(path: str | Path):
    ensure_official_on_path()
    path = Path(path).resolve()
    submission_dir = str(path.parent)
    official_dir = str(OFFICIAL_DIR)
    if submission_dir != official_dir:
        if submission_dir in sys.path:
            sys.path.remove(submission_dir)
        sys.path.insert(1, submission_dir)
    module_name = f"submission_under_test_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load submission from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "custom_kernel"):
        raise AttributeError(f"{path} does not define custom_kernel(data)")
    return module.custom_kernel


def require_cuda(torch_module) -> None:
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmarking; run this on the B200 box")


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "nogit"


def driver_version() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()[0].strip()
    except Exception:
        return "unknown"


def environment_info(torch_module) -> dict[str, Any]:
    info: dict[str, Any] = {
        "gpu": "unavailable",
        "driver": driver_version(),
        "cuda": getattr(torch_module.version, "cuda", None),
        "torch": torch_module.__version__,
        "git_hash": git_hash(),
    }
    if torch_module.cuda.is_available():
        index = torch_module.cuda.current_device()
        info["gpu"] = torch_module.cuda.get_device_name(index)
        info["capability"] = torch_module.cuda.get_device_capability(index)
    return info


def append_jsonl(path: str | Path, rows: list[dict[str, Any]] | dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, dict):
        rows = [rows]
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
