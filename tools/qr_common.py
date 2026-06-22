from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_DIR = ROOT / "official"
BENCHMARK_INPUT_BYTES_TARGET = 256 * 1024 * 1024
MAX_ITERATIONS_PER_BENCHMARK = 50

CANDIDATE_RUNTIME_ENV_KEYS = (
    "FAST_QR_DISABLE_ROUTE_CACHE",
    "FAST_QR_DISABLE_STRUCTURED_ROUTES",
    "FAST_QR_DISABLE_DENSE_TAIL",
    "FAST_QR_DISABLE_DATA_DEPENDENT_ROUTES",
    "FAST_QR_OUTPUT_WORKSPACE_CACHE",
    "FAST_QR_DISABLE_OUTPUT_WORKSPACE_CACHE",
    "FAST_QR_TRUST_SAMPLED_STRUCTURED_GUARDS",
    "FAST_QR_DISABLE_B200_TRUST_SAMPLED_STRUCTURED_GUARDS",
    "FAST_QR_STRUCTURED_ROUTES_BEFORE_CUDA",
    "FAST_QR_DISABLE_B200_DEFAULT_BLOCKED_CUDA",
    "FAST_QR_DISABLE_B200_DEFAULT_STRUCTURED_BEFORE_CUDA",
    "FAST_QR_ENABLE_B200_DEFAULT_STRUCTURED_BEFORE_CUDA",
    "FAST_QR_QR512_STRUCTURED_BEFORE_CUDA",
    "FAST_QR_QR1024_STRUCTURED_BEFORE_CUDA",
    "FAST_QR_DENSE_TAIL_CUT",
    "FAST_QR_DENSE_TAIL_CUT_512",
    "FAST_QR_DENSE_TAIL_CUT_1024",
    "FAST_QR_DENSE_TAIL_CUT_2048",
    "FAST_QR_DENSE_TAIL_CUT_4096",
    "FAST_QR_QR512_TAIL_CUT",
    "FAST_QR_QR1024_TAIL_CUT",
    "FAST_QR_QR2048_TAIL_CUT",
    "FAST_QR_QR4096_TAIL_CUT",
    "FAST_QR_DENSE_TAIL_THRESHOLD",
    "FAST_QR_DENSE_TAIL_THRESHOLD_512",
    "FAST_QR_DENSE_TAIL_THRESHOLD_1024",
    "FAST_QR_DENSE_TAIL_THRESHOLD_2048",
    "FAST_QR_DENSE_TAIL_THRESHOLD_4096",
    "FAST_QR_QR512_TAIL_THRESHOLD",
    "FAST_QR_QR1024_TAIL_THRESHOLD",
    "FAST_QR_QR2048_TAIL_THRESHOLD",
    "FAST_QR_QR4096_TAIL_THRESHOLD",
    "FAST_QR_DENSE_TAIL_FORCE",
    "FAST_QR_DENSE_TAIL_FORCE_512",
    "FAST_QR_DENSE_TAIL_FORCE_1024",
    "FAST_QR_DENSE_TAIL_FORCE_2048",
    "FAST_QR_DENSE_TAIL_FORCE_4096",
    "FAST_QR_QR512_TAIL_FORCE",
    "FAST_QR_QR1024_TAIL_FORCE",
    "FAST_QR_QR2048_TAIL_FORCE",
    "FAST_QR_QR4096_TAIL_FORCE",
    "FAST_QR_MIXED_DENSE_TAIL_CUT",
    "FAST_QR_MIXED_DENSE_TAIL_CUT_512",
    "FAST_QR_MIXED_DENSE_TAIL_CUT_1024",
    "FAST_QR_MIXED_DENSE_TAIL_THRESHOLD",
    "FAST_QR_MIXED_DENSE_TAIL_THRESHOLD_512",
    "FAST_QR_MIXED_DENSE_TAIL_THRESHOLD_1024",
    "FAST_QR_DISABLE_QR32_CUDA",
    "FAST_QR_DISABLE_QR176_CUDA",
    "FAST_QR_DISABLE_QR352_CUDA",
    "FAST_QR_DISABLE_QR512_CUDA",
    "FAST_QR_DISABLE_QR1024_CUDA",
    "FAST_QR_DISABLE_QR512_BLOCKED_CUDA",
    "FAST_QR_DISABLE_QR1024_BLOCKED_CUDA",
    "FAST_QR_DISABLE_QR2048_BLOCKED_CUDA",
    "FAST_QR_DISABLE_QR4096_BLOCKED_CUDA",
    "FAST_QR_ENABLE_QR512_BLOCKED_CUDA",
    "FAST_QR_ENABLE_QR1024_BLOCKED_CUDA",
    "FAST_QR_ENABLE_QR2048_BLOCKED_CUDA",
    "FAST_QR_ENABLE_QR4096_BLOCKED_CUDA",
    "FAST_QR_DISABLE_BLOCKED_AUTO_POLICY",
    "FAST_QR_ENABLE_BLOCKED_AUTO_POLICY",
    "FAST_QR_BLOCKED_AUTO_GROUPS",
    "FAST_QR_DISABLE_BLOCKED_AUTO_GROUPS",
    "FAST_QR_ENABLE_BLOCKED_AUTO_GROUPS",
    "FAST_QR_DISABLE_QR512_BLOCKED_AUTO_POLICY",
    "FAST_QR_ENABLE_QR512_BLOCKED_AUTO_POLICY",
    "FAST_QR_DISABLE_QR1024_BLOCKED_AUTO_POLICY",
    "FAST_QR_ENABLE_QR1024_BLOCKED_AUTO_POLICY",
    "FAST_QR_DISABLE_QR2048_BLOCKED_AUTO_POLICY",
    "FAST_QR_ENABLE_QR2048_BLOCKED_AUTO_POLICY",
    "FAST_QR_DISABLE_QR4096_BLOCKED_AUTO_POLICY",
    "FAST_QR_ENABLE_QR4096_BLOCKED_AUTO_POLICY",
    "FAST_QR_QR512_BLOCKED_AUTO_GROUPS",
    "FAST_QR_QR1024_BLOCKED_AUTO_GROUPS",
    "FAST_QR_QR2048_BLOCKED_AUTO_GROUPS",
    "FAST_QR_QR4096_BLOCKED_AUTO_GROUPS",
    "FAST_QR_QR512_AUTO_GROUPS",
    "FAST_QR_QR1024_AUTO_GROUPS",
    "FAST_QR_QR2048_AUTO_GROUPS",
    "FAST_QR_QR4096_AUTO_GROUPS",
    "FAST_QR_DISABLE_QR512_BLOCKED_AUTO_GROUPS",
    "FAST_QR_ENABLE_QR512_BLOCKED_AUTO_GROUPS",
    "FAST_QR_DISABLE_QR1024_BLOCKED_AUTO_GROUPS",
    "FAST_QR_ENABLE_QR1024_BLOCKED_AUTO_GROUPS",
    "FAST_QR_DISABLE_QR2048_BLOCKED_AUTO_GROUPS",
    "FAST_QR_ENABLE_QR2048_BLOCKED_AUTO_GROUPS",
    "FAST_QR_DISABLE_QR4096_BLOCKED_AUTO_GROUPS",
    "FAST_QR_ENABLE_QR4096_BLOCKED_AUTO_GROUPS",
    "FAST_QR_REQUIRE_QR32_CUDA",
    "FAST_QR_REQUIRE_QR176_CUDA",
    "FAST_QR_REQUIRE_QR352_CUDA",
    "FAST_QR_REQUIRE_QR512_CUDA",
    "FAST_QR_REQUIRE_QR1024_CUDA",
    "FAST_QR_REQUIRE_QR512_BLOCKED_CUDA",
    "FAST_QR_REQUIRE_QR1024_BLOCKED_CUDA",
    "FAST_QR_REQUIRE_QR2048_BLOCKED_CUDA",
    "FAST_QR_REQUIRE_QR4096_BLOCKED_CUDA",
    "FAST_QR_QR32_EXTRA_CUDA_CFLAGS",
    "FAST_QR_QR176_EXTRA_CUDA_CFLAGS",
    "FAST_QR_QR352_EXTRA_CUDA_CFLAGS",
    "FAST_QR_QR512_EXTRA_CUDA_CFLAGS",
    "FAST_QR_QR1024_EXTRA_CUDA_CFLAGS",
    "FAST_QR_QR2048_EXTRA_CUDA_CFLAGS",
    "FAST_QR_QR4096_EXTRA_CUDA_CFLAGS",
    "FAST_QR_QR512_BLOCKED_EXTRA_CUDA_CFLAGS",
    "FAST_QR_QR1024_BLOCKED_EXTRA_CUDA_CFLAGS",
    "FAST_QR_QR2048_BLOCKED_EXTRA_CUDA_CFLAGS",
    "FAST_QR_QR4096_BLOCKED_EXTRA_CUDA_CFLAGS",
    "FAST_QR_QR352_PANEL_B",
    "FAST_QR_QR512_PANEL_B",
    "FAST_QR_QR1024_PANEL_B",
    "FAST_QR_QR2048_PANEL_B",
    "FAST_QR_QR4096_PANEL_B",
    "FAST_QR_QR512_BLOCKED_PANEL_B",
    "FAST_QR_QR1024_BLOCKED_PANEL_B",
    "FAST_QR_QR2048_BLOCKED_PANEL_B",
    "FAST_QR_QR4096_BLOCKED_PANEL_B",
    "FAST_QR_QR512_TILE_N",
    "FAST_QR_QR1024_TILE_N",
    "FAST_QR_QR2048_TILE_N",
    "FAST_QR_QR4096_TILE_N",
    "FAST_QR_QR512_BLOCKED_TILE_N",
    "FAST_QR_QR1024_BLOCKED_TILE_N",
    "FAST_QR_QR2048_BLOCKED_TILE_N",
    "FAST_QR_QR4096_BLOCKED_TILE_N",
    "FAST_QR_BLOCKED_COMPACT_WY_TILE_COLS",
    "FAST_QR_QR512_COMPACT_WY_TILE_COLS",
    "FAST_QR_QR1024_COMPACT_WY_TILE_COLS",
    "FAST_QR_QR2048_COMPACT_WY_TILE_COLS",
    "FAST_QR_QR4096_COMPACT_WY_TILE_COLS",
    "FAST_QR_QR512_BLOCKED_COMPACT_WY_TILE_COLS",
    "FAST_QR_QR1024_BLOCKED_COMPACT_WY_TILE_COLS",
    "FAST_QR_QR2048_BLOCKED_COMPACT_WY_TILE_COLS",
    "FAST_QR_QR4096_BLOCKED_COMPACT_WY_TILE_COLS",
    "FAST_QR_BLOCKED_CTAS_PER_MATRIX",
    "FAST_QR_QR512_CTAS_PER_MATRIX",
    "FAST_QR_QR1024_CTAS_PER_MATRIX",
    "FAST_QR_QR2048_CTAS_PER_MATRIX",
    "FAST_QR_QR4096_CTAS_PER_MATRIX",
    "FAST_QR_QR512_BLOCKED_CTAS_PER_MATRIX",
    "FAST_QR_QR1024_BLOCKED_CTAS_PER_MATRIX",
    "FAST_QR_QR2048_BLOCKED_CTAS_PER_MATRIX",
    "FAST_QR_QR4096_BLOCKED_CTAS_PER_MATRIX",
    "FAST_QR_BLOCKED_CTA_SCHEDULE",
    "FAST_QR_QR512_CTA_SCHEDULE",
    "FAST_QR_QR1024_CTA_SCHEDULE",
    "FAST_QR_QR2048_CTA_SCHEDULE",
    "FAST_QR_QR4096_CTA_SCHEDULE",
    "FAST_QR_QR512_BLOCKED_CTA_SCHEDULE",
    "FAST_QR_QR1024_BLOCKED_CTA_SCHEDULE",
    "FAST_QR_QR2048_BLOCKED_CTA_SCHEDULE",
    "FAST_QR_QR4096_BLOCKED_CTA_SCHEDULE",
    "FAST_QR_BLOCKED_POLICY_SAMPLE_ROWS",
    "FAST_QR_QR512_POLICY_SAMPLE_ROWS",
    "FAST_QR_QR1024_POLICY_SAMPLE_ROWS",
    "FAST_QR_QR2048_POLICY_SAMPLE_ROWS",
    "FAST_QR_QR4096_POLICY_SAMPLE_ROWS",
    "FAST_QR_QR512_BLOCKED_POLICY_SAMPLE_ROWS",
    "FAST_QR_QR1024_BLOCKED_POLICY_SAMPLE_ROWS",
    "FAST_QR_QR2048_BLOCKED_POLICY_SAMPLE_ROWS",
    "FAST_QR_QR4096_BLOCKED_POLICY_SAMPLE_ROWS",
    "FAST_QR_BLOCKED_POLICY_FULL_SCAN",
    "FAST_QR_QR512_POLICY_FULL_SCAN",
    "FAST_QR_QR1024_POLICY_FULL_SCAN",
    "FAST_QR_QR2048_POLICY_FULL_SCAN",
    "FAST_QR_QR4096_POLICY_FULL_SCAN",
    "FAST_QR_QR512_BLOCKED_POLICY_FULL_SCAN",
    "FAST_QR_QR1024_BLOCKED_POLICY_FULL_SCAN",
    "FAST_QR_QR2048_BLOCKED_POLICY_FULL_SCAN",
    "FAST_QR_QR4096_BLOCKED_POLICY_FULL_SCAN",
    "FAST_QR_QR352_UPDATE_MODE",
    "FAST_QR_QR512_UPDATE_MODE",
    "FAST_QR_QR1024_UPDATE_MODE",
    "FAST_QR_QR2048_UPDATE_MODE",
    "FAST_QR_QR4096_UPDATE_MODE",
    "FAST_QR_QR512_BLOCKED_UPDATE_MODE",
    "FAST_QR_QR1024_BLOCKED_UPDATE_MODE",
    "FAST_QR_QR2048_BLOCKED_UPDATE_MODE",
    "FAST_QR_QR4096_BLOCKED_UPDATE_MODE",
    "FAST_QR_QR352_PRECISION_MODE",
    "FAST_QR_QR512_PRECISION_MODE",
    "FAST_QR_QR1024_PRECISION_MODE",
    "FAST_QR_QR2048_PRECISION_MODE",
    "FAST_QR_QR4096_PRECISION_MODE",
    "FAST_QR_QR512_BLOCKED_PRECISION_MODE",
    "FAST_QR_QR1024_BLOCKED_PRECISION_MODE",
    "FAST_QR_QR2048_BLOCKED_PRECISION_MODE",
    "FAST_QR_QR4096_BLOCKED_PRECISION_MODE",
    "FAST_QR_QR352_PANEL_REFRESH_MODE",
    "FAST_QR_QR512_PANEL_REFRESH_MODE",
    "FAST_QR_QR1024_PANEL_REFRESH_MODE",
    "FAST_QR_QR2048_PANEL_REFRESH_MODE",
    "FAST_QR_QR4096_PANEL_REFRESH_MODE",
    "FAST_QR_QR512_BLOCKED_PANEL_REFRESH_MODE",
    "FAST_QR_QR1024_BLOCKED_PANEL_REFRESH_MODE",
    "FAST_QR_QR2048_BLOCKED_PANEL_REFRESH_MODE",
    "FAST_QR_QR4096_BLOCKED_PANEL_REFRESH_MODE",
    "FAST_QR_QR352_R_MAINTENANCE_MODE",
    "FAST_QR_QR512_R_MAINTENANCE_MODE",
    "FAST_QR_QR1024_R_MAINTENANCE_MODE",
    "FAST_QR_QR2048_R_MAINTENANCE_MODE",
    "FAST_QR_QR4096_R_MAINTENANCE_MODE",
    "FAST_QR_QR512_BLOCKED_R_MAINTENANCE_MODE",
    "FAST_QR_QR1024_BLOCKED_R_MAINTENANCE_MODE",
    "FAST_QR_QR2048_BLOCKED_R_MAINTENANCE_MODE",
    "FAST_QR_QR4096_BLOCKED_R_MAINTENANCE_MODE",
    "FAST_QR_BLOCKED_PANEL_REFRESH_PERIOD",
    "FAST_QR_BLOCKED_R_MAINTENANCE_PERIOD",
    "FAST_QR_QR512_PANEL_REFRESH_PERIOD",
    "FAST_QR_QR1024_PANEL_REFRESH_PERIOD",
    "FAST_QR_QR2048_PANEL_REFRESH_PERIOD",
    "FAST_QR_QR4096_PANEL_REFRESH_PERIOD",
    "FAST_QR_QR512_BLOCKED_PANEL_REFRESH_PERIOD",
    "FAST_QR_QR1024_BLOCKED_PANEL_REFRESH_PERIOD",
    "FAST_QR_QR2048_BLOCKED_PANEL_REFRESH_PERIOD",
    "FAST_QR_QR4096_BLOCKED_PANEL_REFRESH_PERIOD",
    "FAST_QR_QR512_R_MAINTENANCE_PERIOD",
    "FAST_QR_QR1024_R_MAINTENANCE_PERIOD",
    "FAST_QR_QR2048_R_MAINTENANCE_PERIOD",
    "FAST_QR_QR4096_R_MAINTENANCE_PERIOD",
    "FAST_QR_QR512_BLOCKED_R_MAINTENANCE_PERIOD",
    "FAST_QR_QR1024_BLOCKED_R_MAINTENANCE_PERIOD",
    "FAST_QR_QR2048_BLOCKED_R_MAINTENANCE_PERIOD",
    "FAST_QR_QR4096_BLOCKED_R_MAINTENANCE_PERIOD",
    "FAST_QR_BLOCKED_SYNC_FREE_AUTO_POLICY",
    "FAST_QR_ENABLE_BLOCKED_SYNC_FREE_AUTO_POLICY",
    "FAST_QR_QR512_SYNC_FREE_AUTO_POLICY",
    "FAST_QR_QR1024_SYNC_FREE_AUTO_POLICY",
    "FAST_QR_QR2048_SYNC_FREE_AUTO_POLICY",
    "FAST_QR_QR4096_SYNC_FREE_AUTO_POLICY",
    "FAST_QR_QR512_BLOCKED_SYNC_FREE_AUTO_POLICY",
    "FAST_QR_QR1024_BLOCKED_SYNC_FREE_AUTO_POLICY",
    "FAST_QR_QR2048_BLOCKED_SYNC_FREE_AUTO_POLICY",
    "FAST_QR_QR4096_BLOCKED_SYNC_FREE_AUTO_POLICY",
    "FAST_QR_QR32_THREADS_PER_CTA",
    "FAST_QR_QR32_WARPS_PER_CTA",
    "FAST_QR_QR176_THREADS_PER_CTA",
    "FAST_QR_QR176_WARPS_PER_CTA",
    "FAST_QR_QR176_UPDATE_COL_TILE",
    "FAST_QR_QR352_THREADS_PER_CTA",
    "FAST_QR_QR352_WARPS_PER_CTA",
    "FAST_QR_QR352_UPDATE_COL_TILE",
    "FAST_QR_QR512_THREADS_PER_CTA",
    "FAST_QR_QR512_WARPS_PER_CTA",
    "FAST_QR_QR512_UPDATE_COL_TILE",
    "FAST_QR_QR1024_THREADS_PER_CTA",
    "FAST_QR_QR1024_WARPS_PER_CTA",
    "FAST_QR_QR1024_UPDATE_COL_TILE",
    "FAST_QR_QR2048_THREADS_PER_CTA",
    "FAST_QR_QR2048_WARPS_PER_CTA",
    "FAST_QR_QR4096_THREADS_PER_CTA",
    "FAST_QR_QR4096_WARPS_PER_CTA",
    "FAST_QR_QR512_BLOCKED_THREADS_PER_CTA",
    "FAST_QR_QR512_BLOCKED_WARPS_PER_CTA",
    "FAST_QR_QR1024_BLOCKED_THREADS_PER_CTA",
    "FAST_QR_QR1024_BLOCKED_WARPS_PER_CTA",
    "FAST_QR_QR2048_BLOCKED_THREADS_PER_CTA",
    "FAST_QR_QR2048_BLOCKED_WARPS_PER_CTA",
    "FAST_QR_QR4096_BLOCKED_THREADS_PER_CTA",
    "FAST_QR_QR4096_BLOCKED_WARPS_PER_CTA",
)

TUNER_RUNTIME_ENV_KEYS = (
    "FAST_QR_OCCUPANCY_REGISTERS_PER_SM",
    "FAST_QR_OCCUPANCY_SHARED_BYTES_PER_SM",
    "FAST_QR_OCCUPANCY_MAX_THREADS_PER_SM",
    "FAST_QR_OCCUPANCY_MAX_CTAS_PER_SM",
)

TRACKED_RUNTIME_ENV_KEYS = (
    "TORCH_CUDA_ARCH_LIST",
    "CUDA_VISIBLE_DEVICES",
    *CANDIDATE_RUNTIME_ENV_KEYS,
    *TUNER_RUNTIME_ENV_KEYS,
)


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


def parse_popcorn_seed_tokens(value: str | None, default: list[int | None] | None = None) -> list[int | None]:
    if not value:
        return list(default or [])
    seeds: list[int | None] = []
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        lowered = token.lower()
        if lowered in {"public", "none", "base"}:
            seeds.append(None)
        else:
            seeds.append(int(token))
    return seeds


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


def git_full_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "nogit"


def git_status_porcelain() -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return ""


def git_dirty() -> bool:
    return bool(git_status_porcelain().strip())


def official_upstream_commit() -> str:
    path = OFFICIAL_DIR / "UPSTREAM_COMMIT"
    try:
        return path.read_text().strip()
    except OSError:
        return "unknown"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def file_provenance(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    resolved = resolved.resolve()
    return {
        "path": relative_path(resolved),
        "sha256": file_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def repo_provenance(include_status: bool = True) -> dict[str, Any]:
    status = git_status_porcelain()
    out: dict[str, Any] = {
        "git_hash": git_hash(),
        "git_full_hash": git_full_hash(),
        "git_dirty": bool(status.strip()),
        "official_upstream_commit": official_upstream_commit(),
    }
    if include_status:
        out["git_status_porcelain"] = status.splitlines()
    return out


def driver_version() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()[0].strip()
    except Exception:
        return "unknown"


def tracked_runtime_env(
    env: dict[str, str] | os._Environ[str] | None = None,
    *,
    include_absent: bool = False,
) -> dict[str, str | None]:
    source = os.environ if env is None else env
    out: dict[str, str | None] = {}
    for key in TRACKED_RUNTIME_ENV_KEYS:
        value = source.get(key)
        if value is not None or include_absent:
            out[key] = value
    return out


def tracked_candidate_env(
    env: dict[str, str] | os._Environ[str] | None = None,
    *,
    include_absent: bool = False,
) -> dict[str, str | None]:
    source = os.environ if env is None else env
    out: dict[str, str | None] = {}
    for key in CANDIDATE_RUNTIME_ENV_KEYS:
        value = source.get(key)
        if value is not None or include_absent:
            out[key] = value
    return out


def environment_info(torch_module) -> dict[str, Any]:
    info: dict[str, Any] = {
        "gpu": "unavailable",
        "driver": driver_version(),
        "cuda": getattr(torch_module.version, "cuda", None),
        "torch": torch_module.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "tracked_env": tracked_runtime_env(),
        "candidate_env": tracked_candidate_env(),
        **repo_provenance(include_status=False),
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
