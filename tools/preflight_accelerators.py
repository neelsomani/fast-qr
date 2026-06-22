from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

from qr_common import ROOT, OFFICIAL_DIR, append_jsonl, ensure_official_on_path
from large_kernel_plan import MODE_CHOICES, PRESETS as LARGE_KERNEL_PRESETS, generate_configs as generate_large_kernel_configs
from tune_tail_policy import load_config_rows, parse_inline_config


ensure_official_on_path()
from reference import check_implementation, generate_input  # noqa: E402


DEFAULT_CONFIG = {"name": "default", "env": {}}

ACCELERATOR_FAMILY_SPECS = {
    "qr512_cuda": [
        ("dense", {"batch": 4, "n": 512, "cond": 2, "seed": 1029}),
        ("mixed", {"batch": 4, "n": 512, "cond": 2, "seed": 770001, "case": "mixed"}),
        ("rankdef", {"batch": 4, "n": 512, "cond": 0, "seed": 770003, "case": "rankdef"}),
        ("clustered", {"batch": 4, "n": 512, "cond": 0, "seed": 770004, "case": "clustered"}),
    ],
    "qr512_blocked_cuda": [
        ("dense", {"batch": 4, "n": 512, "cond": 2, "seed": 1029}),
        ("mixed", {"batch": 4, "n": 512, "cond": 2, "seed": 770001, "case": "mixed"}),
        ("rankdef", {"batch": 4, "n": 512, "cond": 0, "seed": 770003, "case": "rankdef"}),
        ("clustered", {"batch": 4, "n": 512, "cond": 0, "seed": 770004, "case": "clustered"}),
    ],
    "qr1024_cuda": [
        ("dense", {"batch": 2, "n": 1024, "cond": 2, "seed": 75342}),
        ("mixed", {"batch": 4, "n": 1024, "cond": 2, "seed": 770002, "case": "mixed"}),
        ("nearrank", {"batch": 2, "n": 1024, "cond": 0, "seed": 770005, "case": "nearrank"}),
    ],
    "qr1024_blocked_cuda": [
        ("dense", {"batch": 2, "n": 1024, "cond": 2, "seed": 75342}),
        ("mixed", {"batch": 4, "n": 1024, "cond": 2, "seed": 770002, "case": "mixed"}),
        ("nearrank", {"batch": 2, "n": 1024, "cond": 0, "seed": 770005, "case": "nearrank"}),
    ],
    "qr2048_blocked_cuda": [
        ("dense", {"batch": 1, "n": 2048, "cond": 1, "seed": 224466}),
        ("rankdef", {"batch": 1, "n": 2048, "cond": 0, "seed": 224467, "case": "rankdef"}),
        ("mixed", {"batch": 1, "n": 2048, "cond": 2, "seed": 224468, "case": "mixed"}),
    ],
    "qr4096_blocked_cuda": [
        ("dense", {"batch": 1, "n": 4096, "cond": 1, "seed": 32412}),
        ("upper", {"batch": 1, "n": 4096, "cond": 0, "seed": 75343, "case": "upper"}),
    ],
}


ACCELERATOR_FAMILY_SPECS["qr512_blocked_cuda_auto"] = ACCELERATOR_FAMILY_SPECS["qr512_blocked_cuda"]
ACCELERATOR_FAMILY_SPECS["qr1024_blocked_cuda_auto"] = ACCELERATOR_FAMILY_SPECS["qr1024_blocked_cuda"]
ACCELERATOR_FAMILY_SPECS["qr2048_blocked_cuda_auto"] = ACCELERATOR_FAMILY_SPECS["qr2048_blocked_cuda"]
ACCELERATOR_FAMILY_SPECS["qr4096_blocked_cuda_auto"] = ACCELERATOR_FAMILY_SPECS["qr4096_blocked_cuda"]


@contextmanager
def temporary_env(overrides: dict[str, str]):
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[str(key)] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def compile_info_accelerator_name(accelerator: str) -> str:
    if accelerator.endswith("_cuda_auto"):
        return accelerator[: -len("_auto")]
    return accelerator


def accelerator_compile_info(candidate, accelerator: str) -> dict[str, Any]:
    compile_accelerator = compile_info_accelerator_name(accelerator)
    cflags_fn = getattr(candidate, f"_{compile_accelerator}_extra_cuda_cflags", None)
    build_key_fn = getattr(candidate, f"_{compile_accelerator}_extension_build_key", None)
    name_fn = getattr(candidate, f"_{compile_accelerator}_extension_name", None)
    threads_fn = getattr(candidate, f"_{compile_accelerator}_threads_per_cta", None)
    warps_fn = getattr(candidate, f"_{compile_accelerator}_warps_per_cta", None)
    panel_b_fn = getattr(candidate, f"_{compile_accelerator}_panel_b", None)
    update_col_tile_fn = getattr(candidate, f"_{compile_accelerator}_update_col_tile", None)
    tile_n_fn = getattr(candidate, f"_{compile_accelerator}_tile_n", None)
    ctas_per_matrix_fn = getattr(candidate, f"_{compile_accelerator}_ctas_per_matrix", None)
    compact_wy_tile_cols_fn = getattr(candidate, f"_{compile_accelerator}_compact_wy_tile_cols", None)
    update_mode_fn = getattr(candidate, f"_{compile_accelerator}_update_mode", None)
    precision_mode_fn = getattr(candidate, f"_{compile_accelerator}_precision_mode", None)
    panel_refresh_mode_fn = getattr(candidate, f"_{compile_accelerator}_panel_refresh_mode", None)
    r_maintenance_mode_fn = getattr(candidate, f"_{compile_accelerator}_r_maintenance_mode", None)
    return {
        "compile_info_accelerator": compile_accelerator,
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "extra_cuda_cflags": cflags_fn() if callable(cflags_fn) else None,
        "extra_cuda_cflags_env": os.environ.get(
            f"FAST_QR_{compile_accelerator.upper().replace('_CUDA', '')}_EXTRA_CUDA_CFLAGS"
        ),
        "threads_per_cta": threads_fn() if callable(threads_fn) else None,
        "warps_per_cta": warps_fn() if callable(warps_fn) else None,
        "panel_b": panel_b_fn() if callable(panel_b_fn) else None,
        "update_col_tile": update_col_tile_fn() if callable(update_col_tile_fn) else None,
        "tile_n": tile_n_fn() if callable(tile_n_fn) else None,
        "ctas_per_matrix": ctas_per_matrix_fn() if callable(ctas_per_matrix_fn) else None,
        "compact_wy_tile_cols": compact_wy_tile_cols_fn() if callable(compact_wy_tile_cols_fn) else None,
        "update_mode": update_mode_fn() if callable(update_mode_fn) else None,
        "precision_mode": precision_mode_fn() if callable(precision_mode_fn) else None,
        "panel_refresh_mode": panel_refresh_mode_fn() if callable(panel_refresh_mode_fn) else None,
        "r_maintenance_mode": r_maintenance_mode_fn() if callable(r_maintenance_mode_fn) else None,
        "extension_build_key": build_key_fn() if callable(build_key_fn) else None,
        "extension_name": name_fn() if callable(name_fn) else None,
    }


def load_candidate_module(path: str | Path):
    ensure_official_on_path()
    path = Path(path).resolve()
    submission_dir = str(path.parent)
    if submission_dir != str(OFFICIAL_DIR):
        if submission_dir in sys.path:
            sys.path.remove(submission_dir)
        sys.path.insert(1, submission_dir)
    module_name = f"accelerator_preflight_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load submission from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def cuda_qr_preflight(
    candidate,
    *,
    accelerator: str,
    entrypoint: str,
    require_env: str,
    extension_attr: str,
    error_attr: str,
    spec: dict[str, int],
    allow_fallback: bool,
    preflight_case: str = "smoke",
) -> dict[str, Any]:
    previous_require = os.environ.get(require_env)
    if not allow_fallback:
        os.environ[require_env] = "1"
    compile_info = accelerator_compile_info(candidate, accelerator)

    try:
        data = generate_input(**spec)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        try:
            output = getattr(candidate, entrypoint)(data.clone())
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            good, message = check_implementation(data, output)
            extension_loaded = bool(getattr(candidate, extension_attr, None) is not None)
            extension_error = getattr(candidate, error_attr, None)
            return {
                "accelerator": accelerator,
                "allow_fallback": allow_fallback,
                "preflight_case": preflight_case,
                "extension_loaded": extension_loaded,
                "extension_error": extension_error,
                "ok": bool(good and (allow_fallback or extension_loaded)),
                "message": message,
                "spec": spec,
                **compile_info,
            }
        except Exception as exc:
            return {
                "accelerator": accelerator,
                "allow_fallback": allow_fallback,
                "preflight_case": preflight_case,
                "extension_loaded": bool(getattr(candidate, extension_attr, None) is not None),
                "extension_error": getattr(candidate, error_attr, None),
                "ok": False,
                "message": f"{type(exc).__name__}: {exc}",
                "spec": spec,
                **compile_info,
            }
    finally:
        if previous_require is None:
            os.environ.pop(require_env, None)
        else:
            os.environ[require_env] = previous_require


def qr32_preflight(candidate, allow_fallback: bool, spec: dict[str, Any] | None = None, preflight_case: str = "smoke") -> dict[str, Any]:
    return cuda_qr_preflight(
        candidate,
        accelerator="qr32_cuda",
        entrypoint="qr32_fast",
        require_env="FAST_QR_REQUIRE_QR32_CUDA",
        extension_attr="_QR32_CUDA_EXTENSION",
        error_attr="_QR32_CUDA_EXTENSION_ERROR",
        spec=spec or {"batch": 20, "n": 32, "cond": 1, "seed": 53124},
        allow_fallback=allow_fallback,
        preflight_case=preflight_case,
    )


def qr176_preflight(candidate, allow_fallback: bool, spec: dict[str, Any] | None = None, preflight_case: str = "smoke") -> dict[str, Any]:
    return cuda_qr_preflight(
        candidate,
        accelerator="qr176_cuda",
        entrypoint="qr176_fast",
        require_env="FAST_QR_REQUIRE_QR176_CUDA",
        extension_attr="_QR176_CUDA_EXTENSION",
        error_attr="_QR176_CUDA_EXTENSION_ERROR",
        spec=spec or {"batch": 40, "n": 176, "cond": 1, "seed": 3321},
        allow_fallback=allow_fallback,
        preflight_case=preflight_case,
    )


def qr352_preflight(candidate, allow_fallback: bool, spec: dict[str, Any] | None = None, preflight_case: str = "smoke") -> dict[str, Any]:
    return cuda_qr_preflight(
        candidate,
        accelerator="qr352_cuda",
        entrypoint="qr352_fast",
        require_env="FAST_QR_REQUIRE_QR352_CUDA",
        extension_attr="_QR352_CUDA_EXTENSION",
        error_attr="_QR352_CUDA_EXTENSION_ERROR",
        spec=spec or {"batch": 4, "n": 352, "cond": 1, "seed": 1200},
        allow_fallback=allow_fallback,
        preflight_case=preflight_case,
    )


def qr512_preflight(candidate, allow_fallback: bool, spec: dict[str, Any] | None = None, preflight_case: str = "smoke") -> dict[str, Any]:
    return cuda_qr_preflight(
        candidate,
        accelerator="qr512_cuda",
        entrypoint="qr512_cuda_fast",
        require_env="FAST_QR_REQUIRE_QR512_CUDA",
        extension_attr="_QR512_CUDA_EXTENSION",
        error_attr="_QR512_CUDA_EXTENSION_ERROR",
        spec=spec or {"batch": 1, "n": 512, "cond": 2, "seed": 5120},
        allow_fallback=allow_fallback,
        preflight_case=preflight_case,
    )


def qr512_blocked_preflight(
    candidate,
    allow_fallback: bool,
    spec: dict[str, Any] | None = None,
    preflight_case: str = "smoke",
) -> dict[str, Any]:
    return cuda_qr_preflight(
        candidate,
        accelerator="qr512_blocked_cuda",
        entrypoint="qr512_blocked_cuda_fast",
        require_env="FAST_QR_REQUIRE_QR512_BLOCKED_CUDA",
        extension_attr="_QR512_BLOCKED_CUDA_EXTENSION",
        error_attr="_QR512_BLOCKED_CUDA_EXTENSION_ERROR",
        spec=spec or {"batch": 1, "n": 512, "cond": 2, "seed": 5120},
        allow_fallback=allow_fallback,
        preflight_case=preflight_case,
    )


def qr512_blocked_auto_preflight(
    candidate,
    allow_fallback: bool,
    spec: dict[str, Any] | None = None,
    preflight_case: str = "smoke",
) -> dict[str, Any]:
    return cuda_qr_preflight(
        candidate,
        accelerator="qr512_blocked_cuda_auto",
        entrypoint="qr512_blocked_cuda_auto_fast",
        require_env="FAST_QR_REQUIRE_QR512_BLOCKED_CUDA",
        extension_attr="_QR512_BLOCKED_CUDA_EXTENSION",
        error_attr="_QR512_BLOCKED_CUDA_EXTENSION_ERROR",
        spec=spec or {"batch": 1, "n": 512, "cond": 2, "seed": 5120},
        allow_fallback=allow_fallback,
        preflight_case=preflight_case,
    )


def qr1024_preflight(candidate, allow_fallback: bool, spec: dict[str, Any] | None = None, preflight_case: str = "smoke") -> dict[str, Any]:
    return cuda_qr_preflight(
        candidate,
        accelerator="qr1024_cuda",
        entrypoint="qr1024_cuda_fast",
        require_env="FAST_QR_REQUIRE_QR1024_CUDA",
        extension_attr="_QR1024_CUDA_EXTENSION",
        error_attr="_QR1024_CUDA_EXTENSION_ERROR",
        spec=spec or {"batch": 1, "n": 1024, "cond": 2, "seed": 10240},
        allow_fallback=allow_fallback,
        preflight_case=preflight_case,
    )


def qr1024_blocked_preflight(
    candidate,
    allow_fallback: bool,
    spec: dict[str, Any] | None = None,
    preflight_case: str = "smoke",
) -> dict[str, Any]:
    return cuda_qr_preflight(
        candidate,
        accelerator="qr1024_blocked_cuda",
        entrypoint="qr1024_blocked_cuda_fast",
        require_env="FAST_QR_REQUIRE_QR1024_BLOCKED_CUDA",
        extension_attr="_QR1024_BLOCKED_CUDA_EXTENSION",
        error_attr="_QR1024_BLOCKED_CUDA_EXTENSION_ERROR",
        spec=spec or {"batch": 1, "n": 1024, "cond": 2, "seed": 10240},
        allow_fallback=allow_fallback,
        preflight_case=preflight_case,
    )


def qr1024_blocked_auto_preflight(
    candidate,
    allow_fallback: bool,
    spec: dict[str, Any] | None = None,
    preflight_case: str = "smoke",
) -> dict[str, Any]:
    return cuda_qr_preflight(
        candidate,
        accelerator="qr1024_blocked_cuda_auto",
        entrypoint="qr1024_blocked_cuda_auto_fast",
        require_env="FAST_QR_REQUIRE_QR1024_BLOCKED_CUDA",
        extension_attr="_QR1024_BLOCKED_CUDA_EXTENSION",
        error_attr="_QR1024_BLOCKED_CUDA_EXTENSION_ERROR",
        spec=spec or {"batch": 1, "n": 1024, "cond": 2, "seed": 10240},
        allow_fallback=allow_fallback,
        preflight_case=preflight_case,
    )


def qr2048_blocked_preflight(
    candidate,
    allow_fallback: bool,
    spec: dict[str, Any] | None = None,
    preflight_case: str = "smoke",
) -> dict[str, Any]:
    return cuda_qr_preflight(
        candidate,
        accelerator="qr2048_blocked_cuda",
        entrypoint="qr2048_blocked_cuda_fast",
        require_env="FAST_QR_REQUIRE_QR2048_BLOCKED_CUDA",
        extension_attr="_QR2048_BLOCKED_CUDA_EXTENSION",
        error_attr="_QR2048_BLOCKED_CUDA_EXTENSION_ERROR",
        spec=spec or {"batch": 1, "n": 2048, "cond": 1, "seed": 20480},
        allow_fallback=allow_fallback,
        preflight_case=preflight_case,
    )


def qr2048_blocked_auto_preflight(
    candidate,
    allow_fallback: bool,
    spec: dict[str, Any] | None = None,
    preflight_case: str = "smoke",
) -> dict[str, Any]:
    return cuda_qr_preflight(
        candidate,
        accelerator="qr2048_blocked_cuda_auto",
        entrypoint="qr2048_blocked_cuda_auto_fast",
        require_env="FAST_QR_REQUIRE_QR2048_BLOCKED_CUDA",
        extension_attr="_QR2048_BLOCKED_CUDA_EXTENSION",
        error_attr="_QR2048_BLOCKED_CUDA_EXTENSION_ERROR",
        spec=spec or {"batch": 1, "n": 2048, "cond": 1, "seed": 20480},
        allow_fallback=allow_fallback,
        preflight_case=preflight_case,
    )


def qr4096_blocked_preflight(
    candidate,
    allow_fallback: bool,
    spec: dict[str, Any] | None = None,
    preflight_case: str = "smoke",
) -> dict[str, Any]:
    return cuda_qr_preflight(
        candidate,
        accelerator="qr4096_blocked_cuda",
        entrypoint="qr4096_blocked_cuda_fast",
        require_env="FAST_QR_REQUIRE_QR4096_BLOCKED_CUDA",
        extension_attr="_QR4096_BLOCKED_CUDA_EXTENSION",
        error_attr="_QR4096_BLOCKED_CUDA_EXTENSION_ERROR",
        spec=spec or {"batch": 1, "n": 4096, "cond": 1, "seed": 40960},
        allow_fallback=allow_fallback,
        preflight_case=preflight_case,
    )


def qr4096_blocked_auto_preflight(
    candidate,
    allow_fallback: bool,
    spec: dict[str, Any] | None = None,
    preflight_case: str = "smoke",
) -> dict[str, Any]:
    return cuda_qr_preflight(
        candidate,
        accelerator="qr4096_blocked_cuda_auto",
        entrypoint="qr4096_blocked_cuda_auto_fast",
        require_env="FAST_QR_REQUIRE_QR4096_BLOCKED_CUDA",
        extension_attr="_QR4096_BLOCKED_CUDA_EXTENSION",
        error_attr="_QR4096_BLOCKED_CUDA_EXTENSION_ERROR",
        spec=spec or {"batch": 1, "n": 4096, "cond": 1, "seed": 40960},
        allow_fallback=allow_fallback,
        preflight_case=preflight_case,
    )


ACCELERATOR_PREFLIGHTS = {
    "qr32_cuda": qr32_preflight,
    "qr176_cuda": qr176_preflight,
    "qr352_cuda": qr352_preflight,
    "qr512_cuda": qr512_preflight,
    "qr512_blocked_cuda": qr512_blocked_preflight,
    "qr512_blocked_cuda_auto": qr512_blocked_auto_preflight,
    "qr1024_cuda": qr1024_preflight,
    "qr1024_blocked_cuda": qr1024_blocked_preflight,
    "qr1024_blocked_cuda_auto": qr1024_blocked_auto_preflight,
    "qr2048_blocked_cuda": qr2048_blocked_preflight,
    "qr2048_blocked_cuda_auto": qr2048_blocked_auto_preflight,
    "qr4096_blocked_cuda": qr4096_blocked_preflight,
    "qr4096_blocked_cuda_auto": qr4096_blocked_auto_preflight,
}

DEFAULT_ACCELERATOR_PREFLIGHTS = [
    "qr32_cuda",
    "qr176_cuda",
    "qr352_cuda",
    "qr512_cuda",
    "qr512_blocked_cuda",
    "qr1024_cuda",
    "qr1024_blocked_cuda",
    "qr2048_blocked_cuda",
    "qr4096_blocked_cuda",
]

LARGE_KERNEL_SHAPE_ACCELERATORS = {
    "qr512": ["qr512_blocked_cuda_auto"],
    "qr1024": ["qr1024_blocked_cuda_auto"],
    "qr2048": ["qr2048_blocked_cuda_auto"],
    "qr4096": ["qr4096_blocked_cuda_auto"],
}


def selected_accelerator_names(raw: str, shape_label: str | None = None) -> list[str]:
    token = (raw or "auto").strip().lower()
    if token == "auto":
        return list(LARGE_KERNEL_SHAPE_ACCELERATORS.get(shape_label or "", DEFAULT_ACCELERATOR_PREFLIGHTS))
    if token == "all":
        return list(ACCELERATOR_PREFLIGHTS)

    names = [piece.strip() for piece in raw.split(",") if piece.strip()]
    unknown = [name for name in names if name not in ACCELERATOR_PREFLIGHTS]
    if unknown:
        choices = ", ".join(["auto", "all", *ACCELERATOR_PREFLIGHTS])
        raise ValueError(f"unknown accelerator(s): {', '.join(unknown)}; expected one of: {choices}")
    if not names:
        raise ValueError("at least one accelerator must be selected")
    return names


def _normalize_config_rows(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for config in configs:
        name = str(config.get("name", "")).strip()
        if not name:
            raise ValueError("preflight config row is missing a non-empty name")
        if name in seen:
            raise ValueError(f"duplicate preflight config name: {name}")
        env = config.get("env", {})
        if not isinstance(env, dict):
            raise ValueError(f"preflight config {name!r} is missing an env object")
        seen.add(name)
        rows.append({"name": name, "env": {str(key): str(value) for key, value in env.items()}})
    return rows


def load_preflight_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    if args.config_jsonl:
        path = ROOT / args.config_jsonl if not Path(args.config_jsonl).is_absolute() else Path(args.config_jsonl)
        configs.extend(load_config_rows(path))
    configs.extend(parse_inline_config(raw) for raw in args.config)
    if args.large_kernel_plan_shape_label:
        configs.extend(
            {
                "name": row["name"],
                "env": row["env"],
            }
            for row in generate_large_kernel_configs(
                args.large_kernel_plan_shape_label,
                max_configs=args.large_kernel_plan_max_configs,
                env_prefix=args.large_kernel_plan_env_prefix,
                mode=args.large_kernel_plan_mode,
            )
        )
    if not configs:
        configs = [DEFAULT_CONFIG]
    return _normalize_config_rows(configs)


def preflight_rows_for_config(
    submission: Path,
    config: dict[str, Any],
    *,
    config_index: int,
    accelerator_names: list[str],
    allow_fallback: bool,
    family_cases: bool,
) -> list[dict[str, Any]]:
    env = {str(key): str(value) for key, value in config.get("env", {}).items()}
    with temporary_env(env):
        candidate = load_candidate_module(submission)
        rows = []
        for name in accelerator_names:
            specs = ACCELERATOR_FAMILY_SPECS.get(name) if family_cases else None
            if not specs:
                specs = [("smoke", None)]
            for preflight_case, spec in specs:
                row = ACCELERATOR_PREFLIGHTS[name](candidate, allow_fallback, spec=spec, preflight_case=preflight_case)
                row.update(
                    {
                        "config_index": config_index,
                        "config_name": config["name"],
                        "config_env": env,
                        "config_env_keys": sorted(env),
                        "family_cases": bool(family_cases),
                    }
                )
                rows.append(row)
        return rows


def run_preflight_matrix(
    submission: Path,
    configs: list[dict[str, Any]],
    accelerator_names: list[str],
    *,
    allow_fallback: bool,
    family_cases: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, config in enumerate(configs):
        rows.extend(
            preflight_rows_for_config(
                submission,
                config,
                config_index=index,
                accelerator_names=accelerator_names,
                allow_fallback=allow_fallback,
                family_cases=family_cases,
            )
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preflight optional candidate accelerators before B200 timing.",
        allow_abbrev=False,
    )
    parser.add_argument("--submission", default="submissions/candidate.py")
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument(
        "--accelerators",
        default="auto",
        help="Comma-separated accelerators, 'all', or 'auto'. Auto selects the matching QR512/QR1024 accelerator for a large-kernel plan, otherwise all.",
    )
    parser.add_argument("--config-jsonl", default=None, help="JSONL config rows with {'name', 'env'} fields.")
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Inline config as name:KEY=VALUE,KEY2=VALUE2. May be repeated.",
    )
    parser.add_argument("--large-kernel-plan-shape-label", choices=sorted(LARGE_KERNEL_PRESETS), default=None)
    parser.add_argument("--large-kernel-plan-mode", choices=MODE_CHOICES, default="current-candidate")
    parser.add_argument("--large-kernel-plan-max-configs", type=int, default=32)
    parser.add_argument("--large-kernel-plan-env-prefix", default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--family-cases",
        action="store_true",
        help=(
            "For QR512/QR1024 config matrices, preflight representative dense, mixed, "
            "and structured benchmark profiles for each env row instead of only one smoke case."
        ),
    )
    args = parser.parse_args()

    submission = ROOT / args.submission if not Path(args.submission).is_absolute() else Path(args.submission)
    configs = load_preflight_configs(args)
    accelerator_names = selected_accelerator_names(args.accelerators, args.large_kernel_plan_shape_label)
    rows = run_preflight_matrix(
        submission,
        configs,
        accelerator_names,
        allow_fallback=args.allow_fallback,
        family_cases=args.family_cases,
    )
    summary = {
        "ok": all(row["ok"] for row in rows),
        "family_cases": bool(args.family_cases),
        "num_accelerators": len(accelerator_names),
        "num_configs": len(configs),
        "num_config_accelerator_rows": len(rows),
        "num_preflight_case_rows": len(rows),
        "preflight_cases_by_accelerator": {
            name: [case for case, _ in ACCELERATOR_FAMILY_SPECS.get(name, [("smoke", None)])]
            if args.family_cases
            else ["smoke"]
            for name in accelerator_names
        },
        "num_passed": sum(1 for row in rows if row["ok"]),
        "num_failed": sum(1 for row in rows if not row["ok"]),
        "accelerators": accelerator_names,
        "configs": configs,
        "summary": True,
    }
    rows.append(summary)

    for row in rows:
        if args.json:
            print(json.dumps(row, sort_keys=True), flush=True)
        else:
            print(json.dumps(row, sort_keys=True), flush=True)

    if args.out:
        append_jsonl(ROOT / args.out if not Path(args.out).is_absolute() else args.out, rows)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
