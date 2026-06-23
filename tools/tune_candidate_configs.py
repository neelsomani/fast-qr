from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from qr_common import (
    CANDIDATE_RUNTIME_ENV_KEYS,
    ROOT,
    TUNER_RUNTIME_ENV_KEYS,
    append_jsonl,
    file_provenance,
    repo_provenance,
)
from summarize_suite import load_jsonl
from tune_tail_policy import (
    config_prefix as tail_config_prefix,
    load_config_rows,
    merged_env,
    parse_inline_config,
    run_step,
    should_skip_benchmark_after_correctness,
    summarize_benchmark,
)


DEFAULT_CONFIGS: list[dict[str, Any]] = [{"name": "default", "env": {}}]


def config_prefix(index: int, name: str) -> str:
    base = tail_config_prefix(index, name)
    if len(base) <= 96:
        return base
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._-")
    slug = slug[:48].rstrip("._-") or "config"
    return f"{index + 1:03d}_{slug}_{digest}"

PTXAS_USED_RE = re.compile(r"ptxas info\s*:\s*Used\s+(\d+)\s+registers(?P<rest>.*)")
PTXAS_FUNCTION_RE = re.compile(r"ptxas info\s*:\s*Function properties for\s+(.+)")
PTXAS_STACK_RE = re.compile(
    r"(\d+)\s+bytes stack frame,\s*(\d+)\s+bytes spill stores,\s*(\d+)\s+bytes spill loads"
)
PTXAS_BYTES_RE = re.compile(r"(\d+)\s+bytes\s+([A-Za-z_][A-Za-z0-9_\[\]]*)")

DEFAULT_OCCUPANCY_REGISTERS_PER_SM = 65536
DEFAULT_OCCUPANCY_SHARED_BYTES_PER_SM = 227 * 1024
DEFAULT_OCCUPANCY_MAX_THREADS_PER_SM = 2048
DEFAULT_OCCUPANCY_MAX_CTAS_PER_SM = 32

CURRENT_CANDIDATE_CONSUMED_ENV_KEYS = set(CANDIDATE_RUNTIME_ENV_KEYS)

TUNER_CONSUMED_ENV_KEYS = set(TUNER_RUNTIME_ENV_KEYS)

CUDA_ROUTE_BYPASSED_ENV_BY_DISABLE_KEY = {
    "FAST_QR_DISABLE_QR512_CUDA": {
        "FAST_QR_QR512_TAIL_CUT",
        "FAST_QR_QR512_TAIL_THRESHOLD",
        "FAST_QR_QR512_TAIL_FORCE",
        "FAST_QR_DENSE_TAIL_CUT_512",
        "FAST_QR_DENSE_TAIL_THRESHOLD_512",
        "FAST_QR_DENSE_TAIL_FORCE_512",
        "FAST_QR_DENSE_TAIL_FORCE",
        "FAST_QR_MIXED_DENSE_TAIL_CUT_512",
        "FAST_QR_MIXED_DENSE_TAIL_THRESHOLD_512",
    },
    "FAST_QR_DISABLE_QR1024_CUDA": {
        "FAST_QR_QR1024_TAIL_CUT",
        "FAST_QR_QR1024_TAIL_THRESHOLD",
        "FAST_QR_QR1024_TAIL_FORCE",
        "FAST_QR_DENSE_TAIL_CUT_1024",
        "FAST_QR_DENSE_TAIL_THRESHOLD_1024",
        "FAST_QR_DENSE_TAIL_FORCE_1024",
        "FAST_QR_DENSE_TAIL_FORCE",
        "FAST_QR_MIXED_DENSE_TAIL_CUT_1024",
        "FAST_QR_MIXED_DENSE_TAIL_THRESHOLD_1024",
    },
}

STRUCTURED_BEFORE_CUDA_ENV_BY_DISABLE_KEY = {
    "FAST_QR_DISABLE_QR512_CUDA": "FAST_QR_QR512_STRUCTURED_BEFORE_CUDA",
    "FAST_QR_DISABLE_QR1024_CUDA": "FAST_QR_QR1024_STRUCTURED_BEFORE_CUDA",
}

BLOCKED_ROUTE_ENV_BY_DISABLE_KEY = {
    "FAST_QR_DISABLE_QR512_CUDA": (
        "FAST_QR_ENABLE_QR512_BLOCKED_CUDA",
        "FAST_QR_REQUIRE_QR512_BLOCKED_CUDA",
        "FAST_QR_DISABLE_QR512_BLOCKED_CUDA",
    ),
    "FAST_QR_DISABLE_QR1024_CUDA": (
        "FAST_QR_ENABLE_QR1024_BLOCKED_CUDA",
        "FAST_QR_REQUIRE_QR1024_BLOCKED_CUDA",
        "FAST_QR_DISABLE_QR1024_BLOCKED_CUDA",
    ),
}


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _csv_tokens(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def _sanitize_env_part(value: str) -> str:
    out = "".join(char.upper() if char.isalnum() else "_" for char in value)
    out = "_".join(part for part in out.split("_") if part)
    return out or "GLOBAL"


def _sanitize_name_part(value: str) -> str:
    out = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    return out.strip("_") or "x"


def _truthy_env_value(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_positive_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _grid_axis(values: list[str]) -> list[str | None]:
    return values if values else [None]


def _arg_csv(args: argparse.Namespace, name: str) -> list[str]:
    return _csv_tokens(getattr(args, name, ""))


def grid_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    axes = {
        "PANEL_B": _arg_csv(args, "panel_widths"),
        "UPDATE_MODE": _arg_csv(args, "update_modes"),
        "PRECISION_MODE": _arg_csv(args, "precision_modes"),
        "TILE_M": _arg_csv(args, "tile_ms"),
        "TILE_N": _arg_csv(args, "tile_ns"),
        "COMPACT_WY_TILE_COLS": _arg_csv(args, "compact_wy_tile_cols"),
        "WARPS_PER_CTA": _arg_csv(args, "warps_per_cta"),
        "CTAS_PER_MATRIX": _arg_csv(args, "ctas_per_matrix"),
        "CTA_SCHEDULE": _arg_csv(args, "cta_schedules"),
        "SYNC_FREE_AUTO_POLICY": _arg_csv(args, "sync_free_auto_policy"),
        "BLOCKED_AUTO_GROUPS": _arg_csv(args, "auto_policy_groups"),
        "CLUSTER_SIZE": _arg_csv(args, "cluster_sizes"),
        "TAIL_CUT": _arg_csv(args, "tail_cuts"),
        "TAIL_THRESHOLD": _arg_csv(args, "tail_thresholds"),
        "TAIL_FORCE": _arg_csv(args, "tail_force"),
        "PANEL_REFRESH": _arg_csv(args, "panel_refreshes"),
        "PANEL_REFRESH_MODE": _arg_csv(args, "panel_refresh_modes"),
        "R_MAINTENANCE_MODE": _arg_csv(args, "r_maintenance_modes"),
        "STRUCTURED_BEFORE_CUDA": _arg_csv(args, "structured_before_cuda"),
    }
    active = {key: values for key, values in axes.items() if values}
    if not active:
        return []

    label = _sanitize_env_part(args.shape_label)
    prefix = args.env_prefix or f"FAST_QR_{label}"
    configs = []
    keys = list(active)
    for values in itertools.product(*(_grid_axis(active[key]) for key in keys)):
        env = {f"{prefix}_{key}": str(value) for key, value in zip(keys, values) if value is not None}
        name_bits = [_sanitize_name_part(args.shape_label)]
        name_bits.extend(_sanitize_name_part(f"{key.lower()}_{value}") for key, value in zip(keys, values) if value is not None)
        configs.append({"name": "__".join(name_bits), "env": env})
    return configs


def load_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = [] if args.no_default else list(DEFAULT_CONFIGS)
    if args.config_jsonl:
        path = ROOT / args.config_jsonl if not Path(args.config_jsonl).is_absolute() else Path(args.config_jsonl)
        configs.extend(load_config_rows(path))
    configs.extend(parse_inline_config(raw) for raw in args.config)
    configs.extend(grid_configs(args))

    seen = set()
    out = []
    for config in configs:
        name = str(config["name"])
        if name in seen:
            raise ValueError(f"duplicate config name: {name}")
        seen.add(name)
        out.append({"name": name, "env": {str(key): str(value) for key, value in config.get("env", {}).items()}})
    return out


def resource_cflags_env(args: argparse.Namespace) -> str:
    if args.resource_cflags_env:
        return str(args.resource_cflags_env)
    label = _sanitize_env_part(args.shape_label)
    return f"FAST_QR_{label}_EXTRA_CUDA_CFLAGS"


def with_resource_metric_flags(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    if not args.collect_resource_metrics:
        return config
    env = dict(config.get("env", {}))
    key = resource_cflags_env(args)
    existing = str(env.get(key, "")).strip()
    flag = "--ptxas-options=-v"
    parts = existing.split()
    if flag not in parts:
        parts.append(flag)
    env[key] = " ".join(parts).strip()
    return {"name": config["name"], "env": env}


def env_consumption(env: dict[str, str]) -> dict[str, Any]:
    keys = sorted(str(key) for key in env)
    candidate_consumed = [key for key in keys if key in CURRENT_CANDIDATE_CONSUMED_ENV_KEYS]
    tuner_consumed = [key for key in keys if key in TUNER_CONSUMED_ENV_KEYS]
    consumed = sorted(set(candidate_consumed) | set(tuner_consumed))
    inert = [key for key in keys if key not in consumed]
    cuda_route_bypassed = []
    for disable_key, bypassed_keys in CUDA_ROUTE_BYPASSED_ENV_BY_DISABLE_KEY.items():
        if str(env.get(disable_key, "")) == "1":
            continue
        structured_key = STRUCTURED_BEFORE_CUDA_ENV_BY_DISABLE_KEY[disable_key]
        if _truthy_env_value(env.get("FAST_QR_STRUCTURED_ROUTES_BEFORE_CUDA")) or _truthy_env_value(
            env.get(structured_key)
        ):
            continue
        blocked_enable_key, blocked_require_key, blocked_disable_key = BLOCKED_ROUTE_ENV_BY_DISABLE_KEY[disable_key]
        if (
            not _truthy_env_value(env.get(blocked_disable_key))
            and (_truthy_env_value(env.get(blocked_enable_key)) or _truthy_env_value(env.get(blocked_require_key)))
        ):
            continue
        cuda_route_bypassed.extend(key for key in keys if key in bypassed_keys)
    cuda_route_bypassed = sorted(set(cuda_route_bypassed))
    notes = []
    if inert:
        notes.append(
            "inert keys are recorded for future blocked kernels but are not read by the current candidate or tuner"
        )
    if cuda_route_bypassed:
        notes.append(
            "cuda-route-bypassed keys are read only if the current QR512/QR1024 CUDA route is disabled or unavailable"
        )
    if any(
        key.endswith("_TAIL_CUT") or key.endswith("_TAIL_THRESHOLD") or key.endswith("_TAIL_FORCE")
        for key in candidate_consumed
    ):
        notes.append("shape-family tail keys feed the current candidate's dense-tail policy aliases")
    if any(key.endswith("_WARPS_PER_CTA") or key.endswith("_THREADS_PER_CTA") for key in candidate_consumed):
        notes.append("current CUDA kernels consume CTA thread-count keys by source specialization")
    if any(key.endswith("_STRUCTURED_BEFORE_CUDA") or key == "FAST_QR_STRUCTURED_ROUTES_BEFORE_CUDA" for key in candidate_consumed):
        notes.append("structured-before-cuda keys let classifier/structured routes run before the robust CUDA fallback")
    if any(key.endswith("_PANEL_REFRESH_MODE") or key.endswith("_R_MAINTENANCE_MODE") for key in candidate_consumed):
        notes.append("blocked repair-mode keys source-specialize prefix panel refresh and panel-prefix R maintenance")
    if any(key.endswith("_PANEL_REFRESH_MODE") or key.endswith("_R_MAINTENANCE_MODE") for key in inert):
        notes.append(
            "inert repair-mode keys were not recognized by the current candidate; check shape label or env prefix"
        )
    return {
        "candidate_consumed_env_keys": candidate_consumed,
        "tuner_consumed_env_keys": tuner_consumed,
        "consumed_env_keys": consumed,
        "inert_env_keys": inert,
        "cuda_route_bypassed_env_keys": cuda_route_bypassed,
        "has_inert_env": bool(inert),
        "has_cuda_route_bypassed_env": bool(cuda_route_bypassed),
        "notes": notes,
    }


def command_plan(args: argparse.Namespace, out_dir: Path, config: dict[str, Any], index: int) -> list[dict[str, Any]]:
    prefix = config_prefix(index, config["name"])
    correctness_out = out_dir / f"{prefix}_correctness.jsonl"
    benchmark_out = out_dir / f"{prefix}_benchmark.jsonl"
    python = sys.executable

    steps: list[dict[str, Any]] = []
    if not args.skip_correctness:
        cmd = [
            python,
            "tools/seed_sweep.py",
            "--submission",
            args.submission,
            "--cases",
            args.correctness_cases,
            "--indices",
            args.correctness_indices,
            "--popcorn-seeds",
            args.popcorn_seeds,
            "--max-factor-scaled",
            f"{args.max_factor_scaled:g}",
            "--max-orth-scaled",
            f"{args.max_orth_scaled:g}",
            "--record-env",
            "--out",
            str(correctness_out),
        ]
        if not args.skip_diagnostics:
            cmd.append("--diagnose")
        steps.append({"step": "correctness", "out": str(correctness_out), "cmd": cmd})

    if not args.skip_benchmark:
        cmd = [
            python,
            "tools/bench_local.py",
            "--submission",
            args.submission,
            "--cases",
            args.benchmark_cases,
            "--repeats",
            str(args.repeats),
            "--recheck",
            "--record-env",
            "--out",
            str(benchmark_out),
        ]
        if args.benchmark_indices:
            cmd.extend(["--indices", args.benchmark_indices])
        if args.benchmark_popcorn_seed is not None:
            cmd.extend(["--popcorn-seed", str(args.benchmark_popcorn_seed)])
        if args.official_stopping:
            cmd.extend(["--official-stopping", "--leaderboard-warmup", "--max-time-ns", f"{args.max_time_ns:g}"])
        steps.append({"step": "benchmark", "out": str(benchmark_out), "cmd": cmd})
    return steps


def parse_ptxas_resource_metrics(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current_function: str | None = None
    pending_stack: dict[str, int] = {}
    for line in text.splitlines():
        function_match = PTXAS_FUNCTION_RE.search(line)
        if function_match:
            current_function = function_match.group(1).strip()
            pending_stack = {}
            continue

        stack_match = PTXAS_STACK_RE.search(line)
        if stack_match:
            pending_stack = {
                "stack_frame_bytes": int(stack_match.group(1)),
                "spill_store_bytes": int(stack_match.group(2)),
                "spill_load_bytes": int(stack_match.group(3)),
            }
            continue

        used_match = PTXAS_USED_RE.search(line)
        if not used_match:
            continue

        rest = used_match.group("rest") or ""
        entry: dict[str, Any] = {
            "function": current_function,
            "registers_per_thread": int(used_match.group(1)),
            **pending_stack,
        }
        cmem_bytes = 0
        for amount, label in PTXAS_BYTES_RE.findall(rest):
            value = int(amount)
            if label.startswith("cmem"):
                cmem_bytes += value
            else:
                entry[f"{label}_bytes"] = value
        if cmem_bytes:
            entry["cmem_bytes"] = cmem_bytes
        rows.append(entry)
        pending_stack = {}
    return rows


def _env_int_suffix(env: dict[str, str], suffix: str) -> int | None:
    for key in sorted(env):
        if key.endswith(suffix):
            parsed = _parse_positive_int(env[key])
            if parsed is not None:
                return parsed
    return None


def _occupancy_limits(env: dict[str, str]) -> dict[str, int]:
    return {
        "registers_per_sm": _parse_positive_int(env.get("FAST_QR_OCCUPANCY_REGISTERS_PER_SM"))
        or DEFAULT_OCCUPANCY_REGISTERS_PER_SM,
        "shared_bytes_per_sm": _parse_positive_int(env.get("FAST_QR_OCCUPANCY_SHARED_BYTES_PER_SM"))
        or DEFAULT_OCCUPANCY_SHARED_BYTES_PER_SM,
        "max_threads_per_sm": _parse_positive_int(env.get("FAST_QR_OCCUPANCY_MAX_THREADS_PER_SM"))
        or DEFAULT_OCCUPANCY_MAX_THREADS_PER_SM,
        "max_ctas_per_sm": _parse_positive_int(env.get("FAST_QR_OCCUPANCY_MAX_CTAS_PER_SM"))
        or DEFAULT_OCCUPANCY_MAX_CTAS_PER_SM,
    }


def add_occupancy_estimates(entries: list[dict[str, Any]], env: dict[str, str]) -> list[dict[str, Any]]:
    warps_per_cta = _env_int_suffix(env, "WARPS_PER_CTA")
    threads_per_cta = _env_int_suffix(env, "THREADS_PER_CTA") or (warps_per_cta * 32 if warps_per_cta else None)
    if not threads_per_cta:
        return [{**entry, "occupancy_estimate": None, "occupancy_reason": "threads_per_cta_unknown"} for entry in entries]

    limits = _occupancy_limits(env)
    max_warps_per_sm = max(1, limits["max_threads_per_sm"] // 32)
    out = []
    for entry in entries:
        registers = int(entry.get("registers_per_thread") or 0)
        smem = int(entry.get("smem_bytes") or 0)
        ctas_by_regs = limits["max_ctas_per_sm"]
        if registers > 0:
            ctas_by_regs = limits["registers_per_sm"] // max(1, registers * threads_per_cta)
        ctas_by_smem = limits["max_ctas_per_sm"]
        if smem > 0:
            ctas_by_smem = limits["shared_bytes_per_sm"] // smem
        ctas_by_threads = limits["max_threads_per_sm"] // threads_per_cta
        active_ctas = max(0, min(limits["max_ctas_per_sm"], ctas_by_regs, ctas_by_smem, ctas_by_threads))
        active_warps = active_ctas * max(1, threads_per_cta // 32)
        occupancy = min(1.0, active_warps / max_warps_per_sm)
        out.append(
            {
                **entry,
                "occupancy_estimate": {
                    "rough": True,
                    "active_ctas_per_sm": active_ctas,
                    "active_warps_per_sm": active_warps,
                    "occupancy": occupancy,
                    "threads_per_cta": threads_per_cta,
                    "limits": limits,
                },
            }
        )
    return out


def _step_chunks_by_config(log_text: str) -> dict[str, list[str]]:
    chunks: dict[str, list[str]] = {}
    current_prefix: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        if current_prefix is not None and buffer:
            chunks.setdefault(current_prefix, []).append("\n".join(buffer))
        buffer = []

    for line in log_text.splitlines():
        if line.startswith("$ "):
            flush()
            match = re.search(r"([0-9]{3}_[A-Za-z0-9_.-]+)_(?:correctness|benchmark)\.jsonl", line)
            current_prefix = match.group(1) if match else None
            buffer = [line]
        else:
            buffer.append(line)
    flush()
    return chunks


def resource_metrics_by_config(out_dir: Path, configs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    log_path = out_dir / "run.log"
    if not log_path.is_file():
        return {}
    chunks = _step_chunks_by_config(log_path.read_text(errors="replace"))
    out: dict[str, dict[str, Any]] = {}
    for index, config in enumerate(configs):
        prefix = config_prefix(index, config["name"])
        entries: list[dict[str, Any]] = []
        for chunk in chunks.get(prefix, []):
            entries.extend(parse_ptxas_resource_metrics(chunk))
        entries = add_occupancy_estimates(entries, config.get("env", {}))
        if not entries:
            out[config["name"]] = {
                "entries": [],
                "num_entries": 0,
                "source": "run.log",
                "available": False,
            }
            continue
        occupancies = [
            float(entry["occupancy_estimate"]["occupancy"])
            for entry in entries
            if isinstance(entry.get("occupancy_estimate"), dict)
        ]
        out[config["name"]] = {
            "entries": entries,
            "num_entries": len(entries),
            "source": "run.log",
            "available": True,
            "max_registers_per_thread": max(int(entry.get("registers_per_thread") or 0) for entry in entries),
            "max_smem_bytes": max(int(entry.get("smem_bytes") or 0) for entry in entries),
            "max_spill_store_bytes": max(int(entry.get("spill_store_bytes") or 0) for entry in entries),
            "max_spill_load_bytes": max(int(entry.get("spill_load_bytes") or 0) for entry in entries),
            "min_estimated_occupancy": min(occupancies) if occupancies else None,
        }
    return out


def summarize_correctness(path: Path) -> dict[str, Any] | None:
    rows = load_jsonl(path)
    if not rows:
        return None
    cases = [row for row in rows if not row.get("summary")]
    diagnostics = [row.get("diagnostics") for row in cases if isinstance(row.get("diagnostics"), dict)]
    factor_values = [float(row["factor_scaled_max"]) for row in diagnostics]
    orth_values = [float(row["orth_scaled_max"]) for row in diagnostics]
    return {
        "num_rows": len(cases),
        "num_failed": sum(1 for row in cases if not row.get("ok") or row.get("margin_ok") is False),
        "max_factor_scaled": max(factor_values) if factor_values else None,
        "max_orth_scaled": max(orth_values) if orth_values else None,
        "case_indices": sorted({int(row["case_index"]) for row in cases if row.get("case_index") is not None}),
        "popcorn_seeds": sorted({str(row.get("popcorn_seed")) for row in cases}),
    }


def summarize_run(out_dir: Path, configs: list[dict[str, Any]]) -> dict[str, Any]:
    resource_by_name = resource_metrics_by_config(out_dir, configs)
    results = []
    for index, config in enumerate(configs):
        prefix = config_prefix(index, config["name"])
        correctness = summarize_correctness(out_dir / f"{prefix}_correctness.jsonl")
        benchmark = summarize_benchmark(out_dir / f"{prefix}_benchmark.jsonl")
        results.append(
            {
                "name": config["name"],
                "env": config["env"],
                "env_consumption": env_consumption(config.get("env", {})),
                "correctness": correctness,
                "benchmark": benchmark,
                "resource_metrics": resource_by_name.get(config["name"]),
            }
        )
    ranked = sorted(
        [row for row in results if row.get("benchmark") and row["benchmark"].get("geomean_us")],
        key=lambda row: row["benchmark"]["geomean_us"],
    )
    return {
        "ok": True,
        "out_dir": str(out_dir),
        "objective": "minimize_geomean_us",
        "resource_metrics_source": "ptxas verbose output in run.log",
        "num_configs": len(results),
        "num_configs_with_inert_env": sum(
            1 for row in results if row["env_consumption"].get("has_inert_env")
        ),
        "num_configs_with_cuda_route_bypassed_env": sum(
            1 for row in results if row["env_consumption"].get("has_cuda_route_bypassed_env")
        ),
        "num_configs_with_resource_metrics": sum(
            1 for row in results if (row.get("resource_metrics") or {}).get("available")
        ),
        "num_failed_configs": sum(1 for row in results if row.get("correctness") and row["correctness"].get("num_failed", 0) > 0),
        "num_benchmarked_configs": sum(1 for row in results if row.get("benchmark") and row["benchmark"].get("geomean_us")),
        "results": results,
        "best": ranked[0] if ranked else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a generic B200 candidate config autotune grid with correctness gates before benchmarks.",
        allow_abbrev=False,
    )
    parser.add_argument("--submission", default="submissions/candidate.py")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--config-jsonl", default=None)
    parser.add_argument("--config", action="append", default=[], help='Add one config as "name:KEY=VALUE,KEY2=VALUE2".')
    parser.add_argument("--no-default", action="store_true", help="Do not include the default empty-env config.")
    parser.add_argument("--shape-label", default="global", help="Shape/family label used in generated config names/env keys.")
    parser.add_argument("--env-prefix", default=None, help="Env prefix for generated grid knobs. Defaults to FAST_QR_<SHAPE_LABEL>.")
    parser.add_argument("--panel-widths", default="")
    parser.add_argument("--update-modes", default="")
    parser.add_argument("--precision-modes", default="")
    parser.add_argument("--tile-ms", default="")
    parser.add_argument("--tile-ns", default="")
    parser.add_argument("--compact-wy-tile-cols", default="")
    parser.add_argument("--warps-per-cta", default="")
    parser.add_argument("--ctas-per-matrix", default="")
    parser.add_argument("--cta-schedules", default="")
    parser.add_argument("--sync-free-auto-policy", default="")
    parser.add_argument("--auto-policy-groups", default="")
    parser.add_argument("--cluster-sizes", default="")
    parser.add_argument("--tail-cuts", default="")
    parser.add_argument("--tail-thresholds", default="")
    parser.add_argument("--tail-force", default="")
    parser.add_argument("--panel-refreshes", default="")
    parser.add_argument("--panel-refresh-modes", default="")
    parser.add_argument("--r-maintenance-modes", default="")
    parser.add_argument("--structured-before-cuda", default="")
    parser.add_argument("--collect-resource-metrics", action="store_true")
    parser.add_argument(
        "--resource-cflags-env",
        default=None,
        help="Env var that receives --ptxas-options=-v. Defaults to FAST_QR_<SHAPE_LABEL>_EXTRA_CUDA_CFLAGS.",
    )
    parser.add_argument("--correctness-cases", default="cases/public_tests.txt")
    parser.add_argument("--correctness-indices", default="19,20,21")
    parser.add_argument("--benchmark-cases", default="cases/public_benchmarks.txt")
    parser.add_argument("--benchmark-indices", default="", help="Comma-separated benchmark case indexes. Defaults to all cases.")
    parser.add_argument("--popcorn-seeds", default="public,1,2,3")
    parser.add_argument("--benchmark-popcorn-seed", type=int, default=None)
    parser.add_argument("--max-factor-scaled", type=float, default=18.0)
    parser.add_argument("--max-orth-scaled", type=float, default=80.0)
    parser.add_argument("--skip-diagnostics", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--official-stopping", action="store_true")
    parser.add_argument("--max-time-ns", type=float, default=30e9)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--benchmark-failed-configs", action="store_true")
    parser.add_argument("--allow-failed-configs", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = ROOT / "results" / f"candidate_config_tune_{timestamp()}" if args.out_dir is None else Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    log_path = out_dir / "run.log"
    summary_path = out_dir / "summary.json"

    configs = load_configs(args)
    configs = [with_resource_metric_flags(args, config) for config in configs]
    env = os.environ.copy()
    submission = ROOT / args.submission if not Path(args.submission).is_absolute() else Path(args.submission)

    append_jsonl(
        manifest_path,
        {
            "event": "tune_start",
            "time": datetime.now().isoformat(),
            "dry_run": args.dry_run,
            "repo": repo_provenance(),
            "submission": file_provenance(submission),
            "args": vars(args),
        },
    )

    failed = False
    hard_failed = False
    for index, config in enumerate(configs):
        steps = command_plan(args, out_dir, config, index)
        append_jsonl(
            manifest_path,
            {
                "event": "config_plan",
                "index": index,
                "name": config["name"],
                "env": config["env"],
                "env_consumption": env_consumption(config.get("env", {})),
                "steps": steps,
            },
        )
        if args.dry_run:
            continue

        correctness_failed = False
        for step in steps:
            if step["step"] == "benchmark" and should_skip_benchmark_after_correctness(
                correctness_failed,
                args.benchmark_failed_configs,
            ):
                append_jsonl(
                    manifest_path,
                    {
                        "event": "step_skipped",
                        "config": config["name"],
                        "step": step["step"],
                        "out": step["out"],
                        "reason": "correctness_failed",
                    },
                )
                continue

            append_jsonl(manifest_path, {"event": "step_start", "config": config["name"], **step})
            code = run_step(step, merged_env(env, config["env"]), log_path)
            append_jsonl(manifest_path, {"event": "step_finish", "config": config["name"], "step": step["step"], "exit_code": code})
            if code != 0:
                failed = True
                expected_config_failure = False
                if step["step"] == "correctness":
                    correctness_failed = True
                    expected_config_failure = summarize_correctness(Path(step["out"])) is not None
                if not expected_config_failure:
                    hard_failed = True
                if hard_failed or args.fail_fast:
                    break
        if hard_failed or (failed and args.fail_fast):
            break

    summary = summarize_run(out_dir, configs)
    summary["dry_run"] = args.dry_run
    summary["ok"] = not failed
    summary["hard_failed"] = hard_failed
    summary["allow_failed_configs"] = args.allow_failed_configs
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    append_jsonl(manifest_path, {"event": "tune_finish", "ok": not failed, "summary": str(summary_path), "time": datetime.now().isoformat()})

    print(json.dumps(summary, sort_keys=True))
    if hard_failed:
        return 1
    if failed and not args.allow_failed_configs:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
