from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from qr_common import ROOT, format_case


PAIRS = {
    "public": ("baseline_geqrf_public.jsonl", "candidate_public.jsonl"),
    "official_style": ("baseline_geqrf_official_style.jsonl", "candidate_official_style.jsonl"),
    "smoke": ("baseline_geqrf_smoke.jsonl", "candidate_smoke.jsonl"),
}

GUARD_OVERHEAD_FILE = "candidate_guard_overhead_public.jsonl"
DEFAULT_CANDIDATE_FILE = "candidate_public.jsonl"
ABLATION_FILES = {
    "no_route_cache": "candidate_ablation_no_route_cache_public.jsonl",
    "cuda_first_structured_routes": "candidate_ablation_cuda_first_structured_routes_public.jsonl",
    "no_structured_routes": "candidate_ablation_no_structured_routes_public.jsonl",
    "no_dense_tail": "candidate_ablation_no_dense_tail_public.jsonl",
    "no_data_dependent_routes": "candidate_ablation_no_data_dependent_routes_public.jsonl",
    "no_qr512_qr1024_cuda": "candidate_ablation_no_qr512_qr1024_cuda_public.jsonl",
}
QUANTIZATION_SWEEP_FILE = "quantization_seed_sweep.jsonl"
MIXED_SWEEP_FILE = "mixed_seed_sweep.jsonl"
BLOCKED_QR_SWEEP_FILE = "blocked_qr_sweep.jsonl"
TAIL_POLICY_TUNE_SUMMARY = Path("tail_policy_tune") / "summary.json"
CANDIDATE_CONFIG_TUNE_SUMMARY = Path("candidate_config_tune") / "summary.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def benchmark_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        spec = row.get("spec")
        if row.get("summary") or not isinstance(spec, dict) or "mean_us" not in row:
            continue
        out[format_case(spec)] = row
    return out


def geomean(values: list[float]) -> float | None:
    positive = [value for value in values if value > 0.0]
    if not positive:
        return None
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def summarize_pair(name: str, suite_dir: Path, baseline_file: str, candidate_file: str) -> dict[str, Any] | None:
    baseline_path = suite_dir / baseline_file
    candidate_path = suite_dir / candidate_file
    if not baseline_path.is_file() or not candidate_path.is_file():
        return None

    baseline = benchmark_rows(load_jsonl(baseline_path))
    candidate = benchmark_rows(load_jsonl(candidate_path))
    common_specs = [spec for spec in baseline if spec in candidate]
    rows = []
    for spec in common_specs:
        base = baseline[spec]
        cand = candidate[spec]
        base_mean = float(base["mean_us"])
        cand_mean = float(cand["mean_us"])
        rows.append(
            {
                "spec": spec,
                "baseline_mean_us": base_mean,
                "candidate_mean_us": cand_mean,
                "speedup": base_mean / cand_mean if cand_mean > 0.0 else None,
                "candidate_ok": bool(cand.get("ok")),
                "baseline_ok": bool(base.get("ok")),
                "candidate_runs": cand.get("runs"),
                "baseline_runs": base.get("runs"),
            }
        )

    baseline_geomean = geomean([row["baseline_mean_us"] for row in rows])
    candidate_geomean = geomean([row["candidate_mean_us"] for row in rows])
    speedup = None
    if baseline_geomean is not None and candidate_geomean is not None and candidate_geomean > 0.0:
        speedup = baseline_geomean / candidate_geomean

    return {
        "name": name,
        "baseline_file": baseline_file,
        "candidate_file": candidate_file,
        "num_common_cases": len(rows),
        "baseline_geomean_us": baseline_geomean,
        "candidate_geomean_us": candidate_geomean,
        "geomean_speedup": speedup,
        "cases": rows,
    }


def summarize_guard_overhead(suite_dir: Path) -> dict[str, Any] | None:
    path = suite_dir / GUARD_OVERHEAD_FILE
    rows = load_jsonl(path)
    cases = []
    for row in rows:
        if "wall_us" not in row:
            continue
        guard_row = {
            "case_index": row.get("case_index"),
            "spec": row.get("spec", ""),
            "route": row.get("route", ""),
            "cold_wall_us": None if row.get("cold_wall_us") is None else float(row["cold_wall_us"]),
            "cold_cuda_us": None if row.get("cold_cuda_us") is None else float(row["cold_cuda_us"]),
            "wall_us": float(row["wall_us"]),
            "cuda_us": None if row.get("cuda_us") is None else float(row["cuda_us"]),
            "repeats": row.get("repeats"),
            "warmup": row.get("warmup"),
        }
        for key in [
            "case_metadata_available",
            "case_metadata_passed_to_submission",
            "case_info_source",
            "case_selection_info_sources",
            "shape_collision",
            "shape_only_case_selection",
            "shape_only_dispatch",
            "uses_tensor_values_for_dispatch",
            "uses_tensor_values_for_case_selection",
            "requires_tensor_guard_for_case_specific_path",
            "classifier_needed_for_case_specific_path",
            "classifier_needed_for_current_candidate",
            "classifier_on_current_hot_path",
            "dispatch_info_sources",
            "route_decision_source",
        ]:
            if key in row:
                guard_row[key] = row[key]
        cases.append(guard_row)
    if not cases:
        return None

    cuda_values = [row["cuda_us"] for row in cases if row["cuda_us"] is not None]
    cold_wall_values = [row["cold_wall_us"] for row in cases if row["cold_wall_us"] is not None]
    cold_cuda_values = [row["cold_cuda_us"] for row in cases if row["cold_cuda_us"] is not None]
    return {
        "file": GUARD_OVERHEAD_FILE,
        "num_cases": len(cases),
        "cold_wall_geomean_us": geomean(cold_wall_values),
        "cold_wall_max_us": max(cold_wall_values) if cold_wall_values else None,
        "cold_cuda_geomean_us": geomean(cold_cuda_values),
        "cold_cuda_max_us": max(cold_cuda_values) if cold_cuda_values else None,
        "wall_geomean_us": geomean([row["wall_us"] for row in cases]),
        "wall_max_us": max(row["wall_us"] for row in cases),
        "cuda_geomean_us": geomean(cuda_values),
        "cuda_max_us": max(cuda_values) if cuda_values else None,
        "cases": cases,
    }


def summarize_ablation(name: str, suite_dir: Path, ablation_file: str) -> dict[str, Any] | None:
    default_path = suite_dir / DEFAULT_CANDIDATE_FILE
    ablation_path = suite_dir / ablation_file
    if not default_path.is_file() or not ablation_path.is_file():
        return None

    default = benchmark_rows(load_jsonl(default_path))
    ablation = benchmark_rows(load_jsonl(ablation_path))
    common_specs = [spec for spec in default if spec in ablation]
    rows = []
    for spec in common_specs:
        default_row = default[spec]
        ablation_row = ablation[spec]
        default_mean = float(default_row["mean_us"])
        ablation_mean = float(ablation_row["mean_us"])
        rows.append(
            {
                "spec": spec,
                "default_mean_us": default_mean,
                "ablation_mean_us": ablation_mean,
                "ablation_over_default": ablation_mean / default_mean if default_mean > 0.0 else None,
                "default_ok": bool(default_row.get("ok")),
                "ablation_ok": bool(ablation_row.get("ok")),
                "default_runs": default_row.get("runs"),
                "ablation_runs": ablation_row.get("runs"),
            }
        )

    default_geomean = geomean([row["default_mean_us"] for row in rows])
    ablation_geomean = geomean([row["ablation_mean_us"] for row in rows])
    ratio = None
    if default_geomean is not None and ablation_geomean is not None and default_geomean > 0.0:
        ratio = ablation_geomean / default_geomean

    return {
        "name": name,
        "default_file": DEFAULT_CANDIDATE_FILE,
        "ablation_file": ablation_file,
        "num_common_cases": len(rows),
        "default_geomean_us": default_geomean,
        "ablation_geomean_us": ablation_geomean,
        "ablation_over_default": ratio,
        "cases": rows,
    }


def summarize_tune_summary(suite_dir: Path, relative_path: Path) -> dict[str, Any] | None:
    path = suite_dir / relative_path
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return {
            "file": str(relative_path),
            "parse_error": f"{type(exc).__name__}: {exc}",
        }

    results = []
    for row in raw.get("results", []):
        correctness = row.get("correctness") if isinstance(row.get("correctness"), dict) else {}
        benchmark = row.get("benchmark") if isinstance(row.get("benchmark"), dict) else {}
        resources = row.get("resource_metrics") if isinstance(row.get("resource_metrics"), dict) else {}
        consumption = row.get("env_consumption") if isinstance(row.get("env_consumption"), dict) else {}
        inert_keys = list(consumption.get("inert_env_keys") or [])
        cuda_route_bypassed_keys = list(consumption.get("cuda_route_bypassed_env_keys") or [])
        results.append(
            {
                "name": row.get("name"),
                "env": row.get("env", {}),
                "env_consumption": consumption,
                "candidate_consumed_env_keys": list(consumption.get("candidate_consumed_env_keys") or []),
                "tuner_consumed_env_keys": list(consumption.get("tuner_consumed_env_keys") or []),
                "inert_env_keys": inert_keys,
                "num_inert_env_keys": len(inert_keys),
                "cuda_route_bypassed_env_keys": cuda_route_bypassed_keys,
                "num_cuda_route_bypassed_env_keys": len(cuda_route_bypassed_keys),
                "correctness_num_failed": correctness.get("num_failed"),
                "correctness_max_factor_scaled": correctness.get("max_factor_scaled"),
                "correctness_max_orth_scaled": correctness.get("max_orth_scaled"),
                "benchmark_num_cases": benchmark.get("num_cases"),
                "benchmark_geomean_us": benchmark.get("geomean_us"),
                "resource_metrics_available": bool(resources.get("available")),
                "resource_max_registers_per_thread": resources.get("max_registers_per_thread"),
                "resource_max_smem_bytes": resources.get("max_smem_bytes"),
                "resource_max_spill_store_bytes": resources.get("max_spill_store_bytes"),
                "resource_max_spill_load_bytes": resources.get("max_spill_load_bytes"),
                "resource_min_estimated_occupancy": resources.get("min_estimated_occupancy"),
            }
        )

    best = raw.get("best") if isinstance(raw.get("best"), dict) else None
    best_benchmark = best.get("benchmark") if isinstance(best, dict) and isinstance(best.get("benchmark"), dict) else {}
    return {
        "file": str(relative_path),
        "ok": raw.get("ok"),
        "hard_failed": raw.get("hard_failed"),
        "allow_failed_configs": raw.get("allow_failed_configs"),
        "objective": raw.get("objective"),
        "num_configs": raw.get("num_configs", len(results)),
        "num_configs_with_inert_env": raw.get(
            "num_configs_with_inert_env",
            sum(1 for row in results if row.get("num_inert_env_keys")),
        ),
        "num_configs_with_cuda_route_bypassed_env": raw.get(
            "num_configs_with_cuda_route_bypassed_env",
            sum(1 for row in results if row.get("num_cuda_route_bypassed_env_keys")),
        ),
        "num_configs_with_resource_metrics": raw.get(
            "num_configs_with_resource_metrics",
            sum(1 for row in results if row.get("resource_metrics_available")),
        ),
        "num_failed_configs": sum(1 for row in results if (row.get("correctness_num_failed") or 0) > 0),
        "num_benchmarked_configs": sum(1 for row in results if row.get("benchmark_geomean_us") is not None),
        "best_name": best.get("name") if isinstance(best, dict) else None,
        "best_geomean_us": best_benchmark.get("geomean_us"),
        "route_order": summarize_route_order_configs(results),
        "results": results,
    }


def summarize_tail_policy_tune(suite_dir: Path) -> dict[str, Any] | None:
    return summarize_tune_summary(suite_dir, TAIL_POLICY_TUNE_SUMMARY)


def summarize_candidate_config_tune(suite_dir: Path) -> dict[str, Any] | None:
    return summarize_tune_summary(suite_dir, CANDIDATE_CONFIG_TUNE_SUMMARY)


def summarize_quantization_seed_sweep(suite_dir: Path) -> dict[str, Any] | None:
    rows = load_jsonl(suite_dir / QUANTIZATION_SWEEP_FILE)
    cases = [row for row in rows if not row.get("summary")]
    if not cases:
        return None
    summary = next((row for row in rows if row.get("summary")), {})
    by_experiment = []
    for experiment in sorted({str(row.get("experiment")) for row in cases if row.get("experiment")}):
        group = [row for row in cases if row.get("experiment") == experiment]
        factor_values = [float(row["factor_scaled_max"]) for row in group if isinstance(row.get("factor_scaled_max"), (int, float))]
        orth_values = [float(row["orth_scaled_max"]) for row in group if isinstance(row.get("orth_scaled_max"), (int, float))]
        by_experiment.append(
            {
                "experiment": experiment,
                "num_rows": len(group),
                "num_failed": sum(1 for row in group if not row.get("ok")),
                "num_margin_failed": sum(1 for row in group if row.get("margin_ok") is False),
                "max_factor_scaled": max(factor_values) if factor_values else None,
                "max_orth_scaled": max(orth_values) if orth_values else None,
            }
        )
    return {
        "file": QUANTIZATION_SWEEP_FILE,
        "ok": bool(summary.get("ok", all(row.get("ok") and row.get("margin_ok", True) for row in cases))),
        "num_rows": int(summary.get("num_rows", len(cases)) or 0),
        "num_failed": int(summary.get("num_failed", sum(1 for row in cases if not row.get("ok"))) or 0),
        "num_margin_failed": int(summary.get("num_margin_failed", sum(1 for row in cases if row.get("margin_ok") is False)) or 0),
        "num_public_seed_rows": int(summary.get("num_public_seed_rows", 0) or 0),
        "num_popcorn_seed_rows": int(summary.get("num_popcorn_seed_rows", 0) or 0),
        "max_factor_scaled": summary.get("max_factor_scaled"),
        "max_orth_scaled": summary.get("max_orth_scaled"),
        "popcorn_seeds": summary.get("popcorn_seeds", []),
        "by_experiment": by_experiment,
    }


def summarize_mixed_seed_sweep(suite_dir: Path) -> dict[str, Any] | None:
    rows = load_jsonl(suite_dir / MIXED_SWEEP_FILE)
    cases = [row for row in rows if not row.get("summary")]
    if not cases:
        return None
    summary = next((row for row in rows if row.get("summary")), {})
    shapes = sorted(
        {
            f"{int(row['batch'])}x{int(row['n'])}"
            for row in cases
            if row.get("batch") is not None and row.get("n") is not None
        }
    )
    by_shape = []
    for shape in shapes:
        batch, n = [int(part) for part in shape.split("x")]
        group = [row for row in cases if row.get("batch") == batch and row.get("n") == n]
        factor_values = [float(row["factor_scaled_max"]) for row in group if isinstance(row.get("factor_scaled_max"), (int, float))]
        orth_values = [float(row["orth_scaled_max"]) for row in group if isinstance(row.get("orth_scaled_max"), (int, float))]
        by_shape.append(
            {
                "shape": shape,
                "num_rows": len(group),
                "num_failed": sum(1 for row in group if not row.get("ok")),
                "num_margin_failed": sum(1 for row in group if row.get("margin_ok") is False),
                "num_route_mismatch": sum(1 for row in group if row.get("route_ok") is False),
                "max_factor_scaled": max(factor_values) if factor_values else None,
                "max_orth_scaled": max(orth_values) if orth_values else None,
            }
        )
    return {
        "file": MIXED_SWEEP_FILE,
        "ok": bool(summary.get("ok", all(row.get("ok") and row.get("margin_ok", True) for row in cases))),
        "num_rows": int(summary.get("num_rows", len(cases)) or 0),
        "num_failed": int(summary.get("num_failed", sum(1 for row in cases if not row.get("ok"))) or 0),
        "num_margin_failed": int(summary.get("num_margin_failed", sum(1 for row in cases if row.get("margin_ok") is False)) or 0),
        "num_route_mismatch": int(summary.get("num_route_mismatch", sum(1 for row in cases if row.get("route_ok") is False)) or 0),
        "num_public_seed_rows": int(summary.get("num_public_seed_rows", 0) or 0),
        "num_popcorn_seed_rows": int(summary.get("num_popcorn_seed_rows", 0) or 0),
        "popcorn_seeds": summary.get("popcorn_seeds", []),
        "case_sources": summary.get("case_sources", []),
        "shapes": shapes,
        "max_factor_scaled": summary.get("max_factor_scaled"),
        "max_orth_scaled": summary.get("max_orth_scaled"),
        "by_shape": by_shape,
    }


def summarize_blocked_qr_sweep(suite_dir: Path) -> dict[str, Any] | None:
    rows = load_jsonl(suite_dir / BLOCKED_QR_SWEEP_FILE)
    cases = [row for row in rows if not row.get("summary")]
    if not cases:
        return None
    summary = next((row for row in rows if row.get("summary")), {})

    def grouped_row(key: tuple[str, str, str]) -> dict[str, Any]:
        precision_mode, r_mode, refresh_mode = key
        group = [
            row
            for row in cases
            if row.get("precision_mode") == precision_mode
            and row.get("r_maintenance_mode") == r_mode
            and row.get("panel_refresh_mode") == refresh_mode
        ]
        factor_values = [float(row["factor_scaled_max"]) for row in group if isinstance(row.get("factor_scaled_max"), (int, float))]
        orth_values = [float(row["orth_scaled_max"]) for row in group if isinstance(row.get("orth_scaled_max"), (int, float))]
        return {
            "precision_mode": precision_mode,
            "r_maintenance_mode": r_mode,
            "panel_refresh_mode": refresh_mode,
            "num_rows": len(group),
            "num_failed": sum(1 for row in group if not row.get("ok")),
            "max_factor_scaled": max(factor_values) if factor_values else None,
            "max_orth_scaled": max(orth_values) if orth_values else None,
        }

    keys = sorted(
        {
            (
                str(row.get("precision_mode") or ""),
                str(row.get("r_maintenance_mode") or ""),
                str(row.get("panel_refresh_mode") or ""),
            )
            for row in cases
        }
    )
    by_config = [grouped_row(key) for key in keys]
    passing_low_precision = [
        row
        for row in by_config
        if row["precision_mode"] != "fp32" and row["num_rows"] > 0 and row["num_failed"] == 0
    ]

    return {
        "file": BLOCKED_QR_SWEEP_FILE,
        "ok": bool(summary.get("ok", all(row.get("ok") for row in cases))),
        "num_rows": int(summary.get("num_rows", len(cases)) or 0),
        "num_failed": int(summary.get("num_failed", sum(1 for row in cases if not row.get("ok"))) or 0),
        "panel_widths": summary.get("panel_widths", sorted({int(row["panel_width"]) for row in cases if row.get("panel_width") is not None})),
        "update_modes": summary.get("update_modes", sorted({str(row.get("update_mode")) for row in cases if row.get("update_mode")})),
        "precision_modes": summary.get("precision_modes", sorted({str(row.get("precision_mode")) for row in cases if row.get("precision_mode")})),
        "r_maintenance_modes": summary.get("r_maintenance_modes", sorted({str(row.get("r_maintenance_mode")) for row in cases if row.get("r_maintenance_mode")})),
        "panel_refresh_modes": summary.get("panel_refresh_modes", sorted({str(row.get("panel_refresh_mode")) for row in cases if row.get("panel_refresh_mode")})),
        "num_passing_low_precision_configs": len(passing_low_precision),
        "passing_low_precision_configs": passing_low_precision,
        "by_config": by_config,
    }


def summarize_manifest(suite_dir: Path) -> dict[str, Any] | None:
    rows = load_jsonl(suite_dir / "manifest.jsonl")
    if not rows:
        return None

    steps = []
    total_elapsed_s = None
    failed = False
    for row in rows:
        event = row.get("event")
        if event == "finish" and row.get("step") and row.get("elapsed_s") is not None:
            elapsed = float(row["elapsed_s"])
            steps.append(
                {
                    "step": str(row["step"]),
                    "elapsed_s": elapsed,
                    "env_overrides": row.get("env_overrides", {}),
                }
            )
        elif event in {"suite_finish", "local_checks_finish"} and row.get("elapsed_s") is not None:
            total_elapsed_s = float(row["elapsed_s"])
        elif event == "suite_failed":
            failed = True
            if row.get("elapsed_s") is not None:
                total_elapsed_s = float(row["elapsed_s"])

    steps_sorted = sorted(steps, key=lambda row: row["elapsed_s"], reverse=True)
    return {
        "file": "manifest.jsonl",
        "num_steps": len(steps),
        "total_elapsed_s": total_elapsed_s,
        "sum_step_elapsed_s": sum(row["elapsed_s"] for row in steps) if steps else None,
        "failed": failed,
        "steps": steps,
        "slowest_steps": steps_sorted[:8],
    }


def summarize_suite(suite_dir: Path) -> dict[str, Any]:
    comparisons = []
    for name, (baseline_file, candidate_file) in PAIRS.items():
        comparison = summarize_pair(name, suite_dir, baseline_file, candidate_file)
        if comparison is not None:
            comparisons.append(comparison)
    guard_overhead = summarize_guard_overhead(suite_dir)
    ablations = []
    for name, file_name in ABLATION_FILES.items():
        ablation = summarize_ablation(name, suite_dir, file_name)
        if ablation is not None:
            ablations.append(ablation)
    tail_policy_tune = summarize_tail_policy_tune(suite_dir)
    candidate_config_tune = summarize_candidate_config_tune(suite_dir)
    quantization_seed_sweep = summarize_quantization_seed_sweep(suite_dir)
    mixed_seed_sweep = summarize_mixed_seed_sweep(suite_dir)
    blocked_qr_sweep = summarize_blocked_qr_sweep(suite_dir)

    return {
        "suite_dir": str(suite_dir),
        "ok": bool(
            comparisons
            or guard_overhead
            or ablations
            or tail_policy_tune
            or candidate_config_tune
            or quantization_seed_sweep
            or mixed_seed_sweep
            or blocked_qr_sweep
        ),
        "runtime": summarize_manifest(suite_dir),
        "num_comparisons": len(comparisons),
        "comparisons": comparisons,
        "guard_overhead": guard_overhead,
        "ablations": ablations,
        "quantization_seed_sweep": quantization_seed_sweep,
        "mixed_seed_sweep": mixed_seed_sweep,
        "blocked_qr_sweep": blocked_qr_sweep,
        "tail_policy_tune": tail_policy_tune,
        "candidate_config_tune": candidate_config_tune,
    }


def fmt_us(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def fmt_speedup(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}x"


def fmt_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 60.0:
        return f"{value:.1f}s"
    minutes = value / 60.0
    if minutes < 60.0:
        return f"{minutes:.1f}m"
    return f"{minutes / 60.0:.2f}h"


def fmt_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def fmt_count(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def fmt_key_list(value: Any, limit: int = 3) -> str:
    if not value:
        return ""
    keys = [str(item) for item in value]
    if len(keys) <= limit:
        return ", ".join(keys)
    return ", ".join(keys[:limit]) + f", +{len(keys) - limit} more"


def fmt_env_dict(value: Any, limit: int = 4) -> str:
    if not isinstance(value, dict) or not value:
        return ""
    items = [f"{key}={value[key]}" for key in sorted(value)]
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f", +{len(items) - limit} more"


def truthy_env_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def route_order_key(env: dict[str, Any]) -> str | None:
    for key in sorted(env):
        if key == "FAST_QR_STRUCTURED_ROUTES_BEFORE_CUDA" or key.endswith("_STRUCTURED_BEFORE_CUDA"):
            return key
    return None


def route_order_label(env_key: str) -> str:
    if env_key == "FAST_QR_STRUCTURED_ROUTES_BEFORE_CUDA":
        return "global"
    suffix = "_STRUCTURED_BEFORE_CUDA"
    prefix = env_key[: -len(suffix)] if env_key.endswith(suffix) else env_key
    if prefix.startswith("FAST_QR_"):
        prefix = prefix[len("FAST_QR_") :]
    return prefix.lower()


def route_order_decision(ratio: float | None) -> str:
    if ratio is None:
        return "insufficient-data"
    if ratio < 0.98:
        return "prefer-structured-first"
    if ratio > 1.02:
        return "prefer-cuda-first"
    return "neutral-within-noise"


def summarize_route_order_configs(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    route_rows = []
    for row in results:
        env = row.get("env") if isinstance(row.get("env"), dict) else {}
        key = route_order_key(env)
        if key is None:
            continue
        route_rows.append(
            {
                **row,
                "route_order_key": key,
                "route_order_label": route_order_label(key),
                "structured_before_cuda": truthy_env_value(env.get(key)),
                "comparison_env": {env_key: env[env_key] for env_key in sorted(env) if env_key != key},
            }
        )

    if not route_rows:
        return None

    grouped: dict[tuple[str, tuple[tuple[str, str], ...]], list[dict[str, Any]]] = {}
    for row in route_rows:
        comparison_key = tuple((str(key), str(value)) for key, value in row["comparison_env"].items())
        grouped.setdefault((row["route_order_label"], comparison_key), []).append(row)

    compared_pairs = []
    for (label, comparison_key), group in grouped.items():
        cuda_first = [
            row
            for row in group
            if not row["structured_before_cuda"] and isinstance(row.get("benchmark_geomean_us"), (int, float))
        ]
        structured_first = [
            row
            for row in group
            if row["structured_before_cuda"] and isinstance(row.get("benchmark_geomean_us"), (int, float))
        ]
        if not cuda_first or not structured_first:
            continue
        cuda_best = min(cuda_first, key=lambda row: float(row["benchmark_geomean_us"]))
        structured_best = min(structured_first, key=lambda row: float(row["benchmark_geomean_us"]))
        cuda_us = float(cuda_best["benchmark_geomean_us"])
        structured_us = float(structured_best["benchmark_geomean_us"])
        ratio = structured_us / cuda_us if cuda_us > 0.0 else None
        compared_pairs.append(
            {
                "route_order_label": label,
                "comparison_env": dict(comparison_key),
                "cuda_first_name": cuda_best.get("name"),
                "cuda_first_geomean_us": cuda_us,
                "structured_first_name": structured_best.get("name"),
                "structured_first_geomean_us": structured_us,
                "structured_over_cuda": ratio,
                "decision": route_order_decision(ratio),
            }
        )

    cuda_rows = [
        row
        for row in route_rows
        if not row["structured_before_cuda"] and isinstance(row.get("benchmark_geomean_us"), (int, float))
    ]
    structured_rows = [
        row
        for row in route_rows
        if row["structured_before_cuda"] and isinstance(row.get("benchmark_geomean_us"), (int, float))
    ]
    best_cuda = min(cuda_rows, key=lambda row: float(row["benchmark_geomean_us"])) if cuda_rows else None
    best_structured = min(structured_rows, key=lambda row: float(row["benchmark_geomean_us"])) if structured_rows else None
    aggregate_ratio = None
    if best_cuda is not None and best_structured is not None and float(best_cuda["benchmark_geomean_us"]) > 0.0:
        aggregate_ratio = float(best_structured["benchmark_geomean_us"]) / float(best_cuda["benchmark_geomean_us"])

    return {
        "num_route_order_configs": len(route_rows),
        "num_compared_pairs": len(compared_pairs),
        "route_order_labels": sorted({row["route_order_label"] for row in route_rows}),
        "num_cuda_first_benchmarked": len(cuda_rows),
        "num_structured_first_benchmarked": len(structured_rows),
        "best_cuda_first_name": best_cuda.get("name") if best_cuda else None,
        "best_cuda_first_geomean_us": best_cuda.get("benchmark_geomean_us") if best_cuda else None,
        "best_structured_first_name": best_structured.get("name") if best_structured else None,
        "best_structured_first_geomean_us": best_structured.get("benchmark_geomean_us") if best_structured else None,
        "structured_over_cuda": aggregate_ratio,
        "decision": route_order_decision(aggregate_ratio),
        "pairs": compared_pairs,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# QR Suite Summary",
        "",
        f"Suite: `{summary['suite_dir']}`",
        "",
    ]
    if not summary["comparisons"]:
        lines.append("No baseline/candidate benchmark pairs were found.")
        lines.append("")

    runtime = summary.get("runtime")
    if runtime:
        lines.extend(
            [
                "## Runtime",
                "",
                f"- total elapsed: {fmt_seconds(runtime.get('total_elapsed_s'))}",
                f"- summed step elapsed: {fmt_seconds(runtime.get('sum_step_elapsed_s'))}",
                f"- completed steps: {runtime.get('num_steps')}",
                f"- failed: {runtime.get('failed')}",
                "",
                "| step | elapsed |",
                "| --- | ---: |",
            ]
        )
        for row in runtime.get("slowest_steps", []):
            lines.append(f"| {row['step']} | {fmt_seconds(row['elapsed_s'])} |")
        lines.append("")

    for comparison in summary["comparisons"]:
        lines.extend(
            [
                f"## {comparison['name']}",
                "",
                f"- cases: {comparison['num_common_cases']}",
                f"- baseline geomean us: {fmt_us(comparison['baseline_geomean_us'])}",
                f"- candidate geomean us: {fmt_us(comparison['candidate_geomean_us'])}",
                f"- geomean speedup: {fmt_speedup(comparison['geomean_speedup'])}",
                "",
                "| case | baseline us | candidate us | speedup |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in comparison["cases"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["spec"],
                        fmt_us(row["baseline_mean_us"]),
                        fmt_us(row["candidate_mean_us"]),
                        fmt_speedup(row["speedup"]),
                    ]
                )
                + " |"
            )
        lines.append("")

    guard = summary.get("guard_overhead")
    if guard:
        lines.extend(
            [
                "## Guard Overhead",
                "",
                f"- cases: {guard['num_cases']}",
                f"- cold wall geomean us: {fmt_us(guard['cold_wall_geomean_us'])}",
                f"- hot cached wall geomean us: {fmt_us(guard['wall_geomean_us'])}",
                f"- cold wall max us: {fmt_us(guard['cold_wall_max_us'])}",
                f"- hot cached wall max us: {fmt_us(guard['wall_max_us'])}",
                f"- cold CUDA geomean us: {fmt_us(guard['cold_cuda_geomean_us'])}",
                f"- hot cached CUDA geomean us: {fmt_us(guard['cuda_geomean_us'])}",
                "",
                "| case | route | cold wall us | hot wall us | cold CUDA us | hot CUDA us |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in guard["cases"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["spec"],
                        row["route"],
                        fmt_us(row["cold_wall_us"]),
                        fmt_us(row["wall_us"]),
                        fmt_us(row["cold_cuda_us"]),
                        fmt_us(row["cuda_us"]),
                    ]
                )
                + " |"
            )
        lines.append("")

    quantization = summary.get("quantization_seed_sweep")
    if quantization:
        lines.extend(
            [
                "## Quantization Seed Sweep",
                "",
                f"- ok: {quantization.get('ok')}",
                f"- rows: {quantization.get('num_rows')}",
                f"- failed rows: {quantization.get('num_failed')}",
                f"- margin failures: {quantization.get('num_margin_failed')}",
                f"- public-seed rows: {quantization.get('num_public_seed_rows')}",
                f"- POPCORN-seed rows: {quantization.get('num_popcorn_seed_rows')}",
                f"- max factor scaled: {fmt_us(quantization.get('max_factor_scaled'))}",
                f"- max orth scaled: {fmt_us(quantization.get('max_orth_scaled'))}",
                "",
                "| experiment | rows | failed | margin failed | max factor | max orth |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in quantization.get("by_experiment", []):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("experiment") or ""),
                        str(row.get("num_rows")),
                        str(row.get("num_failed")),
                        str(row.get("num_margin_failed")),
                        fmt_us(row.get("max_factor_scaled")),
                        fmt_us(row.get("max_orth_scaled")),
                    ]
                )
                + " |"
            )
        lines.append("")

    mixed = summary.get("mixed_seed_sweep")
    if mixed:
        lines.extend(
            [
                "## Mixed Seed Sweep",
                "",
                f"- ok: {mixed.get('ok')}",
                f"- rows: {mixed.get('num_rows')}",
                f"- failed rows: {mixed.get('num_failed')}",
                f"- margin failures: {mixed.get('num_margin_failed')}",
                f"- route mismatches: {mixed.get('num_route_mismatch')}",
                f"- shapes: {', '.join(mixed.get('shapes') or [])}",
                f"- POPCORN seeds: {', '.join(str(seed) for seed in mixed.get('popcorn_seeds') or [])}",
                f"- max factor scaled: {fmt_us(mixed.get('max_factor_scaled'))}",
                f"- max orth scaled: {fmt_us(mixed.get('max_orth_scaled'))}",
                "",
                "| shape | rows | failed | margin failed | route mismatch | max factor | max orth |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in mixed.get("by_shape", []):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("shape") or ""),
                        str(row.get("num_rows")),
                        str(row.get("num_failed")),
                        str(row.get("num_margin_failed")),
                        str(row.get("num_route_mismatch")),
                        fmt_us(row.get("max_factor_scaled")),
                        fmt_us(row.get("max_orth_scaled")),
                    ]
                )
                + " |"
            )
        lines.append("")

    blocked = summary.get("blocked_qr_sweep")
    if blocked:
        lines.extend(
            [
                "## Blocked QR Sweep",
                "",
                f"- ok: {blocked.get('ok')}",
                f"- rows: {blocked.get('num_rows')}",
                f"- failed rows: {blocked.get('num_failed')}",
                f"- panel widths: {', '.join(str(value) for value in blocked.get('panel_widths') or [])}",
                f"- update modes: {', '.join(str(value) for value in blocked.get('update_modes') or [])}",
                f"- precision modes: {', '.join(str(value) for value in blocked.get('precision_modes') or [])}",
                f"- passing low-precision configs: {blocked.get('num_passing_low_precision_configs')}",
                "",
                "| precision | R maintenance | panel refresh | rows | failed | max factor | max orth |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in blocked.get("by_config", []):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("precision_mode") or ""),
                        str(row.get("r_maintenance_mode") or ""),
                        str(row.get("panel_refresh_mode") or ""),
                        str(row.get("num_rows")),
                        str(row.get("num_failed")),
                        fmt_us(row.get("max_factor_scaled")),
                        fmt_us(row.get("max_orth_scaled")),
                    ]
                )
                + " |"
            )
        lines.append("")

    tune = summary.get("tail_policy_tune")
    if tune:
        lines.extend(
            [
                "## Tail Policy Tune",
                "",
                f"- configs: {tune.get('num_configs')}",
                f"- configs with correctness failures: {tune.get('num_failed_configs')}",
                f"- benchmarked configs: {tune.get('num_benchmarked_configs')}",
                f"- best config: {tune.get('best_name') or 'n/a'}",
                f"- best geomean us: {fmt_us(tune.get('best_geomean_us'))}",
                f"- hard failed: {tune.get('hard_failed')}",
                "",
                "| config | correctness failures | max factor | benchmark geomean us |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in tune.get("results", []):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("name") or ""),
                        str(row.get("correctness_num_failed")),
                        fmt_us(row.get("correctness_max_factor_scaled")),
                        fmt_us(row.get("benchmark_geomean_us")),
                    ]
                )
                + " |"
            )
        lines.append("")

    tune = summary.get("candidate_config_tune")
    if tune:
        lines.extend(
            [
                "## Candidate Config Tune",
                "",
                f"- objective: {tune.get('objective') or 'minimize_geomean_us'}",
                f"- configs: {tune.get('num_configs')}",
                f"- configs with correctness failures: {tune.get('num_failed_configs')}",
                f"- benchmarked configs: {tune.get('num_benchmarked_configs')}",
                f"- configs with resource metrics: {tune.get('num_configs_with_resource_metrics')}",
                f"- configs with inert env keys: {tune.get('num_configs_with_inert_env', 0)}",
                f"- configs with CUDA-route-bypassed env keys: {tune.get('num_configs_with_cuda_route_bypassed_env', 0)}",
                f"- best config: {tune.get('best_name') or 'n/a'}",
                f"- best geomean us: {fmt_us(tune.get('best_geomean_us'))}",
                f"- hard failed: {tune.get('hard_failed')}",
            ]
        )
        route_order = tune.get("route_order")
        if route_order:
            lines.extend(
                [
                    f"- route-order decision: {route_order.get('decision')}",
                    f"- route-order structured/cuda: {fmt_ratio(route_order.get('structured_over_cuda'))}",
                    f"- route-order compared pairs: {route_order.get('num_compared_pairs')}",
                ]
            )
        lines.extend(
            [
                "",
                "| config | env | correctness failures | max factor | benchmark geomean us | inert env keys | CUDA-bypassed env keys | regs/thread | smem bytes | est occupancy |",
                "| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in tune.get("results", []):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("name") or ""),
                        fmt_env_dict(row.get("env")),
                        str(row.get("correctness_num_failed")),
                        fmt_us(row.get("correctness_max_factor_scaled")),
                        fmt_us(row.get("benchmark_geomean_us")),
                        fmt_key_list(row.get("inert_env_keys")),
                        fmt_key_list(row.get("cuda_route_bypassed_env_keys")),
                        fmt_count(row.get("resource_max_registers_per_thread")),
                        fmt_count(row.get("resource_max_smem_bytes")),
                        fmt_ratio(row.get("resource_min_estimated_occupancy")),
                    ]
                )
                + " |"
            )
        lines.append("")
        if route_order and route_order.get("pairs"):
            lines.extend(
                [
                    "| route-order label | shared env | cuda-first config | cuda-first us | structured-first config | structured-first us | structured/cuda | decision |",
                    "| --- | --- | --- | ---: | --- | ---: | ---: | --- |",
                ]
            )
            for row in route_order.get("pairs", []):
                shared_env = ", ".join(f"{key}={value}" for key, value in row.get("comparison_env", {}).items())
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(row.get("route_order_label") or ""),
                            shared_env,
                            str(row.get("cuda_first_name") or ""),
                            fmt_us(row.get("cuda_first_geomean_us")),
                            str(row.get("structured_first_name") or ""),
                            fmt_us(row.get("structured_first_geomean_us")),
                            fmt_ratio(row.get("structured_over_cuda")),
                            str(row.get("decision") or ""),
                        ]
                    )
                    + " |"
                )
            lines.append("")

    for ablation in summary.get("ablations", []):
        lines.extend(
            [
                f"## Ablation: {ablation['name']}",
                "",
                f"- cases: {ablation['num_common_cases']}",
                f"- default geomean us: {fmt_us(ablation['default_geomean_us'])}",
                f"- ablation geomean us: {fmt_us(ablation['ablation_geomean_us'])}",
                f"- ablation/default: {fmt_speedup(ablation['ablation_over_default'])}",
                "",
                "| case | default us | ablation us | ablation/default |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in ablation["cases"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["spec"],
                        fmt_us(row["default_mean_us"]),
                        fmt_us(row["ablation_mean_us"]),
                        fmt_speedup(row["ablation_over_default"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize B200 suite benchmark JSONL outputs.",
        allow_abbrev=False,
    )
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--markdown-out", default=None)
    args = parser.parse_args()

    suite_dir = ROOT / args.suite_dir if not Path(args.suite_dir).is_absolute() else Path(args.suite_dir)
    summary = summarize_suite(suite_dir)
    text = render_markdown(summary)

    if args.json_out:
        out = ROOT / args.json_out if not Path(args.json_out).is_absolute() else Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.markdown_out:
        out = ROOT / args.markdown_out if not Path(args.markdown_out).is_absolute() else Path(args.markdown_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)

    print(text)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
