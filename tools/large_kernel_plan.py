from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

from qr_common import ROOT


PRESETS: dict[str, dict[str, Any]] = {
    "qr512": {
        "benchmark_indices": "3,7,9,10",
        "correctness_indices": "3,6,7,8,9,10,11,19",
        "panel_widths": ["16", "32", "48", "64"],
        "update_modes": ["reflectors", "compact-wy"],
        "precision_modes": ["fp32", "tf32", "fp16-input"],
        "tile_ms": ["64", "128"],
        "tile_ns": ["64", "128"],
        "compact_wy_tile_cols": ["2", "4", "8"],
        "warps_per_cta": ["4", "8", "16", "32"],
        "ctas_per_matrix": ["1", "2"],
        "cta_schedules": ["fixed", "frontload"],
        "sync_free_auto_policy": ["1", "0"],
        "auto_policy_groups": ["1", "0"],
        "policy_full_scan": ["1", "0"],
        "cluster_sizes": ["1"],
        "tail_cuts": ["0", "16", "24", "32"],
        "tail_thresholds": ["0.0", "0.03"],
        "tail_force": ["0", "1"],
        "panel_refreshes": ["1", "2"],
        "panel_refresh_modes": ["none", "prefix"],
        "r_maintenance_modes": ["none", "panel-prefix"],
        "structured_before_cuda": ["0", "1"],
    },
    "qr1024": {
        "benchmark_indices": "4,8,11",
        "correctness_indices": "4,12,13,14,15,20",
        "panel_widths": ["32", "48", "64", "96"],
        "update_modes": ["reflectors", "compact-wy"],
        "precision_modes": ["fp32", "tf32", "fp16-input"],
        "tile_ms": ["128", "256"],
        "tile_ns": ["128", "256"],
        "compact_wy_tile_cols": ["2", "4", "8"],
        "warps_per_cta": ["8", "16", "32"],
        "ctas_per_matrix": ["1", "2", "4"],
        "cta_schedules": ["fixed", "frontload"],
        "sync_free_auto_policy": ["1", "0"],
        "auto_policy_groups": ["1", "0"],
        "policy_full_scan": ["1", "0"],
        "cluster_sizes": ["1"],
        "tail_cuts": ["0", "8", "64", "128"],
        "tail_thresholds": ["0.0", "0.03"],
        "tail_force": ["0", "1"],
        "panel_refreshes": ["1", "2"],
        "panel_refresh_modes": ["none", "prefix"],
        "r_maintenance_modes": ["none", "panel-prefix"],
        "structured_before_cuda": ["0", "1"],
    },
    "qr2048": {
        "benchmark_indices": "5",
        "correctness_indices": "16,21",
        "panel_widths": ["32", "64", "96"],
        "update_modes": ["reflectors", "compact-wy"],
        "precision_modes": ["fp32", "tf32", "fp16-input"],
        "tile_ms": ["128", "256"],
        "tile_ns": ["128", "256"],
        "compact_wy_tile_cols": ["1", "2", "4"],
        "warps_per_cta": ["4", "8"],
        "ctas_per_matrix": ["4", "8"],
        "cta_schedules": ["fixed", "frontload", "all-tiles"],
        "sync_free_auto_policy": ["1", "0"],
        "auto_policy_groups": ["1", "0"],
        "policy_full_scan": ["1", "0"],
        "cluster_sizes": ["1", "2"],
        "tail_cuts": ["0", "64"],
        "tail_thresholds": ["0.0", "0.1", "0.2"],
        "tail_force": ["0", "1"],
        "panel_refreshes": ["1", "2"],
        "panel_refresh_modes": ["none", "prefix"],
        "r_maintenance_modes": ["none", "panel-prefix"],
    },
    "qr4096": {
        "benchmark_indices": "6",
        "correctness_indices": "5,18",
        "panel_widths": ["64", "96"],
        "update_modes": ["reflectors", "compact-wy"],
        "precision_modes": ["fp32", "tf32", "fp16-input"],
        "tile_ms": ["256", "512"],
        "tile_ns": ["256", "512"],
        "compact_wy_tile_cols": ["1", "2", "4"],
        "warps_per_cta": ["8"],
        "ctas_per_matrix": ["8", "16"],
        "cta_schedules": ["fixed", "frontload", "all-tiles"],
        "sync_free_auto_policy": ["1", "0"],
        "auto_policy_groups": ["1", "0"],
        "policy_full_scan": ["1", "0"],
        "cluster_sizes": ["1", "2"],
        "tail_cuts": ["0", "128"],
        "tail_thresholds": ["0.0", "0.1", "0.2"],
        "tail_force": ["0", "1"],
        "panel_refreshes": ["1", "2"],
        "panel_refresh_modes": ["none", "prefix"],
        "r_maintenance_modes": ["none", "panel-prefix"],
    },
}


AXIS_TO_ENV = {
    "panel_widths": "PANEL_B",
    "update_modes": "UPDATE_MODE",
    "precision_modes": "PRECISION_MODE",
    "tile_ms": "TILE_M",
    "tile_ns": "TILE_N",
    "compact_wy_tile_cols": "COMPACT_WY_TILE_COLS",
    "warps_per_cta": "WARPS_PER_CTA",
    "ctas_per_matrix": "CTAS_PER_MATRIX",
    "cta_schedules": "CTA_SCHEDULE",
    "sync_free_auto_policy": "SYNC_FREE_AUTO_POLICY",
    "auto_policy_groups": "BLOCKED_AUTO_GROUPS",
    "policy_full_scan": "POLICY_FULL_SCAN",
    "cluster_sizes": "CLUSTER_SIZE",
    "tail_cuts": "TAIL_CUT",
    "tail_thresholds": "TAIL_THRESHOLD",
    "tail_force": "TAIL_FORCE",
    "panel_refreshes": "PANEL_REFRESH",
    "panel_refresh_modes": "PANEL_REFRESH_MODE",
    "r_maintenance_modes": "R_MAINTENANCE_MODE",
    "structured_before_cuda": "STRUCTURED_BEFORE_CUDA",
}

MODE_CHOICES = ("future-blocked", "current-candidate")
FUTURE_BLOCKED_AXES = tuple(axis for axis in AXIS_TO_ENV if axis != "structured_before_cuda")

CURRENT_CANDIDATE_AXES = {
    # The current blocked CUDA paths consume panel width, tail tile width,
    # trailing-update algorithm/precision, compact-WY tile width, prefix repair
    # modes, CTA sizing, per-matrix column-tile CTA scheduling, and dense-tail
    # policy aliases. They do not consume the future cluster axes yet.
    "qr512": (
        "panel_widths",
        "update_modes",
        "precision_modes",
        "panel_refresh_modes",
        "r_maintenance_modes",
        "tile_ns",
        "compact_wy_tile_cols",
        "warps_per_cta",
        "ctas_per_matrix",
        "cta_schedules",
        "sync_free_auto_policy",
        "auto_policy_groups",
        "policy_full_scan",
        "tail_cuts",
        "tail_thresholds",
        "tail_force",
        "structured_before_cuda",
    ),
    "qr1024": (
        "panel_widths",
        "update_modes",
        "precision_modes",
        "panel_refresh_modes",
        "r_maintenance_modes",
        "tile_ns",
        "compact_wy_tile_cols",
        "warps_per_cta",
        "ctas_per_matrix",
        "cta_schedules",
        "sync_free_auto_policy",
        "auto_policy_groups",
        "policy_full_scan",
        "tail_cuts",
        "tail_thresholds",
        "tail_force",
        "structured_before_cuda",
    ),
    "qr2048": (
        "panel_widths",
        "update_modes",
        "precision_modes",
        "panel_refresh_modes",
        "r_maintenance_modes",
        "tile_ns",
        "compact_wy_tile_cols",
        "warps_per_cta",
        "ctas_per_matrix",
        "cta_schedules",
        "sync_free_auto_policy",
        "auto_policy_groups",
        "policy_full_scan",
        "tail_cuts",
        "tail_thresholds",
        "tail_force",
    ),
    "qr4096": (
        "panel_widths",
        "update_modes",
        "precision_modes",
        "panel_refresh_modes",
        "r_maintenance_modes",
        "tile_ns",
        "compact_wy_tile_cols",
        "warps_per_cta",
        "ctas_per_matrix",
        "cta_schedules",
        "sync_free_auto_policy",
        "auto_policy_groups",
        "policy_full_scan",
        "tail_cuts",
        "tail_thresholds",
        "tail_force",
    ),
}

CURRENT_CANDIDATE_STATIC_ENV = {
    "qr512": {"FAST_QR_ENABLE_QR512_BLOCKED_CUDA": "1"},
    "qr1024": {"FAST_QR_ENABLE_QR1024_BLOCKED_CUDA": "1"},
    "qr2048": {"FAST_QR_ENABLE_QR2048_BLOCKED_CUDA": "1"},
    "qr4096": {"FAST_QR_ENABLE_QR4096_BLOCKED_CUDA": "1"},
}

CURRENT_CANDIDATE_SEED_AXIS_VALUES: dict[str, list[tuple[str, dict[str, str]]]] = {
    "qr512": [
        (
            "b200_default_sync_free_compact_wy_frontload_2cta",
            {
                "panel_widths": "32",
                "update_modes": "compact-wy",
                "precision_modes": "fp32",
                "panel_refresh_modes": "prefix",
                "r_maintenance_modes": "panel-prefix",
                "tile_ns": "128",
                "compact_wy_tile_cols": "4",
                "warps_per_cta": "8",
                "ctas_per_matrix": "2",
                "cta_schedules": "frontload",
                "sync_free_auto_policy": "1",
                "auto_policy_groups": "1",
                "policy_full_scan": "1",
                "tail_cuts": "32",
                "tail_thresholds": "0.03",
                "structured_before_cuda": "0",
            },
        ),
        (
            "sync_free_repair",
            {
                "panel_widths": "32",
                "update_modes": "reflectors",
                "precision_modes": "fp32",
                "panel_refresh_modes": "prefix",
                "r_maintenance_modes": "panel-prefix",
                "tile_ns": "64",
                "compact_wy_tile_cols": "4",
                "warps_per_cta": "8",
                "ctas_per_matrix": "1",
                "cta_schedules": "fixed",
                "sync_free_auto_policy": "1",
                "auto_policy_groups": "1",
                "policy_full_scan": "1",
                "tail_cuts": "32",
                "tail_thresholds": "0.03",
                "tail_force": "1",
                "structured_before_cuda": "0",
            },
        ),
        (
            "sparse_policy_repair",
            {
                "panel_widths": "32",
                "update_modes": "reflectors",
                "precision_modes": "fp32",
                "panel_refresh_modes": "prefix",
                "r_maintenance_modes": "panel-prefix",
                "tile_ns": "64",
                "compact_wy_tile_cols": "4",
                "warps_per_cta": "8",
                "ctas_per_matrix": "1",
                "cta_schedules": "fixed",
                "sync_free_auto_policy": "0",
                "auto_policy_groups": "1",
                "policy_full_scan": "0",
                "tail_cuts": "32",
                "tail_thresholds": "0.03",
                "structured_before_cuda": "0",
            },
        ),
        (
            "no_tail_repair",
            {
                "panel_widths": "32",
                "update_modes": "reflectors",
                "precision_modes": "fp32",
                "panel_refresh_modes": "prefix",
                "r_maintenance_modes": "panel-prefix",
                "tile_ns": "64",
                "compact_wy_tile_cols": "4",
                "warps_per_cta": "8",
                "ctas_per_matrix": "1",
                "cta_schedules": "fixed",
                "sync_free_auto_policy": "0",
                "auto_policy_groups": "1",
                "policy_full_scan": "1",
                "tail_cuts": "0",
                "tail_thresholds": "0.0",
                "structured_before_cuda": "0",
            },
        ),
        (
            "tf32_repair",
            {
                "panel_widths": "32",
                "update_modes": "reflectors",
                "precision_modes": "tf32",
                "panel_refresh_modes": "prefix",
                "r_maintenance_modes": "panel-prefix",
                "tile_ns": "64",
                "compact_wy_tile_cols": "4",
                "warps_per_cta": "8",
                "ctas_per_matrix": "1",
                "cta_schedules": "fixed",
                "sync_free_auto_policy": "0",
                "auto_policy_groups": "1",
                "policy_full_scan": "1",
                "tail_cuts": "32",
                "tail_thresholds": "0.03",
                "structured_before_cuda": "0",
            },
        ),
        (
            "compact_wy_repair",
            {
                "panel_widths": "32",
                "update_modes": "compact-wy",
                "precision_modes": "fp32",
                "panel_refresh_modes": "prefix",
                "r_maintenance_modes": "panel-prefix",
                "tile_ns": "64",
                "compact_wy_tile_cols": "4",
                "warps_per_cta": "8",
                "ctas_per_matrix": "1",
                "cta_schedules": "fixed",
                "sync_free_auto_policy": "0",
                "auto_policy_groups": "1",
                "policy_full_scan": "1",
                "tail_cuts": "32",
                "tail_thresholds": "0.03",
                "structured_before_cuda": "0",
            },
        ),
        (
            "frontload_2cta_repair",
            {
                "panel_widths": "32",
                "update_modes": "reflectors",
                "precision_modes": "fp32",
                "panel_refresh_modes": "prefix",
                "r_maintenance_modes": "panel-prefix",
                "tile_ns": "128",
                "compact_wy_tile_cols": "4",
                "warps_per_cta": "8",
                "ctas_per_matrix": "2",
                "cta_schedules": "frontload",
                "sync_free_auto_policy": "0",
                "auto_policy_groups": "1",
                "policy_full_scan": "1",
                "tail_cuts": "32",
                "tail_thresholds": "0.03",
                "structured_before_cuda": "0",
            },
        ),
        (
            "structured_first_repair",
            {
                "panel_widths": "32",
                "update_modes": "reflectors",
                "precision_modes": "fp32",
                "panel_refresh_modes": "prefix",
                "r_maintenance_modes": "panel-prefix",
                "tile_ns": "64",
                "compact_wy_tile_cols": "4",
                "warps_per_cta": "8",
                "ctas_per_matrix": "1",
                "cta_schedules": "fixed",
                "sync_free_auto_policy": "0",
                "auto_policy_groups": "1",
                "policy_full_scan": "1",
                "tail_cuts": "32",
                "tail_thresholds": "0.03",
                "structured_before_cuda": "1",
            },
        ),
    ],
    "qr1024": [
        (
            "b200_default_sync_free_compact_wy_frontload_2cta",
            {
                "panel_widths": "32",
                "update_modes": "compact-wy",
                "precision_modes": "fp32",
                "panel_refresh_modes": "prefix",
                "r_maintenance_modes": "panel-prefix",
                "tile_ns": "128",
                "compact_wy_tile_cols": "4",
                "warps_per_cta": "8",
                "ctas_per_matrix": "2",
                "cta_schedules": "frontload",
                "sync_free_auto_policy": "1",
                "auto_policy_groups": "1",
                "policy_full_scan": "1",
                "tail_cuts": "64",
                "tail_thresholds": "0.03",
                "structured_before_cuda": "0",
            },
        ),
        (
            "sync_free_repair",
            {
                "panel_widths": "32",
                "update_modes": "reflectors",
                "precision_modes": "fp32",
                "panel_refresh_modes": "prefix",
                "r_maintenance_modes": "panel-prefix",
                "tile_ns": "128",
                "compact_wy_tile_cols": "4",
                "warps_per_cta": "8",
                "ctas_per_matrix": "1",
                "cta_schedules": "fixed",
                "sync_free_auto_policy": "1",
                "auto_policy_groups": "1",
                "policy_full_scan": "1",
                "tail_cuts": "64",
                "tail_thresholds": "0.03",
                "tail_force": "1",
                "structured_before_cuda": "0",
            },
        ),
        (
            "sparse_policy_repair",
            {
                "panel_widths": "32",
                "update_modes": "reflectors",
                "precision_modes": "fp32",
                "panel_refresh_modes": "prefix",
                "r_maintenance_modes": "panel-prefix",
                "tile_ns": "128",
                "compact_wy_tile_cols": "4",
                "warps_per_cta": "8",
                "ctas_per_matrix": "1",
                "cta_schedules": "fixed",
                "sync_free_auto_policy": "0",
                "auto_policy_groups": "1",
                "policy_full_scan": "0",
                "tail_cuts": "64",
                "tail_thresholds": "0.03",
                "structured_before_cuda": "0",
            },
        ),
        (
            "mixed_safe_tail_repair",
            {
                "panel_widths": "32",
                "update_modes": "reflectors",
                "precision_modes": "fp32",
                "panel_refresh_modes": "prefix",
                "r_maintenance_modes": "panel-prefix",
                "tile_ns": "128",
                "compact_wy_tile_cols": "4",
                "warps_per_cta": "8",
                "ctas_per_matrix": "1",
                "cta_schedules": "fixed",
                "sync_free_auto_policy": "0",
                "auto_policy_groups": "1",
                "policy_full_scan": "1",
                "tail_cuts": "8",
                "tail_thresholds": "0.03",
                "structured_before_cuda": "0",
            },
        ),
        (
            "no_tail_repair",
            {
                "panel_widths": "32",
                "update_modes": "reflectors",
                "precision_modes": "fp32",
                "panel_refresh_modes": "prefix",
                "r_maintenance_modes": "panel-prefix",
                "tile_ns": "128",
                "compact_wy_tile_cols": "4",
                "warps_per_cta": "8",
                "ctas_per_matrix": "1",
                "cta_schedules": "fixed",
                "sync_free_auto_policy": "0",
                "auto_policy_groups": "1",
                "policy_full_scan": "1",
                "tail_cuts": "0",
                "tail_thresholds": "0.0",
                "structured_before_cuda": "0",
            },
        ),
        (
            "tf32_repair",
            {
                "panel_widths": "32",
                "update_modes": "reflectors",
                "precision_modes": "tf32",
                "panel_refresh_modes": "prefix",
                "r_maintenance_modes": "panel-prefix",
                "tile_ns": "128",
                "compact_wy_tile_cols": "4",
                "warps_per_cta": "8",
                "ctas_per_matrix": "1",
                "cta_schedules": "fixed",
                "sync_free_auto_policy": "0",
                "auto_policy_groups": "1",
                "policy_full_scan": "1",
                "tail_cuts": "64",
                "tail_thresholds": "0.03",
                "structured_before_cuda": "0",
            },
        ),
        (
            "structured_first_repair",
            {
                "panel_widths": "32",
                "update_modes": "reflectors",
                "precision_modes": "fp32",
                "panel_refresh_modes": "prefix",
                "r_maintenance_modes": "panel-prefix",
                "tile_ns": "128",
                "compact_wy_tile_cols": "4",
                "warps_per_cta": "8",
                "ctas_per_matrix": "1",
                "cta_schedules": "fixed",
                "sync_free_auto_policy": "0",
                "auto_policy_groups": "1",
                "policy_full_scan": "1",
                "tail_cuts": "64",
                "tail_thresholds": "0.03",
                "structured_before_cuda": "1",
            },
        ),
    ],
    "qr2048": [
        (
            "b200_default_compact_wy_all_tiles_8cta",
            {
                "panel_widths": "64",
                "update_modes": "compact-wy",
                "precision_modes": "fp32",
                "panel_refresh_modes": "prefix",
                "r_maintenance_modes": "panel-prefix",
                "tile_ns": "128",
                "compact_wy_tile_cols": "2",
                "warps_per_cta": "8",
                "ctas_per_matrix": "8",
                "cta_schedules": "all-tiles",
                "sync_free_auto_policy": "1",
                "auto_policy_groups": "1",
                "policy_full_scan": "1",
                "tail_cuts": "64",
                "tail_thresholds": "0.2",
                "tail_force": "0",
            },
        ),
    ],
    "qr4096": [
        (
            "b200_default_compact_wy_all_tiles_16cta",
            {
                "panel_widths": "64",
                "update_modes": "compact-wy",
                "precision_modes": "fp32",
                "panel_refresh_modes": "prefix",
                "r_maintenance_modes": "panel-prefix",
                "tile_ns": "256",
                "compact_wy_tile_cols": "2",
                "warps_per_cta": "8",
                "ctas_per_matrix": "16",
                "cta_schedules": "all-tiles",
                "sync_free_auto_policy": "1",
                "auto_policy_groups": "1",
                "policy_full_scan": "1",
                "tail_cuts": "128",
                "tail_thresholds": "0.2",
                "tail_force": "0",
            },
        ),
    ],
}


def _sanitize(value: str) -> str:
    out = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    return out.strip("_") or "x"


def _sample_indices(count: int, limit: int) -> list[int]:
    if limit <= 0 or count <= limit:
        return list(range(count))
    if limit == 1:
        return [0]
    step = (count - 1) / float(limit - 1)
    return sorted({round(index * step) for index in range(limit)})


def _axis_names_for_mode(shape_label: str, mode: str) -> tuple[str, ...]:
    if mode not in MODE_CHOICES:
        choices = ", ".join(MODE_CHOICES)
        raise ValueError(f"unknown large-kernel plan mode {mode!r}; expected one of: {choices}")
    if mode == "current-candidate":
        return CURRENT_CANDIDATE_AXES[shape_label]
    return FUTURE_BLOCKED_AXES


def _seed_allowed_by_overrides(axis_values: dict[str, str], axis_overrides: dict[str, list[str]] | None) -> bool:
    if not axis_overrides:
        return True
    for axis, allowed in axis_overrides.items():
        if axis in axis_values and str(axis_values[axis]) not in {str(value) for value in allowed}:
            return False
    return True


def _row_name(
    shape_label: str,
    values_by_axis: dict[str, str],
    *,
    product_index: int | None = None,
    seed_name: str | None = None,
) -> str:
    name_bits = [shape_label, f"seed_{_sanitize(seed_name)}" if seed_name else f"p{product_index:04d}"]
    for axis, value in values_by_axis.items():
        if axis in {
            "panel_widths",
            "precision_modes",
            "warps_per_cta",
            "ctas_per_matrix",
            "cta_schedules",
            "sync_free_auto_policy",
            "auto_policy_groups",
            "compact_wy_tile_cols",
            "tail_cuts",
            "tail_thresholds",
            "tail_force",
            "panel_refresh_modes",
            "r_maintenance_modes",
        }:
            name_bits.append(_sanitize(f"{AXIS_TO_ENV[axis].lower()}_{value}"))
        if axis == "structured_before_cuda":
            name_bits.append(_sanitize(f"structured_before_cuda_{value}"))
    return "__".join(name_bits)


def _row_from_axis_values(
    shape_label: str,
    prefix: str,
    axes: list[tuple[str, list[str]]],
    values_by_axis: dict[str, str],
    *,
    mode: str,
    plan_index: int,
    product_index: int | None = None,
    seed_name: str | None = None,
) -> dict[str, Any]:
    axis_order = [axis for axis, _ in axes]
    env = {
        f"{prefix}_{AXIS_TO_ENV[axis]}": str(values_by_axis[axis])
        for axis in axis_order
        if axis in values_by_axis
    }
    if mode == "current-candidate":
        env.update(CURRENT_CANDIDATE_STATIC_ENV.get(shape_label, {}))
    return {
        "name": _row_name(shape_label, {axis: values_by_axis[axis] for axis in axis_order if axis in values_by_axis}, product_index=product_index, seed_name=seed_name),
        "env": env,
        "mode": mode,
        "effective_only": mode == "current-candidate",
        "plan_index": plan_index,
        "product_index": product_index,
        **({"seed_name": seed_name} if seed_name else {}),
    }


def _env_signature(env: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in env.items()))


def generate_configs(
    shape_label: str,
    *,
    max_configs: int = 32,
    env_prefix: str | None = None,
    mode: str = "future-blocked",
    axis_overrides: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    if shape_label not in PRESETS:
        known = ", ".join(sorted(PRESETS))
        raise ValueError(f"unknown QR kernel preset {shape_label!r}; expected one of: {known}")
    if axis_overrides:
        unknown = sorted(set(axis_overrides) - set(AXIS_TO_ENV))
        if unknown:
            raise ValueError(f"unknown large-kernel axis override(s): {', '.join(unknown)}")

    preset = PRESETS[shape_label]
    axis_names = _axis_names_for_mode(shape_label, mode)
    prefix = f"FAST_QR_{shape_label.upper()}" if mode == "current-candidate" else (env_prefix or f"FAST_QR_{shape_label.upper()}")
    axes = [
        (axis, axis_overrides.get(axis) if axis_overrides and axis in axis_overrides else preset[axis])
        for axis in axis_names
        if preset.get(axis)
    ]
    products = list(itertools.product(*(values for _, values in axes)))
    selected = _sample_indices(len(products), max_configs)

    rows: list[dict[str, Any]] = []
    seen_envs: set[tuple[tuple[str, str], ...]] = set()
    unlimited = max_configs <= 0

    if mode == "current-candidate":
        default_seed_values = {axis: str(values[0]) for axis, values in axes if values}
        for seed_name, seed_values in CURRENT_CANDIDATE_SEED_AXIS_VALUES.get(shape_label, []):
            if not unlimited and len(rows) >= max_configs:
                break
            complete_seed_values = dict(default_seed_values)
            complete_seed_values.update({axis: str(value) for axis, value in seed_values.items()})
            if not _seed_allowed_by_overrides(complete_seed_values, axis_overrides):
                continue
            if not all(axis in complete_seed_values for axis, _ in axes):
                continue
            row = _row_from_axis_values(
                shape_label,
                prefix,
                axes,
                complete_seed_values,
                mode=mode,
                plan_index=len(rows),
                seed_name=seed_name,
            )
            signature = _env_signature(row["env"])
            if signature in seen_envs:
                continue
            rows.append(row)
            seen_envs.add(signature)

    for ordinal, product_index in enumerate(selected):
        if not unlimited and len(rows) >= max_configs:
            break
        values = products[product_index]
        values_by_axis = {axis: str(value) for (axis, _), value in zip(axes, values)}
        row = _row_from_axis_values(
            shape_label,
            prefix,
            axes,
            values_by_axis,
            mode=mode,
            plan_index=len(rows),
            product_index=product_index,
        )
        signature = _env_signature(row["env"])
        if signature in seen_envs:
            continue
        rows.append(row)
        seen_envs.add(signature)
    return rows


def tune_command(shape_label: str, config_path: str, *, repeats: int = 3, official_stopping: bool = False) -> list[str]:
    preset = PRESETS[shape_label]
    cmd = [
        sys.executable,
        "tools/tune_candidate_configs.py",
        "--shape-label",
        shape_label,
        "--config-jsonl",
        config_path,
        "--correctness-indices",
        str(preset["correctness_indices"]),
        "--benchmark-indices",
        str(preset["benchmark_indices"]),
        "--popcorn-seeds",
        "public,1,2,3",
        "--repeats",
        str(repeats),
        "--allow-failed-configs",
    ]
    if official_stopping:
        cmd.append("--official-stopping")
    return cmd


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"name": row["name"], "env": row["env"]}, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate bounded candidate-config grids for QR benchmark shape families.",
        allow_abbrev=False,
    )
    parser.add_argument("--shape-label", required=True, choices=sorted(PRESETS))
    parser.add_argument("--max-configs", type=int, default=32)
    parser.add_argument("--env-prefix", default=None)
    parser.add_argument(
        "--mode",
        default="future-blocked",
        choices=MODE_CHOICES,
        help=(
            "future-blocked emits the full design grid; current-candidate emits only knobs "
            "that the current candidate implementation can use without bypass."
        ),
    )
    parser.add_argument("--out", default=None, help="Write tune_candidate_configs-compatible JSONL configs.")
    parser.add_argument("--print-command", action="store_true", help="Print the matching tune_candidate_configs.py command.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--official-stopping", action="store_true")
    args = parser.parse_args()

    rows = generate_configs(args.shape_label, max_configs=args.max_configs, env_prefix=args.env_prefix, mode=args.mode)
    out_path: Path | None = None
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        write_jsonl(out_path, rows)

    if not args.out:
        for row in rows:
            print(json.dumps({"name": row["name"], "env": row["env"]}, sort_keys=True))

    if args.print_command:
        config_path = str(out_path if out_path is not None else Path(f"results/{args.shape_label}_large_kernel_configs.jsonl"))
        print(" ".join(tune_command(args.shape_label, config_path, repeats=args.repeats, official_stopping=args.official_stopping)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
