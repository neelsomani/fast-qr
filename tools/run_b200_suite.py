from __future__ import annotations

import argparse
import json
import os
import selectors
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

from large_kernel_plan import (
    MODE_CHOICES as LARGE_KERNEL_PLAN_MODE_CHOICES,
    generate_configs as generate_large_kernel_configs,
    write_jsonl as write_large_kernel_config_jsonl,
)
from qr_common import ROOT, file_provenance, repo_provenance, tracked_runtime_env
from tune_tail_policy import DEFAULT_CONFIGS as DEFAULT_TAIL_TUNE_CONFIGS
from validate_b200_suite import validate_suite


DEFAULT_BENCHMARK_EXPERIMENT_INDEXES = "1,2,3,4,5,6,7,8,9,10,11"
DEFAULT_QUANTIZATION_SWEEP_INDEXES = "3,4,5,6,7,8,9,10,11"
DEFAULT_BLOCKED_QR_SWEEP_INDEXES = "3,4,19,20"
DEFAULT_CANDIDATE_CONFIG_CORRECTNESS_INDICES = "19,20,21"
ROUTE_ABLATIONS = [
    (
        "no_route_cache",
        {"FAST_QR_DISABLE_ROUTE_CACHE": "1"},
        "candidate_ablation_no_route_cache_public.jsonl",
    ),
    (
        "cuda_first_structured_routes",
        {
            "FAST_QR_QR512_STRUCTURED_BEFORE_CUDA": "0",
            "FAST_QR_QR1024_STRUCTURED_BEFORE_CUDA": "0",
            "FAST_QR_DISABLE_B200_DEFAULT_STRUCTURED_BEFORE_CUDA": "1",
        },
        "candidate_ablation_cuda_first_structured_routes_public.jsonl",
    ),
    (
        "no_structured_routes",
        {"FAST_QR_DISABLE_STRUCTURED_ROUTES": "1"},
        "candidate_ablation_no_structured_routes_public.jsonl",
    ),
    (
        "no_dense_tail",
        {"FAST_QR_DISABLE_DENSE_TAIL": "1"},
        "candidate_ablation_no_dense_tail_public.jsonl",
    ),
    (
        "no_data_dependent_routes",
        {"FAST_QR_DISABLE_DATA_DEPENDENT_ROUTES": "1"},
        "candidate_ablation_no_data_dependent_routes_public.jsonl",
    ),
    (
        "no_qr512_qr1024_cuda",
        {
            "FAST_QR_DISABLE_QR512_CUDA": "1",
            "FAST_QR_DISABLE_QR1024_CUDA": "1",
            "FAST_QR_DISABLE_QR512_BLOCKED_CUDA": "1",
            "FAST_QR_DISABLE_QR1024_BLOCKED_CUDA": "1",
        },
        "candidate_ablation_no_qr512_qr1024_cuda_public.jsonl",
    ),
]

CANDIDATE_CONFIG_PREFLIGHT_ACCELERATORS = {
    "qr512": "qr512_blocked_cuda_auto",
    "qr1024": "qr1024_blocked_cuda_auto",
    "qr2048": "qr2048_blocked_cuda_auto",
    "qr4096": "qr4096_blocked_cuda_auto",
}


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_command(cmd: list[str], log_path: Path, env: dict[str, str], timeout_s: float | None = None) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    start = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert proc.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        timed_out = False
        returncode = 0
        try:
            while True:
                elapsed_s = time.perf_counter() - start
                if timeout_s is not None and elapsed_s > timeout_s:
                    timed_out = True
                    break
                for key, _ in selector.select(timeout=1.0):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        break
                    text = chunk.decode("utf-8", errors="replace")
                    log.write(text)
                    log.flush()
                    print(text, end="", flush=True)
                if proc.poll() is not None:
                    for chunk in iter(lambda: os.read(proc.stdout.fileno(), 65536), b""):
                        text = chunk.decode("utf-8", errors="replace")
                        log.write(text)
                        log.flush()
                        print(text, end="", flush=True)
                    break
            if timed_out:
                message = f"\ncommand timed out after {timeout_s:.1f}s\n"
                log.write(message)
                log.flush()
                print(message, end="", flush=True)
                proc.terminate()
                try:
                    proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10.0)
                raise TimeoutError(f"command timed out after {timeout_s:.1f}s: {' '.join(cmd)}")
            returncode = proc.wait()
        finally:
            selector.close()
        log.write(f"\nexit_code={returncode}; elapsed_s={time.perf_counter() - start:.3f}\n")
    if returncode != 0:
        raise RuntimeError(f"command failed with exit code {returncode}: {' '.join(cmd)}")


def append_manifest(manifest_path: Path, row: dict) -> None:
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def suite_provenance(args: argparse.Namespace, env: dict[str, str]) -> dict:
    submission = ROOT / args.submission if not Path(args.submission).is_absolute() else Path(args.submission)
    baseline = ROOT / args.baseline if not Path(args.baseline).is_absolute() else Path(args.baseline)
    return {
        "event": "suite_provenance",
        "time": datetime.now().isoformat(),
        "repo": repo_provenance(),
        "submission": file_provenance(submission),
        "baseline": file_provenance(baseline),
        "args": vars(args),
        "env": tracked_runtime_env(env, include_absent=True),
    }


def suite_env_overrides(args: argparse.Namespace) -> dict[str, str | None]:
    qr32_extra_cuda_cflags = args.qr32_extra_cuda_cflags
    if qr32_extra_cuda_cflags is None and getattr(args, "qr32_sm100", False):
        qr32_extra_cuda_cflags = "-arch=sm_100"
    return {
        "TORCH_CUDA_ARCH_LIST": args.torch_cuda_arch_list,
        "FAST_QR_QR32_EXTRA_CUDA_CFLAGS": qr32_extra_cuda_cflags,
    }


def apply_suite_env_options(env: dict[str, str], args: argparse.Namespace) -> dict[str, str]:
    out = env.copy()
    for key, value in suite_env_overrides(args).items():
        if value is None:
            continue
        if value == "":
            out.pop(key, None)
        else:
            out[key] = value
    return out


def visible_suite_env(env: dict[str, str], args: argparse.Namespace) -> dict[str, str | None]:
    keys = suite_env_overrides(args)
    return {key: env.get(key) for key in keys if env.get(key) is not None}


def merged_env(base: dict[str, str], overrides: dict[str, str] | None) -> dict[str, str]:
    if not overrides:
        return base
    out = base.copy()
    out.update(overrides)
    return out


def parse_int_list(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def tail_tune_config_count(args: argparse.Namespace) -> int:
    inline_count = len(args.tail_tune_config or [])
    if args.tail_tune_config_jsonl:
        path = ROOT / args.tail_tune_config_jsonl if not Path(args.tail_tune_config_jsonl).is_absolute() else Path(args.tail_tune_config_jsonl)
        try:
            file_count = sum(1 for line in path.read_text().splitlines() if line.strip())
        except OSError:
            file_count = 1
        return max(1, file_count + inline_count)
    return len(DEFAULT_TAIL_TUNE_CONFIGS) + inline_count


def candidate_config_tune_count(args: argparse.Namespace) -> int:
    inline_count = len(args.candidate_config_tune_config or [])
    file_count = 0
    if args.candidate_config_tune_config_jsonl:
        path = (
            ROOT / args.candidate_config_tune_config_jsonl
            if not Path(args.candidate_config_tune_config_jsonl).is_absolute()
            else Path(args.candidate_config_tune_config_jsonl)
        )
        try:
            file_count = sum(1 for line in path.read_text().splitlines() if line.strip())
        except OSError:
            file_count = 1
    generated_plan_count = 0
    if getattr(args, "candidate_config_tune_large_kernel_plan_mode", None):
        try:
            generated_plan_count = len(candidate_config_tune_large_kernel_plan_rows(args))
        except (KeyError, ValueError):
            generated_plan_count = 1
    grid_axes = [
        args.candidate_config_tune_panel_widths,
        args.candidate_config_tune_update_modes,
        args.candidate_config_tune_precision_modes,
        args.candidate_config_tune_tile_ms,
        args.candidate_config_tune_tile_ns,
        getattr(args, "candidate_config_tune_compact_wy_tile_cols", ""),
        args.candidate_config_tune_warps_per_cta,
        args.candidate_config_tune_ctas_per_matrix,
        getattr(args, "candidate_config_tune_cta_schedules", ""),
        getattr(args, "candidate_config_tune_sync_free_auto_policy", ""),
        getattr(args, "candidate_config_tune_auto_policy_groups", ""),
        args.candidate_config_tune_cluster_sizes,
        args.candidate_config_tune_tail_cuts,
        getattr(args, "candidate_config_tune_tail_thresholds", ""),
        getattr(args, "candidate_config_tune_tail_force", ""),
        args.candidate_config_tune_panel_refreshes,
        args.candidate_config_tune_panel_refresh_modes,
        args.candidate_config_tune_r_maintenance_modes,
        args.candidate_config_tune_structured_before_cuda,
    ]
    grid_count = 1
    has_grid = False
    for raw in grid_axes:
        values = [item for item in raw.split(",") if item.strip()]
        if values:
            has_grid = True
            grid_count *= len(values)
    default_count = 0 if args.candidate_config_tune_no_default else 1
    return max(1, default_count + file_count + inline_count + generated_plan_count + (grid_count if has_grid else 0))


def candidate_config_tune_large_kernel_plan_rows(args: argparse.Namespace) -> list[dict]:
    return generate_large_kernel_configs(
        args.candidate_config_tune_shape_label,
        max_configs=args.candidate_config_tune_large_kernel_plan_max_configs,
        env_prefix=args.candidate_config_tune_env_prefix,
        mode=args.candidate_config_tune_large_kernel_plan_mode,
        axis_overrides=candidate_config_tune_large_kernel_axis_overrides(args),
    )


def candidate_config_tune_large_kernel_plan_path(suite_dir: Path) -> Path:
    return suite_dir / "candidate_config_tune_large_kernel_configs.jsonl"


def candidate_config_accelerator_preflight_path(suite_dir: Path) -> Path:
    return suite_dir / "candidate_config_accelerator_preflight.jsonl"


def candidate_config_accelerator_for_shape(shape_label: str) -> str | None:
    return CANDIDATE_CONFIG_PREFLIGHT_ACCELERATORS.get(shape_label)


def _required_kernel_priority(case_index: int, row: dict) -> tuple[int, int]:
    batch = int(row.get("batch") or 0)
    n = int(row.get("n") or 0)
    if batch == 640 and n == 512:
        return (0, case_index)
    if batch == 60 and n == 1024:
        return (1, case_index)
    if n in {2048, 4096}:
        return (2, case_index)
    return (3, case_index)


def candidate_config_next_required_target(args: argparse.Namespace) -> dict:
    targets = candidate_config_required_targets(args)
    if not targets:
        raise ValueError("candidate policy has no rows with required_cuda_kernel and candidate_config_shape_label")
    return targets[0]


def candidate_config_required_targets(args: argparse.Namespace) -> list[dict]:
    from candidate_policy import policy_rows

    benchmark_cases = getattr(args, "candidate_config_tune_benchmark_cases", "cases/public_benchmarks.txt")
    cases_path = ROOT / benchmark_cases if not Path(benchmark_cases).is_absolute() else Path(benchmark_cases)
    rows = policy_rows(getattr(args, "submission", "submissions/candidate.py"), cases_path)
    candidates = sorted(
        [
            (case_index, row)
            for case_index, row in enumerate(rows)
            if row.get("required_cuda_kernel") and row.get("candidate_config_shape_label")
        ],
        key=lambda item: _required_kernel_priority(*item),
    )
    targets_by_shape: dict[str, dict] = {}
    for case_index, row in candidates:
        shape_label = str(row["candidate_config_shape_label"])
        if shape_label in targets_by_shape:
            targets_by_shape[shape_label]["source_case_indices"].append(case_index)
            targets_by_shape[shape_label]["source_specs"].append(row.get("spec"))
            continue
        targets_by_shape[shape_label] = _candidate_config_target_from_policy_row(case_index, row)
    return list(targets_by_shape.values())


def _candidate_config_target_from_policy_row(case_index: int, row: dict) -> dict:
    return {
        "source": "candidate_policy",
        "case_index": case_index,
        "source_case_indices": [case_index],
        "spec": row.get("spec"),
        "source_specs": [row.get("spec")],
        "shape_label": row["candidate_config_shape_label"],
        "env_prefix": row.get("candidate_config_env_prefix"),
        "benchmark_indices": row.get("candidate_config_benchmark_indices", ""),
        "correctness_indices": row.get("candidate_config_correctness_indices", ""),
        "required_cuda_kernel": row.get("required_cuda_kernel"),
        "required_repair_modes": row.get("required_repair_modes", []),
    }


def candidate_config_tune_command_for_target(
    target: dict,
    *,
    suite_name: str,
    max_configs: int = 8,
    mode: str = "current-candidate",
    python: str = "python",
) -> list[str]:
    command = [
        python,
        "tools/run_b200_suite.py",
        "--suite-name",
        suite_name,
        "--include-candidate-config-tune",
        "--candidate-config-tune-shape-label",
        str(target["shape_label"]),
        "--candidate-config-tune-large-kernel-plan-mode",
        mode,
        "--candidate-config-tune-large-kernel-plan-max-configs",
        str(max_configs),
    ]
    if target.get("env_prefix"):
        command.extend(["--candidate-config-tune-env-prefix", str(target["env_prefix"])])
    if target.get("correctness_indices"):
        command.extend(["--candidate-config-tune-correctness-indices", str(target["correctness_indices"])])
    if target.get("benchmark_indices"):
        command.extend(["--candidate-config-tune-benchmark-indices", str(target["benchmark_indices"])])
    for axis, value in _repair_mode_axis_constraints(target.get("required_repair_modes", [])).items():
        if axis == "panel_refresh_modes":
            command.extend(["--candidate-config-tune-panel-refresh-modes", value])
        elif axis == "r_maintenance_modes":
            command.extend(["--candidate-config-tune-r-maintenance-modes", value])
    return command


def _csv_override_values(raw: object) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def candidate_config_tune_large_kernel_axis_overrides(args: argparse.Namespace) -> dict[str, list[str]]:
    attr_by_axis = {
        "panel_widths": "candidate_config_tune_panel_widths",
        "update_modes": "candidate_config_tune_update_modes",
        "precision_modes": "candidate_config_tune_precision_modes",
        "tile_ms": "candidate_config_tune_tile_ms",
        "tile_ns": "candidate_config_tune_tile_ns",
        "compact_wy_tile_cols": "candidate_config_tune_compact_wy_tile_cols",
        "warps_per_cta": "candidate_config_tune_warps_per_cta",
        "ctas_per_matrix": "candidate_config_tune_ctas_per_matrix",
        "cta_schedules": "candidate_config_tune_cta_schedules",
        "sync_free_auto_policy": "candidate_config_tune_sync_free_auto_policy",
        "auto_policy_groups": "candidate_config_tune_auto_policy_groups",
        "cluster_sizes": "candidate_config_tune_cluster_sizes",
        "tail_cuts": "candidate_config_tune_tail_cuts",
        "tail_thresholds": "candidate_config_tune_tail_thresholds",
        "tail_force": "candidate_config_tune_tail_force",
        "panel_refreshes": "candidate_config_tune_panel_refreshes",
        "panel_refresh_modes": "candidate_config_tune_panel_refresh_modes",
        "r_maintenance_modes": "candidate_config_tune_r_maintenance_modes",
        "structured_before_cuda": "candidate_config_tune_structured_before_cuda",
    }
    overrides: dict[str, list[str]] = {}
    for axis, attr in attr_by_axis.items():
        values = _csv_override_values(getattr(args, attr, ""))
        if values:
            overrides[axis] = values
    target = getattr(args, "candidate_config_tune_policy_target", None)
    if isinstance(target, dict):
        for axis, value in _repair_mode_axis_constraints(target.get("required_repair_modes", [])).items():
            overrides.setdefault(axis, [value])
    return overrides


def _repair_mode_axis_constraints(required_modes: object) -> dict[str, str]:
    constraints: dict[str, str] = {}
    for raw in required_modes if isinstance(required_modes, list) else []:
        if not isinstance(raw, str) or "=" not in raw:
            continue
        key, value = [part.strip() for part in raw.split("=", 1)]
        if key == "panel_refresh_mode" and value:
            constraints["panel_refresh_modes"] = value
        elif key == "r_maintenance_mode" and value:
            constraints["r_maintenance_modes"] = value
    return constraints


def apply_candidate_config_next_required_target(args: argparse.Namespace) -> dict:
    target = candidate_config_next_required_target(args)
    shape_label = str(target["shape_label"])
    if args.candidate_config_tune_shape_label in {"global", "auto", ""}:
        args.candidate_config_tune_shape_label = shape_label
    elif args.candidate_config_tune_shape_label != shape_label:
        raise ValueError(
            "--candidate-config-tune-next-required selected "
            f"{shape_label!r}, but --candidate-config-tune-shape-label is "
            f"{args.candidate_config_tune_shape_label!r}; remove one of these options"
        )

    if args.candidate_config_tune_env_prefix is None and target.get("env_prefix"):
        args.candidate_config_tune_env_prefix = str(target["env_prefix"])
    if not args.candidate_config_tune_benchmark_indices and target.get("benchmark_indices"):
        args.candidate_config_tune_benchmark_indices = str(target["benchmark_indices"])
    if (
        args.candidate_config_tune_correctness_indices in {"", DEFAULT_CANDIDATE_CONFIG_CORRECTNESS_INDICES}
        and target.get("correctness_indices")
    ):
        args.candidate_config_tune_correctness_indices = str(target["correctness_indices"])
    if args.candidate_config_tune_large_kernel_plan_mode is None:
        args.candidate_config_tune_large_kernel_plan_mode = "current-candidate"
    target["large_kernel_plan_mode"] = args.candidate_config_tune_large_kernel_plan_mode
    target["effective_only"] = args.candidate_config_tune_large_kernel_plan_mode == "current-candidate"
    applied_constraints: dict[str, str] = {}
    constraint_attr_by_axis = {
        "panel_refresh_modes": "candidate_config_tune_panel_refresh_modes",
        "r_maintenance_modes": "candidate_config_tune_r_maintenance_modes",
    }
    for axis, value in _repair_mode_axis_constraints(target.get("required_repair_modes", [])).items():
        attr = constraint_attr_by_axis[axis]
        if not getattr(args, attr, ""):
            applied_constraints[axis] = value
    if applied_constraints:
        target["applied_axis_constraints"] = applied_constraints
    if target["effective_only"]:
        target["note"] = (
            "auto target defaults to current-candidate mode so B200 tuning covers only env knobs "
            "the current submission can consume; pass --candidate-config-tune-large-kernel-plan-mode "
            "future-blocked to emit the blocked-kernel design grid"
        )
    args.candidate_config_tune_policy_target = target
    return target


def write_candidate_config_tune_large_kernel_plan(args: argparse.Namespace, suite_dir: Path) -> tuple[Path, list[dict]]:
    rows = candidate_config_tune_large_kernel_plan_rows(args)
    path = candidate_config_tune_large_kernel_plan_path(suite_dir)
    write_large_kernel_config_jsonl(path, rows)
    return path, rows


def step_timeout_s(step_name: str, args: argparse.Namespace) -> float | None:
    if step_name != "pytest":
        return None
    timeout_s = getattr(args, "pytest_timeout_s", None)
    if timeout_s is None or timeout_s <= 0:
        return None
    return float(timeout_s)


def candidate_config_tune_large_kernel_plan_preview(args: argparse.Namespace, suite_dir: Path) -> dict | None:
    if not getattr(args, "include_candidate_config_tune", False) or not getattr(
        args,
        "candidate_config_tune_large_kernel_plan_mode",
        None,
    ):
        return None
    rows = candidate_config_tune_large_kernel_plan_rows(args)
    return {
        "path": str(candidate_config_tune_large_kernel_plan_path(suite_dir)),
        "mode": args.candidate_config_tune_large_kernel_plan_mode,
        "shape_label": args.candidate_config_tune_shape_label,
        "env_prefix": args.candidate_config_tune_env_prefix,
        "max_configs": args.candidate_config_tune_large_kernel_plan_max_configs,
        "num_configs": len(rows),
        "config_names": [row.get("name") for row in rows],
        "configs": rows,
    }


def candidate_config_benchmark_case_count(args: argparse.Namespace) -> int:
    values = [item for item in args.candidate_config_tune_benchmark_indices.split(",") if item.strip()]
    return len(values) if values else 12


def make_tarball(suite_dir: Path) -> Path:
    tar_path = suite_dir.with_suffix(".tgz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(suite_dir, arcname=suite_dir.name)
    return tar_path


def default_validation_blockers(args: argparse.Namespace) -> list[str]:
    blockers = []
    for name in [
        "skip_policy",
        "skip_submission_validation",
        "skip_route_trace",
        "skip_guard_benchmark",
        "skip_route_ablations",
        "skip_secret_audit",
        "skip_runtime_preflight",
        "skip_seed_sweep",
        "skip_quantization_sweep",
        "skip_mixed_seed_sweep",
        "skip_classifier_sweep",
        "skip_tail_policy_sweep",
        "skip_candidate_tests",
        "skip_benchmark_correctness",
        "skip_dev_robustness",
        "skip_accelerator_preflight",
        "skip_smoke",
        "skip_baseline_public",
        "skip_candidate_public",
        "skip_experiments",
        "skip_official_style",
        "skip_candidate_official_style",
        "skip_pytest",
    ]:
        if getattr(args, name):
            blockers.append(f"--{name.replace('_', '-')}")
    if args.candidate_test_indices:
        blockers.append("--candidate-test-indices")
    return blockers


def step_record(step: tuple[str, list[str]] | tuple[str, list[str], dict[str, str]]) -> dict:
    if len(step) == 2:
        step_name, cmd = step
        env_overrides = {}
    else:
        step_name, cmd, env_overrides = step
    return {
        "step": step_name,
        "cmd": cmd,
        "env_overrides": env_overrides,
    }


def step_category(step_name: str) -> str:
    if step_name in {
        "print_spec",
        "sync_cases_check",
        "secret_audit",
        "submission_validation",
        "runtime_preflight",
        "accelerator_preflight",
        "candidate_config_accelerator_preflight",
        "pytest",
    }:
        return "preflight"
    if step_name in {
        "seed_sweep_margin",
        "quantization_seed_sweep",
        "mixed_seed_sweep",
        "tail_policy_sweep",
        "candidate_public_tests",
        "candidate_public_benchmark_correctness",
        "candidate_dev_robustness",
        "local_official_eval_test",
    }:
        return "correctness"
    if step_name == "tail_policy_tune":
        return "tail_policy_tuning"
    if step_name == "candidate_config_tune":
        return "candidate_config_tuning"
    if step_name == "classifier_seed_sweep":
        return "dispatch_analysis"
    if step_name.startswith("baseline_") or step_name.startswith("candidate_"):
        if "ablation" in step_name:
            return "route_ablation_timing"
        if step_name in {
            "candidate_policy",
            "candidate_implementation_status",
            "candidate_route_trace",
            "candidate_guard_overhead",
        }:
            return "dispatch_analysis"
        return "benchmark_timing"
    if step_name == "blocked_qr_sweep":
        return "verifier_experiment"
    if step_name.startswith("experiments_"):
        return "verifier_experiment"
    if step_name in {"suite_summary", "suite_analysis", "suite_validation"}:
        return "export_analysis"
    if step_name in {"popcorn_test", "popcorn_leaderboard"}:
        return "official_remote"
    return "other"


def dry_run_workload_summary(
    steps: list[tuple[str, list[str]] | tuple[str, list[str], dict[str, str]]],
    args: argparse.Namespace,
) -> dict:
    categories: dict[str, int] = {}
    gpu_heavy_steps = []
    for step in steps:
        step_name = step[0]
        category = step_category(step_name)
        categories[category] = categories.get(category, 0) + 1
        if category in {"correctness", "benchmark_timing", "route_ablation_timing", "verifier_experiment"}:
            gpu_heavy_steps.append(step_name)
        if category in {"tail_policy_tuning", "candidate_config_tuning"}:
            gpu_heavy_steps.append(step_name)
    return {
        "step_counts_by_category": dict(sorted(categories.items())),
        "num_gpu_heavy_steps": len(gpu_heavy_steps),
        "gpu_heavy_steps": gpu_heavy_steps,
        "num_benchmark_timing_steps": categories.get("benchmark_timing", 0),
        "num_route_ablation_timing_steps": categories.get("route_ablation_timing", 0),
        "num_verifier_experiment_steps": categories.get("verifier_experiment", 0),
        "num_tail_policy_tuning_steps": categories.get("tail_policy_tuning", 0),
        "num_candidate_config_tuning_steps": categories.get("candidate_config_tuning", 0),
        "num_official_remote_steps": categories.get("official_remote", 0),
        "runtime_estimate": estimate_b200_runtime(steps, args),
        "note": "Exact elapsed time is hardware- and candidate-dependent; completed runs record timings in suite_summary.md.",
    }


def _add_estimate(
    totals: dict[str, dict[str, float]],
    category: str,
    low_min: float,
    high_min: float,
) -> None:
    bucket = totals.setdefault(category, {"low_minutes": 0.0, "high_minutes": 0.0})
    bucket["low_minutes"] += low_min
    bucket["high_minutes"] += max(low_min, high_min)


def _step_estimate_minutes(step_name: str, args: argparse.Namespace) -> tuple[str, float, float]:
    if step_name in {
        "print_spec",
        "sync_cases_check",
        "secret_audit",
        "submission_validation",
        "runtime_preflight",
        "pytest",
        "suite_summary",
        "suite_analysis",
        "suite_validation",
    }:
        return "fixed_overhead", 0.05, 1.0
    if step_name == "accelerator_preflight":
        return "fixed_overhead", 0.5, 5.0
    if step_name == "candidate_config_accelerator_preflight":
        configs = candidate_config_tune_count(args)
        scale = max(1.0, configs / 8.0)
        return "fixed_overhead", 0.5 * scale, 5.0 * scale
    if step_name in {"candidate_policy", "candidate_route_trace", "candidate_guard_overhead"}:
        return "dispatch_analysis", 0.2, 4.0
    if step_name == "seed_sweep_margin":
        seeds = len([item for item in args.seed_sweep_popcorn_seeds.split(",") if item.strip()])
        cases = len([item for item in args.seed_sweep_indices.split(",") if item.strip()])
        scale = max(1.0, seeds * cases / 12.0)
        return "correctness_and_diagnostics", 2.0 * scale, 10.0 * scale
    if step_name == "quantization_seed_sweep":
        seeds = len([item for item in args.quantization_sweep_popcorn_seeds.split(",") if item.strip()])
        cases = len([item for item in args.quantization_sweep_indices.split(",") if item.strip()])
        experiments = len([item for item in args.quantization_sweep_experiments.split(",") if item.strip()])
        scale = max(1.0, seeds * cases * max(1, experiments) / 36.0)
        return "correctness_and_diagnostics", 5.0 * scale, 45.0 * scale
    if step_name == "mixed_seed_sweep":
        seeds = len([item for item in args.mixed_sweep_popcorn_seeds.split(",") if item.strip()])
        benchmark_cases = len([item for item in args.mixed_sweep_benchmark_indices.split(",") if item.strip()])
        test_cases = len([item for item in args.mixed_sweep_test_indices.split(",") if item.strip()])
        scale = max(1.0, seeds * (benchmark_cases + test_cases) / 20.0)
        return "correctness_and_diagnostics", 4.0 * scale, 35.0 * scale
    if step_name == "classifier_seed_sweep":
        seeds = len([item for item in args.classifier_sweep_popcorn_seeds.split(",") if item.strip()])
        cases = len([item for item in args.classifier_sweep_indices.split(",") if item.strip()])
        scale = max(1.0, seeds * cases / 28.0)
        return "dispatch_analysis", 0.5 * scale, 5.0 * scale
    if step_name == "tail_policy_sweep":
        seeds = len([item for item in args.tail_policy_popcorn_seeds.split(",") if item.strip()])
        cases = len([item for item in args.tail_policy_indices.split(",") if item.strip()])
        scale = max(1.0, seeds * cases / 28.0)
        return "correctness_and_diagnostics", 5.0 * scale, 30.0 * scale
    if step_name == "tail_policy_tune":
        seeds = len([item for item in args.tail_policy_popcorn_seeds.split(",") if item.strip()])
        cases = len([item for item in args.tail_policy_indices.split(",") if item.strip()])
        configs = tail_tune_config_count(args)
        correctness_scale = max(1.0, seeds * cases / 28.0)
        bench_scale = max(1.0, args.tail_tune_repeats / 3.0)
        official_scale = 2.0 if args.tail_tune_official_stopping else 1.0
        return (
            "tail_policy_tuning",
            configs * (2.0 * correctness_scale + 0.75 * bench_scale),
            configs * (10.0 * correctness_scale + 5.0 * bench_scale * official_scale),
        )
    if step_name == "candidate_config_tune":
        seeds = len([item for item in args.candidate_config_tune_popcorn_seeds.split(",") if item.strip()])
        cases = len([item for item in args.candidate_config_tune_correctness_indices.split(",") if item.strip()])
        configs = candidate_config_tune_count(args)
        correctness_scale = max(1.0, seeds * cases / 12.0)
        benchmark_case_scale = max(0.25, candidate_config_benchmark_case_count(args) / 12.0)
        bench_scale = max(1.0, args.candidate_config_tune_repeats / 3.0) * benchmark_case_scale
        official_scale = 2.0 if args.candidate_config_tune_official_stopping else 1.0
        return (
            "candidate_config_tuning",
            configs * (2.0 * correctness_scale + 0.75 * bench_scale),
            configs * (12.0 * correctness_scale + 5.0 * bench_scale * official_scale),
        )
    if step_name in {
        "candidate_public_tests",
        "candidate_public_benchmark_correctness",
        "candidate_dev_robustness",
        "local_official_eval_test",
    }:
        return "correctness_and_diagnostics", 2.0, 18.0
    if step_name in {"baseline_smoke", "candidate_smoke"}:
        return "benchmark_timing", 0.5 * max(1, args.smoke_repeats), 2.0 * max(1, args.smoke_repeats)
    if step_name == "baseline_public":
        return "benchmark_timing", 0.75 * max(1, args.baseline_repeats), 3.0 * max(1, args.baseline_repeats)
    if step_name == "candidate_public":
        return "benchmark_timing", 0.75 * max(1, args.candidate_repeats), 4.0 * max(1, args.candidate_repeats)
    if step_name.startswith("candidate_ablation_"):
        return "benchmark_timing", 0.75 * max(1, args.ablation_repeats), 3.5 * max(1, args.ablation_repeats)
    if step_name in {"baseline_official_style", "candidate_official_style"}:
        # bench_local defaults to a 10s official-style per-case threshold over 12 public cases.
        repeats_scale = min(2.0, max(1.0, args.official_repeats / 1000.0))
        return "official_style_timing", 3.0 * repeats_scale, 12.0 * repeats_scale
    if step_name == "blocked_qr_sweep":
        cases = len([item for item in args.blocked_qr_sweep_indices.split(",") if item.strip()])
        panels = len([item for item in args.blocked_qr_sweep_panel_widths.split(",") if item.strip()])
        update_modes = len([item for item in args.blocked_qr_sweep_update_modes.split(",") if item.strip()])
        precision_modes = len([item for item in args.blocked_qr_sweep_precision_modes.split(",") if item.strip()])
        r_modes = len([item for item in args.blocked_qr_sweep_r_maintenance_modes.split(",") if item.strip()])
        refresh_modes = len([item for item in args.blocked_qr_sweep_panel_refresh_modes.split(",") if item.strip()])
        rows = max(1, cases * panels * update_modes * precision_modes * r_modes * refresh_modes)
        scale = max(1.0, rows / 32.0)
        return "verifier_experiments", 2.0 * scale, 25.0 * scale
    if step_name.startswith("experiments_public_benchmark_"):
        tail_cuts = len([item for item in args.experiment_tail_cuts.split(",") if item.strip()])
        return "verifier_experiments", 1.5, 4.0 + 0.5 * max(0, tail_cuts - 1)
    if step_name in {"popcorn_test", "popcorn_leaderboard"}:
        if args.popcorn_timeout_s:
            high = max(0.1, args.popcorn_timeout_s / 60.0)
            return "official_remote", min(0.1, high), high
        return "official_remote", 3.0, 30.0
    return "other", 0.1, 2.0


def estimate_b200_runtime(
    steps: list[tuple[str, list[str]] | tuple[str, list[str], dict[str, str]]],
    args: argparse.Namespace,
) -> dict:
    by_category: dict[str, dict[str, float]] = {}
    by_step = []
    for step in steps:
        step_name = step[0]
        category, low_min, high_min = _step_estimate_minutes(step_name, args)
        _add_estimate(by_category, category, low_min, high_min)
        by_step.append(
            {
                "step": step_name,
                "category": category,
                "low_minutes": round(low_min, 1),
                "high_minutes": round(high_min, 1),
            }
        )

    low_total = sum(bucket["low_minutes"] for bucket in by_category.values())
    high_total = sum(bucket["high_minutes"] for bucket in by_category.values())
    slowest = sorted(by_step, key=lambda row: row["high_minutes"], reverse=True)[:8]
    return {
        "rough": True,
        "low_minutes": round(low_total, 1),
        "high_minutes": round(high_total, 1),
        "low_hours": round(low_total / 60.0, 2),
        "high_hours": round(high_total / 60.0, 2),
        "by_category": {
            key: {
                "low_minutes": round(value["low_minutes"], 1),
                "high_minutes": round(value["high_minutes"], 1),
            }
            for key, value in sorted(by_category.items())
        },
        "slowest_steps": slowest,
        "assumptions": [
            "Rough planning estimate for a single B200; real timings are recorded in manifest.jsonl.",
            "Large diagnostic and verifier-experiment steps can dominate wall time.",
            "Official-style timing uses adaptive stopping, so repeat counts are upper bounds.",
        ],
    }


def dry_run_plan(
    suite_dir: Path,
    steps: list[tuple[str, list[str]] | tuple[str, list[str], dict[str, str]]],
    validation_blockers: list[str],
    args: argparse.Namespace,
    env: dict[str, str],
) -> dict:
    plan = {
        "dry_run": True,
        "suite_dir": str(suite_dir),
        "suite_env": visible_suite_env(env, args),
        "num_steps": len(steps),
        "will_validate_suite": not args.skip_suite_validation and not validation_blockers,
        "will_validate_completed_export": not args.skip_suite_validation and not validation_blockers,
        "will_require_final_kernels": bool(getattr(args, "require_final_kernels", False)),
        "validation_blockers": validation_blockers,
        "workload": dry_run_workload_summary(steps, args),
        "steps": [step_record(step) for step in steps],
    }
    if getattr(args, "candidate_config_tune_policy_target", None):
        plan["candidate_config_tune_policy_target"] = args.candidate_config_tune_policy_target
    generated_candidate_config_plan = candidate_config_tune_large_kernel_plan_preview(args, suite_dir)
    if generated_candidate_config_plan is not None:
        plan["candidate_config_tune_large_kernel_plan"] = generated_candidate_config_plan
    return plan


def print_dry_run_plan(plan: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(plan, sort_keys=True))
        return

    print(f"DRY RUN: {plan['suite_dir']}")
    print(f"steps: {plan['num_steps']}")
    if plan["will_validate_suite"]:
        print("suite validation: enabled")
        if plan.get("will_require_final_kernels"):
            print("final-kernel readiness: required")
    else:
        blockers = ", ".join(plan["validation_blockers"]) if plan["validation_blockers"] else "disabled by flag"
        print(f"suite validation: skipped ({blockers})")
    if plan["will_validate_completed_export"]:
        print("post-export validation: enabled")
    else:
        print("post-export validation: skipped")
    if plan.get("suite_env"):
        print("suite env: " + json.dumps(plan["suite_env"], sort_keys=True))
    if plan.get("candidate_config_tune_policy_target"):
        target = plan["candidate_config_tune_policy_target"]
        print(
            "candidate config target: "
            f"{target.get('shape_label')} / {target.get('required_cuda_kernel')} "
            f"(case {target.get('case_index')})"
        )
    if plan.get("candidate_config_tune_large_kernel_plan"):
        generated = plan["candidate_config_tune_large_kernel_plan"]
        names = ", ".join(generated.get("config_names") or [])
        print(
            "generated candidate configs: "
            f"{generated.get('num_configs')} -> {generated.get('path')}"
        )
        if names:
            print(f"generated config names: {names}")
    workload = plan.get("workload") or {}
    if workload:
        counts = ", ".join(
            f"{key}={value}" for key, value in workload.get("step_counts_by_category", {}).items()
        )
        print(f"workload: {counts}")
        print(f"gpu-heavy steps: {workload.get('num_gpu_heavy_steps', 0)}")
        estimate = workload.get("runtime_estimate") or {}
        if estimate:
            print(
                "estimated B200 wall time: "
                f"{estimate.get('low_minutes')} - {estimate.get('high_minutes')} min "
                f"({estimate.get('low_hours')} - {estimate.get('high_hours')} hr, rough)"
            )
    for index, step in enumerate(plan["steps"], start=1):
        env = step["env_overrides"]
        env_part = ""
        if env:
            env_part = " env=" + json.dumps(env, sort_keys=True)
        print(f"{index:02d}. {step['step']}{env_part}: {' '.join(step['cmd'])}")


def validate_completed_export(suite_dir: Path, require_final_kernels: bool = False) -> None:
    result = validate_suite(
        suite_dir,
        require_finish=True,
        require_final_kernels=require_final_kernels,
    )
    if result["ok"]:
        print("post-export validation: PASS")
        return
    preview = "; ".join(f"{error['check']}: {error['message']}" for error in result["errors"][:5])
    raise RuntimeError(f"post-export validation failed with {result['num_errors']} errors: {preview}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the first-pass B200 QR lab suite and export JSONL results.",
        allow_abbrev=False,
    )
    parser.add_argument("--submission", default="submissions/candidate.py")
    parser.add_argument("--baseline", default="submissions/baseline_geqrf.py")
    parser.add_argument("--suite-name", default=None)
    parser.add_argument(
        "--torch-cuda-arch-list",
        default="10.0",
        help='Suite-wide TORCH_CUDA_ARCH_LIST. Defaults to "10.0" for B200; pass an empty string to unset.',
    )
    parser.add_argument(
        "--qr32-extra-cuda-cflags",
        default=None,
        help='Suite-wide FAST_QR_QR32_EXTRA_CUDA_CFLAGS for the inline qr32 probe, e.g. "-arch=sm_100".',
    )
    parser.add_argument(
        "--qr32-sm100",
        action="store_true",
        help='Shortcut for --qr32-extra-cuda-cflags="-arch=sm_100".',
    )
    parser.add_argument("--smoke-repeats", type=int, default=1)
    parser.add_argument("--baseline-repeats", type=int, default=3)
    parser.add_argument("--candidate-repeats", type=int, default=3)
    parser.add_argument("--official-repeats", type=int, default=1000)
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--skip-policy", action="store_true")
    parser.add_argument("--skip-submission-validation", action="store_true")
    parser.add_argument("--skip-route-trace", action="store_true")
    parser.add_argument("--skip-guard-benchmark", action="store_true")
    parser.add_argument("--skip-route-ablations", action="store_true")
    parser.add_argument("--skip-secret-audit", action="store_true")
    parser.add_argument("--skip-runtime-preflight", action="store_true")
    parser.add_argument("--skip-suite-validation", action="store_true")
    parser.add_argument(
        "--require-final-kernels",
        action="store_true",
        help="When validating the suite, also require implementation_status to report no remaining non-final routes.",
    )
    parser.add_argument("--skip-seed-sweep", action="store_true")
    parser.add_argument("--skip-quantization-sweep", action="store_true")
    parser.add_argument("--skip-mixed-seed-sweep", action="store_true")
    parser.add_argument("--skip-classifier-sweep", action="store_true")
    parser.add_argument("--skip-tail-policy-sweep", action="store_true")
    parser.add_argument("--skip-candidate-tests", action="store_true")
    parser.add_argument("--skip-benchmark-correctness", action="store_true")
    parser.add_argument("--skip-dev-robustness", action="store_true")
    parser.add_argument("--skip-accelerator-preflight", action="store_true")
    parser.add_argument("--allow-accelerator-fallback", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-baseline-public", action="store_true")
    parser.add_argument("--skip-candidate-public", action="store_true")
    parser.add_argument("--skip-experiments", action="store_true")
    parser.add_argument("--skip-official-style", action="store_true")
    parser.add_argument("--skip-candidate-official-style", action="store_true")
    parser.add_argument("--seed-sweep-indices", default="19,20,21")
    parser.add_argument("--seed-sweep-popcorn-seeds", default="public,1,2,3")
    parser.add_argument("--seed-sweep-max-factor-scaled", type=float, default=18.0)
    parser.add_argument("--seed-sweep-max-orth-scaled", type=float, default=80.0)
    parser.add_argument("--quantization-sweep-indices", default=DEFAULT_QUANTIZATION_SWEEP_INDEXES)
    parser.add_argument("--quantization-sweep-popcorn-seeds", default="public,1,2,3")
    parser.add_argument("--quantization-sweep-experiments", default="fp16-nearby,tf32-input-nearby")
    parser.add_argument("--quantization-sweep-max-factor-scaled", type=float, default=18.0)
    parser.add_argument("--quantization-sweep-max-orth-scaled", type=float, default=80.0)
    parser.add_argument("--mixed-sweep-benchmark-indices", default="7,8")
    parser.add_argument("--mixed-sweep-test-indices", default="19,20,21")
    parser.add_argument("--mixed-sweep-popcorn-seeds", default="public,1,2,3")
    parser.add_argument("--mixed-sweep-max-factor-scaled", type=float, default=18.0)
    parser.add_argument("--mixed-sweep-max-orth-scaled", type=float, default=80.0)
    parser.add_argument("--classifier-sweep-indices", default="3,4,7,8,9,10,11")
    parser.add_argument("--classifier-sweep-popcorn-seeds", default="public,1,2,3")
    parser.add_argument("--tail-policy-indices", default="3,4,5,6,7,8,9,10,11")
    parser.add_argument("--tail-policy-popcorn-seeds", default="public,0,1,2,3")
    parser.add_argument("--tail-policy-cuts", default="candidate")
    parser.add_argument("--tail-policy-max-factor-scaled", type=float, default=18.0)
    parser.add_argument("--tail-policy-max-orth-scaled", type=float, default=80.0)
    parser.add_argument(
        "--include-blocked-qr-sweep",
        action="store_true",
        help=(
            "Also run the PyTorch blocked compact-Householder reference sweep. This is a design probe for "
            "panel refresh/R maintenance, not a timing candidate."
        ),
    )
    parser.add_argument("--blocked-qr-sweep-cases", default="cases/public_tests.txt")
    parser.add_argument("--blocked-qr-sweep-indices", default=DEFAULT_BLOCKED_QR_SWEEP_INDEXES)
    parser.add_argument("--blocked-qr-sweep-panel-widths", default="16,32")
    parser.add_argument("--blocked-qr-sweep-update-modes", default="compact-wy")
    parser.add_argument("--blocked-qr-sweep-precision-modes", default="fp32,tf32-input")
    parser.add_argument("--blocked-qr-sweep-r-maintenance-modes", default="none,panel-prefix")
    parser.add_argument("--blocked-qr-sweep-panel-refresh-modes", default="none,prefix")
    parser.add_argument(
        "--blocked-qr-sweep-require-pass",
        action="store_true",
        help="Fail the suite if any blocked QR sweep row fails. By default expected low-precision failures are allowed.",
    )
    parser.add_argument(
        "--include-tail-policy-tune",
        action="store_true",
        help="Also run tools/tune_tail_policy.py as an opt-in grid after the bounded tail-policy sweep.",
    )
    parser.add_argument("--tail-tune-repeats", type=int, default=3)
    parser.add_argument("--tail-tune-config-jsonl", default=None)
    parser.add_argument(
        "--tail-tune-config",
        action="append",
        default=[],
        help='Forward one tuner config as "name:KEY=VALUE,KEY2=VALUE2". Can be repeated.',
    )
    parser.add_argument("--tail-tune-official-stopping", action="store_true")
    parser.add_argument("--tail-tune-benchmark-failed-configs", action="store_true")
    parser.add_argument("--tail-tune-fail-fast", action="store_true")
    parser.add_argument(
        "--include-candidate-config-tune",
        action="store_true",
        help="Also run tools/tune_candidate_configs.py as an opt-in generic kernel/config grid.",
    )
    parser.add_argument(
        "--candidate-config-tune-next-required",
        action="store_true",
        help=(
            "Enable candidate-config tuning and fill shape label, env prefix, benchmark indices, and "
            "correctness indices from the highest-priority candidate_policy row with required_cuda_kernel."
        ),
    )
    parser.add_argument("--candidate-config-tune-repeats", type=int, default=3)
    parser.add_argument("--candidate-config-tune-config-jsonl", default=None)
    parser.add_argument(
        "--candidate-config-tune-config",
        action="append",
        default=[],
        help='Forward one generic tuner config as "name:KEY=VALUE,KEY2=VALUE2". Can be repeated.',
    )
    parser.add_argument("--candidate-config-tune-no-default", action="store_true")
    parser.add_argument("--candidate-config-tune-shape-label", default="global")
    parser.add_argument("--candidate-config-tune-env-prefix", default=None)
    parser.add_argument(
        "--candidate-config-tune-large-kernel-plan-mode",
        choices=LARGE_KERNEL_PLAN_MODE_CHOICES,
        default=None,
        help=(
            "Generate a tools/large_kernel_plan.py config JSONL inside the suite directory. "
            "Use current-candidate for currently effective knobs or future-blocked for design-space grids."
        ),
    )
    parser.add_argument("--candidate-config-tune-large-kernel-plan-max-configs", type=int, default=32)
    parser.add_argument("--candidate-config-tune-panel-widths", default="")
    parser.add_argument("--candidate-config-tune-update-modes", default="")
    parser.add_argument("--candidate-config-tune-precision-modes", default="")
    parser.add_argument("--candidate-config-tune-tile-ms", default="")
    parser.add_argument("--candidate-config-tune-tile-ns", default="")
    parser.add_argument("--candidate-config-tune-compact-wy-tile-cols", default="")
    parser.add_argument("--candidate-config-tune-warps-per-cta", default="")
    parser.add_argument("--candidate-config-tune-ctas-per-matrix", default="")
    parser.add_argument("--candidate-config-tune-cta-schedules", default="")
    parser.add_argument("--candidate-config-tune-sync-free-auto-policy", default="")
    parser.add_argument("--candidate-config-tune-auto-policy-groups", default="")
    parser.add_argument("--candidate-config-tune-cluster-sizes", default="")
    parser.add_argument("--candidate-config-tune-tail-cuts", default="")
    parser.add_argument("--candidate-config-tune-tail-thresholds", default="")
    parser.add_argument("--candidate-config-tune-tail-force", default="")
    parser.add_argument("--candidate-config-tune-panel-refreshes", default="")
    parser.add_argument("--candidate-config-tune-panel-refresh-modes", default="")
    parser.add_argument("--candidate-config-tune-r-maintenance-modes", default="")
    parser.add_argument("--candidate-config-tune-structured-before-cuda", default="")
    parser.add_argument("--candidate-config-tune-collect-resource-metrics", action="store_true")
    parser.add_argument("--candidate-config-tune-resource-cflags-env", default=None)
    parser.add_argument("--candidate-config-tune-correctness-cases", default="cases/public_tests.txt")
    parser.add_argument("--candidate-config-tune-correctness-indices", default=DEFAULT_CANDIDATE_CONFIG_CORRECTNESS_INDICES)
    parser.add_argument("--candidate-config-tune-benchmark-cases", default="cases/public_benchmarks.txt")
    parser.add_argument("--candidate-config-tune-benchmark-indices", default="")
    parser.add_argument("--candidate-config-tune-popcorn-seeds", default="public,1,2,3")
    parser.add_argument("--candidate-config-tune-benchmark-popcorn-seed", type=int, default=None)
    parser.add_argument("--candidate-config-tune-max-factor-scaled", type=float, default=18.0)
    parser.add_argument("--candidate-config-tune-max-orth-scaled", type=float, default=80.0)
    parser.add_argument("--candidate-config-tune-official-stopping", action="store_true")
    parser.add_argument("--candidate-config-tune-benchmark-failed-configs", action="store_true")
    parser.add_argument("--candidate-config-tune-fail-fast", action="store_true")
    parser.add_argument("--candidate-config-tune-skip-diagnostics", action="store_true")
    parser.add_argument(
        "--skip-candidate-config-accelerator-preflight",
        action="store_true",
        help=(
            "When a generated QR512/QR1024 large-kernel config plan is tuned, skip the "
            "per-config accelerator compile/correctness preflight before timing."
        ),
    )
    parser.add_argument(
        "--experiment-indices",
        default=DEFAULT_BENCHMARK_EXPERIMENT_INDEXES,
        help="Comma-separated public benchmark indexes for verifier experiments.",
    )
    parser.add_argument("--experiment-tail-cuts", default="0,4,8,16,32,64,128")
    parser.add_argument("--benchmark-max-factor-scaled", type=float, default=18.0)
    parser.add_argument("--benchmark-max-orth-scaled", type=float, default=80.0)
    parser.add_argument("--dev-max-factor-scaled", type=float, default=18.0)
    parser.add_argument("--dev-max-orth-scaled", type=float, default=80.0)
    parser.add_argument("--guard-repeats", type=int, default=20)
    parser.add_argument("--guard-warmup", type=int, default=3)
    parser.add_argument("--ablation-repeats", type=int, default=2)
    parser.add_argument("--candidate-test-indices", default="", help="Defaults to all public correctness cases.")
    parser.add_argument(
        "--include-local-official-eval",
        action="store_true",
        help="Also run frozen official/eval.py in test mode. This can add substantial time.",
    )
    parser.add_argument(
        "--include-popcorn-test",
        action="store_true",
        help="Also stage and submit the candidate to Popcorn test mode after local correctness preflights.",
    )
    parser.add_argument(
        "--include-popcorn-leaderboard",
        action="store_true",
        help="Also submit Popcorn mode=leaderboard after the local suite. This implies --include-popcorn-test.",
    )
    parser.add_argument("--popcorn-bin", default="popcorn")
    parser.add_argument("--popcorn-timeout-s", type=float, default=None)
    parser.add_argument("--popcorn-seed", type=int, default=None)
    parser.add_argument(
        "--pytest-timeout-s",
        type=float,
        default=1800.0,
        help="Timeout for the pytest suite step. Defaults to 1800 seconds; pass 0 to disable.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the suite command plan without creating results or running commands.")
    parser.add_argument("--dry-run-json", action="store_true", help="Print the dry-run command plan as JSON; implies --dry-run.")
    args = parser.parse_args()
    if args.dry_run_json:
        args.dry_run = True
    args.candidate_config_tune_policy_target = None
    if args.candidate_config_tune_next_required:
        args.include_candidate_config_tune = True
        try:
            apply_candidate_config_next_required_target(args)
        except ValueError as exc:
            parser.error(str(exc))
    if args.candidate_config_tune_large_kernel_plan_mode and not args.include_candidate_config_tune:
        parser.error("--candidate-config-tune-large-kernel-plan-mode requires --include-candidate-config-tune")
    if args.candidate_config_tune_large_kernel_plan_mode and args.candidate_config_tune_config_jsonl:
        parser.error(
            "--candidate-config-tune-large-kernel-plan-mode cannot be combined with "
            "--candidate-config-tune-config-jsonl"
        )
    if args.candidate_config_tune_large_kernel_plan_mode:
        try:
            candidate_config_tune_large_kernel_plan_rows(args)
        except ValueError as exc:
            parser.error(str(exc))

    name = args.suite_name or f"b200_suite_{timestamp()}"
    suite_dir = ROOT / "results" / name
    log_path = suite_dir / "run.log"
    manifest_path = suite_dir / "manifest.jsonl"

    env = apply_suite_env_options(os.environ.copy(), args)
    python = sys.executable
    popcorn_seed_args = []
    if args.popcorn_seed is not None:
        popcorn_seed_args = ["--popcorn-seed", str(args.popcorn_seed)]
    experiment_indices = parse_int_list(args.experiment_indices)

    steps: list[tuple[str, list[str]] | tuple[str, list[str], dict[str, str]]] = [
        ("print_spec", [python, "tools/print_spec.py"]),
        ("sync_cases_check", [python, "tools/sync_cases_from_task_yml.py", "--check"]),
    ]
    if not args.skip_secret_audit:
        steps.append(
            (
                "secret_audit",
                [
                    python,
                    "tools/audit_secrets.py",
                    "--json",
                    "--out",
                    str(suite_dir / "secret_audit.jsonl"),
                ],
            )
        )
    if not args.skip_runtime_preflight:
        steps.append(
            (
                "runtime_preflight",
                [
                    python,
                    "tools/check_b200_env.py",
                    "--json",
                    "--out",
                    str(suite_dir / "runtime_preflight.jsonl"),
                ],
            )
        )
    if not args.skip_submission_validation:
        steps.append(
            (
                "submission_validation",
                [
                    python,
                    "tools/validate_submission.py",
                    "--submission",
                    args.submission,
                    "--stage-dir",
                    str(suite_dir / "submission_stage"),
                    "--json",
                    "--out",
                    str(suite_dir / "submission_validation.jsonl"),
                ],
            )
        )
    if not args.skip_policy:
        steps.append(
            (
                "candidate_policy",
                [
                    python,
                    "tools/candidate_policy.py",
                    "--submission",
                    args.submission,
                    "--cases",
                    "cases/public_benchmarks.txt",
                    "--json",
                    "--record-env",
                    "--out",
                    str(suite_dir / "candidate_policy_public.jsonl"),
                ],
            )
        )
        steps.append(
            (
                "candidate_implementation_status",
                [
                    python,
                    "tools/implementation_status.py",
                    "--submission",
                    args.submission,
                    "--cases",
                    "cases/public_benchmarks.txt",
                    "--json",
                    "--record-env",
                    "--out",
                    str(suite_dir / "candidate_implementation_status.jsonl"),
                ],
            )
        )
    if not args.skip_route_trace:
        steps.append(
            (
                "candidate_route_trace",
                [
                    python,
                    "tools/trace_candidate_routes.py",
                    "--submission",
                    args.submission,
                    "--cases",
                    "cases/public_benchmarks.txt",
                    "--json",
                    "--record-env",
                    "--out",
                    str(suite_dir / "candidate_route_trace_public.jsonl"),
                    *popcorn_seed_args,
                ],
            )
        )
    if not args.skip_guard_benchmark:
        steps.append(
            (
                "candidate_guard_overhead",
                [
                    python,
                    "tools/benchmark_guards.py",
                    "--submission",
                    args.submission,
                    "--cases",
                    "cases/public_benchmarks.txt",
                    "--repeats",
                    str(args.guard_repeats),
                    "--warmup",
                    str(args.guard_warmup),
                    "--json",
                    "--record-env",
                    "--out",
                    str(suite_dir / "candidate_guard_overhead_public.jsonl"),
                    *popcorn_seed_args,
                ],
            )
        )
    if not args.skip_pytest:
        steps.append(("pytest", [python, "-m", "pytest"]))
    if not args.skip_seed_sweep:
        steps.append(
            (
                "seed_sweep_margin",
                [
                    python,
                    "tools/seed_sweep.py",
                    "--submission",
                    args.submission,
                    "--cases",
                    "cases/public_tests.txt",
                    "--indices",
                    args.seed_sweep_indices,
                    "--popcorn-seeds",
                    args.seed_sweep_popcorn_seeds,
                    "--max-factor-scaled",
                    f"{args.seed_sweep_max_factor_scaled:g}",
                    "--max-orth-scaled",
                    f"{args.seed_sweep_max_orth_scaled:g}",
                    "--record-env",
                    "--out",
                    str(suite_dir / "seed_sweep_margin.jsonl"),
                ],
            )
        )
    if not args.skip_quantization_sweep:
        steps.append(
            (
                "quantization_seed_sweep",
                [
                    python,
                    "tools/quantization_seed_sweep.py",
                    "--submission",
                    args.submission,
                    "--cases",
                    "cases/public_benchmarks.txt",
                    "--indices",
                    args.quantization_sweep_indices,
                    "--popcorn-seeds",
                    args.quantization_sweep_popcorn_seeds,
                    "--experiments",
                    args.quantization_sweep_experiments,
                    "--max-factor-scaled",
                    f"{args.quantization_sweep_max_factor_scaled:g}",
                    "--max-orth-scaled",
                    f"{args.quantization_sweep_max_orth_scaled:g}",
                    "--record-env",
                    "--out",
                    str(suite_dir / "quantization_seed_sweep.jsonl"),
                ],
            )
        )
    if not args.skip_mixed_seed_sweep:
        steps.append(
            (
                "mixed_seed_sweep",
                [
                    python,
                    "tools/mixed_seed_sweep.py",
                    "--submission",
                    args.submission,
                    "--benchmark-cases",
                    "cases/public_benchmarks.txt",
                    "--benchmark-indices",
                    args.mixed_sweep_benchmark_indices,
                    "--test-cases",
                    "cases/public_tests.txt",
                    "--test-indices",
                    args.mixed_sweep_test_indices,
                    "--popcorn-seeds",
                    args.mixed_sweep_popcorn_seeds,
                    "--max-factor-scaled",
                    f"{args.mixed_sweep_max_factor_scaled:g}",
                    "--max-orth-scaled",
                    f"{args.mixed_sweep_max_orth_scaled:g}",
                    "--record-env",
                    "--out",
                    str(suite_dir / "mixed_seed_sweep.jsonl"),
                ],
            )
        )
    if not args.skip_classifier_sweep:
        steps.append(
            (
                "classifier_seed_sweep",
                [
                    python,
                    "tools/classifier_seed_sweep.py",
                    "--submission",
                    args.submission,
                    "--cases",
                    "cases/public_benchmarks.txt",
                    "--indices",
                    args.classifier_sweep_indices,
                    "--popcorn-seeds",
                    args.classifier_sweep_popcorn_seeds,
                    "--include-plan",
                    "--record-env",
                    "--out",
                    str(suite_dir / "classifier_seed_sweep.jsonl"),
                ],
            )
        )
    if not args.skip_tail_policy_sweep:
        steps.append(
            (
                "tail_policy_sweep",
                [
                    python,
                    "tools/tail_policy_sweep.py",
                    "--submission",
                    args.submission,
                    "--cases",
                    "cases/public_benchmarks.txt",
                    "--indices",
                    args.tail_policy_indices,
                    "--popcorn-seeds",
                    args.tail_policy_popcorn_seeds,
                    "--tail-cuts",
                    args.tail_policy_cuts,
                    "--diagnose",
                    "--max-factor-scaled",
                    f"{args.tail_policy_max_factor_scaled:g}",
                    "--max-orth-scaled",
                    f"{args.tail_policy_max_orth_scaled:g}",
                    "--record-env",
                    "--out",
                    str(suite_dir / "candidate_tail_policy_sweep.jsonl"),
                ],
            )
        )
    if args.include_blocked_qr_sweep:
        blocked_qr_sweep_cmd = [
            python,
            "tools/blocked_qr_sweep.py",
            "--cases",
            args.blocked_qr_sweep_cases,
            "--indices",
            args.blocked_qr_sweep_indices,
            "--panel-widths",
            args.blocked_qr_sweep_panel_widths,
            "--update-modes",
            args.blocked_qr_sweep_update_modes,
            "--precision-modes",
            args.blocked_qr_sweep_precision_modes,
            "--r-maintenance-modes",
            args.blocked_qr_sweep_r_maintenance_modes,
            "--panel-refresh-modes",
            args.blocked_qr_sweep_panel_refresh_modes,
            "--diagnose",
            "--json",
            "--out",
            str(suite_dir / "blocked_qr_sweep.jsonl"),
            *popcorn_seed_args,
        ]
        if not args.blocked_qr_sweep_require_pass:
            blocked_qr_sweep_cmd.append("--allow-failures")
        steps.append(("blocked_qr_sweep", blocked_qr_sweep_cmd))
    if args.include_tail_policy_tune:
        tail_tune_cmd = [
            python,
            "tools/tune_tail_policy.py",
            "--submission",
            args.submission,
            "--cases",
            "cases/public_benchmarks.txt",
            "--out-dir",
            str(suite_dir / "tail_policy_tune"),
            "--correctness-indices",
            args.tail_policy_indices,
            "--popcorn-seeds",
            args.tail_policy_popcorn_seeds,
            "--max-factor-scaled",
            f"{args.tail_policy_max_factor_scaled:g}",
            "--max-orth-scaled",
            f"{args.tail_policy_max_orth_scaled:g}",
            "--repeats",
            str(args.tail_tune_repeats),
            "--allow-failed-configs",
        ]
        if args.tail_tune_config_jsonl:
            tail_tune_cmd.extend(["--config-jsonl", args.tail_tune_config_jsonl])
        for config in args.tail_tune_config:
            tail_tune_cmd.extend(["--config", config])
        if args.tail_tune_official_stopping:
            tail_tune_cmd.append("--official-stopping")
        if args.tail_tune_benchmark_failed_configs:
            tail_tune_cmd.append("--benchmark-failed-configs")
        if args.tail_tune_fail_fast:
            tail_tune_cmd.append("--fail-fast")
        steps.append(("tail_policy_tune", tail_tune_cmd))
    if args.include_candidate_config_tune:
        generated_candidate_config_path = (
            candidate_config_tune_large_kernel_plan_path(suite_dir)
            if args.candidate_config_tune_large_kernel_plan_mode
            else None
        )
        preflight_accelerator = candidate_config_accelerator_for_shape(args.candidate_config_tune_shape_label)
        if (
            generated_candidate_config_path is not None
            and preflight_accelerator is not None
            and not args.skip_accelerator_preflight
            and not args.skip_candidate_config_accelerator_preflight
        ):
            candidate_config_preflight_cmd = [
                python,
                "tools/preflight_accelerators.py",
                "--submission",
                args.submission,
                "--config-jsonl",
                str(generated_candidate_config_path),
                "--accelerators",
                preflight_accelerator,
                "--json",
                "--out",
                str(candidate_config_accelerator_preflight_path(suite_dir)),
            ]
            if args.allow_accelerator_fallback:
                candidate_config_preflight_cmd.append("--allow-fallback")
            candidate_config_preflight_cmd.append("--family-cases")
            steps.append(("candidate_config_accelerator_preflight", candidate_config_preflight_cmd))
        candidate_config_tune_cmd = [
            python,
            "tools/tune_candidate_configs.py",
            "--submission",
            args.submission,
            "--out-dir",
            str(suite_dir / "candidate_config_tune"),
            "--shape-label",
            args.candidate_config_tune_shape_label,
            "--correctness-cases",
            args.candidate_config_tune_correctness_cases,
            "--correctness-indices",
            args.candidate_config_tune_correctness_indices,
            "--benchmark-cases",
            args.candidate_config_tune_benchmark_cases,
            "--popcorn-seeds",
            args.candidate_config_tune_popcorn_seeds,
            "--max-factor-scaled",
            f"{args.candidate_config_tune_max_factor_scaled:g}",
            "--max-orth-scaled",
            f"{args.candidate_config_tune_max_orth_scaled:g}",
            "--repeats",
            str(args.candidate_config_tune_repeats),
            "--allow-failed-configs",
        ]
        if args.candidate_config_tune_benchmark_indices:
            candidate_config_tune_cmd.extend(["--benchmark-indices", args.candidate_config_tune_benchmark_indices])
        if generated_candidate_config_path is not None:
            candidate_config_tune_cmd.extend(["--config-jsonl", str(generated_candidate_config_path)])
        if args.candidate_config_tune_config_jsonl:
            candidate_config_tune_cmd.extend(["--config-jsonl", args.candidate_config_tune_config_jsonl])
        for config in args.candidate_config_tune_config:
            candidate_config_tune_cmd.extend(["--config", config])
        if args.candidate_config_tune_no_default:
            candidate_config_tune_cmd.append("--no-default")
        if args.candidate_config_tune_env_prefix:
            candidate_config_tune_cmd.extend(["--env-prefix", args.candidate_config_tune_env_prefix])
        for flag, value in [
            ("--panel-widths", args.candidate_config_tune_panel_widths),
            ("--update-modes", args.candidate_config_tune_update_modes),
            ("--precision-modes", args.candidate_config_tune_precision_modes),
            ("--tile-ms", args.candidate_config_tune_tile_ms),
            ("--tile-ns", args.candidate_config_tune_tile_ns),
            ("--compact-wy-tile-cols", args.candidate_config_tune_compact_wy_tile_cols),
            ("--warps-per-cta", args.candidate_config_tune_warps_per_cta),
            ("--ctas-per-matrix", args.candidate_config_tune_ctas_per_matrix),
            ("--cta-schedules", getattr(args, "candidate_config_tune_cta_schedules", "")),
            ("--sync-free-auto-policy", getattr(args, "candidate_config_tune_sync_free_auto_policy", "")),
            ("--auto-policy-groups", getattr(args, "candidate_config_tune_auto_policy_groups", "")),
            ("--cluster-sizes", args.candidate_config_tune_cluster_sizes),
            ("--tail-cuts", args.candidate_config_tune_tail_cuts),
            ("--tail-thresholds", args.candidate_config_tune_tail_thresholds),
            ("--tail-force", getattr(args, "candidate_config_tune_tail_force", "")),
            ("--panel-refreshes", args.candidate_config_tune_panel_refreshes),
            ("--panel-refresh-modes", args.candidate_config_tune_panel_refresh_modes),
            ("--r-maintenance-modes", args.candidate_config_tune_r_maintenance_modes),
            ("--structured-before-cuda", args.candidate_config_tune_structured_before_cuda),
        ]:
            if value:
                candidate_config_tune_cmd.extend([flag, value])
        if args.candidate_config_tune_benchmark_popcorn_seed is not None:
            candidate_config_tune_cmd.extend(
                ["--benchmark-popcorn-seed", str(args.candidate_config_tune_benchmark_popcorn_seed)]
            )
        if args.candidate_config_tune_collect_resource_metrics:
            candidate_config_tune_cmd.append("--collect-resource-metrics")
        if args.candidate_config_tune_resource_cflags_env:
            candidate_config_tune_cmd.extend(["--resource-cflags-env", args.candidate_config_tune_resource_cflags_env])
        if args.candidate_config_tune_official_stopping:
            candidate_config_tune_cmd.append("--official-stopping")
        if args.candidate_config_tune_benchmark_failed_configs:
            candidate_config_tune_cmd.append("--benchmark-failed-configs")
        if args.candidate_config_tune_fail_fast:
            candidate_config_tune_cmd.append("--fail-fast")
        if args.candidate_config_tune_skip_diagnostics:
            candidate_config_tune_cmd.append("--skip-diagnostics")
        steps.append(("candidate_config_tune", candidate_config_tune_cmd))
    if not args.skip_candidate_tests:
        test_cmd = [
            python,
            "tools/check_cases.py",
            "--submission",
            args.submission,
            "--cases",
            "cases/public_tests.txt",
            "--json",
            "--record-env",
            "--out",
            str(suite_dir / "candidate_public_tests.jsonl"),
            *popcorn_seed_args,
        ]
        if args.candidate_test_indices:
            test_cmd.extend(["--indices", args.candidate_test_indices])
        steps.append(("candidate_public_tests", test_cmd))
    if not args.skip_benchmark_correctness:
        steps.append(
            (
                "candidate_public_benchmark_correctness",
                [
                    python,
                    "tools/check_cases.py",
                    "--submission",
                    args.submission,
                    "--cases",
                    "cases/public_benchmarks.txt",
                    "--diagnose",
                    "--max-factor-scaled",
                    f"{args.benchmark_max_factor_scaled:g}",
                    "--max-orth-scaled",
                    f"{args.benchmark_max_orth_scaled:g}",
                    "--json",
                    "--record-env",
                    "--out",
                    str(suite_dir / "candidate_public_benchmark_correctness.jsonl"),
                    *popcorn_seed_args,
                ],
            )
        )
    if not args.skip_dev_robustness:
        steps.append(
            (
                "candidate_dev_robustness",
                [
                    python,
                    "tools/check_cases.py",
                    "--submission",
                    args.submission,
                    "--cases",
                    "cases/dev_robustness.txt",
                    "--diagnose",
                    "--max-factor-scaled",
                    f"{args.dev_max_factor_scaled:g}",
                    "--max-orth-scaled",
                    f"{args.dev_max_orth_scaled:g}",
                    "--json",
                    "--record-env",
                    "--out",
                    str(suite_dir / "candidate_dev_robustness.jsonl"),
                    *popcorn_seed_args,
                ],
            )
        )
    if not args.skip_accelerator_preflight:
        preflight_cmd = [
            python,
            "tools/preflight_accelerators.py",
            "--submission",
            args.submission,
            "--json",
            "--out",
            str(suite_dir / "accelerator_preflight.jsonl"),
            "--family-cases",
        ]
        if args.allow_accelerator_fallback:
            preflight_cmd.append("--allow-fallback")
        steps.append(("accelerator_preflight", preflight_cmd))
    if args.include_local_official_eval:
        steps.append(
            (
                "local_official_eval_test",
                [
                    python,
                    "tools/run_official_eval.py",
                    "--submission",
                    args.submission,
                    "--cases",
                    "cases/public_tests.txt",
                    "--mode",
                    "test",
                    *popcorn_seed_args,
                ],
            )
        )
    include_popcorn_test = args.include_popcorn_test or args.include_popcorn_leaderboard
    if include_popcorn_test:
        popcorn_cmd = [
            python,
            "tools/submit_popcorn.py",
            "--submission",
            args.submission,
            "--mode",
            "test",
            "--popcorn-bin",
            args.popcorn_bin,
            "--out-dir",
            str(suite_dir / "popcorn_test"),
        ]
        if args.popcorn_timeout_s is not None:
            popcorn_cmd.extend(["--timeout-s", f"{args.popcorn_timeout_s:g}"])
        steps.append(("popcorn_test", popcorn_cmd))
    if not args.skip_smoke:
        steps.append(
            (
                "baseline_smoke",
                [
                    python,
                    "tools/bench_local.py",
                    "--submission",
                    args.baseline,
                    "--cases",
                    "cases/public_benchmarks.txt",
                    "--repeats",
                    str(args.smoke_repeats),
                    "--recheck",
                    "--record-env",
                    "--out",
                    str(suite_dir / "baseline_geqrf_smoke.jsonl"),
                    *popcorn_seed_args,
                ],
            )
        )
        steps.append(
            (
                "candidate_smoke",
                [
                    python,
                    "tools/bench_local.py",
                    "--submission",
                    args.submission,
                    "--cases",
                    "cases/public_benchmarks.txt",
                    "--repeats",
                    str(args.smoke_repeats),
                    "--recheck",
                    "--record-env",
                    "--out",
                    str(suite_dir / "candidate_smoke.jsonl"),
                    *popcorn_seed_args,
                ],
            )
        )
    if not args.skip_baseline_public:
        steps.append(
            (
                "baseline_public",
                [
                    python,
                    "tools/bench_local.py",
                    "--submission",
                    args.baseline,
                    "--cases",
                    "cases/public_benchmarks.txt",
                    "--repeats",
                    str(args.baseline_repeats),
                    "--recheck",
                    "--record-env",
                    "--out",
                    str(suite_dir / "baseline_geqrf_public.jsonl"),
                    *popcorn_seed_args,
                ],
            )
        )
    if not args.skip_candidate_public:
        steps.append(
            (
                "candidate_public",
                [
                    python,
                    "tools/bench_local.py",
                    "--submission",
                    args.submission,
                    "--cases",
                    "cases/public_benchmarks.txt",
                    "--repeats",
                    str(args.candidate_repeats),
                    "--recheck",
                    "--record-env",
                    "--out",
                    str(suite_dir / "candidate_public.jsonl"),
                    *popcorn_seed_args,
                ],
            )
        )
    if not args.skip_route_ablations:
        for label, env_overrides, out_name in ROUTE_ABLATIONS:
            steps.append(
                (
                    f"candidate_ablation_{label}",
                    [
                        python,
                        "tools/bench_local.py",
                        "--submission",
                        args.submission,
                        "--cases",
                        "cases/public_benchmarks.txt",
                        "--repeats",
                        str(args.ablation_repeats),
                        "--recheck",
                        "--record-env",
                        "--out",
                        str(suite_dir / out_name),
                        *popcorn_seed_args,
                    ],
                    env_overrides,
                )
            )
    if not args.skip_experiments:
        for index in experiment_indices:
            steps.append(
                (
                    f"experiments_public_benchmark_{index}",
                    [
                        python,
                        "tools/experiments.py",
                        "--submission",
                        args.submission,
                        "--cases",
                        "cases/public_benchmarks.txt",
                        "--index",
                        str(index),
                        "--experiments",
                        "all",
                        "--tail-cuts",
                        args.experiment_tail_cuts,
                        "--record-env",
                        "--out",
                        str(suite_dir / "experiments_public_benchmarks.jsonl"),
                        *popcorn_seed_args,
                    ],
                )
            )
    if not args.skip_official_style:
        steps.append(
            (
                "baseline_official_style",
                [
                    python,
                    "tools/bench_local.py",
                    "--submission",
                    args.baseline,
                    "--cases",
                    "cases/public_benchmarks.txt",
                    "--repeats",
                    str(args.official_repeats),
                    "--official-stopping",
                    "--leaderboard-warmup",
                    "--recheck",
                    "--record-env",
                    "--out",
                    str(suite_dir / "baseline_geqrf_official_style.jsonl"),
                    *popcorn_seed_args,
                ],
            )
        )
    if not args.skip_official_style and not args.skip_candidate_official_style:
        steps.append(
            (
                "candidate_official_style",
                [
                    python,
                    "tools/bench_local.py",
                    "--submission",
                    args.submission,
                    "--cases",
                    "cases/public_benchmarks.txt",
                    "--repeats",
                    str(args.official_repeats),
                    "--official-stopping",
                    "--leaderboard-warmup",
                    "--recheck",
                    "--record-env",
                    "--out",
                    str(suite_dir / "candidate_official_style.jsonl"),
                    *popcorn_seed_args,
                ],
            )
        )
    if args.include_popcorn_leaderboard:
        popcorn_leaderboard_cmd = [
            python,
            "tools/submit_popcorn.py",
            "--submission",
            args.submission,
            "--mode",
            "leaderboard",
            "--popcorn-bin",
            args.popcorn_bin,
            "--out-dir",
            str(suite_dir / "popcorn_leaderboard"),
        ]
        if args.popcorn_timeout_s is not None:
            popcorn_leaderboard_cmd.extend(["--timeout-s", f"{args.popcorn_timeout_s:g}"])
        steps.append(("popcorn_leaderboard", popcorn_leaderboard_cmd))
    should_summarize = (
        (not args.skip_baseline_public and not args.skip_candidate_public)
        or (not args.skip_official_style and not args.skip_candidate_official_style)
        or not args.skip_smoke
    )
    if should_summarize:
        steps.append(
            (
                "suite_summary",
                [
                    python,
                    "tools/summarize_suite.py",
                    "--suite-dir",
                    str(suite_dir),
                    "--json-out",
                    str(suite_dir / "suite_summary.json"),
                    "--markdown-out",
                    str(suite_dir / "suite_summary.md"),
                ],
            )
        )
    if should_summarize:
        steps.append(
            (
                "suite_analysis",
                [
                    python,
                    "tools/analyze_b200_results.py",
                    "--suite-dir",
                    str(suite_dir),
                    "--json-out",
                    str(suite_dir / "suite_analysis.json"),
                    "--markdown-out",
                    str(suite_dir / "suite_analysis.md"),
                ],
            )
        )
    validation_blockers = default_validation_blockers(args)
    if not args.skip_suite_validation and not validation_blockers:
        suite_validation_cmd = [
            python,
            "tools/validate_b200_suite.py",
            "--suite-dir",
            str(suite_dir),
            "--allow-incomplete",
        ]
        if args.require_final_kernels:
            suite_validation_cmd.append("--require-final-kernels")
        steps.append(
            (
                "suite_validation",
                suite_validation_cmd,
            )
        )

    if args.dry_run:
        print_dry_run_plan(dry_run_plan(suite_dir, steps, validation_blockers, args, env), as_json=args.dry_run_json)
        return 0

    if suite_dir.exists():
        parser.error(
            f"result directory already exists: {suite_dir}; "
            "choose a different --suite-name or remove the existing directory intentionally"
        )

    suite_dir.mkdir(parents=True, exist_ok=False)

    suite_started = time.perf_counter()
    try:
        generated_candidate_config_rows = None
        generated_candidate_config_path = None
        if args.include_candidate_config_tune and args.candidate_config_tune_large_kernel_plan_mode:
            generated_candidate_config_path, generated_candidate_config_rows = write_candidate_config_tune_large_kernel_plan(
                args,
                suite_dir,
            )
        append_manifest(manifest_path, suite_provenance(args, env))
        if generated_candidate_config_rows is not None and generated_candidate_config_path is not None:
            append_manifest(
                manifest_path,
                {
                    "event": "candidate_config_tune_large_kernel_plan",
                    "time": datetime.now().isoformat(),
                    "path": str(generated_candidate_config_path),
                    "mode": args.candidate_config_tune_large_kernel_plan_mode,
                    "shape_label": args.candidate_config_tune_shape_label,
                    "policy_target": args.candidate_config_tune_policy_target,
                    "num_configs": len(generated_candidate_config_rows),
                    "config_names": [row["name"] for row in generated_candidate_config_rows],
                },
            )
        if args.skip_suite_validation:
            append_manifest(
                manifest_path,
                {
                    "event": "suite_validation_skipped",
                    "reason": "--skip-suite-validation",
                    "time": datetime.now().isoformat(),
                },
            )
        elif validation_blockers:
            append_manifest(
                manifest_path,
                {
                    "event": "suite_validation_skipped",
                    "reason": "partial suite",
                    "blockers": validation_blockers,
                    "time": datetime.now().isoformat(),
                },
            )
        for step in steps:
            if len(step) == 2:
                step_name, cmd = step
                env_overrides = None
            else:
                step_name, cmd, env_overrides = step
            step_started = time.perf_counter()
            append_manifest(
                manifest_path,
                {
                    "event": "start",
                    "step": step_name,
                    "env_overrides": env_overrides or {},
                    "time": datetime.now().isoformat(),
                },
            )
            run_command(cmd, log_path, merged_env(env, env_overrides), timeout_s=step_timeout_s(step_name, args))
            append_manifest(
                manifest_path,
                {
                    "event": "finish",
                    "step": step_name,
                    "env_overrides": env_overrides or {},
                    "elapsed_s": time.perf_counter() - step_started,
                    "time": datetime.now().isoformat(),
                },
            )
        tar_path = suite_dir.with_suffix(".tgz")
        append_manifest(
            manifest_path,
            {
                "event": "suite_finish",
                "elapsed_s": time.perf_counter() - suite_started,
                "tarball": str(tar_path),
                "time": datetime.now().isoformat(),
            },
        )
        tar_path = make_tarball(suite_dir)
        if not args.skip_suite_validation and not validation_blockers:
            validate_completed_export(suite_dir, require_final_kernels=args.require_final_kernels)
        print(f"\nDONE\nresults: {suite_dir}\ntarball: {tar_path}")
        return 0
    except Exception as exc:
        append_manifest(
            manifest_path,
            {
                "event": "suite_failed",
                "error": str(exc),
                "elapsed_s": time.perf_counter() - suite_started,
                "time": datetime.now().isoformat(),
            },
        )
        tar_path = make_tarball(suite_dir)
        print(f"\nFAILED: {exc}\npartial results: {suite_dir}\npartial tarball: {tar_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
