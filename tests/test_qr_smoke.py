import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

SYNTHETIC_POPCORN_SUBMISSION = "import torch\n\ndef custom_kernel(data):\n    return torch.geqrf(data)\n"
SYNTHETIC_CANDIDATE_SHA = "82e50331c6aac55c842855a1eea6218e1e27e748ad17e296ec651e547aca4315"
SYNTHETIC_BASELINE_SHA = "b" * 64
SYNTHETIC_BAD_SHA = "c" * 64

from qr_common import (  # noqa: E402
    CANDIDATE_RUNTIME_ENV_KEYS,
    ensure_official_on_path,
    environment_info,
    file_provenance,
    format_case,
    load_submission,
    parse_popcorn_seed_tokens,
    TRACKED_RUNTIME_ENV_KEYS,
    tracked_candidate_env,
)

ensure_official_on_path()
from reference import check_implementation, generate_input  # noqa: E402
from benchmark_guards import measure_route_decision, route_decision  # noqa: E402
from bench_local import parse_indices as parse_benchmark_indices  # noqa: E402
from check_b200_env import evaluate as evaluate_b200_env  # noqa: E402
from check_cases import parse_indices, run_case as run_check_case  # noqa: E402
from classifier_seed_sweep import run_case as run_classifier_case, summarize as summarize_classifier_sweep  # noqa: E402
from diagnose import diagnose  # noqa: E402
from experiments import (  # noqa: E402
    classify_features,
    experiment_column_major,
    experiment_identity_q,
    experiment_r_projection,
    experiment_tail_delete,
    parse_experiments,
)
from quantization_seed_sweep import run_case as run_quantization_case, summarize as summarize_quantization_sweep  # noqa: E402
from mixed_seed_sweep import run_case as run_mixed_seed_case, summarize as summarize_mixed_seed_sweep  # noqa: E402
from implementation_status import readiness_rows, summarize_readiness  # noqa: E402
from large_kernel_plan import generate_configs as generate_large_kernel_configs, tune_command as large_kernel_tune_command  # noqa: E402
from preflight_accelerators import (  # noqa: E402
    ACCELERATOR_FAMILY_SPECS,
    accelerator_compile_info,
    load_preflight_configs,
    qr32_preflight,
    qr176_preflight,
    qr352_preflight,
    qr512_preflight,
    qr1024_preflight,
    run_preflight_matrix,
    selected_accelerator_names,
)
from run_official_eval import write_official_case_file  # noqa: E402
from run_b200_suite import (  # noqa: E402
    apply_suite_env_options,
    candidate_config_accelerator_for_shape,
    candidate_config_accelerator_preflight_path,
    candidate_config_tune_large_kernel_plan_path,
    candidate_config_tune_large_kernel_plan_rows,
    default_validation_blockers,
    dry_run_plan,
    suite_env_overrides,
    suite_provenance,
    write_candidate_config_tune_large_kernel_plan,
)
from seed_sweep import parse_int_list, run_case  # noqa: E402
from submit_popcorn import build_command as build_popcorn_command, selected_modes as selected_popcorn_modes, stage_submission  # noqa: E402
from summarize_suite import (  # noqa: E402
    load_jsonl,
    render_markdown as render_summary_markdown,
    summarize_blocked_qr_sweep as summarize_blocked_qr_suite_sweep,
    summarize_suite,
)
from sweep import diagnostic_row, load_config  # noqa: E402
from tail_policy_sweep import parse_cut_tokens, run_policy_cut  # noqa: E402
from tune_tail_policy import (  # noqa: E402
    DEFAULT_CONFIGS as DEFAULT_TAIL_TUNE_CONFIGS,
    command_plan,
    parse_inline_config,
    should_skip_benchmark_after_correctness,
    summarize_run,
)
from tune_candidate_configs import (  # noqa: E402
    CURRENT_CANDIDATE_CONSUMED_ENV_KEYS,
    command_plan as candidate_config_command_plan,
    env_consumption as candidate_config_env_consumption,
    grid_configs as candidate_grid_configs,
    load_configs as load_candidate_tune_configs,
    parse_ptxas_resource_metrics,
    resource_metrics_by_config,
    summarize_run as summarize_candidate_config_run,
    with_resource_metric_flags,
)
from candidate_policy import policy_rows  # noqa: E402
from trace_candidate_routes import structured_group_counts, trace_route  # noqa: E402
from validate_b200_suite import (  # noqa: E402
    EXPECTED_DEFAULT_STEPS,
    REQUIRED_DEFAULT_FILES,
    validate_candidate_config_tune,
    validate_route_trace,
    validate_suite,
)
from validate_local_checks import EXPECTED_STEPS as LOCAL_EXPECTED_STEPS  # noqa: E402
from validate_local_checks import REQUIRED_FILES as LOCAL_REQUIRED_FILES, validate_local_checks  # noqa: E402
from validate_submission import validate_submission  # noqa: E402
from run_local_checks import next_required_b200_summary  # noqa: E402
from analyze_b200_results import (  # noqa: E402
    analyze_large_cuda_probe_ablation,
    analyze_suite,
    render_markdown as render_analysis_markdown,
)
from blocked_qr_reference import (  # noqa: E402
    _apply_panel_compact_wy,
    _apply_panel_reflectors,
    _panel_left_transform,
    blocked_geqrf_reference,
    run_case as run_blocked_qr_reference_case,
)
from blocked_qr_sweep import (  # noqa: E402
    parse_panel_widths as parse_blocked_qr_panel_widths,
    summarize as summarize_blocked_qr_sweep,
    sweep as run_blocked_qr_sweep,
)
from audit_secrets import scan_text as scan_secret_text  # noqa: E402


def _load_candidate_module():
    spec = importlib.util.spec_from_file_location("candidate_module_under_test", ROOT / "submissions/candidate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate_module_under_test"] = module
    spec.loader.exec_module(module)
    return module


def _small_case(case="dense"):
    spec = {"batch": 3, "n": 8, "cond": 1, "seed": 123}
    if case != "dense":
        spec["case"] = case
        spec["cond"] = 0 if case in {"rankdef", "clustered"} else spec["cond"]
    return spec


@pytest.mark.parametrize("submission", ["submissions/baseline_geqrf.py", "submissions/candidate.py"])
@pytest.mark.parametrize("case", ["dense", "mixed", "rankdef", "clustered"])
def test_submissions_pass_official_checker_on_small_cases(submission, case):
    custom_kernel = load_submission(ROOT / submission)
    spec = _small_case(case)
    data = generate_input(**spec)
    output = custom_kernel(data.clone())
    good, message = check_implementation(data, output)
    assert good, message


def test_diagnostics_are_consistent_with_passing_geqrf():
    data = generate_input(**_small_case("mixed"))
    h, tau = torch.geqrf(data)
    good, message = check_implementation(data, (h, tau))
    assert good, message
    result = diagnose(data, h, tau)
    assert result["factor_scaled_max"] < 20.0
    assert result["orth_scaled_max"] < 100.0
    assert result["worst_factor_matrix"] in range(data.shape[0])


@pytest.mark.parametrize("panel_width", [1, 3, 5, 32])
@pytest.mark.parametrize("update_mode", ["reflectors", "block-full", "compact-wy"])
def test_blocked_qr_reference_passes_official_checker_and_returns_column_major_h(panel_width, update_mode):
    data = generate_input(batch=3, n=8, cond=1, seed=321)
    h, tau = blocked_geqrf_reference(data, panel_width=panel_width, update_mode=update_mode)

    good, message = check_implementation(data, (h, tau))
    assert good, message
    assert h.shape == data.shape
    assert h.stride() == (8 * 8, 1, 8)
    assert tau.shape == (3, 8)
    assert tau.dtype == torch.float32


def test_blocked_qr_reference_tf32_input_update_precision_exposes_r_drift():
    data = generate_input(batch=2, n=8, cond=1, seed=444)
    h, tau = blocked_geqrf_reference(
        data,
        panel_width=3,
        update_mode="compact-wy",
        precision_mode="tf32-input",
    )

    good, message = check_implementation(data, (h, tau))
    assert not good
    assert "R - Q.T @ A is too large" in message
    assert h.stride() == (8 * 8, 1, 8)


def test_blocked_qr_reference_panel_refresh_and_r_maintenance_repair_tf32_update():
    data = generate_input(batch=2, n=8, cond=1, seed=444)
    h_partial, tau_partial = blocked_geqrf_reference(
        data,
        panel_width=3,
        update_mode="compact-wy",
        precision_mode="tf32-input",
        r_maintenance_mode="panel-prefix",
    )
    partial_good, partial_message = check_implementation(data, (h_partial, tau_partial))
    assert not partial_good
    assert "R - Q.T @ A is too large" in partial_message

    h, tau = blocked_geqrf_reference(
        data,
        panel_width=3,
        update_mode="compact-wy",
        precision_mode="tf32-input",
        r_maintenance_mode="panel-prefix",
        panel_refresh_mode="prefix",
    )

    good, message = check_implementation(data, (h, tau))
    assert good, message
    result = diagnose(data, h, tau)
    assert result["factor_scaled_max"] < 20.0
    assert result["orth_scaled_max"] < 100.0


def test_block_panel_updates_match_sequential_reflector_update():
    data = generate_input(batch=2, n=10, cond=2, seed=333, case="mixed")
    h_block, tau_block = blocked_geqrf_reference(data, panel_width=4, update_mode="compact-wy")

    panel = data[:, :, :4].contiguous()
    panel_h, panel_tau = torch.geqrf(panel)
    sequential = data[:, :, 4:].clone()
    _apply_panel_reflectors(sequential, panel_h, panel_tau)
    blocked = _panel_left_transform(panel_h, panel_tau) @ data[:, :, 4:]
    compact = data[:, :, 4:].clone()
    _apply_panel_compact_wy(compact, panel_h, panel_tau)

    assert torch.allclose(sequential, blocked, rtol=2e-5, atol=2e-5)
    assert torch.allclose(sequential, compact, rtol=2e-5, atol=2e-5)
    good, message = check_implementation(data, (h_block, tau_block))
    assert good, message


def test_blocked_qr_reference_cli_row_reports_contract_fields():
    spec = {"batch": 2, "n": 8, "cond": 2, "seed": 222, "case": "mixed"}
    row = run_blocked_qr_reference_case(
        spec,
        panel_width=3,
        column_major_h=True,
        update_mode="compact-wy",
        precision_mode="fp32",
        r_maintenance_mode="panel-prefix",
        panel_refresh_mode="prefix",
        diagnose_output=True,
    )

    assert row["ok"], row
    assert row["case_text"] == format_case(spec)
    assert row["panel_width"] == 3
    assert row["update_mode"] == "compact-wy"
    assert row["precision_mode"] == "fp32"
    assert row["r_maintenance_mode"] == "panel-prefix"
    assert row["panel_refresh_mode"] == "prefix"
    assert row["column_major_h"] is True
    assert row["h_stride"] == [8 * 8, 1, 8]
    assert row["factor_scaled_max"] < 20.0
    assert row["orth_scaled_max"] < 100.0


def test_blocked_qr_sweep_records_precision_mode_failures():
    spec = {"batch": 2, "n": 8, "cond": 1, "seed": 444}
    assert parse_blocked_qr_panel_widths("2,3") == [2, 3]
    rows = run_blocked_qr_sweep(
        [(0, spec)],
        panel_widths=[3],
        update_modes=["compact-wy"],
        precision_modes=["fp32", "tf32-input"],
        r_maintenance_modes=["none", "panel-prefix"],
        panel_refresh_modes=["none", "prefix"],
        diagnose_output=True,
    )

    data_rows = [row for row in rows if not row.get("summary")]
    summary = rows[-1]
    assert len(data_rows) == 8
    assert summary == summarize_blocked_qr_sweep(data_rows)
    assert not summary["ok"]
    assert summary["num_failed"] == 3
    assert any(
        row["ok"] and row["precision_mode"] == "fp32" and row["r_maintenance_mode"] == "none"
        for row in data_rows
    )
    assert any(
        (not row["ok"]) and row["precision_mode"] == "tf32-input" and row["r_maintenance_mode"] == "none"
        for row in data_rows
    )
    assert any(
        row["ok"]
        and row["precision_mode"] == "tf32-input"
        and row["r_maintenance_mode"] == "panel-prefix"
        and row["panel_refresh_mode"] == "prefix"
        for row in data_rows
    )
    assert all("factor_scaled_max" in row for row in data_rows)


def test_verifier_experiments_have_passing_smoke_paths():
    spec = _small_case("dense")
    data = generate_input(**spec)
    custom_kernel = load_submission(ROOT / "submissions/candidate.py")

    rows = []
    rows.extend(experiment_r_projection(custom_kernel, data, spec))
    rows.extend(experiment_column_major(data, spec))
    rows.extend(experiment_tail_delete(data, spec, [0]))

    assert rows
    assert all(row["ok"] for row in rows), rows
    r_projection = rows[0]
    assert "factor_scaled_max_before" in r_projection
    assert "factor_scaled_max_after" in r_projection
    assert "orth_scaled_max_before" in r_projection


def test_identity_q_experiment_passes_upper_and_fails_dense():
    upper_spec = {"batch": 2, "n": 8, "cond": 0, "seed": 12, "case": "upper"}
    upper_data = generate_input(**upper_spec)
    upper_row = experiment_identity_q(upper_data, upper_spec)[0]
    assert upper_row["ok"], upper_row

    dense_spec = {"batch": 2, "n": 8, "cond": 1, "seed": 12}
    dense_data = generate_input(**dense_spec)
    dense_row = experiment_identity_q(dense_data, dense_spec)[0]
    assert not dense_row["ok"], dense_row


def test_all_experiments_excludes_negative_controls():
    assert "identity-q" not in parse_experiments("all")
    assert "identity-q" in parse_experiments("all-with-controls")


def test_tail_delete_skips_cuts_that_remove_all_reflectors():
    spec = {"batch": 2, "n": 8, "cond": 1, "seed": 12}
    data = generate_input(**spec)
    rows = experiment_tail_delete(data, spec, [0, 4, 8, 16])
    assert [row["tail_cut"] for row in rows] == [0, 4]
    assert all(row["reflectors_kept"] > 0 for row in rows)


def test_r_projection_reports_bad_candidate_output_cleanly():
    data = generate_input(**_small_case("dense"))

    def bad_kernel(_data):
        return data

    row = experiment_r_projection(bad_kernel, data, _small_case("dense"))[0]
    assert not row["ok"]
    assert row["before_ok"] is False
    assert "cannot repair R" in row["message"]


def test_r_projection_repairs_bad_r_when_q_structure_is_valid():
    spec = _small_case("dense")
    data = generate_input(**spec)

    def bad_r_kernel(_data):
        h, tau = torch.geqrf(_data)
        h_bad = h.clone()
        h_bad = torch.tril(h_bad, diagonal=-1)
        return h_bad, tau

    row = experiment_r_projection(bad_r_kernel, data, spec)[0]
    assert row["before_ok"] is False, row
    assert row["ok"], row
    assert row["factor_scaled_max_after"] < row["factor_scaled_max_before"]


def test_submission_loader_supports_same_directory_helpers(tmp_path):
    helper = tmp_path / "helper.py"
    helper.write_text(
        "import torch\n\n"
        "def run(data):\n"
        "    return torch.geqrf(data)\n"
    )
    submission = tmp_path / "candidate_with_helper.py"
    submission.write_text(
        "from task import input_t, output_t\n"
        "from helper import run\n\n"
        "def custom_kernel(data: input_t) -> output_t:\n"
        "    return run(data)\n"
    )
    custom_kernel = load_submission(submission)
    assert sys.path[0] == str(ROOT / "official")
    assert sys.path[1] == str(tmp_path)
    data = generate_input(**_small_case("dense"))
    good, message = check_implementation(data, custom_kernel(data.clone()))
    assert good, message


def test_submission_loader_keeps_official_dir_first_for_official_submission():
    custom_kernel = load_submission(ROOT / "official/submission.py")
    assert sys.path[0] == str(ROOT / "official")
    data = generate_input(**_small_case("dense"))
    good, message = check_implementation(data, custom_kernel(data.clone()))
    assert good, message


def test_check_cases_helpers_on_small_case():
    custom_kernel = load_submission(ROOT / "submissions/candidate.py")
    row = run_check_case(custom_kernel, {"batch": 2, "n": 8, "cond": 1, "seed": 11}, diagnose_output=True)
    assert row["ok"], row
    assert "diagnostics" in row
    assert row["h_shape"] == [2, 8, 8]
    assert len(row["h_stride"]) == 3
    assert isinstance(row["column_major_h_actual"], bool)
    assert row["h_layout_actual"] in {"column_major", "torch_contiguous", "other_strided"}
    margin_row = run_check_case(
        custom_kernel,
        {"batch": 2, "n": 8, "cond": 1, "seed": 11},
        diagnose_output=False,
        max_factor_scaled=20.0,
        max_orth_scaled=100.0,
    )
    assert margin_row["ok"], margin_row
    assert margin_row["margin_ok"], margin_row
    assert margin_row["factor_margin_ok"], margin_row
    assert margin_row["orth_margin_ok"], margin_row
    tight_row = run_check_case(
        custom_kernel,
        {"batch": 2, "n": 8, "cond": 1, "seed": 11},
        diagnose_output=False,
        max_factor_scaled=0.0,
    )
    assert tight_row["ok"], tight_row
    assert not tight_row["margin_ok"], tight_row
    assert not tight_row["factor_margin_ok"], tight_row
    assert parse_indices("", 3) == [0, 1, 2]
    assert parse_indices("0,2", 3) == [0, 2]
    with pytest.raises(IndexError):
        parse_indices("3", 3)
    assert parse_benchmark_indices("", 3) == [0, 1, 2]
    assert parse_benchmark_indices("0,2", 3) == [0, 2]
    with pytest.raises(ValueError):
        parse_benchmark_indices("3", 3)
    with pytest.raises(ValueError):
        parse_benchmark_indices(",", 3)


def test_environment_info_contains_compact_result_provenance(monkeypatch):
    monkeypatch.setenv("FAST_QR_QR512_PANEL_B", "32")
    monkeypatch.setenv("FAST_QR_QR512_PRECISION_MODE", "tf32-input")
    monkeypatch.setenv("FAST_QR_QR512_BLOCKED_POLICY_SAMPLE_ROWS", "12")
    monkeypatch.setenv("FAST_QR_QR2048_TAIL_THRESHOLD", "0.2")
    monkeypatch.setenv("FAST_QR_OCCUPANCY_REGISTERS_PER_SM", "65536")
    info = environment_info(torch)
    assert "git_hash" in info
    assert "git_full_hash" in info
    assert "git_dirty" in info
    assert "official_upstream_commit" in info
    assert "git_status_porcelain" not in info
    assert info["candidate_env"] == {
        "FAST_QR_QR512_PANEL_B": "32",
        "FAST_QR_QR512_PRECISION_MODE": "tf32-input",
        "FAST_QR_QR512_BLOCKED_POLICY_SAMPLE_ROWS": "12",
        "FAST_QR_QR2048_TAIL_THRESHOLD": "0.2",
    }
    assert info["tracked_env"]["FAST_QR_QR512_PANEL_B"] == "32"
    assert info["tracked_env"]["FAST_QR_QR512_BLOCKED_POLICY_SAMPLE_ROWS"] == "12"
    assert info["tracked_env"]["FAST_QR_OCCUPANCY_REGISTERS_PER_SM"] == "65536"

    candidate = file_provenance(ROOT / "submissions/candidate.py")
    assert candidate["path"] == "submissions/candidate.py"
    assert len(candidate["sha256"]) == 64


def test_tracked_candidate_env_can_record_absent_keys():
    env = {"FAST_QR_QR512_PANEL_B": "64", "UNRELATED": "ignored"}
    row = tracked_candidate_env(env, include_absent=True)
    assert row["FAST_QR_QR512_PANEL_B"] == "64"
    assert row["FAST_QR_QR1024_PANEL_B"] is None
    assert "UNRELATED" not in row
    assert set(row) == set(CANDIDATE_RUNTIME_ENV_KEYS)


def _suite_args(**overrides):
    values = {
        "skip_policy": False,
        "skip_pytest": False,
        "skip_submission_validation": False,
        "skip_route_trace": False,
        "skip_guard_benchmark": False,
        "skip_route_ablations": False,
        "skip_secret_audit": False,
        "skip_runtime_preflight": False,
        "skip_seed_sweep": False,
        "skip_quantization_sweep": False,
        "skip_mixed_seed_sweep": False,
        "skip_classifier_sweep": False,
        "skip_tail_policy_sweep": False,
        "skip_candidate_tests": False,
        "skip_benchmark_correctness": False,
        "skip_dev_robustness": False,
        "skip_accelerator_preflight": False,
        "skip_smoke": False,
        "skip_baseline_public": False,
        "skip_candidate_public": False,
        "skip_experiments": False,
        "skip_official_style": False,
        "skip_candidate_official_style": False,
        "candidate_test_indices": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_default_suite_validation_blockers_match_required_artifacts():
    assert default_validation_blockers(_suite_args()) == []
    assert default_validation_blockers(_suite_args(skip_pytest=True)) == ["--skip-pytest"]
    assert default_validation_blockers(_suite_args(skip_smoke=True)) == ["--skip-smoke"]
    assert default_validation_blockers(_suite_args(skip_submission_validation=True)) == [
        "--skip-submission-validation"
    ]
    assert default_validation_blockers(_suite_args(candidate_test_indices="0,1")) == ["--candidate-test-indices"]
    assert default_validation_blockers(_suite_args(skip_benchmark_correctness=True)) == [
        "--skip-benchmark-correctness"
    ]
    assert default_validation_blockers(_suite_args(skip_dev_robustness=True)) == ["--skip-dev-robustness"]
    assert default_validation_blockers(_suite_args(skip_secret_audit=True)) == ["--skip-secret-audit"]
    assert default_validation_blockers(_suite_args(skip_runtime_preflight=True)) == ["--skip-runtime-preflight"]
    assert default_validation_blockers(_suite_args(skip_quantization_sweep=True)) == ["--skip-quantization-sweep"]
    assert default_validation_blockers(_suite_args(skip_mixed_seed_sweep=True)) == ["--skip-mixed-seed-sweep"]
    assert default_validation_blockers(_suite_args(skip_tail_policy_sweep=True)) == ["--skip-tail-policy-sweep"]
    assert default_validation_blockers(_suite_args(skip_classifier_sweep=True)) == ["--skip-classifier-sweep"]
    assert default_validation_blockers(_suite_args(skip_policy=True, skip_route_ablations=True)) == [
        "--skip-policy",
        "--skip-route-ablations",
    ]


def test_b200_suite_applies_and_records_compile_environment():
    args = SimpleNamespace(
        submission="submissions/candidate.py",
        baseline="submissions/baseline_geqrf.py",
        torch_cuda_arch_list="10.0",
        qr32_extra_cuda_cflags="-arch=sm_100",
        qr32_sm100=False,
    )
    assert suite_env_overrides(args) == {
        "TORCH_CUDA_ARCH_LIST": "10.0",
        "FAST_QR_QR32_EXTRA_CUDA_CFLAGS": "-arch=sm_100",
    }

    env = apply_suite_env_options({}, args)
    assert env["TORCH_CUDA_ARCH_LIST"] == "10.0"
    assert env["FAST_QR_QR32_EXTRA_CUDA_CFLAGS"] == "-arch=sm_100"
    env["FAST_QR_QR2048_TAIL_CUT"] = "64"
    env["FAST_QR_QR2048_TAIL_THRESHOLD"] = "0.2"
    env["FAST_QR_QR2048_BLOCKED_POLICY_SAMPLE_ROWS"] = "16"
    env["FAST_QR_QR4096_TAIL_CUT"] = "128"
    env["FAST_QR_QR4096_TAIL_THRESHOLD"] = "0.1"
    env["FAST_QR_QR512_THREADS_PER_CTA"] = "512"
    env["FAST_QR_OCCUPANCY_REGISTERS_PER_SM"] = "131072"

    row = suite_provenance(args, env)
    assert row["env"]["TORCH_CUDA_ARCH_LIST"] == "10.0"
    assert row["env"]["FAST_QR_QR32_EXTRA_CUDA_CFLAGS"] == "-arch=sm_100"
    assert row["env"]["FAST_QR_QR2048_TAIL_CUT"] == "64"
    assert row["env"]["FAST_QR_QR2048_TAIL_THRESHOLD"] == "0.2"
    assert row["env"]["FAST_QR_QR2048_BLOCKED_POLICY_SAMPLE_ROWS"] == "16"
    assert row["env"]["FAST_QR_QR4096_TAIL_CUT"] == "128"
    assert row["env"]["FAST_QR_QR4096_TAIL_THRESHOLD"] == "0.1"
    assert row["env"]["FAST_QR_QR512_THREADS_PER_CTA"] == "512"
    assert row["env"]["FAST_QR_QR1024_THREADS_PER_CTA"] is None
    assert row["env"]["FAST_QR_OCCUPANCY_REGISTERS_PER_SM"] == "131072"


def test_candidate_config_consumed_env_key_list_matches_runtime_tracking():
    assert CURRENT_CANDIDATE_CONSUMED_ENV_KEYS == set(CANDIDATE_RUNTIME_ENV_KEYS)


def test_b200_suite_qr32_sm100_shortcut_sets_extra_compile_flags():
    args = SimpleNamespace(torch_cuda_arch_list="10.0", qr32_extra_cuda_cflags=None, qr32_sm100=True)
    assert suite_env_overrides(args)["FAST_QR_QR32_EXTRA_CUDA_CFLAGS"] == "-arch=sm_100"
    assert apply_suite_env_options({}, args)["FAST_QR_QR32_EXTRA_CUDA_CFLAGS"] == "-arch=sm_100"


def test_b200_suite_dry_run_exposes_compile_environment(tmp_path):
    args = SimpleNamespace(
        skip_suite_validation=False,
        require_final_kernels=True,
        torch_cuda_arch_list="10.0",
        qr32_extra_cuda_cflags="-arch=sm_100",
        qr32_sm100=False,
    )
    env = apply_suite_env_options({}, args)
    plan = dry_run_plan(
        tmp_path,
        [("print_spec", ["python", "tools/print_spec.py"])],
        [],
        args,
        env,
    )
    assert plan["suite_env"] == {
        "TORCH_CUDA_ARCH_LIST": "10.0",
        "FAST_QR_QR32_EXTRA_CUDA_CFLAGS": "-arch=sm_100",
    }
    assert plan["will_require_final_kernels"] is True


def test_b200_suite_default_accelerator_preflight_runs_family_cases(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "tools/run_b200_suite.py",
            "--dry-run",
            "--dry-run-json",
            "--suite-name",
            f"pytest_default_preflight_{tmp_path.name}",
            "--skip-suite-validation",
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    plan = json.loads(completed.stdout)
    preflight_cmd = next(row["cmd"] for row in plan["steps"] if row["step"] == "accelerator_preflight")
    assert "tools/preflight_accelerators.py" in preflight_cmd
    assert "--family-cases" in preflight_cmd


def test_b200_suite_leaderboard_dry_run_includes_test_first():
    completed = subprocess.run(
        [
            sys.executable,
            "tools/run_b200_suite.py",
            "--dry-run",
            "--dry-run-json",
            "--suite-name",
            "pytest_popcorn_leaderboard",
            "--skip-suite-validation",
            "--include-popcorn-leaderboard",
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    plan = json.loads(completed.stdout)
    steps = [row["step"] for row in plan["steps"]]
    assert "popcorn_test" in steps
    assert "popcorn_leaderboard" in steps
    assert steps.index("popcorn_test") < steps.index("popcorn_leaderboard")
    assert plan["workload"]["num_official_remote_steps"] == 2


def test_b200_suite_dry_run_can_generate_current_candidate_config_plan(tmp_path):
    suite_name = f"pytest_current_candidate_plan_{tmp_path.name}"
    completed = subprocess.run(
        [
            sys.executable,
            "tools/run_b200_suite.py",
            "--dry-run",
            "--dry-run-json",
            "--suite-name",
            suite_name,
            "--skip-suite-validation",
            "--include-candidate-config-tune",
            "--candidate-config-tune-shape-label",
            "qr512",
            "--candidate-config-tune-large-kernel-plan-mode",
            "current-candidate",
            "--candidate-config-tune-large-kernel-plan-max-configs",
            "16",
            "--candidate-config-tune-correctness-indices",
            "3,6,7,8,9,10,11,19",
            "--candidate-config-tune-benchmark-indices",
            "3,7,9,10",
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    plan = json.loads(completed.stdout)
    suite_dir = Path(plan["suite_dir"])
    assert suite_dir.name == suite_name
    assert not suite_dir.exists()

    steps = [row["step"] for row in plan["steps"]]
    assert "candidate_config_accelerator_preflight" in steps
    assert steps.index("candidate_config_accelerator_preflight") < steps.index("candidate_config_tune")
    preflight_step = next(row for row in plan["steps"] if row["step"] == "candidate_config_accelerator_preflight")
    preflight_cmd = preflight_step["cmd"]
    assert "tools/preflight_accelerators.py" in preflight_cmd
    assert preflight_cmd[preflight_cmd.index("--accelerators") + 1] == "qr512_blocked_cuda_auto"
    assert "--family-cases" in preflight_cmd
    assert Path(preflight_cmd[preflight_cmd.index("--config-jsonl") + 1]) == (
        suite_dir / "candidate_config_tune_large_kernel_configs.jsonl"
    )
    assert Path(preflight_cmd[preflight_cmd.index("--out") + 1]) == candidate_config_accelerator_preflight_path(suite_dir)

    config_step = next(row for row in plan["steps"] if row["step"] == "candidate_config_tune")
    cmd = config_step["cmd"]
    config_path = Path(cmd[cmd.index("--config-jsonl") + 1])
    assert config_path == suite_dir / "candidate_config_tune_large_kernel_configs.jsonl"
    assert cmd[cmd.index("--shape-label") + 1] == "qr512"
    assert cmd[cmd.index("--benchmark-indices") + 1] == "3,7,9,10"
    assert plan["workload"]["num_candidate_config_tuning_steps"] == 1
    assert plan["workload"]["step_counts_by_category"]["preflight"] >= 1
    generated = plan["candidate_config_tune_large_kernel_plan"]
    assert generated["path"] == str(suite_dir / "candidate_config_tune_large_kernel_configs.jsonl")
    assert generated["mode"] == "current-candidate"
    assert generated["shape_label"] == "qr512"
    assert generated["num_configs"] == 16
    assert len(generated["configs"]) == 16
    assert generated["config_names"] == [row["name"] for row in generated["configs"]]
    assert not Path(generated["path"]).exists()


def test_b200_suite_candidate_config_tune_next_required_auto_targets_qr512(tmp_path):
    suite_name = f"pytest_candidate_config_auto_{tmp_path.name}"
    completed = subprocess.run(
        [
            sys.executable,
            "tools/run_b200_suite.py",
            "--dry-run",
            "--dry-run-json",
            "--suite-name",
            suite_name,
            "--skip-suite-validation",
            "--candidate-config-tune-next-required",
            "--candidate-config-tune-large-kernel-plan-max-configs",
            "8",
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    plan = json.loads(completed.stdout)
    suite_dir = Path(plan["suite_dir"])
    target = plan["candidate_config_tune_policy_target"]
    assert target["source"] == "candidate_policy"
    assert target["shape_label"] == "qr512"
    assert target["required_cuda_kernel"] == "qr512_blocked_householder_r_maintenance"
    assert target["benchmark_indices"] == "3,7,9,10"
    assert target["correctness_indices"] == "3,6,7,8,9,10,11,19"
    assert target["large_kernel_plan_mode"] == "current-candidate"
    assert target["effective_only"] is True
    assert target["applied_axis_constraints"] == {
        "panel_refresh_modes": "prefix",
        "r_maintenance_modes": "panel-prefix",
    }

    steps = [row["step"] for row in plan["steps"]]
    assert "candidate_config_accelerator_preflight" in steps
    assert "candidate_config_tune" in steps
    preflight_cmd = next(row["cmd"] for row in plan["steps"] if row["step"] == "candidate_config_accelerator_preflight")
    assert preflight_cmd[preflight_cmd.index("--accelerators") + 1] == "qr512_blocked_cuda_auto"
    assert "--family-cases" in preflight_cmd
    assert Path(preflight_cmd[preflight_cmd.index("--config-jsonl") + 1]) == (
        suite_dir / "candidate_config_tune_large_kernel_configs.jsonl"
    )

    tune_cmd = next(row["cmd"] for row in plan["steps"] if row["step"] == "candidate_config_tune")
    assert tune_cmd[tune_cmd.index("--shape-label") + 1] == "qr512"
    assert tune_cmd[tune_cmd.index("--correctness-indices") + 1] == "3,6,7,8,9,10,11,19"
    assert tune_cmd[tune_cmd.index("--benchmark-indices") + 1] == "3,7,9,10"
    assert tune_cmd[tune_cmd.index("--env-prefix") + 1] == "FAST_QR_QR512"
    assert "--panel-refresh-modes" not in tune_cmd
    assert "--r-maintenance-modes" not in tune_cmd
    assert Path(tune_cmd[tune_cmd.index("--config-jsonl") + 1]) == (
        suite_dir / "candidate_config_tune_large_kernel_configs.jsonl"
    )
    assert plan["workload"]["num_candidate_config_tuning_steps"] == 1
    generated = plan["candidate_config_tune_large_kernel_plan"]
    assert generated["path"] == str(suite_dir / "candidate_config_tune_large_kernel_configs.jsonl")
    assert generated["mode"] == "current-candidate"
    assert generated["shape_label"] == "qr512"
    assert generated["env_prefix"] == "FAST_QR_QR512"
    assert generated["max_configs"] == 8
    assert generated["num_configs"] == 8
    assert len(generated["configs"]) == 8
    assert generated["configs"][0]["name"] == generated["config_names"][0]
    assert {row["env"]["FAST_QR_QR512_PANEL_REFRESH_MODE"] for row in generated["configs"]} == {"prefix"}
    assert {row["env"]["FAST_QR_QR512_R_MAINTENANCE_MODE"] for row in generated["configs"]} == {
        "panel-prefix"
    }

    args = SimpleNamespace(
        candidate_config_tune_shape_label="qr512",
        candidate_config_tune_env_prefix="FAST_QR_QR512",
        candidate_config_tune_benchmark_cases="cases/public_benchmarks.txt",
        candidate_config_tune_benchmark_indices="",
        candidate_config_tune_correctness_indices="19,20,21",
        candidate_config_tune_large_kernel_plan_mode="current-candidate",
        candidate_config_tune_large_kernel_plan_max_configs=8,
        submission="submissions/candidate.py",
    )
    target_rows = candidate_config_tune_large_kernel_plan_rows(args)
    assert {tuple(sorted(row["env"])) for row in target_rows} == {
        (
            "FAST_QR_ENABLE_QR512_BLOCKED_CUDA",
            "FAST_QR_QR512_BLOCKED_AUTO_GROUPS",
            "FAST_QR_QR512_COMPACT_WY_TILE_COLS",
            "FAST_QR_QR512_CTAS_PER_MATRIX",
            "FAST_QR_QR512_CTA_SCHEDULE",
            "FAST_QR_QR512_PANEL_B",
            "FAST_QR_QR512_PANEL_REFRESH_MODE",
            "FAST_QR_QR512_POLICY_FULL_SCAN",
            "FAST_QR_QR512_PRECISION_MODE",
            "FAST_QR_QR512_R_MAINTENANCE_MODE",
            "FAST_QR_QR512_STRUCTURED_BEFORE_CUDA",
            "FAST_QR_QR512_SYNC_FREE_AUTO_POLICY",
            "FAST_QR_QR512_TAIL_CUT",
            "FAST_QR_QR512_TAIL_FORCE",
            "FAST_QR_QR512_TAIL_THRESHOLD",
            "FAST_QR_QR512_TILE_N",
            "FAST_QR_QR512_UPDATE_MODE",
            "FAST_QR_QR512_WARPS_PER_CTA",
        ),
    }


def test_b200_suite_dry_run_can_include_blocked_qr_sweep(tmp_path):
    suite_name = f"pytest_blocked_qr_sweep_{tmp_path.name}"
    completed = subprocess.run(
        [
            sys.executable,
            "tools/run_b200_suite.py",
            "--dry-run",
            "--dry-run-json",
            "--suite-name",
            suite_name,
            "--skip-suite-validation",
            "--include-blocked-qr-sweep",
            "--blocked-qr-sweep-indices",
            "3,19",
            "--blocked-qr-sweep-panel-widths",
            "16",
            "--blocked-qr-sweep-precision-modes",
            "fp32,tf32-input",
            "--blocked-qr-sweep-r-maintenance-modes",
            "none,panel-prefix",
            "--blocked-qr-sweep-panel-refresh-modes",
            "none,prefix",
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    plan = json.loads(completed.stdout)
    steps = [row["step"] for row in plan["steps"]]
    assert "blocked_qr_sweep" in steps
    assert plan["workload"]["num_verifier_experiment_steps"] >= 1
    step = next(row for row in plan["steps"] if row["step"] == "blocked_qr_sweep")
    cmd = step["cmd"]
    assert "tools/blocked_qr_sweep.py" in cmd
    assert cmd[cmd.index("--indices") + 1] == "3,19"
    assert cmd[cmd.index("--panel-widths") + 1] == "16"
    assert cmd[cmd.index("--precision-modes") + 1] == "fp32,tf32-input"
    assert cmd[cmd.index("--r-maintenance-modes") + 1] == "none,panel-prefix"
    assert cmd[cmd.index("--panel-refresh-modes") + 1] == "none,prefix"
    assert "--allow-failures" in cmd
    assert Path(cmd[cmd.index("--out") + 1]) == Path(plan["suite_dir"]) / "blocked_qr_sweep.jsonl"


def test_b200_suite_candidate_config_accelerator_helpers(tmp_path):
    assert candidate_config_accelerator_for_shape("qr512") == "qr512_blocked_cuda_auto"
    assert candidate_config_accelerator_for_shape("qr1024") == "qr1024_blocked_cuda_auto"
    assert candidate_config_accelerator_for_shape("qr2048") == "qr2048_blocked_cuda_auto"
    assert candidate_config_accelerator_for_shape("qr4096") == "qr4096_blocked_cuda_auto"
    assert candidate_config_accelerator_preflight_path(tmp_path) == tmp_path / "candidate_config_accelerator_preflight.jsonl"


def test_b200_suite_materializes_generated_candidate_config_plan(tmp_path):
    args = SimpleNamespace(
        candidate_config_tune_shape_label="qr1024",
        candidate_config_tune_env_prefix=None,
        candidate_config_tune_large_kernel_plan_mode="current-candidate",
        candidate_config_tune_large_kernel_plan_max_configs=16,
    )
    expected_rows = candidate_config_tune_large_kernel_plan_rows(args)
    path, rows = write_candidate_config_tune_large_kernel_plan(args, tmp_path)

    assert path == candidate_config_tune_large_kernel_plan_path(tmp_path)
    assert rows == expected_rows
    written = [json.loads(line) for line in path.read_text().splitlines()]
    assert written == [{"name": row["name"], "env": row["env"]} for row in expected_rows]
    assert {row["env"]["FAST_QR_QR1024_WARPS_PER_CTA"] for row in written} == {"8", "16", "32"}
    assert {row["env"]["FAST_QR_QR1024_STRUCTURED_BEFORE_CUDA"] for row in written} == {"0", "1"}


def test_future_blocked_large_kernel_plan_carries_repair_mode_axes():
    rows = generate_large_kernel_configs("qr512", max_configs=96, mode="future-blocked")
    env_keys = {key for row in rows for key in row["env"]}
    assert "FAST_QR_QR512_PANEL_REFRESH_MODE" in env_keys
    assert "FAST_QR_QR512_R_MAINTENANCE_MODE" in env_keys
    assert "FAST_QR_QR512_BLOCKED_AUTO_GROUPS" in env_keys
    assert "FAST_QR_QR512_POLICY_FULL_SCAN" in env_keys
    assert "FAST_QR_QR512_TAIL_FORCE" in env_keys
    assert {row["env"].get("FAST_QR_QR512_PANEL_REFRESH_MODE") for row in rows} >= {"none", "prefix"}
    assert {row["env"].get("FAST_QR_QR512_R_MAINTENANCE_MODE") for row in rows} >= {"none", "panel-prefix"}
    assert {row["env"].get("FAST_QR_QR512_BLOCKED_AUTO_GROUPS") for row in rows} >= {"0", "1"}
    assert {row["env"].get("FAST_QR_QR512_POLICY_FULL_SCAN") for row in rows} >= {"0", "1"}
    assert {row["env"].get("FAST_QR_QR512_TAIL_FORCE") for row in rows} >= {"0", "1"}
    assert any("panel_refresh_mode_prefix" in row["name"] for row in rows)
    assert any("r_maintenance_mode_panel-prefix" in row["name"] for row in rows)
    assert any("blocked_auto_groups_0" in row["name"] for row in rows)

    mode_env = {
        "FAST_QR_QR512_PANEL_REFRESH_MODE": "prefix",
        "FAST_QR_QR512_R_MAINTENANCE_MODE": "panel-prefix",
        "FAST_QR_QR512_BLOCKED_AUTO_GROUPS": "0",
        "FAST_QR_QR512_POLICY_FULL_SCAN": "1",
        "FAST_QR_QR512_TAIL_FORCE": "1",
    }
    consumption = candidate_config_env_consumption(mode_env)
    assert set(consumption["candidate_consumed_env_keys"]) == set(mode_env)
    assert consumption["inert_env_keys"] == []


def test_b200_suite_can_unset_inherited_compile_environment():
    args = SimpleNamespace(torch_cuda_arch_list="", qr32_extra_cuda_cflags="", qr32_sm100=False)
    env = apply_suite_env_options(
        {
            "TORCH_CUDA_ARCH_LIST": "8.0",
            "FAST_QR_QR32_EXTRA_CUDA_CFLAGS": "-arch=sm_80",
        },
        args,
    )
    assert "TORCH_CUDA_ARCH_LIST" not in env
    assert "FAST_QR_QR32_EXTRA_CUDA_CFLAGS" not in env


def test_b200_runtime_preflight_evaluator_contract():
    info = {
        "torch": {
            "import_ok": True,
            "cuda_available": True,
            "device_count": 1,
            "devices": [
                {
                    "name": "NVIDIA B200",
                    "capability": [10, 0],
                    "total_memory_gib": 180.0,
                }
            ],
        }
    }
    ok, errors = evaluate_b200_env(info, require_name="B200", min_major=10, min_memory_gib=150.0)
    assert ok, errors

    bad = {
        "torch": {
            "import_ok": True,
            "cuda_available": True,
            "device_count": 1,
            "devices": [
                {
                    "name": "NVIDIA A100",
                    "capability": [8, 0],
                    "total_memory_gib": 80.0,
                }
            ],
        }
    }
    ok, errors = evaluate_b200_env(bad, require_name="B200", min_major=10, min_memory_gib=150.0)
    assert not ok
    assert any("does not contain" in error for error in errors)
    assert any("capability" in error for error in errors)
    assert any("memory" in error for error in errors)


def test_secret_scanner_detects_high_signal_tokens_without_exposing_value():
    rows = scan_secret_text(ROOT / "README.md", "OPENAI_API_KEY=sk-" + "a" * 48)
    assert len(rows) == 1
    assert rows[0]["rule"] == "openai_api_key"
    assert rows[0]["file"] == "README.md"
    assert "a" * 24 not in rows[0]["match_preview"]


def test_suite_summary_compares_baseline_and_candidate(tmp_path):
    baseline = tmp_path / "baseline_geqrf_public.jsonl"
    candidate = tmp_path / "candidate_public.jsonl"
    guards = tmp_path / "candidate_guard_overhead_public.jsonl"
    (tmp_path / "manifest.jsonl").write_text(
        '{"event": "finish", "step": "candidate_public", "elapsed_s": 12.0, "time": "2026-06-21T00:00:00"}\n'
        '{"event": "finish", "step": "baseline_public", "elapsed_s": 30.0, "time": "2026-06-21T00:00:01"}\n'
        '{"event": "suite_finish", "elapsed_s": 45.0, "time": "2026-06-21T00:00:02"}\n'
    )
    baseline.write_text(
        '{"ok": true, "spec": {"batch": 1, "n": 8, "cond": 1, "seed": 1}, "mean_us": 100.0, "runs": 3}\n'
        '{"ok": true, "spec": {"batch": 1, "n": 16, "cond": 1, "seed": 2}, "mean_us": 400.0, "runs": 3}\n'
        '{"geomean_us": 200.0, "num_cases": 2}\n'
    )
    candidate.write_text(
        '{"ok": true, "spec": {"batch": 1, "n": 8, "cond": 1, "seed": 1}, "mean_us": 50.0, "runs": 3}\n'
        '{"ok": true, "spec": {"batch": 1, "n": 16, "cond": 1, "seed": 2}, "mean_us": 100.0, "runs": 3}\n'
        '{"geomean_us": 70.7107, "num_cases": 2}\n'
    )
    (tmp_path / "candidate_ablation_no_structured_routes_public.jsonl").write_text(
        '{"ok": true, "spec": {"batch": 1, "n": 8, "cond": 1, "seed": 1}, "mean_us": 100.0, "runs": 2}\n'
        '{"ok": true, "spec": {"batch": 1, "n": 16, "cond": 1, "seed": 2}, "mean_us": 200.0, "runs": 2}\n'
    )
    tune_dir = tmp_path / "tail_policy_tune"
    tune_dir.mkdir()
    (tune_dir / "summary.json").write_text(
        json.dumps(
            {
                "ok": False,
                "hard_failed": False,
                "allow_failed_configs": True,
                "num_configs": 2,
                "best": {"name": "cut16", "benchmark": {"geomean_us": 90.0}},
                "results": [
                    {
                        "name": "cut16",
                        "env": {"FAST_QR_DENSE_TAIL_CUT_512": "16"},
                        "correctness": {"num_failed": 0, "max_factor_scaled": 2.0, "max_orth_scaled": 1.0},
                        "benchmark": {"num_cases": 2, "geomean_us": 90.0},
                    },
                    {
                        "name": "cut64",
                        "env": {"FAST_QR_DENSE_TAIL_CUT_512": "64"},
                        "correctness": {"num_failed": 1, "max_factor_scaled": 22.0, "max_orth_scaled": 1.0},
                        "benchmark": None,
                    },
                ],
            }
        )
        + "\n"
    )
    config_tune_dir = tmp_path / "candidate_config_tune"
    config_tune_dir.mkdir()
    (config_tune_dir / "summary.json").write_text(
        json.dumps(
            {
                "ok": False,
                "hard_failed": False,
                "allow_failed_configs": True,
                "objective": "minimize_geomean_us",
                "num_configs": 2,
                "num_configs_with_inert_env": 0,
                "num_configs_with_cuda_route_bypassed_env": 1,
                "num_configs_with_resource_metrics": 1,
                "best": {"name": "qr512_panel32", "benchmark": {"geomean_us": 80.0}},
                "results": [
                    {
                        "name": "qr512_panel32",
                        "env": {"FAST_QR_QR512_PANEL_B": "32"},
                        "env_consumption": {
                            "candidate_consumed_env_keys": ["FAST_QR_QR512_PANEL_B"],
                            "tuner_consumed_env_keys": [],
                            "inert_env_keys": [],
                            "cuda_route_bypassed_env_keys": [],
                            "has_inert_env": False,
                        },
                        "correctness": {"num_failed": 0, "max_factor_scaled": 3.0, "max_orth_scaled": 2.0},
                        "benchmark": {"num_cases": 4, "geomean_us": 80.0},
                        "resource_metrics": {
                            "available": True,
                            "max_registers_per_thread": 48,
                            "max_smem_bytes": 2048,
                            "max_spill_store_bytes": 0,
                            "max_spill_load_bytes": 0,
                            "min_estimated_occupancy": 0.5,
                        },
                    },
                    {
                        "name": "qr512_panel64",
                        "env": {"FAST_QR_QR512_PANEL_B": "64", "FAST_QR_QR512_TAIL_CUT": "24"},
                        "env_consumption": {
                            "candidate_consumed_env_keys": ["FAST_QR_QR512_PANEL_B", "FAST_QR_QR512_TAIL_CUT"],
                            "tuner_consumed_env_keys": [],
                            "inert_env_keys": [],
                            "cuda_route_bypassed_env_keys": ["FAST_QR_QR512_TAIL_CUT"],
                            "has_inert_env": False,
                        },
                        "correctness": {"num_failed": 1, "max_factor_scaled": 23.0, "max_orth_scaled": 2.0},
                        "benchmark": None,
                    },
                ],
            }
        )
        + "\n"
    )
    guards.write_text(
        '{"case_index": 0, "spec": "batch: 1; n: 8; cond: 1; seed: 1", "route": "torch.geqrf", '
        '"cold_wall_us": 20.0, "cold_cuda_us": null, '
        '"wall_us": 2.0, "cuda_us": null, "repeats": 10, "warmup": 2, '
        '"case_metadata_passed_to_submission": false, '
        '"case_selection_info_sources": ["data.shape"], "dispatch_info_sources": ["data.shape"], '
        '"uses_tensor_values_for_dispatch": false, "uses_tensor_values_for_case_selection": false, '
        '"classifier_needed_for_current_candidate": false, "route_decision_source": "data.shape"}\n'
        '{"case_index": 1, "spec": "batch: 1; n: 16; cond: 1; seed: 2", "route": "qr16_fast", '
        '"cold_wall_us": 80.0, "cold_cuda_us": 10.0, '
        '"wall_us": 8.0, "cuda_us": 1.5, "repeats": 10, "warmup": 2, '
        '"case_metadata_passed_to_submission": false, '
        '"case_selection_info_sources": ["data.shape", "tensor_values"], '
        '"dispatch_info_sources": ["data.shape", "tensor_values"], '
        '"uses_tensor_values_for_dispatch": true, "uses_tensor_values_for_case_selection": true, '
        '"classifier_needed_for_current_candidate": true, "route_decision_source": "data.shape+tensor_values"}\n'
    )
    (tmp_path / "quantization_seed_sweep.jsonl").write_text(
        '{"experiment": "fp16-nearby", "ok": true, "margin_ok": true, '
        '"factor_scaled_max": 2.5, "orth_scaled_max": 1.5, "popcorn_seed": null, '
        '"diagnostics": {"factor_scaled_max": 2.5, "orth_scaled_max": 1.5}}\n'
        '{"experiment": "tf32-input-nearby", "ok": true, "margin_ok": true, '
        '"factor_scaled_max": 3.5, "orth_scaled_max": 1.25, "popcorn_seed": 1, '
        '"diagnostics": {"factor_scaled_max": 3.5, "orth_scaled_max": 1.25}}\n'
        '{"summary": true, "ok": true, "num_rows": 2, "num_failed": 0, "num_margin_failed": 0, '
        '"num_public_seed_rows": 1, "num_popcorn_seed_rows": 1, '
        '"max_factor_scaled": 3.5, "max_orth_scaled": 1.5}\n'
    )
    (tmp_path / "mixed_seed_sweep.jsonl").write_text(
        '{"case_source": "public_benchmarks", "case": "mixed", "batch": 640, "n": 512, '
        '"route": "qr512_mixed_fast", "route_ok": true, "ok": true, "margin_ok": true, '
        '"factor_scaled_max": 2.0, "orth_scaled_max": 1.0, "popcorn_seed": null, '
        '"diagnostics": {"factor_scaled_max": 2.0, "orth_scaled_max": 1.0}}\n'
        '{"case_source": "public_tests", "case": "mixed", "batch": 16, "n": 512, '
        '"route": "torch.geqrf", "route_ok": true, "ok": true, "margin_ok": true, '
        '"factor_scaled_max": 2.25, "orth_scaled_max": 1.25, "popcorn_seed": 1, '
        '"diagnostics": {"factor_scaled_max": 2.25, "orth_scaled_max": 1.25}}\n'
        '{"summary": true, "ok": true, "num_rows": 2, "num_failed": 0, "num_margin_failed": 0, '
        '"num_route_mismatch": 0, "num_public_seed_rows": 1, "num_popcorn_seed_rows": 1, '
        '"popcorn_seeds": ["1", "public"], "case_sources": ["public_benchmarks", "public_tests"], '
        '"shapes": ["16x512", "640x512"], "max_factor_scaled": 2.25, "max_orth_scaled": 1.25}\n'
    )
    (tmp_path / "blocked_qr_sweep.jsonl").write_text(
        '{"ok": true, "message": "", "spec": {"batch": 16, "n": 512, "cond": 2, "seed": 32530}, '
        '"case_index": 19, "case_text": "batch: 16; n: 512; cond: 2; seed: 32530", '
        '"batch": 16, "n": 512, "panel_width": 32, "update_mode": "compact-wy", '
        '"precision_mode": "fp32", "r_maintenance_mode": "none", "panel_refresh_mode": "none", '
        '"wall_us": 10.0, "factor_scaled_max": 0.2, "orth_scaled_max": 0.8}\n'
        '{"ok": false, "message": "factor mismatch", "spec": {"batch": 16, "n": 512, "cond": 2, "seed": 32530}, '
        '"case_index": 19, "case_text": "batch: 16; n: 512; cond: 2; seed: 32530", '
        '"batch": 16, "n": 512, "panel_width": 32, "update_mode": "compact-wy", '
        '"precision_mode": "tf32-input", "r_maintenance_mode": "none", "panel_refresh_mode": "none", '
        '"wall_us": 11.0, "factor_scaled_max": 596.0, "orth_scaled_max": 0.9}\n'
        '{"ok": true, "message": "", "spec": {"batch": 16, "n": 512, "cond": 2, "seed": 32530}, '
        '"case_index": 19, "case_text": "batch: 16; n: 512; cond: 2; seed: 32530", '
        '"batch": 16, "n": 512, "panel_width": 32, "update_mode": "compact-wy", '
        '"precision_mode": "tf32-input", "r_maintenance_mode": "panel-prefix", "panel_refresh_mode": "prefix", '
        '"wall_us": 12.0, "factor_scaled_max": 0.3, "orth_scaled_max": 0.7}\n'
        '{"summary": true, "ok": false, "num_rows": 3, "num_failed": 1, '
        '"panel_widths": [32], "update_modes": ["compact-wy"], "precision_modes": ["fp32", "tf32-input"], '
        '"r_maintenance_modes": ["none", "panel-prefix"], "panel_refresh_modes": ["none", "prefix"]}\n'
    )
    summary = summarize_suite(tmp_path)
    assert summary["ok"], summary
    public = summary["comparisons"][0]
    assert public["name"] == "public"
    assert public["num_common_cases"] == 2
    assert public["geomean_speedup"] == pytest.approx(2.828427, rel=1e-5)
    assert summary["guard_overhead"]["num_cases"] == 2
    assert summary["guard_overhead"]["cold_wall_geomean_us"] == pytest.approx(40.0)
    assert summary["guard_overhead"]["wall_geomean_us"] == pytest.approx(4.0)
    assert summary["guard_overhead"]["wall_max_us"] == pytest.approx(8.0)
    assert summary["guard_overhead"]["cases"][1]["uses_tensor_values_for_case_selection"] is True
    assert summary["ablations"][0]["name"] == "no_structured_routes"
    assert summary["ablations"][0]["ablation_over_default"] == pytest.approx(2.0)
    assert summary["runtime"]["total_elapsed_s"] == pytest.approx(45.0)
    assert summary["runtime"]["slowest_steps"][0]["step"] == "baseline_public"
    assert summary["quantization_seed_sweep"]["num_rows"] == 2
    assert summary["quantization_seed_sweep"]["num_margin_failed"] == 0
    assert summary["quantization_seed_sweep"]["max_factor_scaled"] == pytest.approx(3.5)
    assert summary["quantization_seed_sweep"]["by_experiment"][0]["experiment"] == "fp16-nearby"
    assert summary["mixed_seed_sweep"]["num_rows"] == 2
    assert summary["mixed_seed_sweep"]["num_route_mismatch"] == 0
    assert summary["mixed_seed_sweep"]["max_factor_scaled"] == pytest.approx(2.25)
    assert summary["blocked_qr_sweep"]["num_rows"] == 3
    assert summary["blocked_qr_sweep"]["num_failed"] == 1
    assert summary["blocked_qr_sweep"]["num_passing_low_precision_configs"] == 1
    assert summary["blocked_qr_sweep"]["passing_low_precision_configs"][0]["precision_mode"] == "tf32-input"
    assert summarize_blocked_qr_suite_sweep(tmp_path)["by_config"][1]["max_factor_scaled"] == pytest.approx(596.0)
    assert summary["tail_policy_tune"]["num_configs"] == 2
    assert summary["tail_policy_tune"]["num_failed_configs"] == 1
    assert summary["tail_policy_tune"]["num_benchmarked_configs"] == 1
    assert summary["tail_policy_tune"]["best_name"] == "cut16"
    assert summary["tail_policy_tune"]["best_geomean_us"] == pytest.approx(90.0)
    assert summary["candidate_config_tune"]["objective"] == "minimize_geomean_us"
    assert summary["candidate_config_tune"]["num_configs"] == 2
    assert summary["candidate_config_tune"]["num_failed_configs"] == 1
    assert summary["candidate_config_tune"]["num_benchmarked_configs"] == 1
    assert summary["candidate_config_tune"]["num_configs_with_inert_env"] == 0
    assert summary["candidate_config_tune"]["num_configs_with_cuda_route_bypassed_env"] == 1
    assert summary["candidate_config_tune"]["num_configs_with_resource_metrics"] == 1
    assert summary["candidate_config_tune"]["best_name"] == "qr512_panel32"
    assert summary["candidate_config_tune"]["best_geomean_us"] == pytest.approx(80.0)
    assert summary["candidate_config_tune"]["results"][0]["inert_env_keys"] == []
    assert summary["candidate_config_tune"]["results"][1]["cuda_route_bypassed_env_keys"] == [
        "FAST_QR_QR512_TAIL_CUT"
    ]
    assert summary["candidate_config_tune"]["results"][0]["resource_max_registers_per_thread"] == 48
    assert summary["candidate_config_tune"]["results"][0]["resource_min_estimated_occupancy"] == pytest.approx(0.5)
    markdown = render_summary_markdown(summary)
    assert "Runtime" in markdown
    assert "baseline_public" in markdown
    assert "geomean speedup" in markdown
    assert "batch: 1; n: 8" in markdown
    assert "Guard Overhead" in markdown
    assert "Quantization Seed Sweep" in markdown
    assert "fp16-nearby" in markdown
    assert "Mixed Seed Sweep" in markdown
    assert "640x512" in markdown
    assert "Blocked QR Sweep" in markdown
    assert "tf32-input" in markdown
    assert "panel-prefix" in markdown
    assert "Tail Policy Tune" in markdown
    assert "cut16" in markdown
    assert "Candidate Config Tune" in markdown
    assert "qr512_panel32" in markdown
    assert "FAST_QR_QR512_PANEL_B" in markdown
    assert "FAST_QR_QR512_TAIL_CUT" in markdown
    assert "regs/thread" in markdown
    assert "0.500" in markdown
    assert "Ablation: no_structured_routes" in markdown
    assert "qr16_fast" in markdown


def test_large_cuda_probe_ablation_uses_target_shapes_only(tmp_path):
    target_specs = [
        {"batch": 640, "n": 512, "cond": 2, "seed": 1029},
        {"batch": 640, "n": 512, "cond": 2, "seed": 770001, "case": "mixed"},
        {"batch": 60, "n": 1024, "cond": 2, "seed": 75342},
        {"batch": 60, "n": 1024, "cond": 0, "seed": 770005, "case": "nearrank"},
    ]
    unaffected = {"batch": 20, "n": 32, "cond": 1, "seed": 43214}
    _write_jsonl(
        tmp_path / "candidate_public.jsonl",
        [
            *[{"ok": True, "spec": spec, "mean_us": 100.0, "runs": 2} for spec in target_specs],
            {"ok": True, "spec": unaffected, "mean_us": 10.0, "runs": 2},
        ],
    )
    _write_jsonl(
        tmp_path / "candidate_ablation_no_qr512_qr1024_cuda_public.jsonl",
        [
            *[{"ok": True, "spec": spec, "mean_us": 80.0, "runs": 2} for spec in target_specs],
            {"ok": True, "spec": unaffected, "mean_us": 1000.0, "runs": 2},
        ],
    )

    result = analyze_large_cuda_probe_ablation(tmp_path)
    assert result is not None
    assert result["num_target_cases"] == 4
    assert result["ablation_over_default"] == pytest.approx(0.8)
    assert result["all_case_ablation_over_default"] > 1.0
    assert result["decision"] == "disable-qr512-qr1024-cuda"
    assert {f"{row['batch']}x{row['n']}" for row in result["families"]} == {"640x512", "60x1024"}


def test_suite_summary_extracts_candidate_config_route_order(tmp_path):
    config_tune_dir = tmp_path / "candidate_config_tune"
    config_tune_dir.mkdir()
    (config_tune_dir / "summary.json").write_text(
        json.dumps(
            {
                "ok": True,
                "hard_failed": False,
                "allow_failed_configs": True,
                "objective": "minimize_geomean_us",
                "num_configs": 2,
                "best": {"name": "structured_first", "benchmark": {"geomean_us": 90.0}},
                "results": [
                    {
                        "name": "cuda_first",
                        "env": {
                            "FAST_QR_QR512_WARPS_PER_CTA": "8",
                            "FAST_QR_QR512_STRUCTURED_BEFORE_CUDA": "0",
                        },
                        "env_consumption": {
                            "candidate_consumed_env_keys": [
                                "FAST_QR_QR512_WARPS_PER_CTA",
                                "FAST_QR_QR512_STRUCTURED_BEFORE_CUDA",
                            ],
                            "inert_env_keys": [],
                            "cuda_route_bypassed_env_keys": [],
                        },
                        "correctness": {"num_failed": 0, "max_factor_scaled": 2.0, "max_orth_scaled": 1.0},
                        "benchmark": {"num_cases": 4, "geomean_us": 100.0},
                    },
                    {
                        "name": "structured_first",
                        "env": {
                            "FAST_QR_QR512_WARPS_PER_CTA": "8",
                            "FAST_QR_QR512_STRUCTURED_BEFORE_CUDA": "1",
                        },
                        "env_consumption": {
                            "candidate_consumed_env_keys": [
                                "FAST_QR_QR512_WARPS_PER_CTA",
                                "FAST_QR_QR512_STRUCTURED_BEFORE_CUDA",
                            ],
                            "inert_env_keys": [],
                            "cuda_route_bypassed_env_keys": [],
                        },
                        "correctness": {"num_failed": 0, "max_factor_scaled": 2.0, "max_orth_scaled": 1.0},
                        "benchmark": {"num_cases": 4, "geomean_us": 90.0},
                    },
                ],
            }
        )
        + "\n"
    )

    summary = summarize_suite(tmp_path)
    route_order = summary["candidate_config_tune"]["route_order"]
    assert route_order["num_route_order_configs"] == 2
    assert route_order["num_compared_pairs"] == 1
    assert route_order["decision"] == "prefer-structured-first"
    assert route_order["structured_over_cuda"] == pytest.approx(0.9)
    assert route_order["pairs"][0]["comparison_env"] == {"FAST_QR_QR512_WARPS_PER_CTA": "8"}

    markdown = render_summary_markdown(summary)
    assert "route-order decision: prefer-structured-first" in markdown
    assert "structured_first" in markdown
    assert "cuda_first" in markdown


def test_analysis_recommends_route_order_from_candidate_config_tune(tmp_path):
    _write_jsonl(
        tmp_path / "baseline_geqrf_public.jsonl",
        [
            {"ok": True, "spec": {"batch": 640, "n": 512, "cond": 2, "seed": 1029}, "mean_us": 200.0},
            {"geomean_us": 200.0, "num_cases": 1},
        ],
    )
    _write_jsonl(
        tmp_path / "candidate_public.jsonl",
        [
            {"ok": True, "spec": {"batch": 640, "n": 512, "cond": 2, "seed": 1029}, "mean_us": 100.0},
            {"geomean_us": 100.0, "num_cases": 1},
        ],
    )
    config_tune_dir = tmp_path / "candidate_config_tune"
    config_tune_dir.mkdir()
    (config_tune_dir / "summary.json").write_text(
        json.dumps(
            {
                "ok": True,
                "hard_failed": False,
                "objective": "minimize_geomean_us",
                "best": {"name": "structured_first", "benchmark": {"geomean_us": 90.0}},
                "results": [
                    {
                        "name": "cuda_first",
                        "env": {
                            "FAST_QR_QR512_WARPS_PER_CTA": "8",
                            "FAST_QR_QR512_STRUCTURED_BEFORE_CUDA": "0",
                        },
                        "correctness": {"num_failed": 0},
                        "benchmark": {"num_cases": 4, "geomean_us": 100.0},
                    },
                    {
                        "name": "structured_first",
                        "env": {
                            "FAST_QR_QR512_WARPS_PER_CTA": "8",
                            "FAST_QR_QR512_STRUCTURED_BEFORE_CUDA": "1",
                        },
                        "correctness": {"num_failed": 0},
                        "benchmark": {"num_cases": 4, "geomean_us": 90.0},
                    },
                ],
            }
        )
        + "\n"
    )

    analysis = analyze_suite(tmp_path)
    route_order = analysis["candidate_config_tune"]["route_order"]
    assert route_order["decision"] == "prefer-structured-first"
    assert any(
        action["area"] == "route-order" and "prefer-structured-first" in action["action"]
        for action in analysis["final_algorithm_recommendation"]["priority_actions"]
    )

    markdown = render_analysis_markdown(analysis)
    assert "route-order decision: prefer-structured-first" in markdown
    assert "structured/cuda" in markdown


def test_analysis_recommends_blocked_qr_low_precision_refresh_path(tmp_path):
    spec = {"batch": 640, "n": 512, "cond": 2, "seed": 1029}
    _write_jsonl(
        tmp_path / "baseline_geqrf_public.jsonl",
        [
            {"ok": True, "spec": spec, "mean_us": 200.0},
            {"geomean_us": 200.0, "num_cases": 1},
        ],
    )
    _write_jsonl(
        tmp_path / "candidate_public.jsonl",
        [
            {"ok": True, "spec": spec, "mean_us": 100.0},
            {"geomean_us": 100.0, "num_cases": 1},
        ],
    )
    _write_jsonl(
        tmp_path / "blocked_qr_sweep.jsonl",
        [
            {
                "ok": False,
                "message": "factor mismatch",
                "spec": spec,
                "case_index": 3,
                "panel_width": 32,
                "update_mode": "compact-wy",
                "precision_mode": "tf32-input",
                "r_maintenance_mode": "none",
                "panel_refresh_mode": "none",
                "wall_us": 11.0,
                "factor_scaled_max": 596.0,
                "orth_scaled_max": 0.9,
            },
            {
                "ok": True,
                "message": "",
                "spec": spec,
                "case_index": 3,
                "panel_width": 32,
                "update_mode": "compact-wy",
                "precision_mode": "tf32-input",
                "r_maintenance_mode": "panel-prefix",
                "panel_refresh_mode": "prefix",
                "wall_us": 12.0,
                "factor_scaled_max": 0.3,
                "orth_scaled_max": 0.7,
            },
            {
                "summary": True,
                "ok": False,
                "num_rows": 2,
                "num_failed": 1,
                "panel_widths": [32],
                "update_modes": ["compact-wy"],
                "precision_modes": ["tf32-input"],
                "r_maintenance_modes": ["none", "panel-prefix"],
                "panel_refresh_modes": ["none", "prefix"],
            },
        ],
    )

    analysis = analyze_suite(tmp_path)
    assert analysis["blocked_qr_sweep"]["num_passing_low_precision_configs"] == 1
    assert any(
        action["area"] == "blocked-qr-low-precision"
        and "prefix panel refresh" in action["action"]
        for action in analysis["final_algorithm_recommendation"]["priority_actions"]
    )

    markdown = render_analysis_markdown(analysis)
    assert "Blocked QR Sweep" in markdown
    assert "tf32-input" in markdown
    assert "panel-prefix" in markdown


def _write_jsonl(path: Path, rows):
    path.write_text("".join(f"{row}\n" if isinstance(row, str) else json.dumps(row) + "\n" for row in rows))


def _write_local_checks_tarball(suite_dir: Path, tarball_path: Path):
    with tarfile.open(tarball_path, "w:gz") as tar:
        for file_name in LOCAL_REQUIRED_FILES:
            tar.add(suite_dir / file_name, arcname=f"{suite_dir.name}/{file_name}")


def _local_manifest_rows(tarball_path: Path, skip_pytest=False):
    rows = [
        {
            "event": "local_checks_start",
            "time": "2026-06-21T00:00:00",
            "repo": {"git_hash": "abc1234"},
            "skip_pytest": skip_pytest,
        }
    ]
    for step in LOCAL_EXPECTED_STEPS:
        if skip_pytest and step == "pytest":
            continue
        rows.append({"event": "start", "step": step, "time": "2026-06-21T00:00:00"})
        rows.append({"event": "finish", "step": step, "elapsed_s": 0.1, "time": "2026-06-21T00:00:01"})
    rows.append({"event": "local_checks_finish", "elapsed_s": 1.0, "tarball": str(tarball_path)})
    return rows


def _write_local_checks_suite(tmp_path: Path) -> Path:
    tarball_path = tmp_path.with_suffix(".tgz")
    _write_jsonl(tmp_path / "manifest.jsonl", _local_manifest_rows(tarball_path))
    (tmp_path / "run.log").write_text(
        "$ tools/audit_secrets.py\n$ tools/check_b200_env.py\n$ tools/validate_submission.py\n"
    )
    _write_jsonl(
        tmp_path / "secret_audit.jsonl",
        [{"summary": True, "ok": True, "files_scanned": 42, "num_findings": 0}],
    )
    _write_jsonl(
        tmp_path / "runtime_preflight.jsonl",
        [
            {
                "ok": False,
                "errors": ["torch.cuda.is_available() is false"],
                "torch": {"import_ok": True, "cuda_available": False, "device_count": 0, "devices": []},
            }
        ],
    )
    _write_jsonl(
        tmp_path / "submission_validation.jsonl",
        [{"event": "submission_validation", "ok": True, "errors": []}],
    )
    _write_jsonl(
        tmp_path / "candidate_policy_public.jsonl",
        [
            {
                "spec": f"case {idx}",
                "batch": 640 if idx == 3 else 1,
                "n": 512 if idx == 3 else 8,
                "dispatch": "fallback",
                "primary": "torch.geqrf",
                "shape_collision": idx == 0,
                "required_cuda_kernel": "qr512_blocked_householder_r_maintenance" if idx == 3 else None,
                "required_repair_modes": (
                    ["panel_refresh_mode=prefix", "r_maintenance_mode=panel-prefix"] if idx == 3 else []
                ),
                "candidate_config_shape_label": "qr512" if idx == 3 else None,
                "candidate_config_env_prefix": "FAST_QR_QR512" if idx == 3 else None,
                "candidate_config_benchmark_indices": "3,7,9,10" if idx == 3 else "",
                "candidate_config_correctness_indices": "3,6,7,8,9,10,11,19" if idx == 3 else "",
            }
            for idx in range(12)
        ],
    )
    _write_jsonl(
        tmp_path / "candidate_implementation_status.jsonl",
        [
            *[
                {
                    "case_index": idx,
                    "spec": f"case {idx}",
                    "dispatch": "fallback",
                    "primary": "torch.geqrf",
                    "implementation_kind": "torch_geqrf_fallback",
                    "readiness": "missing_custom_kernel",
                    "uses_torch_geqrf": True,
                    "has_custom_cuda": False,
                    "final_kernel_required": True,
                    "next_work": "write compact Householder CUDA path",
                }
                for idx in range(12)
            ],
            {
                "summary": True,
                "ok": True,
                "ready_for_final_submission": False,
                "num_cases": 12,
                "num_final_kernel_required": 12,
                "by_implementation_kind": {"torch_geqrf_fallback": 12},
                "by_readiness": {"missing_custom_kernel": 12},
                "next_priority_cases": [],
            },
        ],
    )
    (tmp_path / "b200_dry_run_plan.json").write_text(
        json.dumps(
            {
                "dry_run": True,
                "will_validate_suite": True,
                "will_validate_completed_export": True,
                "validation_blockers": [],
                "workload": {
                    "num_gpu_heavy_steps": 30,
                    "num_benchmark_timing_steps": 6,
                    "num_route_ablation_timing_steps": 6,
                    "num_verifier_experiment_steps": 11,
                    "runtime_estimate": {"high_minutes": 120.0},
                },
                "steps": [{"step": step} for step in EXPECTED_DEFAULT_STEPS],
            }
        )
        + "\n"
    )
    (tmp_path / "b200_next_required_dry_run_plan.json").write_text(
        json.dumps(
            {
                "dry_run": True,
                "will_validate_suite": True,
                "will_validate_completed_export": True,
                "validation_blockers": [],
                "candidate_config_tune_policy_target": {
                    "source": "candidate_policy",
                    "case_index": 3,
                    "spec": "case 3",
                    "shape_label": "qr512",
                    "env_prefix": "FAST_QR_QR512",
                    "benchmark_indices": "3,7,9,10",
                    "correctness_indices": "3,6,7,8,9,10,11,19",
                    "required_cuda_kernel": "qr512_blocked_householder_r_maintenance",
                    "required_repair_modes": ["panel_refresh_mode=prefix", "r_maintenance_mode=panel-prefix"],
                    "large_kernel_plan_mode": "current-candidate",
                    "effective_only": True,
                    "applied_axis_constraints": {
                        "panel_refresh_modes": "prefix",
                        "r_maintenance_modes": "panel-prefix",
                    },
                },
                "candidate_config_tune_large_kernel_plan": {
                    "path": str(tmp_path / "candidate_config_tune_large_kernel_configs.jsonl"),
                    "mode": "current-candidate",
                    "shape_label": "qr512",
                    "env_prefix": "FAST_QR_QR512",
                    "max_configs": 8,
                    "num_configs": 1,
                    "config_names": ["qr512_probe"],
                    "configs": [
                        {
                            "name": "qr512_probe",
                            "env": {
                                "FAST_QR_QR512_PANEL_B": "32",
                                "FAST_QR_QR512_PANEL_REFRESH_MODE": "prefix",
                                "FAST_QR_QR512_R_MAINTENANCE_MODE": "panel-prefix",
                                "FAST_QR_QR512_UPDATE_MODE": "reflectors",
                            },
                        }
                    ],
                },
                "steps": [
                    {
                        "step": "candidate_config_accelerator_preflight",
                        "cmd": ["python", "tools/preflight_accelerators.py", "--family-cases"],
                    },
                    {"step": "candidate_config_tune", "cmd": ["python", "tools/tune_candidate_configs.py"]},
                ],
            }
        )
        + "\n"
    )
    _write_local_checks_tarball(tmp_path, tarball_path)
    return tarball_path


def test_validate_local_checks_accepts_complete_non_cuda_export_and_flags_stale_tarball(tmp_path):
    tarball_path = _write_local_checks_suite(tmp_path)
    result = validate_local_checks(tmp_path)
    assert result["ok"], result

    with tarfile.open(tarball_path, "w:gz") as tar:
        stale = tmp_path / "stale_secret_audit.jsonl"
        stale.write_text('{"summary": true, "ok": true, "num_findings": 0}\n')
        tar.add(stale, arcname=f"{tmp_path.name}/secret_audit.jsonl")
        for file_name in LOCAL_REQUIRED_FILES:
            if file_name != "secret_audit.jsonl":
                tar.add(tmp_path / file_name, arcname=f"{tmp_path.name}/{file_name}")
    result = validate_local_checks(tmp_path)
    assert not result["ok"]
    assert any(
        error["check"] == "tarball.file" and error.get("file") == "secret_audit.jsonl"
        for error in result["errors"]
    )

    plan = json.loads((tmp_path / "b200_dry_run_plan.json").read_text())
    plan["steps"] = [row for row in plan["steps"] if row.get("step") != "candidate_official_style"]
    (tmp_path / "b200_dry_run_plan.json").write_text(json.dumps(plan) + "\n")
    _write_local_checks_tarball(tmp_path, tarball_path)
    result = validate_local_checks(tmp_path)
    assert not result["ok"]
    assert any(
        error["check"] == "b200_plan.steps" and "candidate_official_style" in error["message"]
        for error in result["errors"]
    )


def test_validate_local_checks_flags_bad_next_required_b200_plan(tmp_path):
    tarball_path = _write_local_checks_suite(tmp_path)
    result = validate_local_checks(tmp_path)
    assert result["ok"], result

    plan_path = tmp_path / "b200_next_required_dry_run_plan.json"
    valid_plan = json.loads(plan_path.read_text())

    missing_preview = dict(valid_plan)
    missing_preview.pop("candidate_config_tune_large_kernel_plan")
    plan_path.write_text(json.dumps(missing_preview) + "\n")
    _write_local_checks_tarball(tmp_path, tarball_path)
    result = validate_local_checks(tmp_path)
    assert not result["ok"]
    assert any(
        error["check"] == "b200_next_required_plan.generated_configs"
        and "generated config preview is missing" in error["message"]
        for error in result["errors"]
    )

    missing_family_cases = json.loads(json.dumps(valid_plan))
    for step in missing_family_cases["steps"]:
        if step.get("step") == "candidate_config_accelerator_preflight":
            step["cmd"] = [item for item in step["cmd"] if item != "--family-cases"]
    plan_path.write_text(json.dumps(missing_family_cases) + "\n")
    _write_local_checks_tarball(tmp_path, tarball_path)
    result = validate_local_checks(tmp_path)
    assert not result["ok"]
    assert any(
        error["check"] == "b200_next_required_plan.steps"
        and "--family-cases" in error["message"]
        for error in result["errors"]
    )

    repair_disabled = json.loads(json.dumps(valid_plan))
    repair_disabled["candidate_config_tune_large_kernel_plan"]["configs"][0]["env"][
        "FAST_QR_QR512_PANEL_REFRESH_MODE"
    ] = "none"
    plan_path.write_text(json.dumps(repair_disabled) + "\n")
    _write_local_checks_tarball(tmp_path, tarball_path)
    result = validate_local_checks(tmp_path)
    assert not result["ok"]
    assert any(
        error["check"] == "b200_next_required_plan.generated_configs"
        and "FAST_QR_QR512_PANEL_REFRESH_MODE=prefix" in error["message"]
        for error in result["errors"]
    )


def test_local_checks_next_required_summary_points_to_qr512_command(tmp_path):
    _write_local_checks_suite(tmp_path)

    summary = next_required_b200_summary(tmp_path)

    assert "next-required B200 target: qr512 / qr512_blocked_householder_r_maintenance" in summary
    assert "benchmark indices: 3,7,9,10" in summary
    assert "correctness indices: 3,6,7,8,9,10,11,19" in summary
    assert "generated configs: 1 (current-candidate)" in summary
    assert "next B200 command: python tools/run_b200_suite.py --suite-name" in summary
    assert f"--suite-name {tmp_path.name}_b200_next_required" in summary
    assert "--candidate-config-tune-next-required --candidate-config-tune-large-kernel-plan-max-configs 8" in summary


def test_local_checks_print_next_command_cli_reads_existing_export(tmp_path):
    _write_local_checks_suite(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "tools/run_local_checks.py",
            "--print-next-command",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert "next-required B200 target: qr512 / qr512_blocked_householder_r_maintenance" in completed.stdout
    assert "next B200 command: python tools/run_b200_suite.py --suite-name" in completed.stdout
    assert "--candidate-config-tune-next-required" in completed.stdout
    assert "Traceback" not in completed.stderr


def test_local_checks_print_next_command_refuses_invalid_export(tmp_path):
    tarball_path = _write_local_checks_suite(tmp_path)
    plan_path = tmp_path / "b200_next_required_dry_run_plan.json"
    plan = json.loads(plan_path.read_text())
    plan.pop("candidate_config_tune_large_kernel_plan")
    plan_path.write_text(json.dumps(plan) + "\n")
    _write_local_checks_tarball(tmp_path, tarball_path)

    completed = subprocess.run(
        [
            sys.executable,
            "tools/run_local_checks.py",
            "--print-next-command",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert completed.returncode == 1
    assert "cannot print next B200 command" in completed.stderr
    assert "b200_next_required_plan.generated_configs" in completed.stderr
    assert "next B200 command:" not in completed.stdout
    assert "Traceback" not in completed.stderr


def _write_suite_tarball(suite_dir: Path, tarball_path: Path):
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(suite_dir, arcname=suite_dir.name)


def _suite_provenance_row():
    return {
        "event": "suite_provenance",
        "repo": {
            "git_hash": "abc1234",
            "git_full_hash": "abc1234" * 6,
            "git_dirty": True,
            "git_status_porcelain": [" M submissions/candidate.py"],
            "official_upstream_commit": "e224fc20c430",
        },
        "submission": {"path": "submissions/candidate.py", "sha256": SYNTHETIC_CANDIDATE_SHA, "bytes": 100},
        "baseline": {"path": "submissions/baseline_geqrf.py", "sha256": SYNTHETIC_BASELINE_SHA, "bytes": 50},
        "args": {},
        "env": {key: None for key in TRACKED_RUNTIME_ENV_KEYS},
    }


def _suite_manifest_rows(
    tarball_path: Path,
    include_tail_policy_tune: bool = False,
    include_blocked_qr_sweep: bool = False,
    include_candidate_config_accelerator_preflight: bool = False,
    include_candidate_config_tune: bool = False,
    include_popcorn_test: bool = False,
    include_popcorn_leaderboard: bool = False,
):
    rows = [_suite_provenance_row()]
    for step in EXPECTED_DEFAULT_STEPS:
        rows.append({"event": "start", "step": step, "env_overrides": {}, "time": "2026-06-21T00:00:00"})
        rows.append(
            {
                "event": "finish",
                "step": step,
                "env_overrides": {},
                "elapsed_s": 0.1,
                "time": "2026-06-21T00:00:01",
            }
        )
        if include_tail_policy_tune and step == "tail_policy_sweep":
            rows.append(
                {"event": "start", "step": "tail_policy_tune", "env_overrides": {}, "time": "2026-06-21T00:00:00"}
            )
            rows.append(
                {
                    "event": "finish",
                    "step": "tail_policy_tune",
                    "env_overrides": {},
                    "elapsed_s": 0.1,
                    "time": "2026-06-21T00:00:01",
                }
            )
        if include_blocked_qr_sweep and step == "tail_policy_sweep":
            rows.append(
                {"event": "start", "step": "blocked_qr_sweep", "env_overrides": {}, "time": "2026-06-21T00:00:00"}
            )
            rows.append(
                {
                    "event": "finish",
                    "step": "blocked_qr_sweep",
                    "env_overrides": {},
                    "elapsed_s": 0.1,
                    "time": "2026-06-21T00:00:01",
                }
            )
        if include_candidate_config_accelerator_preflight and step == "tail_policy_sweep":
            rows.append(
                {
                    "event": "start",
                    "step": "candidate_config_accelerator_preflight",
                    "env_overrides": {},
                    "time": "2026-06-21T00:00:00",
                }
            )
            rows.append(
                {
                    "event": "finish",
                    "step": "candidate_config_accelerator_preflight",
                    "env_overrides": {},
                    "elapsed_s": 0.1,
                    "time": "2026-06-21T00:00:01",
                }
            )
        if include_candidate_config_tune and step == "tail_policy_sweep":
            rows.append(
                {
                    "event": "start",
                    "step": "candidate_config_tune",
                    "env_overrides": {},
                    "time": "2026-06-21T00:00:00",
                }
            )
            rows.append(
                {
                    "event": "finish",
                    "step": "candidate_config_tune",
                    "env_overrides": {},
                    "elapsed_s": 0.1,
                    "time": "2026-06-21T00:00:01",
                }
            )
        if include_popcorn_test and step == "accelerator_preflight":
            rows.append({"event": "start", "step": "popcorn_test", "env_overrides": {}, "time": "2026-06-21T00:00:00"})
            rows.append(
                {
                    "event": "finish",
                    "step": "popcorn_test",
                    "env_overrides": {},
                    "elapsed_s": 0.1,
                    "time": "2026-06-21T00:00:01",
                }
            )
        if include_popcorn_leaderboard and step == "candidate_official_style":
            rows.append({"event": "start", "step": "popcorn_leaderboard", "env_overrides": {}, "time": "2026-06-21T00:00:00"})
            rows.append(
                {
                    "event": "finish",
                    "step": "popcorn_leaderboard",
                    "env_overrides": {},
                    "elapsed_s": 0.1,
                    "time": "2026-06-21T00:00:01",
                }
            )
    rows.append({"event": "suite_finish", "elapsed_s": 1.0, "tarball": str(tarball_path)})
    return rows


def test_candidate_config_tune_validator_links_generated_repair_plan(tmp_path):
    plan_path = tmp_path / "candidate_config_tune_large_kernel_configs.jsonl"
    generated_rows = [
        {
            "name": "qr512_panel32",
            "env": {
                "FAST_QR_ENABLE_QR512_BLOCKED_CUDA": "1",
                "FAST_QR_QR512_PANEL_B": "32",
                "FAST_QR_QR512_PANEL_REFRESH_MODE": "prefix",
                "FAST_QR_QR512_R_MAINTENANCE_MODE": "panel-prefix",
            },
        },
        {
            "name": "qr512_panel64",
            "env": {
                "FAST_QR_ENABLE_QR512_BLOCKED_CUDA": "1",
                "FAST_QR_QR512_PANEL_B": "64",
                "FAST_QR_QR512_PANEL_REFRESH_MODE": "prefix",
                "FAST_QR_QR512_R_MAINTENANCE_MODE": "panel-prefix",
            },
        },
    ]
    _write_jsonl(plan_path, generated_rows)
    _write_jsonl(
        tmp_path / "manifest.jsonl",
        [
            _suite_provenance_row(),
            {
                "event": "candidate_config_tune_large_kernel_plan",
                "path": str(plan_path),
                "mode": "current-candidate",
                "shape_label": "qr512",
                "policy_target": {
                    "shape_label": "qr512",
                    "env_prefix": "FAST_QR_QR512",
                    "required_repair_modes": ["panel_refresh_mode=prefix", "r_maintenance_mode=panel-prefix"],
                    "applied_axis_constraints": {
                        "panel_refresh_modes": "prefix",
                        "r_maintenance_modes": "panel-prefix",
                    },
                },
                "num_configs": len(generated_rows),
                "config_names": [row["name"] for row in generated_rows],
            },
            {"event": "start", "step": "candidate_config_tune"},
            {"event": "finish", "step": "candidate_config_tune", "elapsed_s": 1.0},
        ],
    )

    tune_dir = tmp_path / "candidate_config_tune"
    tune_dir.mkdir()
    tune_summary = {
        "hard_failed": False,
        "num_configs": 2,
        "best": {"name": "qr512_panel32", "benchmark": {"geomean_us": 8.0}},
        "results": [
            {
                "name": "qr512_panel32",
                "env": dict(generated_rows[0]["env"]),
                "correctness": {"num_failed": 0},
                "benchmark": {"geomean_us": 8.0},
            },
            {
                "name": "qr512_panel64",
                "env": dict(generated_rows[1]["env"]),
                "correctness": {"num_failed": 1},
                "benchmark": None,
            },
        ],
    }
    (tune_dir / "summary.json").write_text(json.dumps(tune_summary) + "\n")

    errors: list[dict[str, str]] = []
    validate_candidate_config_tune(tmp_path, errors)
    assert errors == []

    stale_generated = [
        generated_rows[0],
        {
            **generated_rows[1],
            "env": {
                **generated_rows[1]["env"],
                "FAST_QR_QR512_PANEL_REFRESH_MODE": "none",
            },
        },
    ]
    _write_jsonl(plan_path, stale_generated)
    errors = []
    validate_candidate_config_tune(tmp_path, errors)
    assert any(error["check"] == "candidate_config_tune.generated_configs" for error in errors)
    assert any("FAST_QR_QR512_PANEL_REFRESH_MODE=prefix" in error["message"] for error in errors)

    _write_jsonl(plan_path, generated_rows)
    broken_summary = {
        **tune_summary,
        "results": [
            tune_summary["results"][0],
            {
                **tune_summary["results"][1],
                "env": {
                    **tune_summary["results"][1]["env"],
                    "FAST_QR_QR512_R_MAINTENANCE_MODE": "none",
                },
            },
        ],
    }
    (tune_dir / "summary.json").write_text(json.dumps(broken_summary) + "\n")
    errors = []
    validate_candidate_config_tune(tmp_path, errors)
    assert any(error["check"] == "candidate_config_tune.generated_configs" for error in errors)
    assert any("does not preserve generated env keys" in error["message"] for error in errors)


def _write_popcorn_artifacts(suite_dir: Path, artifact_dir: str, mode: str, exit_code: int = 0) -> None:
    popcorn_dir = suite_dir / artifact_dir
    popcorn_dir.mkdir(exist_ok=True)
    candidate_provenance = {"path": "submissions/candidate.py", "sha256": SYNTHETIC_CANDIDATE_SHA, "bytes": 100}
    staged_provenance = {
        "path": f"results/suite/{artifact_dir}/submission.py",
        "sha256": SYNTHETIC_CANDIDATE_SHA,
        "bytes": len(SYNTHETIC_POPCORN_SUBMISSION.encode()),
    }
    validation = {
        "event": "submission_validation",
        "ok": True,
        "static_ok": True,
        "import_ok": True,
        "errors": [],
        "source_submission": candidate_provenance,
        "staged_submission": staged_provenance,
    }
    (popcorn_dir / "submission.py").write_text(SYNTHETIC_POPCORN_SUBMISSION)
    _write_jsonl(popcorn_dir / "submission_validation.jsonl", [validation])
    (popcorn_dir / "popcorn.log").write_text(
        f"$ popcorn submit --leaderboard qr_v2 --gpu B200 --mode {mode} submission.py\n"
        f"exit_code={exit_code}; elapsed_s=0.1\n"
    )
    _write_jsonl(
        popcorn_dir / "manifest.jsonl",
        [
            {
                "event": "submit_start",
                "dry_run": False,
                "args": {"mode": mode, "gpu": "B200", "leaderboard": "qr_v2"},
                "source_submission": candidate_provenance,
                "staged_submission": staged_provenance,
                "validation": validation,
            },
            {
                "event": "mode_plan",
                "mode": mode,
                "cmd": ["popcorn", "submit", "--leaderboard", "qr_v2", "--gpu", "B200", "--mode", mode, "submission.py"],
            },
            {
                "event": "mode_finish",
                "mode": mode,
                "exit_code": exit_code,
                "error": None if exit_code == 0 else "failed",
                "elapsed_s": 0.1,
            },
            {"event": "submit_finish", "ok": exit_code == 0, "dry_run": False},
        ],
    )


def _write_popcorn_test_artifacts(suite_dir: Path, exit_code: int = 0) -> None:
    _write_popcorn_artifacts(suite_dir, "popcorn_test", "test", exit_code=exit_code)


def _write_popcorn_leaderboard_artifacts(suite_dir: Path, exit_code: int = 0) -> None:
    _write_popcorn_artifacts(suite_dir, "popcorn_leaderboard", "leaderboard", exit_code=exit_code)


def test_validate_b200_suite_accepts_complete_export_and_flags_missing_provenance(tmp_path):
    candidate_provenance = {
        "gpu": "NVIDIA B200",
        "driver": "570.124.06",
        "cuda": "12.8",
        "torch": "2.8.0",
        "git_hash": "abc1234",
        "git_full_hash": "abc1234" * 6,
        "git_dirty": True,
        "official_upstream_commit": "e224fc20c430",
        "submission": "submissions/candidate.py",
        "submission_sha256": SYNTHETIC_CANDIDATE_SHA,
    }
    baseline_provenance = {
        **candidate_provenance,
        "submission": "submissions/baseline_geqrf.py",
        "submission_sha256": SYNTHETIC_BASELINE_SHA,
    }
    specs = [{"batch": 1, "n": 8, "cond": 1, "seed": seed} for seed in range(12)]
    candidate_benchmark_rows = [
        {"ok": True, "spec": spec, "mean_us": 10.0 + idx, "runs": 3, **candidate_provenance}
        for idx, spec in enumerate(specs)
    ]
    candidate_benchmark_rows.append({"geomean_us": 15.0, "num_cases": 12})
    baseline_benchmark_rows = [
        {"ok": True, "spec": spec, "mean_us": 20.0 + idx, "runs": 3, **baseline_provenance}
        for idx, spec in enumerate(specs)
    ]
    baseline_benchmark_rows.append({"geomean_us": 25.0, "num_cases": 12})
    tarball_path = tmp_path.with_suffix(".tgz")

    _write_jsonl(
        tmp_path / "manifest.jsonl",
        _suite_manifest_rows(tarball_path),
    )
    (tmp_path / "run.log").write_text("$ /venv/bin/python -m pytest\nok\n")
    _write_jsonl(
        tmp_path / "secret_audit.jsonl",
        [
            {
                "summary": True,
                "ok": True,
                "files_considered": 42,
                "files_scanned": 42,
                "files_skipped": 0,
                "num_findings": 0,
                "rules": ["openai_api_key"],
            }
        ],
    )
    _write_jsonl(
        tmp_path / "runtime_preflight.jsonl",
        [
            {
                "ok": True,
                "require_name": "B200",
                "min_compute_major": 10,
                "min_memory_gib": 150.0,
                "selected_device": {
                    "index": 0,
                    "name": "NVIDIA B200",
                    "capability": [10, 0],
                    "total_memory_gib": 180.0,
                },
                "torch": {
                    "import_ok": True,
                    "torch": "2.8.0",
                    "torch_cuda": "12.8",
                    "cuda_available": True,
                    "device_count": 1,
                    "devices": [
                        {
                            "index": 0,
                            "name": "NVIDIA B200",
                            "capability": [10, 0],
                            "total_memory_gib": 180.0,
                        }
                    ],
                },
                "nvidia_smi": {"available": True, "gpus": [{"name": "NVIDIA B200", "driver": "570.124.06"}]},
                "errors": [],
            }
        ],
    )
    _write_jsonl(
        tmp_path / "submission_validation.jsonl",
        [
            {
                "event": "submission_validation",
                "ok": True,
                "static_ok": True,
                "import_ok": True,
                "errors": [],
                "warnings": [],
                "source_submission": {"path": "submissions/candidate.py", "sha256": SYNTHETIC_CANDIDATE_SHA, "bytes": 100},
                "staged_submission": {"path": "results/suite/submission_stage/submission.py", "sha256": SYNTHETIC_CANDIDATE_SHA, "bytes": 100},
            }
        ],
    )
    (tmp_path / "suite_summary.md").write_text("# summary\n\n## Runtime\n")
    (tmp_path / "suite_summary.json").write_text(
        json.dumps(
            {
                "ok": True,
                "runtime": {
                    "num_steps": len(EXPECTED_DEFAULT_STEPS),
                    "total_elapsed_s": 1.0,
                    "sum_step_elapsed_s": 0.1 * len(EXPECTED_DEFAULT_STEPS),
                    "failed": False,
                    "slowest_steps": [{"step": EXPECTED_DEFAULT_STEPS[0], "elapsed_s": 0.1, "env_overrides": {}}],
                },
                "comparisons": [{"name": "public"}, {"name": "official_style"}, {"name": "smoke"}],
                "ablations": [{}, {}, {}, {}, {}, {}],
            }
        )
        + "\n"
    )
    (tmp_path / "suite_analysis.md").write_text(
        "# analysis\n\n"
        "## Final Algorithm Recommendation\n\n"
        "## Shape Family Priorities\n\n"
        "## Data-Dependent Dispatch\n\n"
        "## Output H Layout\n"
    )
    (tmp_path / "suite_analysis.json").write_text(
        json.dumps(
            {
                "ok": True,
                "comparison": {"name": "official_style", "num_common_cases": 12},
                "final_algorithm_recommendation": {
                    "status": "ready-for-next-kernel-decision",
                    "primary_next_step": "prioritize 1x8: replace torch.geqrf fallback",
                    "classifier_required_by_api": False,
                    "classifier_required_for_case_specific_paths": True,
                    "priority_actions": [
                        {
                            "area": "shape-family",
                            "shape": "1x8",
                            "action": "replace torch.geqrf fallback",
                            "evidence": {"num_cases": 12},
                        }
                    ],
                },
                "shape_family_priorities": [
                    {
                        "batch": 1,
                        "n": 8,
                        "num_cases": 12,
                        "candidate_total_us": 186.0,
                        "candidate_geomean_us": 15.0,
                        "action": "replace torch.geqrf fallback",
                    }
                ],
                "data_dependent_dispatch": {
                    "case_metadata_passed_to_submission": False,
                    "num_shape_families": 1,
                    "families": [
                        {
                            "batch": 1,
                            "n": 8,
                            "num_cases": 2,
                            "shape_collision_cases": ["dense", "mixed"],
                            "case_metadata_passed_to_submission": False,
                            "case_info_source": "tensor_values",
                            "route_decision_sources": ["data.shape", "tensor_values"],
                            "uses_tensor_values_for_dispatch": True,
                            "uses_tensor_values_for_case_selection": True,
                            "routes": ["qr32_fast"],
                            "requires_tensor_guard_for_case_specific_path": True,
                            "classifier_needed_for_case_specific_path": True,
                            "classifier_needed_for_current_candidate": True,
                            "classifier_on_current_hot_path": True,
                            "default_geomean_us": 10.0,
                            "cuda_first_structured_geomean_us": 10.0,
                            "cuda_first_structured_over_default": 1.0,
                            "no_structured_geomean_us": 10.0,
                            "no_structured_over_default": 1.0,
                            "no_data_dependent_geomean_us": 10.0,
                            "no_data_dependent_over_default": 1.0,
                            "decision": "neutral-measure-again",
                            "classifier_decision": "neutral-measure-again",
                            "route_order_decision": "neutral-measure-again",
                            "data_dependent_decision": "neutral-measure-again",
                        }
                    ],
                },
                "large_cuda_probe_ablation": {
                    "name": "no_qr512_qr1024_cuda",
                    "ablation_file": "candidate_ablation_no_qr512_qr1024_cuda_public.jsonl",
                    "target_shapes": ["640x512", "60x1024"],
                    "num_target_cases": 7,
                    "default_geomean_us": 10.0,
                    "ablation_geomean_us": 10.0,
                    "ablation_over_default": 1.0,
                    "all_case_ablation_over_default": 1.0,
                    "decision": "neutral-measure-again",
                    "families": [
                        {
                            "batch": 640,
                            "n": 512,
                            "num_cases": 4,
                            "default_geomean_us": 10.0,
                            "ablation_geomean_us": 10.0,
                            "ablation_over_default": 1.0,
                            "decision": "neutral-measure-again",
                            "cases": [],
                        },
                        {
                            "batch": 60,
                            "n": 1024,
                            "num_cases": 3,
                            "default_geomean_us": 10.0,
                            "ablation_geomean_us": 10.0,
                            "ablation_over_default": 1.0,
                            "decision": "neutral-measure-again",
                            "cases": [],
                        },
                    ],
                },
                "output_layouts": {
                    "num_cases": 12,
                    "layout_counts": {"torch_contiguous": 12},
                    "num_column_major": 0,
                    "num_torch_contiguous": 12,
                    "num_policy_mismatch": 0,
                    "shape_families": [
                        {
                            "batch": 1,
                            "n": 8,
                            "num_cases": 12,
                            "layout_counts": {"torch_contiguous": 12},
                            "num_column_major": 0,
                            "num_torch_contiguous": 12,
                            "num_policy_mismatch": 0,
                            "cases": [format_case(spec) for spec in specs],
                        }
                    ],
                    "cases": [],
                },
            }
        )
        + "\n"
    )

    for name in [
        "baseline_geqrf_public.jsonl",
        "baseline_geqrf_official_style.jsonl",
        "baseline_geqrf_smoke.jsonl",
    ]:
        _write_jsonl(tmp_path / name, baseline_benchmark_rows)

    for name in [
        "candidate_public.jsonl",
        "candidate_official_style.jsonl",
        "candidate_smoke.jsonl",
        "candidate_ablation_no_route_cache_public.jsonl",
        "candidate_ablation_cuda_first_structured_routes_public.jsonl",
        "candidate_ablation_no_structured_routes_public.jsonl",
        "candidate_ablation_no_dense_tail_public.jsonl",
        "candidate_ablation_no_data_dependent_routes_public.jsonl",
        "candidate_ablation_no_qr512_qr1024_cuda_public.jsonl",
    ]:
        _write_jsonl(tmp_path / name, candidate_benchmark_rows)

    public_test_rows = [
        *[
            {
                "ok": True,
                "spec": {"batch": 1, "n": 8, "cond": 1, "seed": seed},
                "case_index": seed,
                **candidate_provenance,
            }
            for seed in range(22)
        ],
        {"summary": True, "ok": True, "num_cases": 22, "num_passed": 22, "num_failed": 0},
    ]
    _write_jsonl(tmp_path / "candidate_public_tests.jsonl", public_test_rows)

    public_benchmark_correctness_rows = [
        *[
            {
                "ok": True,
                "spec": spec,
                "case_index": idx,
                "case_text": format_case(spec),
                "diagnostics": {"factor_scaled_max": 2.0, "orth_scaled_max": 1.0},
                "h_shape": [1, 8, 8],
                "h_stride": [64, 8, 1],
                "h_is_contiguous": True,
                "column_major_h_actual": False,
                "h_layout_actual": "torch_contiguous",
                "margin_ok": True,
                "factor_margin_ok": True,
                "orth_margin_ok": True,
                **candidate_provenance,
            }
            for idx, spec in enumerate(specs)
        ],
        {"summary": True, "ok": True, "num_cases": 12, "num_passed": 12, "num_failed": 0},
    ]
    _write_jsonl(tmp_path / "candidate_public_benchmark_correctness.jsonl", public_benchmark_correctness_rows)

    dev_robustness_rows = [
        *[
            {
                "ok": True,
                "spec": {"batch": 1, "n": 8, "cond": 1, "seed": seed},
                "case_index": seed,
                "diagnostics": {"factor_scaled_max": 1.0, "orth_scaled_max": 1.0},
                "margin_ok": True,
                "factor_margin_ok": True,
                "orth_margin_ok": True,
                **candidate_provenance,
            }
            for seed in range(11)
        ],
        {"summary": True, "ok": True, "num_cases": 11, "num_passed": 11, "num_failed": 0},
    ]
    _write_jsonl(tmp_path / "candidate_dev_robustness.jsonl", dev_robustness_rows)
    preflight_cases_by_accelerator = {
        "qr32_cuda": ["smoke"],
        "qr176_cuda": ["smoke"],
        "qr352_cuda": ["smoke"],
        "qr512_cuda": ["dense", "mixed", "rankdef", "clustered"],
        "qr512_blocked_cuda": ["dense", "mixed", "rankdef", "clustered"],
        "qr1024_cuda": ["dense", "mixed", "nearrank"],
        "qr1024_blocked_cuda": ["dense", "mixed", "nearrank"],
        "qr2048_blocked_cuda": ["dense", "rankdef", "mixed"],
        "qr4096_blocked_cuda": ["dense", "upper"],
    }
    accelerator_preflight_rows = [
        {
            "accelerator": accelerator,
            "preflight_case": preflight_case,
            "family_cases": True,
            "ok": True,
            "extension_loaded": True,
        }
        for accelerator, preflight_cases in preflight_cases_by_accelerator.items()
        for preflight_case in preflight_cases
    ]
    _write_jsonl(
        tmp_path / "accelerator_preflight.jsonl",
        [
            *accelerator_preflight_rows,
            {
                "summary": True,
                "ok": True,
                "family_cases": True,
                "num_accelerators": len(preflight_cases_by_accelerator),
                "num_config_accelerator_rows": len(accelerator_preflight_rows),
                "num_preflight_case_rows": len(accelerator_preflight_rows),
                "num_passed": len(accelerator_preflight_rows),
                "num_failed": 0,
                "accelerators": list(preflight_cases_by_accelerator),
                "preflight_cases_by_accelerator": preflight_cases_by_accelerator,
            },
        ],
    )
    policy_rows_for_suite = [
        {
            "spec": format_case(spec),
            "dispatch": "qr32_fast",
            "primary": "torch.geqrf",
            "submission_entrypoint": "custom_kernel(data)",
            "case_metadata_available": False,
            "case_metadata_passed_to_submission": False,
            "case_info_source": "tensor_values" if idx < 2 else "data.shape",
            "case_selection_info_sources": ["data.shape", "tensor_values"] if idx < 2 else ["data.shape"],
            "dispatch_info_sources": ["data.shape", "tensor_values"] if idx < 2 else ["data.shape"],
            "shape_only_case_selection": idx >= 2,
            "shape_only_dispatch": idx >= 2,
            "uses_tensor_values_for_dispatch": idx < 2,
            "uses_tensor_values_for_case_selection": idx < 2,
            "requires_tensor_guard_for_case_specific_path": idx < 2,
            "classifier_needed_for_current_candidate": idx < 2,
            "classifier_needed_for_case_specific_path": idx < 2,
            "classifier_on_current_hot_path": idx < 2,
            "classifier_reason": "shape collision" if idx < 2 else "shape unique",
            "classifier_decision_rule": "compare ablation" if idx < 2 else "not_applicable_shape_unique",
            "column_major_h": False,
            "h_layout": "torch.geqrf_default",
            "shape_collision": idx < 2,
            "shape_collision_cases": ["dense", "mixed"] if idx < 2 else [],
            "requires_tensor_guard_for_case_specific_path": idx < 2,
            **candidate_provenance,
        }
        for idx, spec in enumerate(specs)
    ]
    _write_jsonl(tmp_path / "candidate_policy_public.jsonl", policy_rows_for_suite)
    implementation_rows = [
        {
            "case_index": idx,
            "spec": format_case(spec),
            "batch": int(spec["batch"]),
            "n": int(spec["n"]),
            "case": "dense",
            "dispatch": "qr32_fast",
            "primary": "torch.geqrf",
            "implementation_kind": "torch_geqrf_fallback",
            "readiness": "missing_custom_kernel",
            "priority": "normal",
            "uses_torch_geqrf": True,
            "has_custom_cuda": False,
            "final_kernel_required": True,
            "column_major_h": False,
            "h_layout": "torch.geqrf_default",
            "shape_collision": idx < 2,
            "uses_tensor_values_for_dispatch": idx < 2,
            "classifier_needed_for_current_candidate": idx < 2,
            "next_work": "write compact Householder CUDA path",
            **candidate_provenance,
        }
        for idx, spec in enumerate(specs)
    ]
    implementation_rows.append(
        {
            "summary": True,
            "ok": True,
            "ready_for_final_submission": False,
            "num_cases": 12,
            "num_final_kernel_required": 12,
            "num_custom_cuda_partial": 0,
            "num_torch_composite_experiment": 0,
            "num_torch_geqrf_fallback": 12,
            "by_implementation_kind": {"torch_geqrf_fallback": 12},
            "by_readiness": {"missing_custom_kernel": 12},
            "next_priority_cases": [
                {
                    "case_index": 0,
                    "spec": format_case(specs[0]),
                    "dispatch": "qr32_fast",
                    "readiness": "missing_custom_kernel",
                    "next_work": "write compact Householder CUDA path",
                }
            ],
            **candidate_provenance,
        }
    )
    _write_jsonl(tmp_path / "candidate_implementation_status.jsonl", implementation_rows)

    route_trace_rows = [
        {
            "spec": format_case(spec),
            "batch": int(spec["batch"]),
            "n": int(spec["n"]),
            "dispatch": "qr32_fast",
            "route": "qr32_fast",
            "case_metadata_available": False,
            "case_metadata_passed_to_submission": False,
            "case_info_source": "tensor_values" if idx < 2 else "data.shape",
            "case_selection_info_sources": ["data.shape", "tensor_values"] if idx < 2 else ["data.shape"],
            "shape_collision": idx < 2,
            "shape_only_case_selection": idx >= 2,
            "shape_only_dispatch": idx >= 2,
            "uses_tensor_values_for_dispatch": idx < 2,
            "uses_tensor_values_for_case_selection": idx < 2,
            "requires_tensor_guard_for_case_specific_path": idx < 2,
            "classifier_needed_for_current_candidate": idx < 2,
            "classifier_needed_for_case_specific_path": idx < 2,
            "classifier_on_current_hot_path": idx < 2,
            "dispatch_info_sources": ["data.shape", "tensor_values"] if idx < 2 else ["data.shape"],
            "route_decision_source": "data.shape+tensor_values" if idx < 2 else "data.shape",
            **candidate_provenance,
        }
        for idx, spec in enumerate(specs)
    ]
    _write_jsonl(tmp_path / "candidate_route_trace_public.jsonl", route_trace_rows)

    guard_rows = [
        {
            "case_index": idx,
            "spec": format_case(spec),
            "route": "qr32_fast",
            "cold_wall_us": 1.0,
            "wall_us": 0.1,
            "repeats": 20,
            "warmup": 3,
            "case_metadata_passed_to_submission": False,
            "case_selection_info_sources": ["data.shape", "tensor_values"] if idx < 2 else ["data.shape"],
            "dispatch_info_sources": ["data.shape", "tensor_values"] if idx < 2 else ["data.shape"],
            "uses_tensor_values_for_dispatch": idx < 2,
            "uses_tensor_values_for_case_selection": idx < 2,
            "classifier_needed_for_current_candidate": idx < 2,
            "classifier_needed_for_case_specific_path": idx < 2,
            "classifier_on_current_hot_path": idx < 2,
            "route_decision_source": "data.shape+tensor_values" if idx < 2 else "data.shape",
            **candidate_provenance,
        }
        for idx, spec in enumerate(specs)
    ]
    _write_jsonl(tmp_path / "candidate_guard_overhead_public.jsonl", guard_rows)
    seed_sweep_rows = [
        {"ok": True, "margin_ok": True, "case_index": 19, "popcorn_seed": None, **candidate_provenance},
        {"ok": True, "margin_ok": True, "case_index": 19, "popcorn_seed": 1, **candidate_provenance},
    ]
    _write_jsonl(tmp_path / "seed_sweep_margin.jsonl", seed_sweep_rows)

    quantization_specs = [
        (3, {"batch": 640, "n": 512, "cond": 2, "seed": 1029}, None),
        (4, {"batch": 60, "n": 1024, "cond": 2, "seed": 75342}, None),
        (6, {"batch": 2, "n": 4096, "cond": 1, "seed": 32412}, 1),
    ]
    quantization_rows = []
    for experiment in ["fp16-nearby", "tf32-input-nearby"]:
        for case_index, spec, popcorn_seed in quantization_specs:
            diagnostics = {
                "factor_scaled_max": 2.5,
                "orth_scaled_max": 1.5,
                "worst_factor_matrix": 0,
                "worst_orth_matrix": 0,
            }
            quantization_rows.append(
                {
                    "experiment": experiment,
                    "quantization": "fp16-input" if experiment == "fp16-nearby" else "tf32-input",
                    "ok": True,
                    "margin_ok": True,
                    "message": "",
                    "spec": spec,
                    "case_text": format_case(spec),
                    "case_index": case_index,
                    "popcorn_seed": popcorn_seed,
                    "batch": spec["batch"],
                    "n": spec["n"],
                    "wall_us": 10.0,
                    "diagnostics": diagnostics,
                    **diagnostics,
                    **candidate_provenance,
                }
            )
    quantization_rows.append(
        {
            "summary": True,
            "ok": True,
            "num_rows": 6,
            "num_failed": 0,
            "num_margin_failed": 0,
            "num_passed": 6,
            "num_public_seed_rows": 4,
            "num_popcorn_seed_rows": 2,
            "experiments": ["fp16-nearby", "tf32-input-nearby"],
            "popcorn_seeds": ["1", "public"],
            "max_factor_scaled": 2.5,
            "max_orth_scaled": 1.5,
        }
    )
    _write_jsonl(tmp_path / "quantization_seed_sweep.jsonl", quantization_rows)

    mixed_seed_specs = [
        ("public_benchmarks", 7, {"batch": 640, "n": 512, "cond": 2, "seed": 770001, "case": "mixed"}, None, "qr512_mixed_fast"),
        ("public_benchmarks", 8, {"batch": 60, "n": 1024, "cond": 2, "seed": 770002, "case": "mixed"}, None, "qr1024_mixed_fast"),
        ("public_tests", 19, {"batch": 16, "n": 512, "cond": 2, "seed": 32530, "case": "mixed"}, 1, "qr512_mixed_fast"),
        ("public_tests", 20, {"batch": 4, "n": 1024, "cond": 2, "seed": 4332, "case": "mixed"}, 1, "qr1024_mixed_fast"),
        ("public_tests", 21, {"batch": 2, "n": 2048, "cond": 2, "seed": 224468, "case": "mixed"}, 1, "torch.geqrf"),
    ]
    mixed_seed_rows = []
    for case_source, case_index, spec, popcorn_seed, route in mixed_seed_specs:
        diagnostics = {
            "factor_scaled_max": 2.75,
            "orth_scaled_max": 1.75,
            "worst_factor_matrix": 0,
            "worst_orth_matrix": 0,
        }
        mixed_seed_rows.append(
            {
                "ok": True,
                "margin_ok": True,
                "route_ok": True,
                "message": "",
                "case_source": case_source,
                "case_index": case_index,
                "popcorn_seed": popcorn_seed,
                "spec": spec,
                "case_text": format_case(spec),
                "batch": spec["batch"],
                "n": spec["n"],
                "case": "mixed",
                "route": route,
                "expected_route": route if case_source == "public_benchmarks" else None,
                "kernel_wall_us": 10.0,
                "diagnostics": diagnostics,
                **diagnostics,
                **candidate_provenance,
            }
        )
    mixed_seed_rows.append(
        {
            "summary": True,
            "ok": True,
            "num_rows": 5,
            "num_failed": 0,
            "num_margin_failed": 0,
            "num_route_mismatch": 0,
            "num_public_seed_rows": 2,
            "num_popcorn_seed_rows": 3,
            "case_sources": ["public_benchmarks", "public_tests"],
            "shapes": ["16x512", "2x2048", "4x1024", "60x1024", "640x512"],
            "popcorn_seeds": ["1", "public"],
            "max_factor_scaled": 2.75,
            "max_orth_scaled": 1.75,
        }
    )
    _write_jsonl(tmp_path / "mixed_seed_sweep.jsonl", mixed_seed_rows)

    classifier_sweep_rows = [
        {
            "ok": True,
            "case_index": 3,
            "popcorn_seed": None,
            "spec": {"batch": 640, "n": 512, "cond": 2, "seed": 1029},
            "batch": 640,
            "n": 512,
            "sampled_class": "dense",
            "expected_sampled_class": "dense",
            "classifier_ok": True,
            "route": "qr512_dense_fast",
            "expected_route": "qr512_dense_fast",
            "route_ok": True,
            "sampled_class_wall_us": 10.0,
            "route_wall_us": 12.0,
            **candidate_provenance,
        },
        {
            "ok": True,
            "case_index": 8,
            "popcorn_seed": None,
            "spec": {"batch": 60, "n": 1024, "cond": 2, "seed": 770002, "case": "mixed"},
            "batch": 60,
            "n": 1024,
            "sampled_class": "mixed",
            "expected_sampled_class": "mixed",
            "classifier_ok": True,
            "route": "qr1024_mixed_fast",
            "expected_route": "qr1024_mixed_fast",
            "route_ok": True,
            "sampled_class_wall_us": 9.0,
            "route_wall_us": 11.0,
            **candidate_provenance,
        },
        {
            "ok": True,
            "case_index": 9,
            "popcorn_seed": 1,
            "spec": {"batch": 640, "n": 512, "cond": 0, "seed": 770004, "case": "rankdef"},
            "batch": 640,
            "n": 512,
            "sampled_class": "rankdef",
            "expected_sampled_class": "rankdef",
            "classifier_ok": True,
            "route": "qr512_rankdef_fast",
            "expected_route": "qr512_rankdef_fast",
            "route_ok": True,
            "sampled_class_wall_us": 10.0,
            "route_wall_us": 12.0,
            **candidate_provenance,
        },
        {
            "ok": True,
            "case_index": 11,
            "popcorn_seed": 1,
            "spec": {"batch": 60, "n": 1024, "cond": 0, "seed": 770006, "case": "nearrank"},
            "batch": 60,
            "n": 1024,
            "sampled_class": "nearrank",
            "expected_sampled_class": "nearrank",
            "classifier_ok": True,
            "route": "qr1024_nearrank_fast",
            "expected_route": "qr1024_nearrank_fast",
            "route_ok": True,
            "sampled_class_wall_us": 9.0,
            "route_wall_us": 11.0,
            **candidate_provenance,
        },
        {
            "summary": True,
            "ok": True,
            "num_rows": 4,
            "num_failed": 0,
            "num_classifier_mismatch": 0,
            "num_route_mismatch": 0,
            "num_route_cuda_bypass": 0,
            "num_public_seed_rows": 2,
            "num_popcorn_seed_rows": 2,
            "popcorn_seeds": ["1", "public"],
        },
    ]
    _write_jsonl(tmp_path / "classifier_seed_sweep.jsonl", classifier_sweep_rows)

    tail_specs = [
        (3, {"batch": 640, "n": 512, "cond": 2, "seed": 1029}, "qr512_dense_fast", 32),
        (4, {"batch": 60, "n": 1024, "cond": 2, "seed": 75342}, "qr1024_dense_fast", 96),
        (5, {"batch": 8, "n": 2048, "cond": 1, "seed": 224466}, "qr2048_dense_fast", 64),
        (6, {"batch": 2, "n": 4096, "cond": 1, "seed": 32412}, "qr4096_dense_fast", 256),
    ]
    tail_policy_rows = []
    for popcorn_seed in (None, 1):
        for case_index, spec, route, cut in tail_specs:
            tail_policy_rows.append(
                {
                    "ok": True,
                    "spec": spec,
                    "case_index": case_index,
                    "case_text": format_case(spec),
                    "candidate_route": route,
                    "candidate_policy_cut": cut,
                    "tail_cut": cut,
                    "cut_source": "candidate",
                    "strategy": "candidate_custom_kernel",
                    "popcorn_seed": popcorn_seed,
                    "diagnostics": {"factor_scaled_max": 3.0, "orth_scaled_max": 1.0},
                    "margin_ok": True,
                    "factor_margin_ok": True,
                    "orth_margin_ok": True,
                    **candidate_provenance,
                }
            )
    tail_policy_rows.append(
        {"summary": True, "ok": True, "num_rows": len(tail_policy_rows), "num_passed": len(tail_policy_rows), "num_failed": 0}
    )
    _write_jsonl(tmp_path / "candidate_tail_policy_sweep.jsonl", tail_policy_rows)
    tune_dir = tmp_path / "tail_policy_tune"
    tune_dir.mkdir()
    tune_summary = {
        "ok": False,
        "hard_failed": False,
        "allow_failed_configs": True,
        "num_configs": 2,
        "best": {"name": "default", "benchmark": {"geomean_us": 9.0}},
        "results": [
            {
                "name": "default",
                "env": {},
                "correctness": {"num_failed": 0, "max_factor_scaled": 2.0, "max_orth_scaled": 1.0},
                "benchmark": {"num_cases": 12, "geomean_us": 9.0},
            },
            {
                "name": "bad_tail",
                "env": {"FAST_QR_DENSE_TAIL_CUT_512": "128"},
                "correctness": {"num_failed": 1, "max_factor_scaled": 24.0, "max_orth_scaled": 1.0},
                "benchmark": None,
            },
        ],
    }
    (tune_dir / "summary.json").write_text(json.dumps(tune_summary) + "\n")
    experiment_rows = [
        {"experiment": name, "ok": True, "spec": specs[0], **candidate_provenance}
        for name in ["r-projection", "fp16-nearby", "tf32-input-nearby", "tail-delete", "column-major", "classify"]
    ]
    _write_jsonl(tmp_path / "experiments_public_benchmarks.jsonl", experiment_rows)
    _write_suite_tarball(tmp_path, tarball_path)

    assert validate_suite(tmp_path)["ok"]
    final_readiness = validate_suite(tmp_path, require_final_kernels=True)
    assert not final_readiness["ok"]
    assert any(error["check"] == "implementation_status.final_readiness" for error in final_readiness["errors"])

    missing_env_rows = _suite_manifest_rows(tarball_path)
    del missing_env_rows[0]["env"]["FAST_QR_QR512_PANEL_B"]
    _write_jsonl(tmp_path / "manifest.jsonl", missing_env_rows)
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(
        error["check"] == "manifest.provenance" and "FAST_QR_QR512_PANEL_B" in error["message"]
        for error in result["errors"]
    )
    _write_jsonl(tmp_path / "manifest.jsonl", _suite_manifest_rows(tarball_path))

    _write_jsonl(
        tmp_path / "manifest.jsonl",
        _suite_manifest_rows(tarball_path, include_tail_policy_tune=True),
    )
    tune_report = {
        "num_configs": 2,
        "num_failed_configs": 1,
        "num_benchmarked_configs": 1,
        "best_name": "default",
        "best_geomean_us": 9.0,
        "results": [],
    }
    summary_doc = json.loads((tmp_path / "suite_summary.json").read_text())
    summary_doc["tail_policy_tune"] = {**tune_report, "hard_failed": False}
    (tmp_path / "suite_summary.json").write_text(json.dumps(summary_doc) + "\n")
    (tmp_path / "suite_summary.md").write_text("# summary\n\n## Runtime\n\n## Tail Policy Tune\n")
    analysis_doc = json.loads((tmp_path / "suite_analysis.json").read_text())
    analysis_doc["tail_policy_tune"] = tune_report
    (tmp_path / "suite_analysis.json").write_text(json.dumps(analysis_doc) + "\n")
    (tmp_path / "suite_analysis.md").write_text(
        "# analysis\n\n"
        "## Final Algorithm Recommendation\n\n"
        "## Shape Family Priorities\n\n"
        "## Data-Dependent Dispatch\n\n"
        "## Output H Layout\n\n"
        "## Tail Policy Tune\n"
    )
    _write_suite_tarball(tmp_path, tarball_path)
    assert validate_suite(tmp_path)["ok"]

    with tarfile.open(tarball_path, "w:gz") as tar:
        for file_name in REQUIRED_DEFAULT_FILES:
            tar.add(tmp_path / file_name, arcname=f"{tmp_path.name}/{file_name}")
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(
        error["check"] == "tarball.file" and error.get("file") == "tail_policy_tune/summary.json"
        for error in result["errors"]
    )

    hard_failed_tune = {**tune_summary, "hard_failed": True}
    (tune_dir / "summary.json").write_text(json.dumps(hard_failed_tune) + "\n")
    _write_suite_tarball(tmp_path, tarball_path)
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(error["check"] == "tail_policy_tune.hard_failed" for error in result["errors"])

    (tune_dir / "summary.json").write_text(json.dumps(tune_summary) + "\n")
    config_tune_dir = tmp_path / "candidate_config_tune"
    config_tune_dir.mkdir(exist_ok=True)
    config_tune_summary = {
        "ok": False,
        "hard_failed": False,
        "allow_failed_configs": True,
        "objective": "minimize_geomean_us",
        "num_configs": 2,
        "num_configs_with_inert_env": 0,
        "num_configs_with_cuda_route_bypassed_env": 1,
        "best": {"name": "qr512_panel32", "benchmark": {"geomean_us": 8.0}},
        "results": [
            {
                "name": "qr512_panel32",
                "env": {"FAST_QR_QR512_PANEL_B": "32"},
                "env_consumption": {
                    "candidate_consumed_env_keys": ["FAST_QR_QR512_PANEL_B"],
                    "inert_env_keys": [],
                },
                "correctness": {"num_failed": 0, "max_factor_scaled": 2.0, "max_orth_scaled": 1.0},
                "benchmark": {"num_cases": 4, "geomean_us": 8.0},
            },
            {
                "name": "qr512_panel64",
                "env": {"FAST_QR_QR512_PANEL_B": "64", "FAST_QR_QR512_TAIL_CUT": "24"},
                "env_consumption": {
                    "candidate_consumed_env_keys": ["FAST_QR_QR512_PANEL_B", "FAST_QR_QR512_TAIL_CUT"],
                    "inert_env_keys": [],
                    "cuda_route_bypassed_env_keys": ["FAST_QR_QR512_TAIL_CUT"],
                },
                "correctness": {"num_failed": 1, "max_factor_scaled": 24.0, "max_orth_scaled": 1.0},
                "benchmark": None,
            },
        ],
    }
    (config_tune_dir / "summary.json").write_text(json.dumps(config_tune_summary) + "\n")
    _write_jsonl(
        tmp_path / "manifest.jsonl",
        _suite_manifest_rows(tarball_path, include_tail_policy_tune=True, include_candidate_config_tune=True),
    )
    config_tune_report = {
        "objective": "minimize_geomean_us",
        "num_configs": 2,
        "num_failed_configs": 1,
        "num_benchmarked_configs": 1,
        "num_configs_with_inert_env": 0,
        "num_configs_with_cuda_route_bypassed_env": 1,
        "best_name": "qr512_panel32",
        "best_geomean_us": 8.0,
        "results": [
            {
                "name": "qr512_panel32",
                "inert_env_keys": [],
                "cuda_route_bypassed_env_keys": ["FAST_QR_QR512_TAIL_CUT"],
            }
        ],
    }
    summary_doc = json.loads((tmp_path / "suite_summary.json").read_text())
    summary_doc["candidate_config_tune"] = {**config_tune_report, "hard_failed": False}
    (tmp_path / "suite_summary.json").write_text(json.dumps(summary_doc) + "\n")
    (tmp_path / "suite_summary.md").write_text("# summary\n\n## Runtime\n\n## Tail Policy Tune\n\n## Candidate Config Tune\n")
    analysis_doc = json.loads((tmp_path / "suite_analysis.json").read_text())
    analysis_doc["candidate_config_tune"] = config_tune_report
    (tmp_path / "suite_analysis.json").write_text(json.dumps(analysis_doc) + "\n")
    (tmp_path / "suite_analysis.md").write_text(
        "# analysis\n\n"
        "## Final Algorithm Recommendation\n\n"
        "## Shape Family Priorities\n\n"
        "## Data-Dependent Dispatch\n\n"
        "## Output H Layout\n\n"
        "## Tail Policy Tune\n\n"
        "## Candidate Config Tune\n"
    )
    _write_suite_tarball(tmp_path, tarball_path)
    assert validate_suite(tmp_path)["ok"]

    blocked_sweep_rows = [
        {
            "ok": True,
            "message": "",
            "spec": specs[0],
            "case_index": 0,
            "case_text": format_case(specs[0]),
            "batch": specs[0]["batch"],
            "n": specs[0]["n"],
            "panel_width": 32,
            "update_mode": "compact-wy",
            "precision_mode": "fp32",
            "r_maintenance_mode": "none",
            "panel_refresh_mode": "none",
            "wall_us": 10.0,
            "factor_scaled_max": 0.2,
            "orth_scaled_max": 0.8,
        },
        {
            "ok": False,
            "message": "factor mismatch",
            "spec": specs[0],
            "case_index": 0,
            "case_text": format_case(specs[0]),
            "batch": specs[0]["batch"],
            "n": specs[0]["n"],
            "panel_width": 32,
            "update_mode": "compact-wy",
            "precision_mode": "tf32-input",
            "r_maintenance_mode": "none",
            "panel_refresh_mode": "none",
            "wall_us": 11.0,
            "factor_scaled_max": 596.0,
            "orth_scaled_max": 0.9,
        },
        {
            "ok": True,
            "message": "",
            "spec": specs[0],
            "case_index": 0,
            "case_text": format_case(specs[0]),
            "batch": specs[0]["batch"],
            "n": specs[0]["n"],
            "panel_width": 32,
            "update_mode": "compact-wy",
            "precision_mode": "tf32-input",
            "r_maintenance_mode": "panel-prefix",
            "panel_refresh_mode": "prefix",
            "wall_us": 12.0,
            "factor_scaled_max": 0.3,
            "orth_scaled_max": 0.7,
        },
        {
            "summary": True,
            "ok": False,
            "num_rows": 3,
            "num_failed": 1,
            "panel_widths": [32],
            "update_modes": ["compact-wy"],
            "precision_modes": ["fp32", "tf32-input"],
            "r_maintenance_modes": ["none", "panel-prefix"],
            "panel_refresh_modes": ["none", "prefix"],
        },
    ]
    _write_jsonl(tmp_path / "blocked_qr_sweep.jsonl", blocked_sweep_rows)
    blocked_report = {
        "file": "blocked_qr_sweep.jsonl",
        "ok": False,
        "num_rows": 3,
        "num_failed": 1,
        "panel_widths": [32],
        "update_modes": ["compact-wy"],
        "precision_modes": ["fp32", "tf32-input"],
        "r_maintenance_modes": ["none", "panel-prefix"],
        "panel_refresh_modes": ["none", "prefix"],
        "num_passing_low_precision_configs": 1,
        "passing_low_precision_configs": [
            {
                "precision_mode": "tf32-input",
                "r_maintenance_mode": "panel-prefix",
                "panel_refresh_mode": "prefix",
                "num_rows": 1,
                "num_failed": 0,
                "max_factor_scaled": 0.3,
                "max_orth_scaled": 0.7,
            }
        ],
        "by_config": [
            {
                "precision_mode": "fp32",
                "r_maintenance_mode": "none",
                "panel_refresh_mode": "none",
                "num_rows": 1,
                "num_failed": 0,
                "max_factor_scaled": 0.2,
                "max_orth_scaled": 0.8,
            },
            {
                "precision_mode": "tf32-input",
                "r_maintenance_mode": "none",
                "panel_refresh_mode": "none",
                "num_rows": 1,
                "num_failed": 1,
                "max_factor_scaled": 596.0,
                "max_orth_scaled": 0.9,
            },
            {
                "precision_mode": "tf32-input",
                "r_maintenance_mode": "panel-prefix",
                "panel_refresh_mode": "prefix",
                "num_rows": 1,
                "num_failed": 0,
                "max_factor_scaled": 0.3,
                "max_orth_scaled": 0.7,
            },
        ],
    }
    _write_jsonl(
        tmp_path / "manifest.jsonl",
        _suite_manifest_rows(
            tarball_path,
            include_tail_policy_tune=True,
            include_candidate_config_tune=True,
            include_blocked_qr_sweep=True,
        ),
    )
    summary_doc = json.loads((tmp_path / "suite_summary.json").read_text())
    summary_doc["blocked_qr_sweep"] = blocked_report
    (tmp_path / "suite_summary.json").write_text(json.dumps(summary_doc) + "\n")
    (tmp_path / "suite_summary.md").write_text(
        "# summary\n\n## Runtime\n\n## Tail Policy Tune\n\n## Candidate Config Tune\n\n## Blocked QR Sweep\n"
    )
    analysis_doc = json.loads((tmp_path / "suite_analysis.json").read_text())
    analysis_doc["blocked_qr_sweep"] = blocked_report
    analysis_doc["final_algorithm_recommendation"]["priority_actions"].append(
        {
            "area": "blocked-qr-low-precision",
            "action": "port prefix panel refresh plus panel-prefix R maintenance into QR512/QR1024 CUDA blocked updates",
            "evidence": {"passing_low_precision_configs": blocked_report["passing_low_precision_configs"]},
        }
    )
    (tmp_path / "suite_analysis.json").write_text(json.dumps(analysis_doc) + "\n")
    (tmp_path / "suite_analysis.md").write_text(
        "# analysis\n\n"
        "## Final Algorithm Recommendation\n\n"
        "## Shape Family Priorities\n\n"
        "## Data-Dependent Dispatch\n\n"
        "## Output H Layout\n\n"
        "## Tail Policy Tune\n\n"
        "## Candidate Config Tune\n\n"
        "## Blocked QR Sweep\n"
    )
    _write_suite_tarball(tmp_path, tarball_path)
    assert validate_suite(tmp_path)["ok"]

    (tmp_path / "blocked_qr_sweep.jsonl").unlink()
    _write_suite_tarball(tmp_path, tarball_path)
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(error["check"] == "blocked_qr_sweep.exists" for error in result["errors"])
    _write_jsonl(tmp_path / "blocked_qr_sweep.jsonl", blocked_sweep_rows)
    _write_suite_tarball(tmp_path, tarball_path)

    config_preflight_rows = [
        {
            "accelerator": "qr512_blocked_cuda_auto",
            "ok": True,
            "preflight_case": preflight_case,
            "family_cases": True,
            "config_index": 0,
            "config_name": "qr512__warps_per_cta_4",
            "config_env": {"FAST_QR_QR512_WARPS_PER_CTA": "4"},
            "threads_per_cta": 128,
        }
        for preflight_case in ["dense", "mixed", "rankdef", "clustered"]
    ]
    config_preflight_rows.append(
        {
            "summary": True,
            "ok": True,
            "family_cases": True,
            "preflight_cases_by_accelerator": {
                "qr512_blocked_cuda_auto": ["dense", "mixed", "rankdef", "clustered"]
            },
            "num_configs": 1,
            "num_config_accelerator_rows": 4,
            "num_preflight_case_rows": 4,
            "num_failed": 0,
        }
    )
    _write_jsonl(tmp_path / "candidate_config_accelerator_preflight.jsonl", config_preflight_rows)
    _write_jsonl(
        tmp_path / "manifest.jsonl",
        _suite_manifest_rows(
            tarball_path,
            include_tail_policy_tune=True,
            include_candidate_config_accelerator_preflight=True,
            include_candidate_config_tune=True,
        ),
    )
    _write_suite_tarball(tmp_path, tarball_path)
    assert validate_suite(tmp_path)["ok"]

    _write_jsonl(
        tmp_path / "candidate_config_accelerator_preflight.jsonl",
        [
            {**config_preflight_rows[0], "ok": False},
            {**config_preflight_rows[1], "ok": False, "num_failed": 1},
        ],
    )
    _write_suite_tarball(tmp_path, tarball_path)
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(error["check"] == "candidate_config_accelerator_preflight.summary" for error in result["errors"])
    assert any(error["check"] == "candidate_config_accelerator_preflight.ok" for error in result["errors"])

    missing_family_case_rows = [
        row
        for row in config_preflight_rows
        if row.get("summary") or row.get("preflight_case") != "clustered"
    ]
    missing_family_case_rows[-1] = {
        **missing_family_case_rows[-1],
        "preflight_cases_by_accelerator": {"qr512_blocked_cuda_auto": ["dense", "mixed", "rankdef"]},
        "num_config_accelerator_rows": 3,
        "num_preflight_case_rows": 3,
    }
    _write_jsonl(tmp_path / "candidate_config_accelerator_preflight.jsonl", missing_family_case_rows)
    _write_suite_tarball(tmp_path, tarball_path)
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(
        error["check"] == "candidate_config_accelerator_preflight.family_cases"
        and "clustered" in error["message"]
        for error in result["errors"]
    )

    _write_jsonl(tmp_path / "candidate_config_accelerator_preflight.jsonl", config_preflight_rows)

    with tarfile.open(tarball_path, "w:gz") as tar:
        for file_name in REQUIRED_DEFAULT_FILES:
            tar.add(tmp_path / file_name, arcname=f"{tmp_path.name}/{file_name}")
        tar.add(tune_dir / "summary.json", arcname=f"{tmp_path.name}/tail_policy_tune/summary.json")
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(
        error["check"] == "tarball.file" and error.get("file") == "candidate_config_tune/summary.json"
        for error in result["errors"]
    )

    hard_failed_config_tune = {**config_tune_summary, "hard_failed": True}
    (config_tune_dir / "summary.json").write_text(json.dumps(hard_failed_config_tune) + "\n")
    _write_suite_tarball(tmp_path, tarball_path)
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(error["check"] == "candidate_config_tune.hard_failed" for error in result["errors"])

    (config_tune_dir / "summary.json").write_text(json.dumps(config_tune_summary) + "\n")
    _write_popcorn_test_artifacts(tmp_path)
    _write_jsonl(
        tmp_path / "manifest.jsonl",
        _suite_manifest_rows(
            tarball_path,
            include_tail_policy_tune=True,
            include_candidate_config_tune=True,
            include_popcorn_test=True,
        ),
    )
    _write_suite_tarball(tmp_path, tarball_path)
    assert validate_suite(tmp_path)["ok"]

    _write_popcorn_test_artifacts(tmp_path, exit_code=1)
    _write_suite_tarball(tmp_path, tarball_path)
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(error["check"] == "popcorn_test.exit_code" for error in result["errors"])

    _write_popcorn_test_artifacts(tmp_path)
    _write_popcorn_leaderboard_artifacts(tmp_path)
    _write_jsonl(
        tmp_path / "manifest.jsonl",
        _suite_manifest_rows(
            tarball_path,
            include_tail_policy_tune=True,
            include_popcorn_test=True,
            include_popcorn_leaderboard=True,
        ),
    )
    _write_suite_tarball(tmp_path, tarball_path)
    assert validate_suite(tmp_path)["ok"]

    _write_popcorn_leaderboard_artifacts(tmp_path, exit_code=1)
    _write_suite_tarball(tmp_path, tarball_path)
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(error["check"] == "popcorn_leaderboard.exit_code" for error in result["errors"])

    _write_popcorn_test_artifacts(tmp_path)
    bad_popcorn_validation_rows = [
        {
            "event": "submission_validation",
            "ok": True,
            "static_ok": True,
            "import_ok": True,
            "errors": [],
            "source_submission": {"path": "submissions/candidate.py", "sha256": SYNTHETIC_BAD_SHA, "bytes": 100},
            "staged_submission": {
                "path": "results/suite/popcorn_test/submission.py",
                "sha256": SYNTHETIC_CANDIDATE_SHA,
                "bytes": len(SYNTHETIC_POPCORN_SUBMISSION.encode()),
            },
        }
    ]
    _write_jsonl(tmp_path / "popcorn_test/submission_validation.jsonl", bad_popcorn_validation_rows)
    _write_suite_tarball(tmp_path, tarball_path)
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(
        error["check"] == "popcorn_test.provenance"
        and error.get("file") == "popcorn_test/submission_validation.jsonl"
        and "suite submission sha256" in error["message"]
        for error in result["errors"]
    )

    _write_popcorn_test_artifacts(tmp_path)
    popcorn_manifest_rows = load_jsonl(tmp_path / "popcorn_test/manifest.jsonl")
    popcorn_manifest_rows[0]["source_submission"] = {
        "path": "submissions/candidate.py",
        "sha256": SYNTHETIC_BAD_SHA,
        "bytes": 100,
    }
    _write_jsonl(tmp_path / "popcorn_test/manifest.jsonl", popcorn_manifest_rows)
    _write_suite_tarball(tmp_path, tarball_path)
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(
        error["check"] == "popcorn_test.provenance"
        and error.get("file") == "popcorn_test/manifest.jsonl"
        and "source_submission.sha256" in error["message"]
        for error in result["errors"]
    )

    _write_popcorn_test_artifacts(tmp_path)
    (tmp_path / "popcorn_test/submission.py").write_text(
        "import torch\n\n"
        "def custom_kernel(data):\n"
        "    return torch.geqrf(data.contiguous())\n"
    )
    _write_suite_tarball(tmp_path, tarball_path)
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(
        error["check"] == "popcorn_test.provenance"
        and error.get("file") == "popcorn_test/submission.py"
        and "nested submission.py sha256" in error["message"]
        for error in result["errors"]
    )

    _write_popcorn_test_artifacts(tmp_path)

    _write_jsonl(tmp_path / "manifest.jsonl", _suite_manifest_rows(tarball_path))
    with tarfile.open(tarball_path, "w:gz") as tar:
        for file_name in REQUIRED_DEFAULT_FILES:
            if file_name == "candidate_public.jsonl":
                continue
            tar.add(tmp_path / file_name, arcname=f"{tmp_path.name}/{file_name}")
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(
        error["check"] == "tarball.file" and error.get("file") == "candidate_public.jsonl"
        for error in result["errors"]
    )
    _write_suite_tarball(tmp_path, tarball_path)

    with tarfile.open(tarball_path, "w:gz") as tar:
        stale_candidate = tmp_path / "stale_candidate_public.jsonl"
        stale_candidate.write_text('{"stale": true}\n')
        for file_name in REQUIRED_DEFAULT_FILES:
            source = stale_candidate if file_name == "candidate_public.jsonl" else tmp_path / file_name
            tar.add(source, arcname=f"{tmp_path.name}/{file_name}")
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(
        error["check"] == "tarball.file"
        and error.get("file") == "candidate_public.jsonl"
        and "content" in error["message"]
        for error in result["errors"]
    )
    _write_suite_tarball(tmp_path, tarball_path)

    analysis = analyze_suite(tmp_path)
    assert analysis["ok"], analysis
    assert analysis["comparison"]["name"] == "official_style"
    assert analysis["comparison"]["num_common_cases"] == 12
    assert len(analysis["weakest_speedup_cases"]) == 5
    assert analysis["shape_family_priorities"][0]["batch"] == 1
    assert analysis["shape_family_priorities"][0]["n"] == 8
    assert analysis["shape_family_priorities"][0]["num_cases"] == 12
    assert analysis["shape_family_priorities"][0]["classifier_needed"] is True
    assert analysis["shape_family_priorities"][0]["action"] == "replace torch.geqrf fallback"
    recommendation = analysis["final_algorithm_recommendation"]
    assert recommendation["status"] == "ready-for-next-kernel-decision"
    assert recommendation["classifier_required_by_api"] is False
    assert recommendation["classifier_required_for_case_specific_paths"] is True
    assert recommendation["primary_next_step"] == "prioritize 1x8: replace torch.geqrf fallback"
    assert recommendation["priority_actions"][0]["area"] == "shape-family"
    assert recommendation["dispatch_decisions"][0]["action"] == "neutral-measure-again"
    assert analysis["guard_overhead"]["hot_wall_pct_max"] == pytest.approx(1.0)
    assert analysis["guard_overhead"]["case_selection_num_cases"] == 2
    assert analysis["guard_overhead"]["case_selection_hot_wall_pct_max"] == pytest.approx(1.0)
    assert analysis["data_dependent_dispatch"]["case_metadata_passed_to_submission"] is False
    assert analysis["data_dependent_dispatch"]["num_shape_families"] == 1
    dispatch_family = analysis["data_dependent_dispatch"]["families"][0]
    assert dispatch_family["case_info_source"] == "tensor_values"
    assert dispatch_family["classifier_needed_for_current_candidate"] is True
    assert dispatch_family["cuda_first_structured_over_default"] == pytest.approx(1.0)
    assert dispatch_family["no_structured_over_default"] == pytest.approx(1.0)
    assert dispatch_family["no_data_dependent_over_default"] == pytest.approx(1.0)
    assert dispatch_family["decision"] == "neutral-measure-again"
    assert dispatch_family["classifier_decision"] == "neutral-measure-again"
    assert dispatch_family["route_order_decision"] == "neutral-measure-again"
    assert dispatch_family["data_dependent_decision"] == "neutral-measure-again"
    assert analysis["classifier_seed_sweep"]["ok"] is True
    assert analysis["classifier_seed_sweep"]["num_classifier_mismatch"] == 0
    assert analysis["ablation_decisions"][0]["decision"] == "neutral-within-noise"
    assert any(row["name"] == "no_qr512_qr1024_cuda" for row in analysis["ablation_decisions"])
    assert analysis["benchmark_correctness"]["max_factor_scaled"] == pytest.approx(2.0)
    assert analysis["benchmark_correctness"]["num_margin_failed"] == 0
    assert analysis["output_layouts"]["num_cases"] == 12
    assert analysis["output_layouts"]["layout_counts"] == {"torch_contiguous": 12}
    assert analysis["output_layouts"]["num_policy_mismatch"] == 0
    assert analysis["output_layouts"]["shape_families"][0]["num_torch_contiguous"] == 12
    assert analysis["quantization_seed_sweep"]["max_factor_scaled"] == pytest.approx(2.5)
    assert analysis["quantization_seed_sweep"]["num_margin_failed"] == 0
    assert analysis["mixed_seed_sweep"]["max_factor_scaled"] == pytest.approx(2.75)
    assert analysis["mixed_seed_sweep"]["num_margin_failed"] == 0
    assert analysis["tail_policy_sweep"]["max_factor_scaled"] == pytest.approx(3.0)
    assert analysis["tail_policy_sweep"]["num_margin_failed"] == 0
    assert analysis["tail_policy_tune"]["best_name"] == "default"
    assert analysis["tail_policy_tune"]["num_failed_configs"] == 1
    assert analysis["tail_policy_tune"]["best_geomean_us"] == pytest.approx(9.0)
    assert analysis["candidate_config_tune"]["best_name"] == "qr512_panel32"
    assert analysis["candidate_config_tune"]["objective"] == "minimize_geomean_us"
    assert analysis["candidate_config_tune"]["num_failed_configs"] == 1
    assert analysis["candidate_config_tune"]["num_configs_with_inert_env"] == 0
    assert analysis["candidate_config_tune"]["num_configs_with_cuda_route_bypassed_env"] == 1
    assert analysis["candidate_config_tune"]["best_geomean_us"] == pytest.approx(8.0)
    assert analysis["dev_robustness"]["max_factor_scaled"] == pytest.approx(1.0)
    assert analysis["dev_robustness"]["num_margin_failed"] == 0
    assert analysis["experiments"]["experiments"]["tail-delete"]["passed"] == 1
    analysis_markdown = render_analysis_markdown(analysis)
    assert "Weakest Speedups" in analysis_markdown
    assert "Final Algorithm Recommendation" in analysis_markdown
    assert "classifier required by API: False" in analysis_markdown
    assert "prioritize 1x8: replace torch.geqrf fallback" in analysis_markdown
    assert "Shape Family Priorities" in analysis_markdown
    assert "replace torch.geqrf fallback" in analysis_markdown
    assert "Ablation Decisions" in analysis_markdown
    assert "Guard Overhead" in analysis_markdown
    assert "Data-Dependent Dispatch" in analysis_markdown
    assert "Large CUDA Probe Ablation" in analysis_markdown
    assert "Output H Layout" in analysis_markdown
    assert "neutral-measure-again" in analysis_markdown
    assert "Public Benchmark Correctness" in analysis_markdown
    assert "Quantization Seed Sweep" in analysis_markdown
    assert "Mixed Seed Sweep" in analysis_markdown
    assert "Tail Policy Sweep" in analysis_markdown
    assert "Tail Policy Tune" in analysis_markdown
    assert "Candidate Config Tune" in analysis_markdown
    assert "default" in analysis_markdown
    assert "qr512_panel32" in analysis_markdown
    assert "FAST_QR_QR512_PANEL_B" in analysis_markdown
    assert "Dev Robustness" in analysis_markdown

    bad_rows = list(candidate_benchmark_rows)
    bad_first = dict(bad_rows[0])
    bad_first.pop("submission_sha256")
    bad_rows[0] = bad_first
    _write_jsonl(tmp_path / "candidate_public.jsonl", bad_rows)
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(error["check"] == "benchmark.provenance" for error in result["errors"])

    bad_rows = list(candidate_benchmark_rows)
    bad_rows[0] = {**bad_rows[0], "submission_sha256": SYNTHETIC_BAD_SHA}
    _write_jsonl(tmp_path / "candidate_public.jsonl", bad_rows)
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(
        error["check"] == "benchmark.provenance"
        and error.get("file") == "candidate_public.jsonl"
        and "suite submission sha256" in error["message"]
        for error in result["errors"]
    )

    bad_rows = list(baseline_benchmark_rows)
    bad_rows[0] = {**bad_rows[0], "submission_sha256": SYNTHETIC_CANDIDATE_SHA}
    _write_jsonl(tmp_path / "baseline_geqrf_public.jsonl", bad_rows)
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(
        error["check"] == "benchmark.provenance"
        and error.get("file") == "baseline_geqrf_public.jsonl"
        and "suite baseline sha256" in error["message"]
        for error in result["errors"]
    )

    _write_jsonl(tmp_path / "candidate_public.jsonl", candidate_benchmark_rows)
    _write_jsonl(tmp_path / "baseline_geqrf_public.jsonl", baseline_benchmark_rows)

    bad_validation_rows = [
        {
            "event": "submission_validation",
            "ok": True,
            "static_ok": True,
            "import_ok": True,
            "errors": [],
            "warnings": [],
            "source_submission": {"path": "submissions/candidate.py", "sha256": SYNTHETIC_BAD_SHA, "bytes": 100},
            "staged_submission": {"path": "results/suite/submission_stage/submission.py", "sha256": SYNTHETIC_CANDIDATE_SHA, "bytes": 100},
        }
    ]
    _write_jsonl(tmp_path / "submission_validation.jsonl", bad_validation_rows)
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(
        error["check"] == "submission_validation.provenance"
        and "suite submission sha256" in error["message"]
        for error in result["errors"]
    )

    _write_jsonl(
        tmp_path / "submission_validation.jsonl",
        [
            {
                "event": "submission_validation",
                "ok": True,
                "static_ok": True,
                "import_ok": True,
                "errors": [],
                "warnings": [],
                "source_submission": {"path": "submissions/candidate.py", "sha256": SYNTHETIC_CANDIDATE_SHA, "bytes": 100},
                "staged_submission": {"path": "results/suite/submission_stage/submission.py", "sha256": SYNTHETIC_CANDIDATE_SHA, "bytes": 100},
            }
        ],
    )

    bad_rows = list(public_benchmark_correctness_rows)
    bad_rows[0] = {**bad_rows[0], "submission_sha256": SYNTHETIC_BAD_SHA}
    _write_jsonl(tmp_path / "candidate_public_benchmark_correctness.jsonl", bad_rows)
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(
        error["check"] == "benchmark_correctness.provenance"
        and "suite submission sha256" in error["message"]
        for error in result["errors"]
    )
    _write_jsonl(tmp_path / "candidate_public_benchmark_correctness.jsonl", public_benchmark_correctness_rows)

    bad_rows = list(public_test_rows)
    bad_first = dict(bad_rows[0])
    bad_first.pop("submission_sha256")
    bad_rows[0] = bad_first
    _write_jsonl(tmp_path / "candidate_public_tests.jsonl", bad_rows)
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(error["check"] == "public_tests.provenance" for error in result["errors"])
    _write_jsonl(tmp_path / "candidate_public_tests.jsonl", public_test_rows)

    bad_rows = list(seed_sweep_rows)
    bad_rows[0] = {**bad_rows[0], "submission_sha256": SYNTHETIC_BAD_SHA}
    _write_jsonl(tmp_path / "seed_sweep_margin.jsonl", bad_rows)
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(
        error["check"] == "seed_sweep.provenance" and "suite submission sha256" in error["message"]
        for error in result["errors"]
    )
    _write_jsonl(tmp_path / "seed_sweep_margin.jsonl", seed_sweep_rows)

    bad_rows = list(quantization_rows)
    bad_rows[0] = {**bad_rows[0], "submission_sha256": SYNTHETIC_BAD_SHA}
    _write_jsonl(tmp_path / "quantization_seed_sweep.jsonl", bad_rows)
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(
        error["check"] == "quantization_seed_sweep.provenance" and "suite submission sha256" in error["message"]
        for error in result["errors"]
    )
    _write_jsonl(tmp_path / "quantization_seed_sweep.jsonl", quantization_rows)

    bad_rows = list(mixed_seed_rows)
    bad_rows[0] = {**bad_rows[0], "submission_sha256": SYNTHETIC_BAD_SHA}
    _write_jsonl(tmp_path / "mixed_seed_sweep.jsonl", bad_rows)
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(
        error["check"] == "mixed_seed_sweep.provenance" and "suite submission sha256" in error["message"]
        for error in result["errors"]
    )
    _write_jsonl(tmp_path / "mixed_seed_sweep.jsonl", mixed_seed_rows)

    bad_rows = list(classifier_sweep_rows)
    bad_rows[0] = {**bad_rows[0], "submission_sha256": SYNTHETIC_BAD_SHA}
    _write_jsonl(tmp_path / "classifier_seed_sweep.jsonl", bad_rows)
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(
        error["check"] == "classifier_seed_sweep.provenance" and "suite submission sha256" in error["message"]
        for error in result["errors"]
    )
    _write_jsonl(tmp_path / "classifier_seed_sweep.jsonl", classifier_sweep_rows)

    missing_large_tail_shape_rows = [
        row
        for row in tail_policy_rows
        if row.get("summary") or not (isinstance(row.get("spec"), dict) and row["spec"].get("n") == 4096)
    ]
    _write_jsonl(tmp_path / "candidate_tail_policy_sweep.jsonl", missing_large_tail_shape_rows)
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(error["check"] == "tail_policy.coverage" and "2x4096" in error["message"] for error in result["errors"])
    _write_jsonl(tmp_path / "candidate_tail_policy_sweep.jsonl", tail_policy_rows)

    bad_rows = list(tail_policy_rows)
    bad_rows[0] = {**bad_rows[0], "submission_sha256": SYNTHETIC_BAD_SHA}
    _write_jsonl(tmp_path / "candidate_tail_policy_sweep.jsonl", bad_rows)
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(
        error["check"] == "tail_policy.provenance" and "suite submission sha256" in error["message"]
        for error in result["errors"]
    )
    _write_jsonl(tmp_path / "candidate_tail_policy_sweep.jsonl", tail_policy_rows)

    bad_rows = list(policy_rows_for_suite)
    bad_rows[0] = {**bad_rows[0], "submission_sha256": SYNTHETIC_BAD_SHA}
    _write_jsonl(tmp_path / "candidate_policy_public.jsonl", bad_rows)
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(
        error["check"] == "policy.provenance" and "suite submission sha256" in error["message"]
        for error in result["errors"]
    )
    _write_jsonl(tmp_path / "candidate_policy_public.jsonl", policy_rows_for_suite)

    bad_rows = list(route_trace_rows)
    bad_rows[0] = {**bad_rows[0], "submission_sha256": SYNTHETIC_BAD_SHA}
    _write_jsonl(tmp_path / "candidate_route_trace_public.jsonl", bad_rows)
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(
        error["check"] == "route_trace.provenance" and "suite submission sha256" in error["message"]
        for error in result["errors"]
    )
    _write_jsonl(tmp_path / "candidate_route_trace_public.jsonl", route_trace_rows)

    bad_rows = list(guard_rows)
    bad_rows[0] = {**bad_rows[0], "submission_sha256": SYNTHETIC_BAD_SHA}
    _write_jsonl(tmp_path / "candidate_guard_overhead_public.jsonl", bad_rows)
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(
        error["check"] == "guard.provenance" and "suite submission sha256" in error["message"]
        for error in result["errors"]
    )
    _write_jsonl(tmp_path / "candidate_guard_overhead_public.jsonl", guard_rows)

    bad_rows = list(experiment_rows)
    bad_rows[0] = {**bad_rows[0], "submission_sha256": SYNTHETIC_BAD_SHA}
    _write_jsonl(tmp_path / "experiments_public_benchmarks.jsonl", bad_rows)
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(
        error["check"] == "experiments.provenance" and "suite submission sha256" in error["message"]
        for error in result["errors"]
    )
    _write_jsonl(tmp_path / "experiments_public_benchmarks.jsonl", experiment_rows)

    analysis_doc = json.loads((tmp_path / "suite_analysis.json").read_text())
    analysis_doc["data_dependent_dispatch"]["families"][0]["decision"] = "insufficient-data"
    (tmp_path / "suite_analysis.json").write_text(json.dumps(analysis_doc) + "\n")
    _write_suite_tarball(tmp_path, tarball_path)
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(
        error["check"] == "analysis.data_dependent_dispatch"
        and "insufficient-data" in error["message"]
        for error in result["errors"]
    )

    analysis_doc["data_dependent_dispatch"]["families"][0]["decision"] = "neutral-measure-again"
    (tmp_path / "suite_analysis.json").write_text(json.dumps(analysis_doc) + "\n")
    _write_suite_tarball(tmp_path, tarball_path)

    analysis_doc["output_layouts"]["num_policy_mismatch"] = 1
    (tmp_path / "suite_analysis.json").write_text(json.dumps(analysis_doc) + "\n")
    _write_suite_tarball(tmp_path, tarball_path)
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(
        error["check"] == "analysis.output_layouts" and "mismatches" in error["message"]
        for error in result["errors"]
    )

    analysis_doc["output_layouts"]["num_policy_mismatch"] = 0
    (tmp_path / "suite_analysis.json").write_text(json.dumps(analysis_doc) + "\n")
    _write_suite_tarball(tmp_path, tarball_path)

    _write_jsonl(
        tmp_path / "seed_sweep_margin.jsonl",
        [{"ok": True, "margin_ok": True, "case_index": 19, "popcorn_seed": 1}],
    )
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(error["check"] == "seed_sweep.coverage" for error in result["errors"])

    _write_jsonl(
        tmp_path / "manifest.jsonl",
        [*_suite_manifest_rows(tarball_path), {"event": "finish", "step": "late_step"}],
    )
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(error["check"] == "manifest.status" and "final" in error["message"] for error in result["errors"])

    _write_jsonl(
        tmp_path / "manifest.jsonl",
        _suite_manifest_rows(tarball_path),
    )
    with tarfile.open(tarball_path, "w:gz") as tar:
        stale_manifest = tmp_path / "stale_manifest.jsonl"
        _write_jsonl(stale_manifest, [{"event": "suite_provenance"}])
        tar.add(stale_manifest, arcname=f"{tmp_path.name}/manifest.jsonl")
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(error["check"] == "manifest.tarball" and "suite_finish" in error["message"] for error in result["errors"])

    current_rows = _suite_manifest_rows(tarball_path)
    _write_jsonl(tmp_path / "manifest.jsonl", current_rows)
    stale_completed_manifest = tmp_path / "stale_completed_manifest.jsonl"
    stale_rows = list(current_rows)
    stale_rows[-1] = {**stale_rows[-1], "elapsed_s": 999.0}
    _write_jsonl(stale_completed_manifest, stale_rows)
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(stale_completed_manifest, arcname=f"{tmp_path.name}/manifest.jsonl")
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(error["check"] == "manifest.tarball" and "does not match" in error["message"] for error in result["errors"])

    manifest_rows = _suite_manifest_rows(tarball_path)
    missing_finish = next(
        index
        for index, row in enumerate(manifest_rows)
        if row.get("event") == "finish" and row.get("step") == "candidate_public"
    )
    del manifest_rows[missing_finish]
    _write_jsonl(tmp_path / "manifest.jsonl", manifest_rows)
    _write_suite_tarball(tmp_path, tarball_path)
    result = validate_suite(tmp_path)
    assert not result["ok"]
    assert any(
        error["check"] == "manifest.steps" and "candidate_public" in error["message"]
        for error in result["errors"]
    )


def test_route_trace_structured_group_counts_on_synthetic_profiles():
    candidate = _load_candidate_module()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(123)
    data = torch.randn((4, 16, 16), generator=gen)
    rank = candidate._rankdef_effective_cols(16)
    clustered_cols = candidate._clustered_effective_cols(16)
    data[0, :, rank:] = 0.0
    data[1, :, clustered_cols:] *= 1.0e-7
    scales = torch.logspace(0.0, -2.0, 16, dtype=torch.float32)
    data[2, :, rank:] = data[2, :, : 16 - rank] * (scales[rank:] / scales[: 16 - rank]).view(1, -1)

    counts = structured_group_counts(candidate, data, cond=2)
    assert counts == {
        "rankdef": 1,
        "clustered": 1,
        "scaled_nearrank": 1,
        "tiny_dense_tail": 0,
        "fallback": 1,
    }


def test_route_trace_exports_mixed_plan_sampling_counters():
    candidate = _load_candidate_module()
    row = trace_route(candidate, {"batch": 640, "n": 512, "cond": 2, "seed": 770001, "case": "mixed"})

    assert row["route"] == "qr512_mixed_fast"
    assert row["structured_sampled_plan"] is True
    assert row["structured_trusted_sampled_guards"] is False
    assert row["structured_sampled_matrix_count"] == 640
    assert row["structured_sampled_row_count"] <= 32
    assert row["structured_candidate_counts"]["rankdef"] > 0
    assert row["structured_candidate_counts"]["clustered"] > 0
    assert row["structured_candidate_counts"]["scaled_nearrank"] > 0
    assert row["structured_exact_check_counts"] == row["structured_candidate_counts"]
    assert max(row["structured_exact_check_counts"].values()) < 640 // 4


def test_route_trace_marks_dense_fallthrough_classifier_hot_path():
    candidate = _load_candidate_module()

    row512 = trace_route(candidate, {"batch": 640, "n": 512, "cond": 2, "seed": 1029})
    assert row512["sampled_class"] == "dense"
    assert row512["route"] == "qr512_dense_fast"
    assert row512["classifier_on_current_hot_path"] is True
    assert row512["classifier_needed_for_current_candidate"] is True
    assert row512["uses_tensor_values_for_dispatch"] is True
    assert row512["route_decision_source"] == "data.shape+tensor_values"
    assert row512["cuda_route_bypasses_classifier"] is False

    row1024 = trace_route(candidate, {"batch": 60, "n": 1024, "cond": 2, "seed": 75342})
    assert row1024["sampled_class"] == "dense"
    assert row1024["route"] == "qr1024_dense_fast"
    assert row1024["classifier_on_current_hot_path"] is True
    assert row1024["classifier_needed_for_current_candidate"] is True
    assert row1024["uses_tensor_values_for_dispatch"] is True
    assert row1024["route_decision_source"] == "data.shape+tensor_values"
    assert row1024["cuda_route_bypasses_classifier"] is False


def test_structured_before_cuda_changes_qr512_route_order(monkeypatch):
    candidate = _load_candidate_module()
    fake_data = SimpleNamespace(shape=(640, 512, 512))

    class TrueLike:
        def all(self):
            return self

        def item(self):
            return True

    monkeypatch.delenv("FAST_QR_QR512_STRUCTURED_BEFORE_CUDA", raising=False)
    monkeypatch.delenv("FAST_QR_STRUCTURED_ROUTES_BEFORE_CUDA", raising=False)
    monkeypatch.setenv("FAST_QR_DENSE_TAIL_CUT_512", "0")
    monkeypatch.setattr(candidate, "_qr512_cuda_route_enabled", lambda _data: True)
    monkeypatch.setattr(candidate, "classify_512_sampled", lambda _data: "rankdef")
    monkeypatch.setattr(candidate, "_batch_tail_columns_are_exact_zero", lambda _data, _rank: TrueLike())

    assert candidate._compute_route_plan(fake_data)[0] == "qr512_cuda_fast"

    monkeypatch.setenv("FAST_QR_QR512_STRUCTURED_BEFORE_CUDA", "1")
    assert candidate._compute_route_plan(fake_data)[0] == "qr512_rankdef_fast"

    monkeypatch.setattr(candidate, "_qr512_cuda_try", lambda _data: "cuda")
    monkeypatch.setattr(candidate, "_rankdef_assumed", lambda _data: "rankdef")
    assert candidate._dispatch_route("qr512_rankdef_fast", fake_data) == "rankdef"

    monkeypatch.setenv("FAST_QR_QR512_STRUCTURED_BEFORE_CUDA", "0")
    assert candidate._dispatch_route("qr512_rankdef_fast", fake_data) == "cuda"


def test_blocked_cuda_first_skips_classifier_when_structured_before_cuda_is_off(monkeypatch):
    candidate = _load_candidate_module()
    fake512 = SimpleNamespace(shape=(640, 512, 512))
    fake1024 = SimpleNamespace(shape=(60, 1024, 1024))

    def fail_classifier(_data):
        raise AssertionError("classifier should not run for blocked CUDA-first dispatch")

    monkeypatch.delenv("FAST_QR_QR512_STRUCTURED_BEFORE_CUDA", raising=False)
    monkeypatch.delenv("FAST_QR_QR1024_STRUCTURED_BEFORE_CUDA", raising=False)
    monkeypatch.delenv("FAST_QR_STRUCTURED_ROUTES_BEFORE_CUDA", raising=False)
    monkeypatch.setenv("FAST_QR_DENSE_TAIL_CUT_512", "0")
    monkeypatch.setenv("FAST_QR_DENSE_TAIL_CUT_1024", "0")
    monkeypatch.setattr(candidate, "_qr512_blocked_cuda_route_enabled", lambda _data: True)
    monkeypatch.setattr(candidate, "_qr1024_blocked_cuda_route_enabled", lambda _data: True)
    monkeypatch.setattr(candidate, "_qr512_cuda_route_enabled", lambda _data: True)
    monkeypatch.setattr(candidate, "_qr1024_cuda_route_enabled", lambda _data: True)
    monkeypatch.setattr(candidate, "classify_512_sampled", fail_classifier)
    monkeypatch.setattr(candidate, "classify_1024_sampled", fail_classifier)

    assert candidate._compute_route_plan(fake512) == ("qr512_blocked_cuda_fast", None)
    assert candidate._compute_route_plan(fake1024) == ("qr1024_blocked_cuda_fast", None)


def test_structured_before_cuda_preempts_blocked_auto_for_homogeneous_structured_cases(monkeypatch):
    candidate = _load_candidate_module()
    fake512 = SimpleNamespace(shape=(640, 512, 512))
    fake1024 = SimpleNamespace(shape=(60, 1024, 1024))

    def fail_mixed_plan(*_args, **_kwargs):
        raise AssertionError("homogeneous structured routes should not build a mixed plan")

    monkeypatch.setenv("FAST_QR_QR512_STRUCTURED_BEFORE_CUDA", "1")
    monkeypatch.setenv("FAST_QR_QR1024_STRUCTURED_BEFORE_CUDA", "1")
    monkeypatch.setenv("FAST_QR_DENSE_TAIL_CUT_512", "0")
    monkeypatch.setenv("FAST_QR_DENSE_TAIL_CUT_1024", "0")
    monkeypatch.setattr(candidate, "_qr512_blocked_cuda_route_enabled", lambda _data: True)
    monkeypatch.setattr(candidate, "_qr1024_blocked_cuda_route_enabled", lambda _data: True)
    monkeypatch.setattr(candidate, "_qr512_cuda_route_enabled", lambda _data: False)
    monkeypatch.setattr(candidate, "_qr1024_cuda_route_enabled", lambda _data: False)
    monkeypatch.setattr(candidate, "_blocked_auto_policy_enabled", lambda _data, _n: True)
    monkeypatch.setattr(candidate, "_trust_sampled_structured_guards", lambda _data: True)
    monkeypatch.setattr(candidate, "classify_512_sampled", lambda _data: "rankdef")
    monkeypatch.setattr(candidate, "classify_1024_sampled", lambda _data: "nearrank")
    monkeypatch.setattr(candidate, "_mixed_structured_plan", fail_mixed_plan)

    assert candidate._compute_route_plan(fake512) == ("qr512_rankdef_fast", None)
    assert candidate._compute_route_plan(fake1024) == ("qr1024_nearrank_fast", None)

    monkeypatch.setenv("FAST_QR_QR512_STRUCTURED_BEFORE_CUDA", "0")
    monkeypatch.setenv("FAST_QR_QR1024_STRUCTURED_BEFORE_CUDA", "0")
    monkeypatch.setattr(candidate, "classify_512_sampled", lambda _data: (_ for _ in ()).throw(AssertionError("classifier should be skipped")))
    monkeypatch.setattr(candidate, "classify_1024_sampled", lambda _data: (_ for _ in ()).throw(AssertionError("classifier should be skipped")))

    assert candidate._compute_route_plan(fake512) == ("qr512_blocked_cuda_auto_fast", None)
    assert candidate._compute_route_plan(fake1024) == ("qr1024_blocked_cuda_auto_fast", None)


def test_b200_default_promotes_blocked_cuda_routes(monkeypatch):
    candidate = _load_candidate_module()
    candidate._B200_DEVICE_CACHE.clear()
    props = SimpleNamespace(name="NVIDIA B200", major=10, total_memory=180 * 1024**3)

    monkeypatch.setattr(candidate.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(candidate.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(candidate.torch.cuda, "get_device_properties", lambda _index: props)
    monkeypatch.delenv("FAST_QR_DISABLE_B200_DEFAULT_BLOCKED_CUDA", raising=False)

    fakes = {
        512: SimpleNamespace(is_cuda=True, dtype=torch.float32, ndim=3, shape=(640, 512, 512), device=torch.device("cuda", 0)),
        1024: SimpleNamespace(is_cuda=True, dtype=torch.float32, ndim=3, shape=(60, 1024, 1024), device=torch.device("cuda", 0)),
        2048: SimpleNamespace(is_cuda=True, dtype=torch.float32, ndim=3, shape=(8, 2048, 2048), device=torch.device("cuda", 0)),
        4096: SimpleNamespace(is_cuda=True, dtype=torch.float32, ndim=3, shape=(2, 4096, 4096), device=torch.device("cuda", 0)),
    }

    assert candidate._qr512_blocked_cuda_route_enabled(fakes[512])
    assert candidate._qr1024_blocked_cuda_route_enabled(fakes[1024])
    assert candidate._qr2048_blocked_cuda_route_enabled(fakes[2048])
    assert candidate._qr4096_blocked_cuda_route_enabled(fakes[4096])
    assert candidate._qr32_cuda_warps_per_cta() == 8
    assert candidate._qr32_cuda_threads_per_cta() == 256
    assert candidate._qr176_cuda_update_col_tile() == 16
    assert candidate._qr512_blocked_cuda_loader_state(fakes[512])[1] is True
    assert candidate._qr1024_blocked_cuda_loader_state(fakes[1024])[1] is True
    assert candidate._generic_blocked_cuda_loader_state(2048, fakes[2048])[1] is True
    assert candidate._qr512_blocked_cuda_panel_refresh_mode() == "prefix"
    assert candidate._qr512_blocked_cuda_r_maintenance_mode() == "panel-prefix"
    assert candidate._qr512_blocked_cuda_update_mode() == "compact-wy"
    assert candidate._qr352_cuda_update_mode() == "compact-wy"
    assert candidate._qr512_blocked_cuda_tile_n() == 128
    assert candidate._qr512_blocked_cuda_ctas_per_matrix() == 2
    assert candidate._qr512_blocked_cuda_cta_schedule() == "frontload"
    assert candidate._qr512_blocked_cuda_sync_free_auto_policy()
    assert candidate._qr1024_blocked_cuda_panel_refresh_mode() == "prefix"
    assert candidate._qr1024_blocked_cuda_update_mode() == "compact-wy"
    assert candidate._qr1024_blocked_cuda_ctas_per_matrix() == 2
    assert candidate._qr1024_blocked_cuda_cta_schedule() == "frontload"
    assert candidate._qr1024_blocked_cuda_sync_free_auto_policy()
    assert candidate._qr352_cuda_update_col_tile() == 16
    assert candidate._qr352_cuda_panel_b() == 64
    assert candidate._generic_blocked_cuda_panel_refresh_mode(2048) == "prefix"
    assert candidate._generic_blocked_cuda_r_maintenance_mode(4096) == "panel-prefix"
    assert candidate._generic_blocked_cuda_update_mode(2048) == "compact-wy"
    assert candidate._generic_blocked_cuda_ctas_per_matrix(2048) == 8
    assert candidate._generic_blocked_cuda_cta_schedule(2048) == "all-tiles"
    assert candidate._generic_blocked_cuda_ctas_per_matrix(4096) == 16
    assert candidate._generic_blocked_cuda_cta_schedule(4096) == "all-tiles"
    assert candidate._generic_blocked_cuda_sync_free_auto_policy(2048)
    assert candidate._blocked_auto_policy_grouping_enabled(fakes[512], 512)
    assert candidate._blocked_auto_policy_grouping_enabled(fakes[1024], 1024)
    assert candidate._blocked_auto_policy_grouping_enabled(fakes[2048], 2048)
    assert candidate._blocked_auto_policy_grouping_enabled(fakes[4096], 4096)
    assert "constexpr int QR32_WARPS_PER_CTA = 8;" in candidate._qr32_cuda_source()
    assert "__global__ void __launch_bounds__(256) geqrf32_kernel(" in candidate._qr32_cuda_source()
    assert "constexpr int UPDATE_COL_TILE = 16;" in candidate._qr176_cuda_source()
    assert "constexpr int UPDATE_COL_TILE = 16;" in candidate._qr352_cuda_source()
    assert "constexpr int PANEL_B = 64;" in candidate._qr352_cuda_source()
    assert "constexpr int SYNC_FREE_AUTO_POLICY = 1;" in candidate._qr512_blocked_cuda_source()
    assert "constexpr int USE_COMPACT_WY_UPDATE = 1;" in candidate._qr352_cuda_source()
    assert "constexpr int USE_COMPACT_WY_UPDATE = 1;" in candidate._qr512_blocked_cuda_source()
    assert "constexpr int TILE_N = 128;" in candidate._qr512_blocked_cuda_source()
    assert "constexpr int CTAS_PER_MATRIX = 2;" in candidate._qr512_blocked_cuda_source()
    assert "constexpr int CTA_SCHEDULE_FRONTLOAD = 1;" in candidate._qr512_blocked_cuda_source()
    assert "constexpr int USE_COMPACT_WY_UPDATE = 1;" in candidate._qr1024_blocked_cuda_source()
    assert "constexpr int CTAS_PER_MATRIX = 2;" in candidate._qr1024_blocked_cuda_source()
    assert "constexpr int CTA_SCHEDULE_FRONTLOAD = 1;" in candidate._qr1024_blocked_cuda_source()
    assert "constexpr int USE_COMPACT_WY_UPDATE = 1;" in candidate._generic_blocked_cuda_source(2048)
    assert "constexpr int CTAS_PER_MATRIX = 8;" in candidate._generic_blocked_cuda_source(2048)
    assert "constexpr int CTA_SCHEDULE_ALL_TILES = 1;" in candidate._generic_blocked_cuda_source(2048)
    assert "constexpr int CTAS_PER_MATRIX = 16;" in candidate._generic_blocked_cuda_source(4096)
    assert "constexpr int CTA_SCHEDULE_ALL_TILES = 1;" in candidate._generic_blocked_cuda_source(4096)
    assert "constexpr int SYNC_FREE_AUTO_POLICY = 1;" in candidate._generic_blocked_cuda_source(4096)

    monkeypatch.setenv("FAST_QR_QR352_UPDATE_MODE", "reflectors")
    assert candidate._qr352_cuda_update_mode() == "reflectors"
    assert "constexpr int USE_COMPACT_WY_UPDATE = 0;" in candidate._qr352_cuda_source()

    monkeypatch.setenv("FAST_QR_QR352_UPDATE_COL_TILE", "8")
    assert candidate._qr352_cuda_update_col_tile() == 8
    assert "constexpr int UPDATE_COL_TILE = 8;" in candidate._qr352_cuda_source()

    monkeypatch.setenv("FAST_QR_QR176_UPDATE_COL_TILE", "8")
    assert candidate._qr176_cuda_update_col_tile() == 8
    assert "constexpr int UPDATE_COL_TILE = 8;" in candidate._qr176_cuda_source()

    for n in fakes:
        monkeypatch.setenv(f"FAST_QR_DENSE_TAIL_CUT_{n}", "0")
    monkeypatch.setenv("FAST_QR_DISABLE_STRUCTURED_ROUTES", "1")

    assert candidate._compute_route_plan(fakes[512]) == ("qr512_blocked_cuda_auto_fast", None)
    assert candidate._compute_route_plan(fakes[1024]) == ("qr1024_blocked_cuda_auto_fast", None)
    assert candidate._compute_route_plan(fakes[2048]) == ("qr2048_blocked_cuda_auto_fast", None)
    assert candidate._compute_route_plan(fakes[4096]) == ("qr4096_blocked_cuda_auto_fast", None)

    monkeypatch.setenv("FAST_QR_DISABLE_B200_DEFAULT_BLOCKED_CUDA", "1")
    candidate._B200_DEVICE_CACHE.clear()
    assert not candidate._qr512_blocked_cuda_route_enabled(fakes[512])
    assert candidate._qr512_blocked_cuda_panel_refresh_mode() == "none"
    assert candidate._qr512_blocked_cuda_r_maintenance_mode() == "none"
    assert candidate._qr512_blocked_cuda_update_mode() == "reflectors"
    assert candidate._qr512_blocked_cuda_ctas_per_matrix() == 1
    assert candidate._qr512_blocked_cuda_cta_schedule() == "fixed"
    assert candidate._qr1024_blocked_cuda_ctas_per_matrix() == 1
    assert candidate._qr1024_blocked_cuda_cta_schedule() == "fixed"
    assert candidate._generic_blocked_cuda_ctas_per_matrix(2048) == 4
    assert candidate._generic_blocked_cuda_cta_schedule(2048) == "frontload"
    assert candidate._generic_blocked_cuda_ctas_per_matrix(4096) == 8
    assert candidate._generic_blocked_cuda_cta_schedule(4096) == "frontload"
    assert not candidate._qr512_blocked_cuda_sync_free_auto_policy()
    assert not candidate._blocked_auto_policy_grouping_enabled(fakes[2048], 2048)

    monkeypatch.delenv("FAST_QR_DISABLE_B200_DEFAULT_BLOCKED_CUDA")
    monkeypatch.setenv("FAST_QR_ENABLE_BLOCKED_SYNC_FREE_AUTO_POLICY", "1")
    assert candidate._qr512_blocked_cuda_sync_free_auto_policy()
    monkeypatch.delenv("FAST_QR_ENABLE_BLOCKED_SYNC_FREE_AUTO_POLICY")
    monkeypatch.setenv("FAST_QR_QR512_BLOCKED_SYNC_FREE_AUTO_POLICY", "0")
    assert not candidate._qr512_blocked_cuda_sync_free_auto_policy()
    monkeypatch.delenv("FAST_QR_QR512_BLOCKED_SYNC_FREE_AUTO_POLICY")
    monkeypatch.setenv("FAST_QR_QR512_BLOCKED_AUTO_GROUPS", "0")
    assert not candidate._blocked_auto_policy_grouping_enabled(fakes[512], 512)
    monkeypatch.setenv("FAST_QR_QR512_BLOCKED_AUTO_GROUPS", "1")
    assert candidate._blocked_auto_policy_grouping_enabled(fakes[512], 512)
    monkeypatch.delenv("FAST_QR_QR512_BLOCKED_AUTO_GROUPS")
    monkeypatch.setenv("FAST_QR_BLOCKED_AUTO_GROUPS", "0")
    assert not candidate._blocked_auto_policy_grouping_enabled(fakes[2048], 2048)
    monkeypatch.delenv("FAST_QR_BLOCKED_AUTO_GROUPS")
    monkeypatch.setenv("FAST_QR_ENABLE_QR512_BLOCKED_CUDA", "0")
    candidate._B200_DEVICE_CACHE.clear()
    assert not candidate._qr512_blocked_cuda_route_enabled(fakes[512])

    monkeypatch.setenv("FAST_QR_ENABLE_QR512_BLOCKED_CUDA", "1")
    monkeypatch.setenv("FAST_QR_DISABLE_QR512_BLOCKED_AUTO_POLICY", "1")
    assert candidate._compute_route_plan(fakes[512]) == ("qr512_blocked_cuda_fast", None)


def test_b200_default_uses_blocked_cuda_first_and_structured_first_is_opt_in(monkeypatch):
    candidate = _load_candidate_module()
    candidate._B200_DEVICE_CACHE.clear()
    props = SimpleNamespace(name="NVIDIA B200", major=10, total_memory=180 * 1024**3)
    fake512 = SimpleNamespace(is_cuda=True, dtype=torch.float32, ndim=3, shape=(640, 512, 512), device=torch.device("cuda", 0))
    fake1024 = SimpleNamespace(
        is_cuda=True,
        dtype=torch.float32,
        ndim=3,
        shape=(60, 1024, 1024),
        device=torch.device("cuda", 0),
    )

    class TrueLike:
        def all(self):
            return self

        def item(self):
            return True

    monkeypatch.setattr(candidate.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(candidate.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(candidate.torch.cuda, "get_device_properties", lambda _index: props)
    monkeypatch.delenv("FAST_QR_QR512_STRUCTURED_BEFORE_CUDA", raising=False)
    monkeypatch.delenv("FAST_QR_QR1024_STRUCTURED_BEFORE_CUDA", raising=False)
    monkeypatch.delenv("FAST_QR_STRUCTURED_ROUTES_BEFORE_CUDA", raising=False)
    monkeypatch.delenv("FAST_QR_DISABLE_B200_DEFAULT_STRUCTURED_BEFORE_CUDA", raising=False)
    monkeypatch.delenv("FAST_QR_ENABLE_B200_DEFAULT_STRUCTURED_BEFORE_CUDA", raising=False)
    monkeypatch.setenv("FAST_QR_DISABLE_BLOCKED_AUTO_POLICY", "1")
    monkeypatch.setenv("FAST_QR_DENSE_TAIL_CUT_512", "0")
    monkeypatch.setenv("FAST_QR_DENSE_TAIL_CUT_1024", "0")
    monkeypatch.setattr(candidate, "classify_512_sampled", lambda _data: "rankdef")
    monkeypatch.setattr(candidate, "classify_1024_sampled", lambda _data: "nearrank")
    monkeypatch.setattr(candidate, "_batch_tail_columns_are_exact_zero", lambda _data, _rank: TrueLike())
    monkeypatch.setattr(candidate, "_tail_matches_head_columns", lambda _data, _rank: True)

    assert not candidate._structured_before_cuda(512)
    assert not candidate._structured_before_cuda(1024)
    assert candidate._compute_route_plan(fake512) == ("qr512_blocked_cuda_fast", None)
    assert candidate._compute_route_plan(fake1024) == ("qr1024_blocked_cuda_fast", None)

    monkeypatch.setenv("FAST_QR_ENABLE_B200_DEFAULT_STRUCTURED_BEFORE_CUDA", "1")
    assert candidate._structured_before_cuda(512)
    assert candidate._structured_before_cuda(1024)
    assert candidate._compute_route_plan(fake512) == ("qr512_rankdef_fast", None)
    assert candidate._compute_route_plan(fake1024) == ("qr1024_nearrank_fast", None)

    monkeypatch.setenv("FAST_QR_QR512_STRUCTURED_BEFORE_CUDA", "0")
    assert not candidate._structured_before_cuda(512)
    assert candidate._compute_route_plan(fake512) == ("qr512_blocked_cuda_fast", None)

    monkeypatch.delenv("FAST_QR_QR512_STRUCTURED_BEFORE_CUDA")
    monkeypatch.setenv("FAST_QR_DISABLE_B200_DEFAULT_STRUCTURED_BEFORE_CUDA", "1")
    assert not candidate._structured_before_cuda(512)
    assert not candidate._structured_before_cuda(1024)


def test_b200_default_structured_first_samples_before_auto_policy(monkeypatch):
    candidate = _load_candidate_module()
    candidate._B200_DEVICE_CACHE.clear()
    props = SimpleNamespace(name="NVIDIA B200", major=10, total_memory=180 * 1024**3)
    fake512 = SimpleNamespace(is_cuda=True, dtype=torch.float32, ndim=3, shape=(640, 512, 512), device=torch.device("cuda", 0))
    fake1024 = SimpleNamespace(
        is_cuda=True,
        dtype=torch.float32,
        ndim=3,
        shape=(60, 1024, 1024),
        device=torch.device("cuda", 0),
    )
    calls = []

    monkeypatch.setattr(candidate.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(candidate.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(candidate.torch.cuda, "get_device_properties", lambda _index: props)
    monkeypatch.delenv("FAST_QR_ENABLE_B200_DEFAULT_STRUCTURED_BEFORE_CUDA", raising=False)
    monkeypatch.setenv("FAST_QR_DENSE_TAIL_CUT_512", "0")
    monkeypatch.setenv("FAST_QR_DENSE_TAIL_CUT_1024", "0")
    monkeypatch.setattr(candidate, "classify_512_sampled", lambda _data: calls.append("512") or "dense")
    monkeypatch.setattr(candidate, "classify_1024_sampled", lambda _data: calls.append("1024") or "dense")

    assert candidate._compute_route_plan(fake512) == ("qr512_blocked_cuda_auto_fast", None)
    assert candidate._compute_route_plan(fake1024) == ("qr1024_blocked_cuda_auto_fast", None)
    assert calls == []

    monkeypatch.setenv("FAST_QR_ENABLE_B200_DEFAULT_STRUCTURED_BEFORE_CUDA", "1")
    assert candidate._compute_route_plan(fake512) == ("qr512_blocked_cuda_auto_fast", None)
    assert candidate._compute_route_plan(fake1024) == ("qr1024_blocked_cuda_auto_fast", None)
    assert calls == ["512", "1024"]

    def fail_classifier(_data):
        raise AssertionError("auto blocked policy should bypass Python sampled classifier when structured-first is off")

    monkeypatch.delenv("FAST_QR_ENABLE_B200_DEFAULT_STRUCTURED_BEFORE_CUDA")
    monkeypatch.setattr(candidate, "classify_512_sampled", fail_classifier)
    monkeypatch.setattr(candidate, "classify_1024_sampled", fail_classifier)

    assert candidate._compute_route_plan(fake512) == ("qr512_blocked_cuda_auto_fast", None)
    assert candidate._compute_route_plan(fake1024) == ("qr1024_blocked_cuda_auto_fast", None)


def test_structured_first_dense_classification_uses_threshold_checked_tail_route(monkeypatch):
    candidate = _load_candidate_module()
    fake512 = SimpleNamespace(shape=(640, 512, 512))
    fake1024 = SimpleNamespace(shape=(60, 1024, 1024))

    def fail_auto_policy(*_args, **_kwargs):
        raise AssertionError("threshold-accepted dense route should not fall through to blocked auto policy")

    monkeypatch.setenv("FAST_QR_QR512_STRUCTURED_BEFORE_CUDA", "1")
    monkeypatch.setenv("FAST_QR_QR1024_STRUCTURED_BEFORE_CUDA", "1")
    monkeypatch.setattr(candidate, "_qr512_blocked_cuda_route_enabled", lambda _data: True)
    monkeypatch.setattr(candidate, "_qr1024_blocked_cuda_route_enabled", lambda _data: True)
    monkeypatch.setattr(candidate, "_qr512_cuda_route_enabled", lambda _data: False)
    monkeypatch.setattr(candidate, "_qr1024_cuda_route_enabled", lambda _data: False)
    monkeypatch.setattr(candidate, "classify_512_sampled", lambda _data: "dense")
    monkeypatch.setattr(candidate, "classify_1024_sampled", lambda _data: "dense")
    monkeypatch.setattr(candidate, "_tail_columns_are_tiny_relative_sampled", lambda *_args: True)
    monkeypatch.setattr(candidate, "_blocked_auto_policy_enabled", fail_auto_policy)

    assert candidate._compute_route_plan(fake512) == ("qr512_dense_fast", None)
    assert candidate._compute_route_plan(fake1024) == ("qr1024_dense_fast", None)

    monkeypatch.setattr(candidate, "_blocked_auto_policy_enabled", lambda _data, _n: False)
    monkeypatch.setattr(candidate, "_tail_columns_are_tiny_relative_sampled", lambda *_args: False)
    assert candidate._compute_route_plan(fake512) == ("qr512_blocked_cuda_fast", None)
    assert candidate._compute_route_plan(fake1024) == ("qr1024_blocked_cuda_fast", None)

    monkeypatch.setattr(candidate, "_tail_columns_are_tiny_relative_sampled", lambda *_args: True)
    monkeypatch.setenv("FAST_QR_DENSE_TAIL_CUT_512", "0")
    assert candidate._compute_route_plan(fake512) == ("qr512_blocked_cuda_fast", None)
    monkeypatch.setenv("FAST_QR_DENSE_TAIL_CUT_1024", "0")
    assert candidate._compute_route_plan(fake1024) == ("qr1024_blocked_cuda_fast", None)


@pytest.mark.parametrize(
    ("n", "fast_name", "route_name", "auto_name", "classifier_name"),
    [
        (512, "qr512_fast", "_qr512_blocked_cuda_route_enabled", "_qr512_blocked_cuda_auto_fast", "classify_512_sampled"),
        (1024, "qr1024_fast", "_qr1024_blocked_cuda_route_enabled", "_qr1024_blocked_cuda_auto_fast", "classify_1024_sampled"),
    ],
)
def test_candidate_shape_fast_uses_blocked_auto_before_classifier(
    monkeypatch,
    n,
    fast_name,
    route_name,
    auto_name,
    classifier_name,
):
    candidate = _load_candidate_module()
    data = SimpleNamespace(shape=(640 if n == 512 else 60, n, n))
    sentinel = object()

    def fail_classifier(_data):
        raise AssertionError("shape fast path should use blocked auto before Python classifier")

    monkeypatch.setattr(candidate, route_name, lambda _data: True)
    monkeypatch.setattr(candidate, "_blocked_auto_policy_enabled", lambda _data, shape_n: shape_n == n)
    monkeypatch.setattr(candidate, auto_name, lambda _data: sentinel)
    monkeypatch.setattr(candidate, classifier_name, fail_classifier)

    assert getattr(candidate, fast_name)(data) is sentinel


def test_blocked_structured_first_still_uses_classifier(monkeypatch):
    candidate = _load_candidate_module()
    fake512 = SimpleNamespace(shape=(640, 512, 512))
    fake1024 = SimpleNamespace(shape=(60, 1024, 1024))

    class TrueLike:
        def all(self):
            return self

        def item(self):
            return True

    monkeypatch.setenv("FAST_QR_QR512_STRUCTURED_BEFORE_CUDA", "1")
    monkeypatch.setenv("FAST_QR_QR1024_STRUCTURED_BEFORE_CUDA", "1")
    monkeypatch.setattr(candidate, "_qr512_blocked_cuda_route_enabled", lambda _data: True)
    monkeypatch.setattr(candidate, "_qr1024_blocked_cuda_route_enabled", lambda _data: True)
    monkeypatch.setattr(candidate, "_qr512_cuda_route_enabled", lambda _data: True)
    monkeypatch.setattr(candidate, "_qr1024_cuda_route_enabled", lambda _data: True)
    monkeypatch.setattr(candidate, "classify_512_sampled", lambda _data: "rankdef")
    monkeypatch.setattr(candidate, "classify_1024_sampled", lambda _data: "nearrank")
    monkeypatch.setattr(candidate, "_batch_tail_columns_are_exact_zero", lambda _data, _rank: TrueLike())
    monkeypatch.setattr(candidate, "_tail_matches_head_columns", lambda _data, _rank: True)

    assert candidate._compute_route_plan(fake512) == ("qr512_rankdef_fast", None)
    assert candidate._compute_route_plan(fake1024) == ("qr1024_nearrank_fast", None)


def _route_trace_suite_rows(first_row: dict) -> list[dict]:
    rows = [first_row]
    for index in range(1, 12):
        rows.append(
            {
                "spec": f"batch: 20; n: 32; cond: 1; seed: {index}",
                "batch": 20,
                "n": 32,
                "dispatch": "qr32_fast",
                "route": "qr32_fast",
                "case_metadata_available": False,
                "case_metadata_passed_to_submission": False,
                "case_info_source": "data.shape",
                "case_selection_info_sources": ["data.shape"],
                "shape_collision": False,
                "shape_only_case_selection": True,
                "shape_only_dispatch": True,
                "uses_tensor_values_for_dispatch": False,
                "uses_tensor_values_for_case_selection": False,
                "requires_tensor_guard_for_case_specific_path": False,
                "classifier_needed_for_case_specific_path": False,
                "classifier_needed_for_current_candidate": False,
                "classifier_on_current_hot_path": False,
                "dispatch_info_sources": ["data.shape"],
                "route_decision_source": "data.shape",
                "submission": "submissions/candidate.py",
                "submission_sha256": SYNTHETIC_CANDIDATE_SHA,
            }
        )
    return rows


def _write_route_trace_validation_fixture(tmp_path: Path, first_row: dict) -> None:
    _write_jsonl(
        tmp_path / "manifest.jsonl",
        [
            {
                "event": "suite_provenance",
                "submission": {
                    "path": "submissions/candidate.py",
                    "sha256": SYNTHETIC_CANDIDATE_SHA,
                    "bytes": 1,
                },
                "baseline": {
                    "path": "submissions/baseline_geqrf.py",
                    "sha256": SYNTHETIC_BASELINE_SHA,
                    "bytes": 1,
                },
                "repo": {"git_status_porcelain": []},
            }
        ],
    )
    _write_jsonl(tmp_path / "candidate_route_trace_public.jsonl", _route_trace_suite_rows(first_row))


def test_validate_route_trace_requires_bounded_sampled_plan_for_colliding_public_shapes(tmp_path):
    row = {
        "spec": "batch: 640; n: 512; cond: 2; seed: 770001; case: mixed",
        "batch": 640,
        "n": 512,
        "dispatch": "qr512_fast",
        "route": "qr512_mixed_fast",
        "case_metadata_available": False,
        "case_metadata_passed_to_submission": False,
        "case_info_source": "tensor_values",
        "case_selection_info_sources": ["data.shape", "tensor_values"],
        "shape_collision": True,
        "shape_only_case_selection": False,
        "shape_only_dispatch": False,
        "uses_tensor_values_for_dispatch": True,
        "uses_tensor_values_for_case_selection": True,
        "requires_tensor_guard_for_case_specific_path": True,
        "classifier_needed_for_case_specific_path": True,
        "classifier_needed_for_current_candidate": True,
        "classifier_on_current_hot_path": True,
        "dispatch_info_sources": ["data.shape", "tensor_values"],
        "route_decision_source": "data.shape+tensor_values",
        "structured_sampled_plan": True,
        "structured_sampled_matrix_count": 640,
        "structured_sampled_row_count": 32,
        "structured_candidate_counts": {"rankdef": 80, "clustered": 80, "scaled_nearrank": 80, "tiny_dense_tail": 0},
        "structured_exact_check_counts": {"rankdef": 80, "clustered": 80, "scaled_nearrank": 80, "tiny_dense_tail": 0},
        "submission": "submissions/candidate.py",
        "submission_sha256": SYNTHETIC_CANDIDATE_SHA,
    }
    _write_route_trace_validation_fixture(tmp_path, row)

    errors: list[dict[str, str]] = []
    validate_route_trace(tmp_path, errors)
    assert not [error for error in errors if error["check"] == "route_trace.sampled_plan"]

    bad = dict(row)
    bad.pop("structured_sampled_plan")
    _write_route_trace_validation_fixture(tmp_path, bad)
    errors = []
    validate_route_trace(tmp_path, errors)
    assert any(error["check"] == "route_trace.sampled_plan" for error in errors)

    bad = dict(row)
    bad["structured_sampled_row_count"] = 64
    _write_route_trace_validation_fixture(tmp_path, bad)
    errors = []
    validate_route_trace(tmp_path, errors)
    assert any(
        error["check"] == "route_trace.sampled_plan" and "bounded sampled-classifier budget" in error["message"]
        for error in errors
    )


def test_route_trace_small_fallback_case():
    candidate = _load_candidate_module()
    row = trace_route(candidate, {"batch": 2, "n": 8, "cond": 1, "seed": 11})
    assert row["route"] == "torch.geqrf"
    assert row["dispatch"] == "fallback"


def test_route_trace_uses_candidate_actual_route_plan(monkeypatch):
    candidate = _load_candidate_module()
    monkeypatch.setattr(candidate, "_route_plan_for_data", lambda _data: ("sentinel_route", None))
    row = trace_route(candidate, {"batch": 2, "n": 8, "cond": 1, "seed": 11})
    assert row["route"] == "sentinel_route"
    assert row["dispatch"] == "fallback"


def test_guard_benchmark_measures_route_decision_without_qr():
    candidate = _load_candidate_module()
    data = generate_input(batch=2, n=8, cond=1, seed=11)
    assert route_decision(candidate, data) == "torch.geqrf"
    row = measure_route_decision(candidate, data, repeats=2, warmup=1)
    assert row["route"] == "torch.geqrf"
    assert row["cold_wall_us"] >= 0.0
    assert row["cold_cuda_us"] is None
    assert row["wall_us"] >= 0.0
    assert row["cuda_us"] is None


@pytest.mark.parametrize(
    ("batch", "n", "fast_name"),
    [
        (20, 32, "qr32_fast"),
        (40, 176, "qr176_fast"),
        (40, 352, "qr352_fast"),
        (8, 2048, "qr2048_fast"),
        (2, 4096, "qr4096_fast"),
    ],
)
def test_custom_kernel_unique_public_shapes_bypass_route_planning(monkeypatch, batch, n, fast_name):
    candidate = _load_candidate_module()
    data = SimpleNamespace(shape=(batch, n, n))
    sentinel = object()

    monkeypatch.setattr(
        candidate,
        "_route_plan_for_data",
        lambda _data: (_ for _ in ()).throw(AssertionError("small public shapes should not route-plan")),
    )
    monkeypatch.setattr(candidate, fast_name, lambda _data: sentinel)

    assert candidate.custom_kernel(data) is sentinel


@pytest.mark.parametrize(
    ("batch", "n", "blocked_route_name", "auto_name"),
    [
        (640, 512, "_qr512_blocked_cuda_route_enabled", "qr512_blocked_cuda_auto_fast"),
        (60, 1024, "_qr1024_blocked_cuda_route_enabled", "qr1024_blocked_cuda_auto_fast"),
    ],
)
def test_custom_kernel_public_ambiguous_shapes_bypass_route_planning_on_b200_cuda_first(
    monkeypatch,
    batch,
    n,
    blocked_route_name,
    auto_name,
):
    candidate = _load_candidate_module()
    data = SimpleNamespace(shape=(batch, n, n))
    sentinel = object()

    monkeypatch.setattr(candidate, "_structured_before_cuda", lambda shape_n: False)
    monkeypatch.setattr(candidate, blocked_route_name, lambda _data: True)
    monkeypatch.setattr(candidate, "_blocked_auto_policy_enabled", lambda _data, shape_n: shape_n == n)
    monkeypatch.setattr(candidate, auto_name, lambda _data: sentinel)
    monkeypatch.setattr(
        candidate,
        "_route_plan_for_data",
        lambda _data: (_ for _ in ()).throw(AssertionError("B200 public hot path should not route-plan")),
    )

    assert candidate.custom_kernel(data) is sentinel


@pytest.mark.parametrize(
    ("batch", "n", "blocked_route_name", "auto_name"),
    [
        (640, 512, "_qr512_blocked_cuda_route_enabled", "qr512_blocked_cuda_auto_fast"),
        (60, 1024, "_qr1024_blocked_cuda_route_enabled", "qr1024_blocked_cuda_auto_fast"),
    ],
)
def test_custom_kernel_public_ambiguous_shapes_keep_route_planning_when_structured_first(
    monkeypatch,
    batch,
    n,
    blocked_route_name,
    auto_name,
):
    candidate = _load_candidate_module()
    data = SimpleNamespace(shape=(batch, n, n))
    sentinel = object()
    calls = []

    monkeypatch.setattr(candidate, "_structured_before_cuda", lambda shape_n: shape_n == n)
    monkeypatch.setattr(candidate, blocked_route_name, lambda _data: True)
    monkeypatch.setattr(candidate, "_blocked_auto_policy_enabled", lambda _data, shape_n: shape_n == n)
    monkeypatch.setattr(
        candidate,
        auto_name,
        lambda _data: (_ for _ in ()).throw(AssertionError("structured-first tuning should route-plan")),
    )
    monkeypatch.setattr(candidate, "_route_plan_for_data", lambda data_arg: ("structured_route", {"n": data_arg.shape[-1]}))

    def dispatch(route, data_arg, plan):
        calls.append((route, data_arg, plan))
        return sentinel

    monkeypatch.setattr(candidate, "_dispatch_route", dispatch)

    assert candidate.custom_kernel(data) is sentinel
    assert calls == [("structured_route", data, {"n": n})]


@pytest.mark.parametrize(
    ("n", "fast_name", "route_enabled_name", "auto_name", "blocked_name"),
    [
        (
            2048,
            "qr2048_fast",
            "_qr2048_blocked_cuda_route_enabled",
            "_qr2048_blocked_cuda_auto_fast",
            "_qr2048_blocked_cuda_fast",
        ),
        (
            4096,
            "qr4096_fast",
            "_qr4096_blocked_cuda_route_enabled",
            "_qr4096_blocked_cuda_auto_fast",
            "_qr4096_blocked_cuda_fast",
        ),
    ],
)
def test_candidate_large_unique_fast_paths_prefer_blocked_auto_policy(
    monkeypatch,
    n,
    fast_name,
    route_enabled_name,
    auto_name,
    blocked_name,
):
    candidate = _load_candidate_module()
    data = SimpleNamespace(shape=(1, n, n))
    sentinel = object()

    monkeypatch.setattr(candidate, "_dense_tail_route_or_fallback", lambda _data, _route: "torch.geqrf")
    monkeypatch.setattr(candidate, route_enabled_name, lambda _data: True)
    monkeypatch.setattr(candidate, "_blocked_auto_policy_enabled", lambda _data, shape_n: shape_n == n)
    monkeypatch.setattr(candidate, auto_name, lambda _data: sentinel)
    monkeypatch.setattr(
        candidate,
        blocked_name,
        lambda _data: (_ for _ in ()).throw(AssertionError("large fast path should use blocked auto policy")),
    )

    assert getattr(candidate, fast_name)(data) is sentinel


@pytest.mark.parametrize(
    ("n", "fast_name", "route_enabled_name", "auto_name", "blocked_name"),
    [
        (
            2048,
            "qr2048_fast",
            "_qr2048_blocked_cuda_route_enabled",
            "_qr2048_blocked_cuda_auto_fast",
            "_qr2048_blocked_cuda_fast",
        ),
        (
            4096,
            "qr4096_fast",
            "_qr4096_blocked_cuda_route_enabled",
            "_qr4096_blocked_cuda_auto_fast",
            "_qr4096_blocked_cuda_fast",
        ),
    ],
)
def test_candidate_large_unique_fast_paths_keep_plain_blocked_fallback_when_auto_disabled(
    monkeypatch,
    n,
    fast_name,
    route_enabled_name,
    auto_name,
    blocked_name,
):
    candidate = _load_candidate_module()
    data = SimpleNamespace(shape=(1, n, n))
    sentinel = object()

    monkeypatch.setattr(candidate, "_dense_tail_route_or_fallback", lambda _data, _route: "torch.geqrf")
    monkeypatch.setattr(candidate, route_enabled_name, lambda _data: True)
    monkeypatch.setattr(candidate, "_blocked_auto_policy_enabled", lambda _data, _shape_n: False)
    monkeypatch.setattr(
        candidate,
        auto_name,
        lambda _data: (_ for _ in ()).throw(AssertionError("auto policy is disabled")),
    )
    monkeypatch.setattr(candidate, blocked_name, lambda _data: sentinel)

    assert getattr(candidate, fast_name)(data) is sentinel


def test_candidate_route_cache_reuses_same_tensor_route(monkeypatch):
    candidate = _load_candidate_module()
    candidate._ROUTE_CACHE.clear()
    data = generate_input(batch=20, n=32, cond=1, seed=53124)

    assert candidate._route_for_data(data) == "qr32_fast"
    assert len(candidate._ROUTE_CACHE) == 1

    def fail_compute(_data):
        raise AssertionError("cached route was not reused")

    monkeypatch.setattr(candidate, "_compute_route_plan", fail_compute)
    assert candidate._route_for_data(data) == "qr32_fast"

    monkeypatch.setenv("FAST_QR_DISABLE_ROUTE_CACHE", "1")
    monkeypatch.setattr(candidate, "_compute_route_plan", lambda _data: ("sentinel_route", None))
    assert candidate._route_for_data(data) == "sentinel_route"


def test_candidate_route_cache_includes_public_512_1024_correctness_batches():
    candidate = _load_candidate_module()

    assert candidate._cacheable_route_shape(16, 512)
    assert candidate._cacheable_route_shape(4, 1024)
    assert candidate._cacheable_route_shape(2, 2048)
    assert candidate._cacheable_route_shape(1, 4096)


def test_candidate_route_cache_invalidates_on_inplace_mutation(monkeypatch):
    candidate = _load_candidate_module()
    candidate._ROUTE_CACHE.clear()
    data = generate_input(batch=20, n=32, cond=1, seed=53124)
    calls = []

    def versioned_route(_data):
        calls.append(float(_data[0, 0, 0].item()))
        route = "positive_route" if calls[-1] > 0.0 else "nonpositive_route"
        return route, None

    data[0, 0, 0] = 1.0
    monkeypatch.setattr(candidate, "_compute_route_plan", versioned_route)
    assert candidate._route_for_data(data) == "positive_route"
    assert candidate._route_for_data(data) == "positive_route"
    assert len(calls) == 1

    data[0, 0, 0] = -1.0
    assert candidate._route_for_data(data) == "nonpositive_route"
    assert len(calls) == 2


def test_candidate_route_cache_invalidates_on_ablation_env_change(monkeypatch):
    candidate = _load_candidate_module()
    candidate._ROUTE_CACHE.clear()
    data = generate_input(batch=20, n=32, cond=1, seed=53124)

    def env_route(_data):
        route = "structured_on" if candidate._structured_routes_enabled() else "structured_off"
        return route, None

    monkeypatch.setattr(candidate, "_compute_route_plan", env_route)
    assert candidate._route_for_data(data) == "structured_on"

    monkeypatch.setenv("FAST_QR_DISABLE_STRUCTURED_ROUTES", "1")
    assert candidate._route_for_data(data) == "structured_off"


def test_candidate_route_cache_invalidates_on_tail_policy_env_change(monkeypatch):
    candidate = _load_candidate_module()
    candidate._ROUTE_CACHE.clear()
    data = generate_input(batch=20, n=32, cond=1, seed=53124)

    def tail_route(_data):
        return f"cut_{candidate._dense_tail_cut(512)}", None

    monkeypatch.setattr(candidate, "_compute_route_plan", tail_route)
    assert candidate._route_for_data(data) == "cut_32"
    assert candidate._route_for_data(data) == "cut_32"

    monkeypatch.setenv("FAST_QR_DENSE_TAIL_CUT_512", "12")
    assert candidate._route_for_data(data) == "cut_12"


def test_candidate_route_cache_invalidates_on_large_auto_policy_env_change(monkeypatch):
    candidate = _load_candidate_module()
    candidate._ROUTE_CACHE.clear()
    data = generate_input(batch=20, n=32, cond=1, seed=53124)

    def auto_route(_data):
        disabled = os.environ.get("FAST_QR_DISABLE_QR2048_BLOCKED_AUTO_POLICY") == "1"
        return ("large_auto_disabled" if disabled else "large_auto_enabled"), None

    monkeypatch.setattr(candidate, "_compute_route_plan", auto_route)
    assert candidate._route_for_data(data) == "large_auto_enabled"
    assert candidate._route_for_data(data) == "large_auto_enabled"

    monkeypatch.setenv("FAST_QR_DISABLE_QR2048_BLOCKED_AUTO_POLICY", "1")
    assert candidate._route_for_data(data) == "large_auto_disabled"


def test_candidate_route_cache_invalidates_on_strict_cuda_env_change(monkeypatch):
    candidate = _load_candidate_module()
    candidate._ROUTE_CACHE.clear()
    data = generate_input(batch=20, n=32, cond=1, seed=53124)

    def strict_route(_data):
        required = os.environ.get("FAST_QR_REQUIRE_QR512_CUDA") == "1"
        return ("qr512_probe_required" if required else "qr512_probe_optional"), None

    monkeypatch.setattr(candidate, "_compute_route_plan", strict_route)
    assert candidate._route_for_data(data) == "qr512_probe_optional"
    assert candidate._route_for_data(data) == "qr512_probe_optional"

    monkeypatch.setenv("FAST_QR_REQUIRE_QR512_CUDA", "1")
    assert candidate._route_for_data(data) == "qr512_probe_required"


def test_candidate_large_cuda_thread_knobs_specialize_source(monkeypatch):
    candidate = _load_candidate_module()

    assert candidate._qr512_cuda_threads_per_cta() == 256
    default_name = candidate._qr512_cuda_extension_name()
    default_source = candidate._qr512_cuda_source()
    assert "__shared__ float scratch[32];" in default_source
    assert "constexpr int block = 256;" in default_source
    assert "__global__ void __launch_bounds__(256) geqrf512_kernel(" in default_source
    assert candidate._qr512_cuda_panel_b() == 32
    assert "constexpr int PANEL_B = 32;" in default_source
    assert candidate._qr512_cuda_update_col_tile() == 4
    assert "constexpr int UPDATE_COL_TILE = 4;" in default_source
    assert "float dot_parts[UPDATE_COL_TILE];" in default_source
    assert "col_start += UPDATE_COL_TILE" in default_source
    assert candidate._qr512_cuda_update_mode() == "reflectors"
    assert candidate._qr512_cuda_precision_mode() == "fp32"
    assert "constexpr int USE_TF32_INPUT_UPDATE = 0;" in default_source
    assert "update_operand_512" in default_source
    assert "constexpr int USE_COMPACT_WY_UPDATE = 0;" in default_source
    assert candidate._qr512_cuda_panel_refresh_mode() == "none"
    assert candidate._qr512_cuda_r_maintenance_mode() == "none"
    assert "constexpr int USE_PANEL_REFRESH_PREFIX = 0;" in default_source
    assert "constexpr int USE_R_MAINTENANCE_PANEL_PREFIX = 0;" in default_source
    for token in [
        "HouseholderParams512",
        "hh_norm_kernel_512",
        "hh_generate_reflector_512",
        "hh_normalize_reflector_512",
        "hh_apply_single_reflector_512",
        "hh_apply_single_reflector_to_vector_512",
        "panel_factor_kernel_512",
        "refresh_panel_from_original_kernel_512",
        "repair_panel_r_from_original_kernel_512",
        "form_block_reflector_T_kernel_512",
        "apply_block_reflector_kernel_512",
        "block_trailing_update_kernel_512",
    ]:
        assert token in default_source

    monkeypatch.setenv("FAST_QR_QR512_UPDATE_MODE", "compact-wy")
    assert candidate._qr512_cuda_update_mode() == "compact-wy"
    qr512_compact_source = candidate._qr512_cuda_source()
    assert "constexpr int USE_COMPACT_WY_UPDATE = 1;" in qr512_compact_source
    assert candidate._qr512_cuda_extension_name() != default_name
    monkeypatch.delenv("FAST_QR_QR512_UPDATE_MODE")

    monkeypatch.setenv("FAST_QR_QR512_UPDATE_COL_TILE", "8")
    assert candidate._qr512_cuda_update_col_tile() == 8
    qr512_update_tile_source = candidate._qr512_cuda_source()
    assert "constexpr int UPDATE_COL_TILE = 8;" in qr512_update_tile_source
    assert candidate._qr512_cuda_extension_name() != default_name
    monkeypatch.delenv("FAST_QR_QR512_UPDATE_COL_TILE")

    monkeypatch.setenv("FAST_QR_QR512_PRECISION_MODE", "tf32")
    assert candidate._qr512_cuda_precision_mode() == "tf32-input"
    qr512_tf32_source = candidate._qr512_cuda_source()
    assert "constexpr int USE_TF32_INPUT_UPDATE = 1;" in qr512_tf32_source
    assert "constexpr int USE_FP16_INPUT_UPDATE = 0;" in qr512_tf32_source
    assert "__float_as_uint(value)" in qr512_tf32_source
    assert candidate._qr512_cuda_extension_name() != default_name
    monkeypatch.delenv("FAST_QR_QR512_PRECISION_MODE")

    monkeypatch.setenv("FAST_QR_QR512_PRECISION_MODE", "fp16")
    assert candidate._qr512_cuda_precision_mode() == "fp16-input"
    qr512_fp16_source = candidate._qr512_cuda_source()
    assert "constexpr int USE_TF32_INPUT_UPDATE = 0;" in qr512_fp16_source
    assert "constexpr int USE_FP16_INPUT_UPDATE = 1;" in qr512_fp16_source
    assert "__float2half_rn(value)" in qr512_fp16_source
    assert candidate._qr512_cuda_extension_name() != default_name
    monkeypatch.delenv("FAST_QR_QR512_PRECISION_MODE")

    monkeypatch.setenv("FAST_QR_QR512_PANEL_REFRESH_MODE", "prefix")
    monkeypatch.setenv("FAST_QR_QR512_R_MAINTENANCE_MODE", "panel-prefix")
    assert candidate._qr512_cuda_panel_refresh_mode() == "prefix"
    assert candidate._qr512_cuda_r_maintenance_mode() == "panel-prefix"
    qr512_repair_source = candidate._qr512_cuda_source()
    assert "constexpr int USE_PANEL_REFRESH_PREFIX = 1;" in qr512_repair_source
    assert "constexpr int USE_R_MAINTENANCE_PANEL_PREFIX = 1;" in qr512_repair_source
    refresh_body = qr512_repair_source.split("__device__ void refresh_panel_from_original_kernel_512", 1)[1].split(
        "__device__ void repair_panel_r_from_original_kernel_512",
        1,
    )[0]
    assert "refresh_vector[row] = data" in refresh_body
    assert "for (int row = panel_start + threadIdx.x; row < N; row += blockDim.x)" in refresh_body
    assert "h[b * h_s0 + row * h_s1 + col * h_s2] = refresh_vector[row];" in refresh_body
    assert candidate._qr512_cuda_extension_name() != default_name
    monkeypatch.delenv("FAST_QR_QR512_PANEL_REFRESH_MODE")
    monkeypatch.delenv("FAST_QR_QR512_R_MAINTENANCE_MODE")

    monkeypatch.setenv("FAST_QR_QR512_PANEL_B", "48")
    assert candidate._qr512_cuda_panel_b() == 48
    qr512_panel_source = candidate._qr512_cuda_source()
    assert "constexpr int PANEL_B = 48;" in qr512_panel_source
    assert candidate._qr512_cuda_extension_name() != default_name
    monkeypatch.delenv("FAST_QR_QR512_PANEL_B")

    monkeypatch.setenv("FAST_QR_QR512_WARPS_PER_CTA", "16")
    assert candidate._qr512_cuda_threads_per_cta() == 512
    qr512_source = candidate._qr512_cuda_source()
    assert "__shared__ float scratch[64];" in qr512_source
    assert "constexpr int block = 512;" in qr512_source
    assert "__global__ void __launch_bounds__(512) geqrf512_kernel(" in qr512_source
    assert candidate._qr512_cuda_extension_name() != default_name

    monkeypatch.setenv("FAST_QR_QR512_THREADS_PER_CTA", "128")
    assert candidate._qr512_cuda_threads_per_cta() == 128
    qr512_source = candidate._qr512_cuda_source()
    assert "__shared__ float scratch[16];" in qr512_source
    assert "constexpr int block = 128;" in qr512_source
    assert "__global__ void __launch_bounds__(128) geqrf512_kernel(" in qr512_source

    monkeypatch.setenv("FAST_QR_QR1024_WARPS_PER_CTA", "32")
    assert candidate._qr1024_cuda_threads_per_cta() == 1024
    qr1024_source = candidate._qr1024_cuda_source()
    assert "__shared__ float scratch[128];" in qr1024_source
    assert "constexpr int block = 1024;" in qr1024_source
    assert "__global__ void __launch_bounds__(1024) geqrf1024_kernel(" in qr1024_source
    assert candidate._qr1024_cuda_panel_b() == 32
    assert candidate._qr1024_cuda_update_col_tile() == 4
    assert "constexpr int UPDATE_COL_TILE = 4;" in qr1024_source
    assert candidate._qr1024_cuda_update_mode() == "reflectors"
    assert candidate._qr1024_cuda_precision_mode() == "fp32"
    assert candidate._qr1024_cuda_panel_refresh_mode() == "none"
    assert candidate._qr1024_cuda_r_maintenance_mode() == "none"
    for token in [
        "HouseholderParams1024",
        "hh_norm_kernel_1024",
        "hh_generate_reflector_1024",
        "hh_normalize_reflector_1024",
        "hh_apply_single_reflector_1024",
        "hh_apply_single_reflector_to_vector_1024",
        "panel_factor_kernel_1024",
        "refresh_panel_from_original_kernel_1024",
        "repair_panel_r_from_original_kernel_1024",
        "form_block_reflector_T_kernel_1024",
        "apply_block_reflector_kernel_1024",
        "block_trailing_update_kernel_1024",
    ]:
        assert token in qr1024_source
    monkeypatch.setenv("FAST_QR_QR1024_PANEL_B", "96")
    assert candidate._qr1024_cuda_panel_b() == 96
    assert "constexpr int PANEL_B = 96;" in candidate._qr1024_cuda_source()
    monkeypatch.setenv("FAST_QR_QR1024_UPDATE_MODE", "compact-wy")
    assert candidate._qr1024_cuda_update_mode() == "compact-wy"
    assert "constexpr int USE_COMPACT_WY_UPDATE = 1;" in candidate._qr1024_cuda_source()
    monkeypatch.setenv("FAST_QR_QR1024_UPDATE_COL_TILE", "8")
    assert candidate._qr1024_cuda_update_col_tile() == 8
    assert "constexpr int UPDATE_COL_TILE = 8;" in candidate._qr1024_cuda_source()
    monkeypatch.setenv("FAST_QR_QR1024_PRECISION_MODE", "tf32-input")
    assert candidate._qr1024_cuda_precision_mode() == "tf32-input"
    assert "constexpr int USE_TF32_INPUT_UPDATE = 1;" in candidate._qr1024_cuda_source()
    monkeypatch.setenv("FAST_QR_QR1024_PRECISION_MODE", "fp16-input")
    assert candidate._qr1024_cuda_precision_mode() == "fp16-input"
    assert "constexpr int USE_FP16_INPUT_UPDATE = 1;" in candidate._qr1024_cuda_source()
    monkeypatch.setenv("FAST_QR_QR1024_PANEL_REFRESH_MODE", "prefix")
    monkeypatch.setenv("FAST_QR_QR1024_R_MAINTENANCE_MODE", "panel-prefix")
    assert candidate._qr1024_cuda_panel_refresh_mode() == "prefix"
    assert candidate._qr1024_cuda_r_maintenance_mode() == "panel-prefix"
    assert "constexpr int USE_PANEL_REFRESH_PREFIX = 1;" in candidate._qr1024_cuda_source()
    assert "constexpr int USE_R_MAINTENANCE_PANEL_PREFIX = 1;" in candidate._qr1024_cuda_source()

    monkeypatch.setenv("FAST_QR_QR1024_THREADS_PER_CTA", "96")
    monkeypatch.setenv("FAST_QR_QR1024_WARPS_PER_CTA", "99")
    assert candidate._qr1024_cuda_threads_per_cta() == 256


def test_candidate_blocked_cuda_update_col_end_is_on_trailing_kernel_signature():
    candidate = _load_candidate_module()

    for n in (512, 1024, 2048, 4096):
        if n == 512:
            source = candidate._qr512_blocked_cuda_source()
            cpp_source = candidate._QR512_BLOCKED_CPP_SOURCE
        elif n == 1024:
            source = candidate._qr1024_blocked_cuda_source()
            cpp_source = candidate._QR1024_BLOCKED_CPP_SOURCE
        else:
            source = candidate._generic_blocked_cuda_source(n)
            cpp_source = candidate._blocked_cpp_source_for_n(n)

        assert re.findall(r"__[A-Z0-9_]+__", source + cpp_source) == []
        assert f"void geqrf{n}_blocked_indexed_cuda(" in cpp_source
        assert f"void geqrf{n}_blocked_indexed(" in cpp_source
        assert f"void geqrf{n}_blocked_auto_cuda(" in cpp_source
        assert f"void geqrf{n}_blocked_auto(" in cpp_source
        assert f"void geqrf{n}_blocked_make_policy_cuda(" in cpp_source
        assert f"void geqrf{n}_blocked_make_policy(" in cpp_source
        assert f"void geqrf{n}_blocked_make_policy_metadata_cuda(" in cpp_source
        assert f"void geqrf{n}_blocked_make_policy_metadata(" in cpp_source
        assert f"void geqrf{n}_blocked_policy_cuda(" in cpp_source
        assert f"void geqrf{n}_blocked_policy(" in cpp_source
        assert "TORCH_CHECK(indices.scalar_type() == torch::kInt64" in cpp_source
        assert f"geqrf{n}_blocked_indexed_cuda(data, h, tau, indices, factor_cols, project_tail);" in cpp_source
        assert f"geqrf{n}_blocked_auto_cuda(data, h, tau);" in cpp_source
        assert f"geqrf{n}_blocked_make_policy_cuda(data, factor_cols, project_tail);" in cpp_source
        assert (
            f"geqrf{n}_blocked_make_policy_metadata_cuda(data, factor_cols, project_tail, metadata);"
            in cpp_source
        )
        assert "int64_t max_factor_cols" in cpp_source
        assert "bool any_project_tail" in cpp_source
        assert "int64_t min_project_factor_cols" in cpp_source
        assert "metadata.size(0) >= 6" in cpp_source
        assert f"geqrf{n}_blocked_policy_cuda(" in cpp_source
        for argument in ("factor_cols", "project_tail", "max_factor_cols", "any_project_tail", "min_project_factor_cols"):
            assert argument in cpp_source
        panel_signature = source.split(f"blocked{n}_panel_factor_kernel(", 1)[1].split(") {", 1)[0]
        trailing_signature = source.split(f"blocked{n}_trailing_update_kernel(", 1)[1].split(") {", 1)[0]
        copy_signature = source.split(f"blocked{n}_copy_kernel(", 1)[1].split(") {", 1)[0]
        kernel_prefix = "__global__ void __launch_bounds__(BLOCK_THREADS)"
        copy_body = source.split(f"{kernel_prefix} blocked{n}_copy_kernel(", 1)[1].split(
            f"{kernel_prefix} blocked{n}_tail_projection_kernel(", 1
        )[0]
        host_body = source.split(f"void geqrf{n}_blocked_cuda_impl(", 1)[1].split(
            f"void geqrf{n}_blocked_cuda(",
            1,
        )[0]
        wrapper_body = source.split(f"void geqrf{n}_blocked_cuda(", 1)[1].split(
            f"void geqrf{n}_blocked_indexed_cuda(",
            1,
        )[0]
        indexed_wrapper_body = source.split(f"void geqrf{n}_blocked_indexed_cuda(", 1)[1]
        trailing_launch = source.split(f"blocked{n}_trailing_update_kernel<<<", 1)[1].split(");", 1)[0]
        tail_body = source.split(f"{kernel_prefix} blocked{n}_tail_projection_kernel(", 1)[1].split(
            f"{kernel_prefix} blocked{n}_panel_refresh_kernel(", 1
        )[0]
        refresh_body = source.split(f"{kernel_prefix} blocked{n}_panel_refresh_kernel(", 1)[1].split(
            f"{kernel_prefix} blocked{n}_panel_r_repair_kernel(", 1
        )[0]
        repair_body = source.split(f"{kernel_prefix} blocked{n}_panel_r_repair_kernel(", 1)[1].split(
            f"{kernel_prefix} blocked{n}_panel_factor_kernel(", 1
        )[0]
        panel_body = source.split(f"{kernel_prefix} blocked{n}_panel_factor_kernel(", 1)[1].split(
            f"{kernel_prefix} blocked{n}_trailing_update_kernel(", 1
        )[0]
        sum_body = source.split(f"__device__ __forceinline__ float blocked{n}_sum(", 1)[1].split(
            f"__device__ void blocked{n}_apply_prefix_to_vector(", 1
        )[0]
        trailing_body = source.split(f"{kernel_prefix} blocked{n}_trailing_update_kernel(", 1)[1].split(
            f"void geqrf{n}_blocked_cuda", 1
        )[0]
        policy_body = source.split(f"{kernel_prefix} blocked{n}_policy_kernel(", 1)[1].split(
            f"{kernel_prefix} blocked{n}_dense_tail_policy_kernel(", 1
        )[0]

        for kernel in (
            "copy",
            "tail_projection",
            "panel_refresh",
            "panel_r_repair",
            "panel_factor",
            "trailing_update",
        ):
            assert f"{kernel_prefix} blocked{n}_{kernel}_kernel(" in source
            kernel_signature = source.split(f"{kernel_prefix} blocked{n}_{kernel}_kernel(", 1)[1].split(") {", 1)[0]
            assert "const int64_t* __restrict__ indices" in kernel_signature
        assert "int update_col_end" not in panel_signature
        assert "int update_col_end" in trailing_signature
        assert "factor_cols," in trailing_launch
        expected_ctas = 8 if n >= 4096 else 4 if n >= 2048 else 1
        assert f"constexpr int CTAS_PER_MATRIX = {expected_ctas};" in source
        expected_frontload = 1 if n >= 2048 else 0
        assert f"constexpr int CTA_SCHEDULE_FRONTLOAD = {expected_frontload};" in source
        assert "constexpr int CTA_SCHEDULE_ALL_TILES = 0;" in source
        expected_compact_tile_cols = 2 if n >= 2048 else 4
        assert f"constexpr int COMPACT_WY_TILE_COLS = {expected_compact_tile_cols};" in source
        assert f"constexpr float POLICY_SCALED_TAIL_RATIO = {candidate._policy_scaled_tail_ratio(n):.9e}f;" in source
        assert "constexpr int USE_COMPACT_WY_UPDATE = 0;" in source
        assert "__CTAS_PER_MATRIX__" not in source
        assert "__CTA_SCHEDULE_FRONTLOAD__" not in source
        assert "__CTA_SCHEDULE_ALL_TILES__" not in source
        assert "__COMPACT_WY_TILE_COLS__" not in source
        assert "__POLICY_SCALED_TAIL_RATIO__" not in source
        assert "__USE_COMPACT_WY_UPDATE__" not in source
        assert f"inline int blocked{n}_launch_col_tiles(int col_tiles)" in source
        assert "if (CTA_SCHEDULE_ALL_TILES) {" in source
        assert "if (CTA_SCHEDULE_FRONTLOAD) {" in source
        assert "const int first_tile = blockIdx.x;" in trailing_body
        assert "const int first_tile = blockIdx.x;" in tail_body
        assert "for (int tile = first_tile; ; tile += gridDim.x)" in trailing_body
        assert "for (int tile = first_tile; ; tile += gridDim.x)" in tail_body
        assert "const int tile_reduce_floats = COMPACT_WY_TILE_COLS * warp_count;" in trailing_body
        assert "float* block_t = scratch + tile_reduce_floats;" in trailing_body
        assert "block_t[jj * PANEL_B + jj] = tau_j;" in trailing_body
        assert "reflector_dots[COMPACT_WY_TILE_COLS]" in trailing_body
        assert "panel_dot_shared[COMPACT_WY_TILE_COLS]" in panel_body
        assert "chunk_col_start += COMPACT_WY_TILE_COLS" in panel_body
        assert "float dot_parts[COMPACT_WY_TILE_COLS];" in panel_body
        assert "dot_parts[cc] += v * h[" in panel_body
        assert f"blocked{n}_sum_tile(dot_parts, chunk_width, scratch, panel_dot_shared);" in panel_body
        assert "tau_k * v * panel_dot_shared[cc]" in panel_body
        assert trailing_body.count("chunk_col_start += COMPACT_WY_TILE_COLS") >= 2
        assert "float dot_parts[COMPACT_WY_TILE_COLS];" in trailing_body
        assert f"blocked{n}_sum_tile(dot_parts, chunk_width, scratch, reflector_dots);" in trailing_body
        assert f"blocked{n}_update_operand(reflector_dots[cc])" in trailing_body
        assert (
            f"blocked{n}_sum_tile(\n"
            "                        dot_parts,\n"
            "                        chunk_width,\n"
            "                        scratch,\n"
            "                        block_p + jj * COMPACT_WY_TILE_COLS\n"
            "                    );"
        ) in trailing_body
        assert "block_w[jj * COMPACT_WY_TILE_COLS + cc] = accum;" in trailing_body
        assert "if (USE_COMPACT_WY_UPDATE) {" in trailing_body
        assert "if (USE_COMPACT_WY_UPDATE && trailing_shmem > 49152)" in host_body
        assert f"cudaFuncSetAttribute(\n            blocked{n}_trailing_update_kernel" in host_body
        assert f"const int launch_col_tiles = blocked{n}_launch_col_tiles(col_tiles);" in host_body
        assert "dim3 grid(launch_col_tiles, batch);" in host_body
        assert "dim3 grid(col_tiles, batch);" not in source
        assert "tile_col_start >= factor_cols_b" in source
        assert "min(factor_cols_b, tile_col_start + TILE_N)" in source
        assert copy_signature.count("int factor_cols") == 1
        assert "int copy_col_end" in copy_signature
        assert "int project_tail" not in copy_signature
        assert "const int linear_col_end = copy_col_end;" in copy_body
        assert "batch * int64_t(N) * int64_t(linear_col_end)" in copy_body
        assert "linear % linear_col_end" in copy_body
        assert "col < factor_cols || project_tail" not in copy_body
        assert "if (copy_col_end < N) {" in copy_body
        assert "const int tail_cols = N - copy_col_end;" in copy_body
        assert "batch * int64_t(N) * int64_t(tail_cols)" in copy_body
        assert "if (factor_cols_by_batch != nullptr) {\n            continue;" not in copy_body
        assert "const int col = copy_col_end + int(linear % tail_cols);" in copy_body
        assert "h[b * h_s0 + row * h_s1 + col * h_s2] = 0.0f;" in copy_body
        assert "const int local_b = int(linear / (int64_t(N) * int64_t(linear_col_end)));" in copy_body
        assert "const int b = indices == nullptr ? local_b : int(indices[local_b]);" in copy_body
        assert "const int local_b = int(linear / N);" in copy_body
        assert f"blocked{n}_policy_kernel" in source
        assert f"__device__ __forceinline__ float blocked{n}_max(float value, float* scratch)" in source
        assert f"__device__ __forceinline__ void blocked{n}_max8(float values[8], float* scratch)" in source
        assert "factor_cols_by_batch" in source
        assert "project_tail_by_batch" in source
        assert "has_structured_batch" in source
        assert "atomicExch(has_structured_batch, 1)" in source
        assert "nearrank_plain_err" in source
        assert "const int plain_nearrank" in source
        assert "const int scaled_nearrank" in source
        assert "plain_nearrank || scaled_nearrank" in source
        assert "powf(10.0f" not in policy_body
        assert "head_value * POLICY_SCALED_TAIL_RATIO" in policy_body
        assert "linear < N * N" not in policy_body
        assert "POLICY_SAMPLE_OFFSETS" in policy_body
        assert "POLICY_RANDOM_ROWS" in policy_body
        assert "float policy_maxima[8];" in policy_body
        assert f"blocked{n}_max8(policy_maxima, scratch);" in policy_body
        assert f"blocked{n}_max(" not in policy_body
        assert "__POLICY_RANDOM_ROWS__" not in source
        if n == 512:
            expected_policy_rows = candidate._qr512_blocked_cuda_policy_sample_rows()
        elif n == 1024:
            expected_policy_rows = candidate._qr1024_blocked_cuda_policy_sample_rows()
        else:
            expected_policy_rows = candidate._generic_blocked_cuda_policy_sample_rows(n)
        assert f"constexpr int POLICY_RANDOM_ROWS = {expected_policy_rows};" in source
        assert "const int row = (row_slot == 0) ? tail_col" in policy_body
        assert "const int head_col = offset;" in policy_body
        assert "const int tail_col = RANK_COLS + offset;" in policy_body
        assert f"blocked{n}_dense_tail_allowed(" in source
        assert f"blocked{n}_dense_tail_policy_kernel" in source
        assert f"void geqrf{n}_blocked_auto_cuda(" in source
        assert f"void geqrf{n}_blocked_make_policy_cuda(" in source
        assert f"void geqrf{n}_blocked_make_policy_metadata_cuda(" in source
        assert f"void geqrf{n}_blocked_policy_cuda(" in source
        assert f"blocked{n}_policy_metadata_init_kernel" in source
        assert f"blocked{n}_policy_metadata_kernel" not in source
        assert "int* __restrict__ metadata" in source
        assert "atomicMax(&metadata[0], factor_cols);" in source
        assert "all_dense_tail_allowed" not in source
        assert "metadata[0] = dense_factor_cols;" not in source
        assert "atomicMin(&metadata[4], dense_factor_cols);" in source
        assert "atomicExch(&metadata[5], 0);" in source
        assert "int adjust_blocks = int((batch + threads - 1) / threads);" in source
        assert "atomicMin(&metadata[4], factor_cols);" in source
        assert "metadata.data_ptr<int>()," in source
        assert "int64_t max_factor_cols_arg" in source
        assert "bool any_project_tail" in source
        assert "int64_t min_project_factor_cols_arg" in source
        assert "const int tail_start = per_matrix_policy ? min_project_factor_cols : factor_cols;" in source
        assert "const int tail_span = N - tail_start;" in source
        assert "max_factor_cols_arg," in source
        auto_body = source.split(f"void geqrf{n}_blocked_auto_cuda(", 1)[1]
        assert "auto metadata = torch::empty({6}, int_options);" in auto_body
        assert (
            auto_body.count(
                f"geqrf{n}_blocked_make_policy_metadata_cuda(data, factor_cols, project_tail, metadata);"
            )
            == 1
        )
        assert (
            f"geqrf{n}_blocked_make_policy_metadata_cuda(data, factor_cols, project_tail, metadata);"
            in auto_body
        )
        assert "auto metadata_cpu = metadata.cpu();" in auto_body
        assert "const int* metadata_ptr = metadata_cpu.data_ptr<int>();" in auto_body
        assert "int max_factor_cols = metadata_ptr[0];" in auto_body
        assert "bool any_project_tail = metadata_ptr[2] != 0;" in auto_body
        assert "if (metadata_ptr[5] != 0 && DENSE_TAIL_CUT > 0 && DENSE_TAIL_CUT < N) {" in auto_body
        assert "const int dense_factor_cols = N - DENSE_TAIL_CUT;" in auto_body
        assert "max_factor_cols = dense_factor_cols;" in auto_body
        assert "any_project_tail = true;" in auto_body
        assert "min_project_factor_cols = dense_factor_cols;" in auto_body
        assert "int min_project_factor_cols = metadata_ptr[4];" in auto_body
        assert "if (SYNC_FREE_AUTO_POLICY) {" in auto_body
        assert f"geqrf{n}_blocked_make_policy_cuda(data, factor_cols, project_tail);" in auto_body
        assert "N,\n            true,\n            (3 * N) / 4" in auto_body
        assert "max_factor_cols," in auto_body
        assert "any_project_tail," in auto_body
        assert "min_project_factor_cols" in auto_body
        assert f"geqrf{n}_blocked_policy_cuda(data, h, tau, factor_cols, project_tail, N, true, N);" not in source
        assert "if (project_tail && ((factor_cols < N) || per_matrix_policy))" in source
        assert "if (per_matrix_policy) {\n        factor_cols = N;" not in source
        assert "const size_t policy_shmem = size_t(8 * warp_count) * sizeof(float);" in source
        assert f"blocked{n}_policy_kernel<<<batch, threads, policy_shmem, stream>>>" in source
        assert f"blocked{n}_dense_tail_policy_kernel<<<adjust_blocks, threads, 0, stream>>>" in source
        assert "cudaMemsetAsync(has_structured.data_ptr<int>(), 0, sizeof(int), stream)" in source
        assert f"constexpr int DENSE_TAIL_CUT = {candidate._dense_tail_cut(n)};" in source
        assert f"constexpr int MIXED_DENSE_TAIL_CUT = {candidate._mixed_dense_tail_cut(n)};" in source
        assert f"constexpr float DENSE_TAIL_THRESHOLD = {candidate._dense_tail_threshold(n):.9e}f;" in source
        assert (
            f"constexpr float MIXED_DENSE_TAIL_THRESHOLD = {candidate._mixed_dense_tail_threshold(n):.9e}f;"
            in source
        )
        assert f"constexpr int DENSE_TAIL_FORCE = {1 if candidate._dense_tail_force(n) else 0};" in source
        assert "__MIXED_DENSE_TAIL_CUT__" not in source
        assert "__DENSE_TAIL_THRESHOLD__" not in source
        assert "__MIXED_DENSE_TAIL_THRESHOLD__" not in source
        assert "__DENSE_TAIL_FORCE__" not in source
        assert "const int no_structured = has_structured_batch[0] == 0;" in source
        assert "const int tail_cut = no_structured ? DENSE_TAIL_CUT : MIXED_DENSE_TAIL_CUT;" in source
        assert "const float tail_threshold = no_structured ? DENSE_TAIL_THRESHOLD : MIXED_DENSE_TAIL_THRESHOLD;" in source
        assert "const int tail_force = no_structured ? DENSE_TAIL_FORCE : 0;" in source
        assert "const int dense_factor_cols = N - tail_cut;" in source
        assert "tail / fmaxf(head, 1.0e-30f) < threshold" in source
        assert source.count("if (DENSE_TAIL_CUT > 0 || MIXED_DENSE_TAIL_CUT > 0)") == 2
        assert "factor_cols_by_batch[b] != N || project_tail_by_batch[b] != 0" in source
        assert "atomicMin(&metadata[4], dense_factor_cols);" in source
        assert f"__device__ __forceinline__ float blocked{n}_update_operand(float value)" in source
        assert f"__device__ __forceinline__ float blocked{n}_sum(float value, float* scratch)" in source
        assert f"__device__ __forceinline__ Blocked{n}Householder blocked{n}_make_reflector" in source
        assert "__shfl_down_sync" in sum_body
        assert "const int warp_count = (blockDim.x + 31) >> 5;" in sum_body
        assert "for (int stride = blockDim.x >> 1" not in sum_body
        assert f"__device__ __forceinline__ void blocked{n}_sum_tile(" in source
        assert f"blocked{n}_sum(cc < chunk_width" not in source
        assert "const int tile_reduce_floats = COMPACT_WY_TILE_COLS * warp_count;" in refresh_body
        assert "const int tile_reduce_floats = COMPACT_WY_TILE_COLS * warp_count;" in repair_body
        assert "float* refresh_vectors = scratch + tile_reduce_floats;" in refresh_body
        assert "float* repair_vectors = scratch + tile_reduce_floats;" in repair_body
        assert "float* prefix_dots = refresh_vectors + COMPACT_WY_TILE_COLS * N;" in refresh_body
        assert "float* prefix_dots = repair_vectors + COMPACT_WY_TILE_COLS * N;" in repair_body
        assert "chunk_col_start += COMPACT_WY_TILE_COLS" in refresh_body
        assert "chunk_col_start += COMPACT_WY_TILE_COLS" in repair_body
        assert "blocked" in refresh_body and "_apply_prefix_to_vector(" in refresh_body
        assert "blocked" in repair_body and "_apply_prefix_to_vector(" in repair_body
        assert "chunk_width," in refresh_body
        assert "chunk_width," in repair_body
        assert "scratch + blockDim.x" not in refresh_body
        assert "scratch + blockDim.x" not in repair_body
        assert host_body.count("at::cuda::getCurrentCUDAStream()") == 1
        assert "auto stream = at::cuda::getCurrentCUDAStream();" in host_body
        assert "const int64_t* indices" in source
        assert "indices," in host_body
        assert "nullptr," in wrapper_body
        assert "data.size(0)," in wrapper_body
        assert "indices.data_ptr<int64_t>()," in indexed_wrapper_body
        assert "indices.numel()," in indexed_wrapper_body
        assert "cudaMemsetAsync(h.data_ptr<float>()" not in host_body
        assert f"blocked{n}_copy_kernel<<<copy_blocks, threads, 0, stream>>>" in host_body
        assert f"blocked{n}_panel_refresh_kernel<<<batch, threads, panel_shmem, stream>>>" in host_body
        assert f"blocked{n}_panel_factor_kernel<<<batch, threads, tile_reduce_shmem, stream>>>" in host_body
        assert f"blocked{n}_trailing_update_kernel<<<grid, threads, trailing_shmem, stream>>>" in host_body
        assert f"blocked{n}_panel_r_repair_kernel<<<repair_grid, threads, panel_shmem, stream>>>" in host_body
        assert f"blocked{n}_tail_projection_kernel<<<grid, threads, tail_projection_shmem, stream>>>" in host_body
        assert "const int copy_col_end = (project_tail || factor_cols == N) ? N : factor_cols;" in host_body
        assert f"blocked{n}_update_operand" not in tail_body
        assert "const int tile_reduce_floats = COMPACT_WY_TILE_COLS * warp_count;" in tail_body
        assert "float* tail_dots = scratch + tile_reduce_floats;" in tail_body
        assert "chunk_col_start += COMPACT_WY_TILE_COLS" in tail_body
        assert "float dot_parts[COMPACT_WY_TILE_COLS];" in tail_body
        assert "dot_parts[cc] += v * h[" in tail_body
        assert f"blocked{n}_sum_tile(dot_parts, chunk_width, scratch, tail_dots);" in tail_body
        assert "tau_k * v * tail_dots[cc]" in tail_body
        assert f"blocked{n}_update_operand" in trailing_body
        assert "__USE_PANEL_REFRESH_PREFIX__" not in source
        assert "__USE_R_MAINTENANCE_PANEL_PREFIX__" not in source
        assert "__PANEL_REFRESH_PERIOD__" not in source
        assert "__R_MAINTENANCE_PERIOD__" not in source
        assert "__SYNC_FREE_AUTO_POLICY__" not in source
        assert f"blocked{n}_apply_prefix_to_vector" in source
        assert f"blocked{n}_panel_refresh_kernel" in source
        assert f"blocked{n}_panel_r_repair_kernel" in source
        assert "const int first_tile = blockIdx.x;" in repair_body
        assert "const int local_b = blockIdx.y;" in repair_body
        assert "for (int tile = first_tile; ; tile += gridDim.x)" in repair_body
        assert "const int tile_col_start = panel_start + tile * TILE_N;" in repair_body
        assert "const int tile_col_end = min(col_end, tile_col_start + TILE_N);" in repair_body
        assert "constexpr int USE_PANEL_REFRESH_PREFIX = 0;" in source
        assert "constexpr int USE_R_MAINTENANCE_PANEL_PREFIX = 0;" in source
        assert "constexpr int PANEL_REFRESH_PERIOD = 1;" in source
        assert "constexpr int R_MAINTENANCE_PERIOD = 1;" in source
        assert "constexpr int SYNC_FREE_AUTO_POLICY = 0;" in source
        assert "const int panel_index = panel_start / PANEL_B;" in host_body
        assert "((panel_index % PANEL_REFRESH_PERIOD) == 0)" in host_body
        assert "const int final_panel = panel_end >= factor_cols;" in host_body
        assert "const int next_panel_index = panel_index + 1;" in host_body
        assert "(final_panel || ((next_panel_index % R_MAINTENANCE_PERIOD) == 0))" in host_body
        assert "const int repair_col_tiles = (factor_cols - panel_start + TILE_N - 1) / TILE_N;" in host_body
        assert f"const int launch_repair_col_tiles = blocked{n}_launch_col_tiles(repair_col_tiles);" in host_body
        assert "dim3 repair_grid(launch_repair_col_tiles, batch);" in host_body
        assert "const int warp_count = (threads + 31) >> 5;" in host_body
        assert "const size_t reduce_shmem = size_t(warp_count) * sizeof(float);" in host_body
        assert "const size_t tile_reduce_shmem = size_t(COMPACT_WY_TILE_COLS * warp_count) * sizeof(float);" in host_body
        assert (
            "const size_t tail_projection_shmem = tile_reduce_shmem + size_t(COMPACT_WY_TILE_COLS) * sizeof(float);"
            in host_body
        )
        assert (
            "const size_t panel_shmem =\n        size_t(COMPACT_WY_TILE_COLS * warp_count + COMPACT_WY_TILE_COLS * N + COMPACT_WY_TILE_COLS) * sizeof(float);"
            in host_body
        )
        assert "const size_t panel_shmem = size_t(threads + N) * sizeof(float);" not in source
        assert "if (USE_PANEL_REFRESH_PREFIX && panel_shmem > 49152)" in host_body
        assert "if (USE_R_MAINTENANCE_PANEL_PREFIX && panel_shmem > 49152)" in host_body
        assert f"cudaFuncSetAttribute(\n            blocked{n}_panel_refresh_kernel" in host_body
        assert f"cudaFuncSetAttribute(\n            blocked{n}_panel_r_repair_kernel" in host_body
        assert source.index(f"blocked{n}_panel_refresh_kernel<<<") < source.index(f"blocked{n}_panel_factor_kernel<<<")
        assert source.index(f"blocked{n}_trailing_update_kernel<<<") < source.index(
            f"blocked{n}_panel_r_repair_kernel<<<"
        )


def test_candidate_blocked_cuda_consumes_shape_family_env_aliases(monkeypatch):
    candidate = _load_candidate_module()

    monkeypatch.setenv("FAST_QR_QR512_PANEL_B", "48")
    monkeypatch.setenv("FAST_QR_QR512_TILE_N", "16")
    monkeypatch.setenv("FAST_QR_QR512_COMPACT_WY_TILE_COLS", "6")
    monkeypatch.setenv("FAST_QR_QR512_WARPS_PER_CTA", "4")
    monkeypatch.setenv("FAST_QR_QR512_UPDATE_MODE", "compact-wy")
    monkeypatch.setenv("FAST_QR_QR512_PRECISION_MODE", "tf32")
    monkeypatch.setenv("FAST_QR_QR512_EXTRA_CUDA_CFLAGS", "--maxrregcount=128")
    assert candidate._qr512_blocked_cuda_ctas_per_matrix() == 1
    assert candidate._qr512_blocked_cuda_cta_schedule() == "fixed"
    pre_cta_key = candidate._qr512_blocked_cuda_extension_build_key()
    monkeypatch.setenv("FAST_QR_QR512_CTAS_PER_MATRIX", "2")
    monkeypatch.setenv("FAST_QR_QR512_CTA_SCHEDULE", "frontload")
    default_key = candidate._qr512_blocked_cuda_extension_build_key()

    assert candidate._qr512_blocked_cuda_panel_b() == 48
    assert candidate._qr512_blocked_cuda_tile_n() == 16
    assert candidate._qr512_blocked_cuda_compact_wy_tile_cols() == 6
    assert candidate._qr512_blocked_cuda_ctas_per_matrix() == 2
    assert candidate._qr512_blocked_cuda_cta_schedule() == "frontload"
    assert candidate._qr512_blocked_cuda_threads_per_cta() == 128
    assert candidate._qr512_blocked_cuda_update_mode() == "compact-wy"
    assert candidate._qr512_blocked_cuda_precision_mode() == "tf32-input"
    assert candidate._qr512_blocked_cuda_panel_refresh_mode() == "none"
    assert candidate._qr512_blocked_cuda_r_maintenance_mode() == "none"
    assert candidate._qr512_blocked_cuda_panel_refresh_period() == 1
    assert candidate._qr512_blocked_cuda_r_maintenance_period() == 1
    assert "--maxrregcount=128" in candidate._qr512_blocked_cuda_extra_cuda_cflags()
    source = candidate._qr512_blocked_cuda_source()
    assert "constexpr int PANEL_B = 48;" in source
    assert "constexpr int TILE_N = 16;" in source
    assert "constexpr int COMPACT_WY_TILE_COLS = 6;" in source
    assert "constexpr int CTAS_PER_MATRIX = 2;" in source
    assert "constexpr int CTA_SCHEDULE_FRONTLOAD = 1;" in source
    assert "constexpr int CTA_SCHEDULE_ALL_TILES = 0;" in source
    assert "constexpr int BLOCK_THREADS = 128;" in source
    assert "constexpr int USE_COMPACT_WY_UPDATE = 1;" in source
    assert "constexpr int USE_TF32_INPUT_UPDATE = 1;" in source
    assert "constexpr int USE_FP16_INPUT_UPDATE = 0;" in source
    assert "constexpr int USE_PANEL_REFRESH_PREFIX = 0;" in source
    assert "constexpr int USE_R_MAINTENANCE_PANEL_PREFIX = 0;" in source
    assert default_key != pre_cta_key

    monkeypatch.setenv("FAST_QR_QR512_BLOCKED_PANEL_B", "64")
    monkeypatch.setenv("FAST_QR_QR512_BLOCKED_TILE_N", "8")
    monkeypatch.setenv("FAST_QR_QR512_BLOCKED_COMPACT_WY_TILE_COLS", "5")
    monkeypatch.setenv("FAST_QR_QR512_BLOCKED_CTAS_PER_MATRIX", "3")
    monkeypatch.setenv("FAST_QR_QR512_BLOCKED_CTA_SCHEDULE", "all")
    monkeypatch.setenv("FAST_QR_QR512_BLOCKED_WARPS_PER_CTA", "8")
    monkeypatch.setenv("FAST_QR_QR512_BLOCKED_UPDATE_MODE", "reflectors")
    monkeypatch.setenv("FAST_QR_QR512_BLOCKED_PRECISION_MODE", "fp16")
    monkeypatch.setenv("FAST_QR_QR512_BLOCKED_PANEL_REFRESH_MODE", "prefix")
    monkeypatch.setenv("FAST_QR_QR512_BLOCKED_R_MAINTENANCE_MODE", "panel-prefix")
    monkeypatch.setenv("FAST_QR_QR512_BLOCKED_PANEL_REFRESH_PERIOD", "2")
    monkeypatch.setenv("FAST_QR_QR512_BLOCKED_R_MAINTENANCE_PERIOD", "3")
    monkeypatch.setenv("FAST_QR_QR512_BLOCKED_SYNC_FREE_AUTO_POLICY", "1")
    assert candidate._qr512_blocked_cuda_panel_b() == 64
    assert candidate._qr512_blocked_cuda_tile_n() == 8
    assert candidate._qr512_blocked_cuda_compact_wy_tile_cols() == 5
    assert candidate._qr512_blocked_cuda_ctas_per_matrix() == 3
    assert candidate._qr512_blocked_cuda_cta_schedule() == "all-tiles"
    assert candidate._qr512_blocked_cuda_threads_per_cta() == 256
    assert candidate._qr512_blocked_cuda_update_mode() == "reflectors"
    assert candidate._qr512_blocked_cuda_precision_mode() == "fp16-input"
    assert candidate._qr512_blocked_cuda_panel_refresh_mode() == "prefix"
    assert candidate._qr512_blocked_cuda_r_maintenance_mode() == "panel-prefix"
    assert candidate._qr512_blocked_cuda_panel_refresh_period() == 2
    assert candidate._qr512_blocked_cuda_r_maintenance_period() == 3
    assert candidate._qr512_blocked_cuda_sync_free_auto_policy()
    repair_source = candidate._qr512_blocked_cuda_source()
    assert "constexpr int COMPACT_WY_TILE_COLS = 5;" in repair_source
    assert "constexpr int CTAS_PER_MATRIX = 3;" in repair_source
    assert "constexpr int CTA_SCHEDULE_FRONTLOAD = 0;" in repair_source
    assert "constexpr int CTA_SCHEDULE_ALL_TILES = 1;" in repair_source
    assert "constexpr int USE_COMPACT_WY_UPDATE = 0;" in repair_source
    assert "constexpr int USE_TF32_INPUT_UPDATE = 0;" in repair_source
    assert "constexpr int USE_FP16_INPUT_UPDATE = 1;" in repair_source
    assert "constexpr int USE_PANEL_REFRESH_PREFIX = 1;" in repair_source
    assert "constexpr int USE_R_MAINTENANCE_PANEL_PREFIX = 1;" in repair_source
    assert "constexpr int PANEL_REFRESH_PERIOD = 2;" in repair_source
    assert "constexpr int R_MAINTENANCE_PERIOD = 3;" in repair_source
    assert "constexpr int SYNC_FREE_AUTO_POLICY = 1;" in repair_source
    assert candidate._qr512_blocked_cuda_extension_build_key() != default_key

    monkeypatch.setenv("FAST_QR_QR512_BLOCKED_PANEL_REFRESH_PERIOD", "0")
    assert candidate._qr512_blocked_cuda_panel_refresh_period() == 1

    monkeypatch.setenv("FAST_QR_QR512_BLOCKED_TILE_N", "128")
    assert candidate._qr512_blocked_cuda_tile_n() == 128

    monkeypatch.setenv("FAST_QR_QR1024_PANEL_REFRESH_MODE", "prefix")
    monkeypatch.setenv("FAST_QR_QR1024_R_MAINTENANCE_MODE", "panel-prefix")
    monkeypatch.setenv("FAST_QR_QR1024_PANEL_REFRESH_PERIOD", "4")
    monkeypatch.setenv("FAST_QR_QR1024_R_MAINTENANCE_PERIOD", "5")
    monkeypatch.setenv("FAST_QR_QR1024_TILE_N", "256")
    monkeypatch.setenv("FAST_QR_QR1024_COMPACT_WY_TILE_COLS", "7")
    monkeypatch.setenv("FAST_QR_QR1024_CTAS_PER_MATRIX", "4")
    monkeypatch.setenv("FAST_QR_QR1024_CTA_SCHEDULE", "frontload")
    monkeypatch.setenv("FAST_QR_QR1024_UPDATE_MODE", "compact-wy")
    assert candidate._qr1024_blocked_cuda_tile_n() == 256
    assert candidate._qr1024_blocked_cuda_compact_wy_tile_cols() == 7
    assert candidate._qr1024_blocked_cuda_ctas_per_matrix() == 4
    assert candidate._qr1024_blocked_cuda_cta_schedule() == "frontload"
    assert candidate._qr1024_blocked_cuda_update_mode() == "compact-wy"
    assert candidate._qr1024_blocked_cuda_panel_refresh_mode() == "prefix"
    assert candidate._qr1024_blocked_cuda_r_maintenance_mode() == "panel-prefix"
    assert candidate._qr1024_blocked_cuda_panel_refresh_period() == 4
    assert candidate._qr1024_blocked_cuda_r_maintenance_period() == 5
    assert "constexpr int USE_PANEL_REFRESH_PREFIX = 1;" in candidate._qr1024_blocked_cuda_source()
    assert "constexpr int USE_R_MAINTENANCE_PANEL_PREFIX = 1;" in candidate._qr1024_blocked_cuda_source()
    assert "constexpr int COMPACT_WY_TILE_COLS = 7;" in candidate._qr1024_blocked_cuda_source()
    assert "constexpr int CTAS_PER_MATRIX = 4;" in candidate._qr1024_blocked_cuda_source()
    assert "constexpr int CTA_SCHEDULE_FRONTLOAD = 1;" in candidate._qr1024_blocked_cuda_source()
    assert "constexpr int CTA_SCHEDULE_ALL_TILES = 0;" in candidate._qr1024_blocked_cuda_source()
    assert "constexpr int USE_COMPACT_WY_UPDATE = 1;" in candidate._qr1024_blocked_cuda_source()
    assert "constexpr int PANEL_REFRESH_PERIOD = 4;" in candidate._qr1024_blocked_cuda_source()
    assert "constexpr int R_MAINTENANCE_PERIOD = 5;" in candidate._qr1024_blocked_cuda_source()

    monkeypatch.setenv("FAST_QR_QR2048_PANEL_B", "96")
    monkeypatch.setenv("FAST_QR_QR2048_TILE_N", "256")
    monkeypatch.setenv("FAST_QR_QR2048_COMPACT_WY_TILE_COLS", "3")
    monkeypatch.setenv("FAST_QR_QR2048_CTAS_PER_MATRIX", "8")
    monkeypatch.setenv("FAST_QR_QR2048_CTA_SCHEDULE", "all-tiles")
    monkeypatch.setenv("FAST_QR_QR2048_WARPS_PER_CTA", "8")
    monkeypatch.setenv("FAST_QR_QR2048_UPDATE_MODE", "compact-wy")
    monkeypatch.setenv("FAST_QR_QR2048_PRECISION_MODE", "tf32-input")
    monkeypatch.setenv("FAST_QR_QR2048_PANEL_REFRESH_MODE", "prefix")
    monkeypatch.setenv("FAST_QR_QR2048_R_MAINTENANCE_MODE", "panel-prefix")
    monkeypatch.setenv("FAST_QR_BLOCKED_PANEL_REFRESH_PERIOD", "6")
    monkeypatch.setenv("FAST_QR_BLOCKED_R_MAINTENANCE_PERIOD", "7")
    assert candidate._generic_blocked_cuda_panel_b(2048) == 96
    assert candidate._generic_blocked_cuda_tile_n(2048) == 256
    assert candidate._generic_blocked_cuda_compact_wy_tile_cols(2048) == 3
    assert candidate._generic_blocked_cuda_ctas_per_matrix(2048) == 8
    assert candidate._generic_blocked_cuda_cta_schedule(2048) == "all-tiles"
    assert candidate._qr2048_blocked_cuda_ctas_per_matrix() == 8
    assert candidate._qr2048_blocked_cuda_cta_schedule() == "all-tiles"
    assert candidate._qr2048_blocked_cuda_compact_wy_tile_cols() == 3
    assert candidate._generic_blocked_cuda_threads_per_cta(2048) == 256
    assert candidate._generic_blocked_cuda_update_mode(2048) == "compact-wy"
    assert candidate._qr2048_blocked_cuda_update_mode() == "compact-wy"
    assert candidate._generic_blocked_cuda_precision_mode(2048) == "tf32-input"
    assert candidate._generic_blocked_cuda_panel_refresh_mode(2048) == "prefix"
    assert candidate._generic_blocked_cuda_r_maintenance_mode(2048) == "panel-prefix"
    assert candidate._generic_blocked_cuda_panel_refresh_period(2048) == 6
    assert candidate._generic_blocked_cuda_r_maintenance_period(2048) == 7
    assert not candidate._generic_blocked_cuda_sync_free_auto_policy(2048)
    assert "constexpr int COMPACT_WY_TILE_COLS = 3;" in candidate._generic_blocked_cuda_source(2048)
    assert "constexpr int CTAS_PER_MATRIX = 8;" in candidate._generic_blocked_cuda_source(2048)
    assert "constexpr int CTA_SCHEDULE_FRONTLOAD = 0;" in candidate._generic_blocked_cuda_source(2048)
    assert "constexpr int CTA_SCHEDULE_ALL_TILES = 1;" in candidate._generic_blocked_cuda_source(2048)
    assert "constexpr int USE_COMPACT_WY_UPDATE = 1;" in candidate._generic_blocked_cuda_source(2048)
    assert candidate._qr2048_blocked_cuda_panel_refresh_period() == 6
    assert candidate._qr2048_blocked_cuda_r_maintenance_period() == 7
    assert not candidate._qr2048_blocked_cuda_sync_free_auto_policy()

    monkeypatch.setenv("FAST_QR_ENABLE_BLOCKED_SYNC_FREE_AUTO_POLICY", "1")
    assert candidate._qr1024_blocked_cuda_sync_free_auto_policy()
    assert candidate._generic_blocked_cuda_sync_free_auto_policy(2048)
    assert candidate._qr2048_blocked_cuda_sync_free_auto_policy()
    assert "constexpr int SYNC_FREE_AUTO_POLICY = 1;" in candidate._qr1024_blocked_cuda_source()
    assert "constexpr int SYNC_FREE_AUTO_POLICY = 1;" in candidate._generic_blocked_cuda_source(2048)

    monkeypatch.setenv("FAST_QR_QR1024_BLOCKED_SYNC_FREE_AUTO_POLICY", "0")
    assert not candidate._qr1024_blocked_cuda_sync_free_auto_policy()

    monkeypatch.setenv("FAST_QR_QR4096_TILE_N", "512")
    monkeypatch.setenv("FAST_QR_QR4096_BLOCKED_COMPACT_WY_TILE_COLS", "2")
    monkeypatch.setenv("FAST_QR_QR4096_BLOCKED_CTAS_PER_MATRIX", "16")
    monkeypatch.setenv("FAST_QR_QR4096_BLOCKED_CTA_SCHEDULE", "fixed")
    monkeypatch.setenv("FAST_QR_QR4096_BLOCKED_UPDATE_MODE", "compact-wy")
    assert candidate._generic_blocked_cuda_tile_n(4096) == 512
    assert candidate._qr4096_blocked_cuda_compact_wy_tile_cols() == 2
    assert candidate._qr4096_blocked_cuda_ctas_per_matrix() == 16
    assert candidate._qr4096_blocked_cuda_cta_schedule() == "fixed"
    assert candidate._qr4096_blocked_cuda_update_mode() == "compact-wy"


def test_candidate_blocked_cuda_default_tiles_and_cta_schedules_match_kernel_policy(monkeypatch):
    candidate = _load_candidate_module()
    for env_name in (
        "FAST_QR_QR512_BLOCKED_TILE_N",
        "FAST_QR_QR512_TILE_N",
        "FAST_QR_QR512_BLOCKED_CTAS_PER_MATRIX",
        "FAST_QR_QR512_CTAS_PER_MATRIX",
        "FAST_QR_QR512_BLOCKED_CTA_SCHEDULE",
        "FAST_QR_QR512_CTA_SCHEDULE",
        "FAST_QR_QR1024_BLOCKED_TILE_N",
        "FAST_QR_QR1024_TILE_N",
        "FAST_QR_QR1024_BLOCKED_CTAS_PER_MATRIX",
        "FAST_QR_QR1024_CTAS_PER_MATRIX",
        "FAST_QR_QR1024_BLOCKED_CTA_SCHEDULE",
        "FAST_QR_QR1024_CTA_SCHEDULE",
        "FAST_QR_QR2048_BLOCKED_TILE_N",
        "FAST_QR_QR2048_TILE_N",
        "FAST_QR_QR2048_BLOCKED_CTAS_PER_MATRIX",
        "FAST_QR_QR2048_CTAS_PER_MATRIX",
        "FAST_QR_QR2048_BLOCKED_CTA_SCHEDULE",
        "FAST_QR_QR2048_CTA_SCHEDULE",
        "FAST_QR_QR4096_BLOCKED_TILE_N",
        "FAST_QR_QR4096_TILE_N",
        "FAST_QR_QR4096_BLOCKED_CTAS_PER_MATRIX",
        "FAST_QR_QR4096_CTAS_PER_MATRIX",
        "FAST_QR_QR4096_BLOCKED_CTA_SCHEDULE",
        "FAST_QR_QR4096_CTA_SCHEDULE",
        "FAST_QR_BLOCKED_CTA_SCHEDULE",
    ):
        monkeypatch.delenv(env_name, raising=False)

    assert candidate._qr512_blocked_cuda_tile_n() == 64
    assert candidate._qr512_blocked_cuda_ctas_per_matrix() == 1
    assert candidate._qr512_blocked_cuda_cta_schedule() == "fixed"
    assert candidate._qr1024_blocked_cuda_tile_n() == 128
    assert candidate._qr1024_blocked_cuda_ctas_per_matrix() == 1
    assert candidate._qr1024_blocked_cuda_cta_schedule() == "fixed"
    assert candidate._generic_blocked_cuda_tile_n(2048) == 128
    assert candidate._generic_blocked_cuda_ctas_per_matrix(2048) == 4
    assert candidate._generic_blocked_cuda_cta_schedule(2048) == "frontload"
    assert candidate._generic_blocked_cuda_tile_n(4096) == 256
    assert candidate._generic_blocked_cuda_ctas_per_matrix(4096) == 8
    assert candidate._generic_blocked_cuda_cta_schedule(4096) == "frontload"

    assert "constexpr int TILE_N = 64;" in candidate._qr512_blocked_cuda_source()
    assert "constexpr int CTAS_PER_MATRIX = 1;" in candidate._qr512_blocked_cuda_source()
    assert "constexpr int CTA_SCHEDULE_FRONTLOAD = 0;" in candidate._qr512_blocked_cuda_source()
    assert "constexpr int CTA_SCHEDULE_ALL_TILES = 0;" in candidate._qr512_blocked_cuda_source()
    assert "constexpr int TILE_N = 128;" in candidate._qr1024_blocked_cuda_source()
    assert "constexpr int CTAS_PER_MATRIX = 1;" in candidate._qr1024_blocked_cuda_source()
    assert "constexpr int CTA_SCHEDULE_FRONTLOAD = 0;" in candidate._qr1024_blocked_cuda_source()
    assert "constexpr int CTA_SCHEDULE_ALL_TILES = 0;" in candidate._qr1024_blocked_cuda_source()
    assert "constexpr int TILE_N = 128;" in candidate._generic_blocked_cuda_source(2048)
    assert "constexpr int CTAS_PER_MATRIX = 4;" in candidate._generic_blocked_cuda_source(2048)
    assert "constexpr int CTA_SCHEDULE_FRONTLOAD = 1;" in candidate._generic_blocked_cuda_source(2048)
    assert "constexpr int CTA_SCHEDULE_ALL_TILES = 0;" in candidate._generic_blocked_cuda_source(2048)
    assert "constexpr int TILE_N = 256;" in candidate._generic_blocked_cuda_source(4096)
    assert "constexpr int CTAS_PER_MATRIX = 8;" in candidate._generic_blocked_cuda_source(4096)
    assert "constexpr int CTA_SCHEDULE_FRONTLOAD = 1;" in candidate._generic_blocked_cuda_source(4096)
    assert "constexpr int CTA_SCHEDULE_ALL_TILES = 0;" in candidate._generic_blocked_cuda_source(4096)


def test_candidate_blocked_cuda_build_key_tracks_abi_version(monkeypatch):
    candidate = _load_candidate_module()

    original = (
        candidate._qr512_blocked_cuda_extension_build_key(),
        candidate._qr1024_blocked_cuda_extension_build_key(),
        candidate._generic_blocked_cuda_extension_build_key(2048),
    )
    monkeypatch.setattr(candidate, "_BLOCKED_CUDA_ABI_VERSION", "blocked-indexed-v2")

    assert candidate._qr512_blocked_cuda_extension_build_key() != original[0]
    assert candidate._qr1024_blocked_cuda_extension_build_key() != original[1]
    assert candidate._generic_blocked_cuda_extension_build_key(2048) != original[2]


def test_candidate_blocked_auto_policy_source_uses_dense_tail_env(monkeypatch):
    candidate = _load_candidate_module()

    assert "constexpr int DENSE_TAIL_CUT = 32;" in candidate._qr512_blocked_cuda_source()
    assert "constexpr int MIXED_DENSE_TAIL_CUT = 0;" in candidate._qr512_blocked_cuda_source()
    assert "constexpr float DENSE_TAIL_THRESHOLD = 3.000000000e-02f;" in candidate._qr512_blocked_cuda_source()
    assert "constexpr float MIXED_DENSE_TAIL_THRESHOLD = 0.000000000e+00f;" in candidate._qr512_blocked_cuda_source()
    assert "constexpr int DENSE_TAIL_FORCE = 0;" in candidate._qr512_blocked_cuda_source()
    monkeypatch.setenv("FAST_QR_DENSE_TAIL_CUT_512", "12")
    assert "constexpr int DENSE_TAIL_CUT = 12;" in candidate._qr512_blocked_cuda_source()
    monkeypatch.setenv("FAST_QR_DENSE_TAIL_THRESHOLD_512", "0.125")
    assert "constexpr float DENSE_TAIL_THRESHOLD = 1.250000000e-01f;" in candidate._qr512_blocked_cuda_source()
    monkeypatch.setenv("FAST_QR_DENSE_TAIL_FORCE_512", "1")
    assert "constexpr int DENSE_TAIL_FORCE = 1;" in candidate._qr512_blocked_cuda_source()

    assert "constexpr int DENSE_TAIL_CUT = 64;" in candidate._qr1024_blocked_cuda_source()
    assert "constexpr int MIXED_DENSE_TAIL_CUT = 8;" in candidate._qr1024_blocked_cuda_source()
    assert "constexpr float MIXED_DENSE_TAIL_THRESHOLD = 2.000000000e-02f;" in candidate._qr1024_blocked_cuda_source()
    monkeypatch.setenv("FAST_QR_QR1024_TAIL_CUT", "8")
    assert "constexpr int DENSE_TAIL_CUT = 8;" in candidate._qr1024_blocked_cuda_source()
    monkeypatch.setenv("FAST_QR_MIXED_DENSE_TAIL_CUT_1024", "4")
    assert "constexpr int MIXED_DENSE_TAIL_CUT = 4;" in candidate._qr1024_blocked_cuda_source()
    monkeypatch.setenv("FAST_QR_MIXED_DENSE_TAIL_THRESHOLD_1024", "0.03125")
    assert "constexpr float MIXED_DENSE_TAIL_THRESHOLD = 3.125000000e-02f;" in candidate._qr1024_blocked_cuda_source()


def test_candidate_blocked_auto_policy_source_uses_sample_rows_env(monkeypatch):
    candidate = _load_candidate_module()

    default_keys = (
        candidate._qr512_blocked_cuda_extension_build_key(),
        candidate._qr1024_blocked_cuda_extension_build_key(),
        candidate._generic_blocked_cuda_extension_build_key(2048),
    )

    assert candidate._qr512_blocked_cuda_policy_sample_rows() == 8
    assert candidate._qr1024_blocked_cuda_policy_sample_rows() == 8
    assert candidate._generic_blocked_cuda_policy_sample_rows(2048) == 8
    assert "constexpr int POLICY_RANDOM_ROWS = 8;" in candidate._qr512_blocked_cuda_source()
    assert "constexpr int POLICY_RANDOM_ROWS = 8;" in candidate._qr1024_blocked_cuda_source()
    assert "constexpr int POLICY_RANDOM_ROWS = 8;" in candidate._generic_blocked_cuda_source(2048)
    assert "__POLICY_RANDOM_ROWS__" not in candidate._qr512_blocked_cuda_source()

    monkeypatch.setenv("FAST_QR_QR512_BLOCKED_POLICY_SAMPLE_ROWS", "12")
    assert candidate._qr512_blocked_cuda_policy_sample_rows() == 12
    assert candidate._qr1024_blocked_cuda_policy_sample_rows() == 8
    assert "constexpr int POLICY_RANDOM_ROWS = 12;" in candidate._qr512_blocked_cuda_source()
    assert candidate._qr512_blocked_cuda_extension_build_key() != default_keys[0]

    monkeypatch.delenv("FAST_QR_QR512_BLOCKED_POLICY_SAMPLE_ROWS")
    monkeypatch.setenv("FAST_QR_BLOCKED_POLICY_SAMPLE_ROWS", "6")
    assert candidate._qr512_blocked_cuda_policy_sample_rows() == 6
    assert candidate._qr1024_blocked_cuda_policy_sample_rows() == 6
    assert candidate._generic_blocked_cuda_policy_sample_rows(2048) == 6
    assert "constexpr int POLICY_RANDOM_ROWS = 6;" in candidate._generic_blocked_cuda_source(2048)
    assert candidate._qr1024_blocked_cuda_extension_build_key() != default_keys[1]
    assert candidate._generic_blocked_cuda_extension_build_key(2048) != default_keys[2]

    monkeypatch.setenv("FAST_QR_QR1024_POLICY_SAMPLE_ROWS", "10")
    assert candidate._qr1024_blocked_cuda_policy_sample_rows() == 10
    assert "constexpr int POLICY_RANDOM_ROWS = 10;" in candidate._qr1024_blocked_cuda_source()

    monkeypatch.setenv("FAST_QR_QR512_POLICY_SAMPLE_ROWS", "0")
    assert candidate._qr512_blocked_cuda_policy_sample_rows() == 6


def test_candidate_blocked_auto_policy_source_uses_full_scan_env(monkeypatch):
    candidate = _load_candidate_module()

    monkeypatch.setenv("FAST_QR_BLOCKED_POLICY_FULL_SCAN", "0")
    default_keys = (
        candidate._qr512_blocked_cuda_extension_build_key(),
        candidate._qr1024_blocked_cuda_extension_build_key(),
        candidate._generic_blocked_cuda_extension_build_key(2048),
    )

    assert not candidate._qr512_blocked_cuda_policy_full_scan()
    assert not candidate._qr1024_blocked_cuda_policy_full_scan()
    assert not candidate._generic_blocked_cuda_policy_full_scan(2048)
    qr512_sparse_source = candidate._qr512_blocked_cuda_source()
    assert "constexpr int USE_FULL_POLICY_SCAN = 0;" in qr512_sparse_source
    assert "constexpr int USE_FULL_POLICY_SCAN = 0;" in candidate._generic_blocked_cuda_source(2048)
    assert "__USE_FULL_POLICY_SCAN__" not in qr512_sparse_source
    dense_tail_body = qr512_sparse_source.split(
        "__device__ __forceinline__ int blocked512_dense_tail_allowed(", 1
    )[1].split("__global__ void __launch_bounds__(BLOCK_THREADS) blocked512_dense_tail_policy_kernel", 1)[0]
    assert "if (USE_FULL_POLICY_SCAN) {" in dense_tail_body
    assert "row_slot < POLICY_RANDOM_ROWS + 1" in dense_tail_body
    assert "const int row = (row_slot == 0) ? tail_col" in dense_tail_body
    assert "for (int row = 0; row < N; ++row)" in dense_tail_body

    monkeypatch.setenv("FAST_QR_QR512_BLOCKED_POLICY_FULL_SCAN", "1")
    assert candidate._qr512_blocked_cuda_policy_full_scan()
    assert not candidate._qr1024_blocked_cuda_policy_full_scan()
    assert "constexpr int USE_FULL_POLICY_SCAN = 1;" in candidate._qr512_blocked_cuda_source()
    assert candidate._qr512_blocked_cuda_extension_build_key() != default_keys[0]

    monkeypatch.delenv("FAST_QR_QR512_BLOCKED_POLICY_FULL_SCAN")
    monkeypatch.setenv("FAST_QR_BLOCKED_POLICY_FULL_SCAN", "1")
    assert candidate._qr512_blocked_cuda_policy_full_scan()
    assert candidate._qr1024_blocked_cuda_policy_full_scan()
    assert candidate._generic_blocked_cuda_policy_full_scan(2048)
    assert "constexpr int USE_FULL_POLICY_SCAN = 1;" in candidate._qr1024_blocked_cuda_source()
    assert candidate._qr1024_blocked_cuda_extension_build_key() != default_keys[1]
    assert candidate._generic_blocked_cuda_extension_build_key(2048) != default_keys[2]

    monkeypatch.setenv("FAST_QR_QR1024_POLICY_FULL_SCAN", "0")
    assert not candidate._qr1024_blocked_cuda_policy_full_scan()
    assert "constexpr int USE_FULL_POLICY_SCAN = 0;" in candidate._qr1024_blocked_cuda_source()


def test_candidate_large_cuda_loader_state_tracks_compile_knobs(monkeypatch):
    candidate = _load_candidate_module()

    sentinel512 = object()
    candidate._QR512_CUDA_EXTENSION = sentinel512
    candidate._QR512_CUDA_EXTENSION_STATE = candidate._qr512_cuda_loader_state()
    assert candidate._load_qr512_cuda_extension() is sentinel512

    monkeypatch.setenv("FAST_QR_QR512_WARPS_PER_CTA", "16")
    assert candidate._load_qr512_cuda_extension() is None
    assert candidate._QR512_CUDA_EXTENSION is None
    assert candidate._QR512_CUDA_EXTENSION_FAILED is True
    assert candidate._QR512_CUDA_EXTENSION_FAILED_STATE == candidate._qr512_cuda_loader_state()
    assert "requires CUDA" in candidate._QR512_CUDA_EXTENSION_ERROR

    monkeypatch.setenv("FAST_QR_DISABLE_QR512_CUDA", "1")
    assert candidate._load_qr512_cuda_extension() is None
    assert "disabled" in candidate._QR512_CUDA_EXTENSION_ERROR

    monkeypatch.delenv("FAST_QR_DISABLE_QR512_CUDA")
    assert candidate._load_qr512_cuda_extension() is None
    assert "requires CUDA" in candidate._QR512_CUDA_EXTENSION_ERROR

    sentinel1024 = object()
    candidate._QR1024_CUDA_EXTENSION = sentinel1024
    candidate._QR1024_CUDA_EXTENSION_STATE = candidate._qr1024_cuda_loader_state()
    assert candidate._load_qr1024_cuda_extension() is sentinel1024

    monkeypatch.setenv("FAST_QR_QR1024_THREADS_PER_CTA", "512")
    assert candidate._load_qr1024_cuda_extension() is None
    assert candidate._QR1024_CUDA_EXTENSION is None
    assert candidate._QR1024_CUDA_EXTENSION_FAILED is True
    assert candidate._QR1024_CUDA_EXTENSION_FAILED_STATE == candidate._qr1024_cuda_loader_state()
    assert "requires CUDA" in candidate._QR1024_CUDA_EXTENSION_ERROR


def test_candidate_generic_blocked_loader_exports_indexed_entrypoint(monkeypatch):
    candidate = _load_candidate_module()
    captured = {}
    sentinel = object()
    fake_data = SimpleNamespace(device=torch.device("cuda", 0))

    def fake_load_inline(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setenv("FAST_QR_ENABLE_QR2048_BLOCKED_CUDA", "1")
    monkeypatch.setattr(candidate.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr("torch.utils.cpp_extension.load_inline", fake_load_inline)

    assert candidate._load_generic_blocked_cuda_extension(2048, fake_data) is sentinel
    assert captured["functions"] == [
        "geqrf2048_blocked",
        "geqrf2048_blocked_indexed",
        "geqrf2048_blocked_auto",
        "geqrf2048_blocked_auto_workspace",
        "geqrf2048_blocked_make_policy",
        "geqrf2048_blocked_make_policy_metadata",
        "geqrf2048_blocked_policy",
    ]


def test_candidate_generic_blocked_try_into_calls_indexed_entrypoint(monkeypatch):
    candidate = _load_candidate_module()
    calls = []
    device = torch.device("cpu")
    data = SimpleNamespace(is_cuda=True, dtype=torch.float32, ndim=3, shape=(8, 2048, 2048), device=device)
    h = object()
    tau = object()

    class FakeIndex:
        dtype = torch.long

        def __init__(self, index_device):
            self.device = index_device

        def numel(self):
            return 2

        def contiguous(self):
            calls.append(("contiguous",))
            return self

    class FakeExtension:
        def geqrf2048_blocked_indexed(self, data_arg, h_arg, tau_arg, idx_arg, factor_cols, project_tail):
            calls.append((data_arg, h_arg, tau_arg, idx_arg, int(factor_cols), bool(project_tail)))

    monkeypatch.setattr(candidate, "_generic_blocked_cuda_required", lambda _n: False)
    monkeypatch.setattr(candidate, "_load_generic_blocked_cuda_extension", lambda _n, _data: FakeExtension())

    idx = FakeIndex(device)
    assert candidate._generic_blocked_cuda_try_into(data, h, tau, idx, 2048, factor_cols=1536, project_tail=True)
    assert calls == [
        ("contiguous",),
        (data, h, tau, idx, 1536, True),
    ]


@pytest.mark.parametrize("sync_free", [False, True])
def test_candidate_generic_blocked_auto_try_calls_auto_entrypoint(monkeypatch, sync_free):
    candidate = _load_candidate_module()
    calls = []
    device = torch.device("cpu")
    data = SimpleNamespace(is_cuda=True, dtype=torch.float32, ndim=3, shape=(8, 2048, 2048), device=device)

    class FakeExtension:
        def geqrf2048_blocked_auto(self, data_arg, h_arg, tau_arg):
            calls.append((data_arg, h_arg.shape, h_arg.stride(), tau_arg.shape))
            h_arg[:, :, :] = 11.0
            tau_arg[:, :] = 5.0

    monkeypatch.setattr(candidate, "_generic_blocked_cuda_required", lambda _n: False)
    monkeypatch.setattr(candidate, "_generic_blocked_cuda_sync_free_auto_policy", lambda _n: sync_free)
    monkeypatch.setattr(candidate, "_load_generic_blocked_cuda_extension", lambda _n, _data: FakeExtension())

    h, tau = candidate._generic_blocked_cuda_auto_try(data, 2048)

    assert calls == [(data, (8, 2048, 2048), (2048 * 2048, 1, 2048), (8, 2048))]
    assert h[0, 0, 0].item() == 11.0
    assert tau[0, 0].item() == 5.0


def test_candidate_generic_blocked_auto_try_prefers_sync_free_workspace_entrypoint(monkeypatch):
    candidate = _load_candidate_module()
    n = 4096
    data = SimpleNamespace(is_cuda=True, dtype=torch.float32, ndim=3, shape=(2, n, n), device=torch.device("cpu"))
    calls = []

    class FakeExtension:
        def geqrf4096_blocked_auto_workspace(
            self,
            data_arg,
            h_arg,
            tau_arg,
            factor_cols_arg,
            project_tail_arg,
        ):
            calls.append(
                (
                    "workspace",
                    data_arg,
                    h_arg.shape,
                    h_arg.stride(),
                    tau_arg.shape,
                    factor_cols_arg.shape,
                    factor_cols_arg.dtype,
                    project_tail_arg.shape,
                    project_tail_arg.dtype,
                )
            )
            h_arg[:, :, :] = 19.0
            tau_arg[:, :] = 23.0

        def geqrf4096_blocked_auto(self, *_args):
            raise AssertionError("sync-free generic auto should use preallocated policy workspace when available")

    monkeypatch.setattr(candidate, "_generic_blocked_cuda_required", lambda _n: False)
    monkeypatch.setattr(candidate, "_generic_blocked_cuda_sync_free_auto_policy", lambda _n: True)
    monkeypatch.setattr(candidate, "_load_generic_blocked_cuda_extension", lambda _n, _data: FakeExtension())

    h, tau = candidate._generic_blocked_cuda_auto_try(data, n)

    assert calls == [
        (
            "workspace",
            data,
            (2, n, n),
            (n * n, 1, n),
            (2, n),
            (2,),
            torch.int32,
            (2,),
            torch.int32,
        )
    ]
    assert h[0, 0, 0].item() == 19.0
    assert tau[0, 0].item() == 23.0


def test_candidate_generic_blocked_auto_try_prefers_policy_metadata(monkeypatch):
    candidate = _load_candidate_module()
    candidate._BLOCKED_AUTO_POLICY_CACHE.clear()
    n = 2048

    class FakeData:
        is_cuda = True
        dtype = torch.float32
        ndim = 3
        shape = (2, n, n)
        device = torch.device("cpu")
        _version = 0

    data = FakeData()
    calls = []

    class FakeExtension:
        pass

    extension = FakeExtension()

    def fake_make_policy(*_args):
        raise AssertionError("metadata policy path should replace separate make-policy reductions")

    def fake_make_policy_metadata(data_arg, factor_cols_arg, project_tail_arg, metadata_arg):
        factor_cols_arg[0] = n - 96
        factor_cols_arg[1] = n - 64
        project_tail_arg[0] = 0
        project_tail_arg[1] = 1
        metadata_arg.copy_(torch.tensor([n - 64, n - 96, 1, 0, n - 64, 0], dtype=torch.int32))
        calls.append(("metadata", data_arg, factor_cols_arg, project_tail_arg, metadata_arg))

    def fake_policy(
        data_arg,
        h_arg,
        tau_arg,
        factor_cols_arg,
        project_tail_arg,
        max_factor_cols_arg,
        any_project_tail_arg,
        min_project_factor_cols_arg,
    ):
        calls.append(
            (
                "policy",
                data_arg,
                h_arg.shape,
                h_arg.stride(),
                tau_arg.shape,
                factor_cols_arg,
                project_tail_arg,
                max_factor_cols_arg,
                any_project_tail_arg,
                min_project_factor_cols_arg,
            )
        )

    setattr(extension, "geqrf2048_blocked_make_policy", fake_make_policy)
    setattr(extension, "geqrf2048_blocked_make_policy_metadata", fake_make_policy_metadata)
    setattr(extension, "geqrf2048_blocked_policy", fake_policy)
    monkeypatch.setattr(candidate, "_generic_blocked_cuda_required", lambda _n: False)
    monkeypatch.setattr(candidate, "_generic_blocked_cuda_sync_free_auto_policy", lambda _n: False)
    monkeypatch.setattr(candidate, "_load_generic_blocked_cuda_extension", lambda _n, _data: extension)

    h, tau = candidate._generic_blocked_cuda_auto_try(data, n)
    h2, tau2 = candidate._generic_blocked_cuda_auto_try(data, n)

    assert h.shape == (2, n, n)
    assert h.stride() == (n * n, 1, n)
    assert tau.shape == (2, n)
    assert h2.shape == (2, n, n)
    assert tau2.shape == (2, n)
    assert [call[0] for call in calls] == ["metadata", "policy", "policy"]
    assert calls[1][-5] is calls[0][2]
    assert calls[1][-4] is calls[0][3]
    assert calls[1][-3] == n - 64
    assert calls[1][-2] is True
    assert calls[1][-1] == n - 64
    assert calls[2][-5] is calls[0][2]
    assert calls[2][-4] is calls[0][3]
    assert calls[2][-3] == n - 64
    assert calls[2][-2] is True
    assert calls[2][-1] == n - 64


def test_candidate_generic_blocked_auto_try_uses_homogeneous_policy_fast_path(monkeypatch):
    candidate = _load_candidate_module()
    candidate._BLOCKED_AUTO_POLICY_CACHE.clear()
    n = 4096
    rank = candidate._rankdef_effective_cols(n)

    class FakeData:
        is_cuda = True
        dtype = torch.float32
        ndim = 3
        shape = (2, n, n)
        device = torch.device("cpu")
        _version = 0

    data = FakeData()
    calls = []

    class FakeExtension:
        pass

    extension = FakeExtension()

    def fake_make_policy_metadata(data_arg, factor_cols_arg, project_tail_arg, metadata_arg):
        factor_cols_arg[:] = rank
        project_tail_arg[:] = 1
        metadata_arg.copy_(torch.tensor([rank, rank, 1, 1, rank, 0], dtype=torch.int32))
        calls.append(("metadata", data_arg, factor_cols_arg, project_tail_arg))

    def fake_blocked(data_arg, h_arg, tau_arg, factor_cols_arg, project_tail_arg):
        calls.append(("blocked", data_arg, int(factor_cols_arg), bool(project_tail_arg)))
        h_arg[:, :, :] = 13.0
        tau_arg[:, :] = 17.0

    def fail_policy(*_args):
        raise AssertionError("homogeneous policy should use the direct blocked fast path")

    setattr(extension, "geqrf4096_blocked_make_policy_metadata", fake_make_policy_metadata)
    setattr(extension, "geqrf4096_blocked_policy", fail_policy)
    setattr(extension, "geqrf4096_blocked", fake_blocked)
    monkeypatch.setattr(candidate, "_generic_blocked_cuda_required", lambda _n: False)
    monkeypatch.setattr(candidate, "_generic_blocked_cuda_sync_free_auto_policy", lambda _n: False)
    monkeypatch.setattr(candidate, "_load_generic_blocked_cuda_extension", lambda _n, _data: extension)

    h, tau = candidate._generic_blocked_cuda_auto_try(data, n)

    assert [call[0] for call in calls] == ["metadata", "blocked"]
    assert calls[1][2:] == (rank, True)
    assert h[0, 0, 0].item() == 13.0
    assert tau[0, 0].item() == 17.0


def test_candidate_generic_blocked_auto_try_groups_nonhomogeneous_policy_with_indexed_calls(monkeypatch):
    candidate = _load_candidate_module()
    candidate._BLOCKED_AUTO_POLICY_CACHE.clear()
    n = 2048
    rank = candidate._rankdef_effective_cols(n)
    clustered = candidate._clustered_effective_cols(n)
    dense_tail = n - candidate._dense_tail_cut(n)

    class FakeData:
        is_cuda = True
        dtype = torch.float32
        ndim = 3
        shape = (5, n, n)
        device = torch.device("cpu")
        _version = 0

    data = FakeData()
    calls = []

    class FakeExtension:
        pass

    extension = FakeExtension()

    def fake_make_policy_metadata(data_arg, factor_cols_arg, project_tail_arg, metadata_arg):
        factor_cols_arg.copy_(torch.tensor([rank, clustered, rank, dense_tail, n], dtype=torch.int32))
        project_tail_arg.copy_(torch.tensor([0, 0, 1, 1, 0], dtype=torch.int32))
        metadata_arg.copy_(torch.tensor([n, clustered, 1, 0, min(rank, dense_tail), 0], dtype=torch.int32))
        calls.append(("metadata", data_arg, factor_cols_arg.clone(), project_tail_arg.clone()))

    def fake_policy(*_args):
        raise AssertionError("grouped generic auto policy should use indexed blocked calls")

    def fake_indexed(data_arg, h_arg, tau_arg, idx_arg, factor_cols_arg, project_tail_arg):
        calls.append(("indexed", data_arg, idx_arg.tolist(), int(factor_cols_arg), bool(project_tail_arg)))
        h_arg[idx_arg, :, :] = float(len(calls))
        tau_arg[idx_arg, :] = float(len(calls))

    setattr(extension, "geqrf2048_blocked_make_policy_metadata", fake_make_policy_metadata)
    setattr(extension, "geqrf2048_blocked_policy", fake_policy)
    setattr(extension, "geqrf2048_blocked_indexed", fake_indexed)
    monkeypatch.setattr(candidate, "_generic_blocked_cuda_required", lambda _n: False)
    monkeypatch.setattr(candidate, "_generic_blocked_cuda_sync_free_auto_policy", lambda _n: False)
    monkeypatch.setattr(candidate, "_blocked_auto_policy_grouping_enabled", lambda _data, shape_n: shape_n == n)
    monkeypatch.setattr(candidate, "_load_generic_blocked_cuda_extension", lambda _n, _data: extension)

    h, tau = candidate._generic_blocked_cuda_auto_try(data, n)

    assert h.shape == (5, n, n)
    assert tau.shape == (5, n)
    assert [call[0] for call in calls] == ["metadata", "indexed", "indexed", "indexed", "indexed", "indexed"]
    assert calls[1][2:] == ([0], rank, False)
    assert calls[2][2:] == ([1], clustered, False)
    assert calls[3][2:] == ([2], rank, True)
    assert calls[4][2:] == ([3], dense_tail, True)
    assert calls[5][2:] == ([4], n, False)
    assert h[:, 0, 0].tolist() == [2.0, 3.0, 4.0, 5.0, 6.0]
    assert tau[:, 0].tolist() == [2.0, 3.0, 4.0, 5.0, 6.0]


@pytest.mark.parametrize(
    (
        "n",
        "loader_name",
        "try_name",
        "auto_name",
        "workspace_name",
        "make_policy_name",
        "policy_name",
        "env_name",
    ),
    [
        (
            512,
            "_load_qr512_blocked_cuda_extension",
            "_qr512_blocked_cuda_auto_try",
            "geqrf512_blocked_auto",
            "geqrf512_blocked_auto_workspace",
            "geqrf512_blocked_make_policy",
            "geqrf512_blocked_policy",
            "FAST_QR_QR512_BLOCKED_SYNC_FREE_AUTO_POLICY",
        ),
        (
            1024,
            "_load_qr1024_blocked_cuda_extension",
            "_qr1024_blocked_cuda_auto_try",
            "geqrf1024_blocked_auto",
            "geqrf1024_blocked_auto_workspace",
            "geqrf1024_blocked_make_policy",
            "geqrf1024_blocked_policy",
            "FAST_QR_QR1024_BLOCKED_SYNC_FREE_AUTO_POLICY",
        ),
    ],
)
def test_candidate_blocked_auto_try_can_use_sync_free_auto_entrypoint(
    monkeypatch,
    n,
    loader_name,
    try_name,
    auto_name,
    workspace_name,
    make_policy_name,
    policy_name,
    env_name,
):
    candidate = _load_candidate_module()
    candidate._BLOCKED_AUTO_POLICY_CACHE.clear()
    monkeypatch.setenv(env_name, "1")

    class FakeData:
        is_cuda = True
        dtype = torch.float32
        ndim = 3
        shape = (2, n, n)
        device = torch.device("cpu")
        _version = 0

    data = FakeData()
    calls = []

    class FakeExtension:
        pass

    extension = FakeExtension()

    def fake_workspace(data_arg, h_arg, tau_arg, factor_cols_arg, project_tail_arg):
        calls.append(
            (
                "workspace",
                data_arg,
                h_arg.shape,
                h_arg.stride(),
                tau_arg.shape,
                factor_cols_arg.shape,
                factor_cols_arg.dtype,
                project_tail_arg.shape,
                project_tail_arg.dtype,
            )
        )
        h_arg[:, :, :] = 7.0
        tau_arg[:, :] = 3.0

    def fail_auto(*_args):
        raise AssertionError("sync-free auto path should use the preallocated workspace entrypoint")

    def fail_policy(*_args):
        raise AssertionError("sync-free auto path should not build Python policy tensors")

    setattr(extension, auto_name, fail_auto)
    setattr(extension, workspace_name, fake_workspace)
    setattr(extension, make_policy_name, fail_policy)
    setattr(extension, f"{make_policy_name}_metadata", fail_policy)
    setattr(extension, policy_name, fail_policy)
    monkeypatch.setattr(candidate, loader_name, lambda _data: extension)

    h, tau = getattr(candidate, try_name)(data)

    assert calls == [
        (
            "workspace",
            data,
            (2, n, n),
            (n * n, 1, n),
            (2, n),
            (2,),
            torch.int32,
            (2,),
            torch.int32,
        )
    ]
    assert h[0, 0, 0].item() == 7.0
    assert tau[0, 0].item() == 3.0


@pytest.mark.parametrize(
    ("n", "loader_name", "try_name", "make_policy_name", "policy_name"),
    [
        (
            512,
            "_load_qr512_blocked_cuda_extension",
            "_qr512_blocked_cuda_auto_try",
            "geqrf512_blocked_make_policy",
            "geqrf512_blocked_policy",
        ),
        (
            1024,
            "_load_qr1024_blocked_cuda_extension",
            "_qr1024_blocked_cuda_auto_try",
            "geqrf1024_blocked_make_policy",
            "geqrf1024_blocked_policy",
        ),
    ],
)
def test_candidate_blocked_auto_try_caches_policy_tensors(
    monkeypatch,
    n,
    loader_name,
    try_name,
    make_policy_name,
    policy_name,
):
    candidate = _load_candidate_module()
    candidate._BLOCKED_AUTO_POLICY_CACHE.clear()

    class FakeData:
        is_cuda = True
        dtype = torch.float32
        ndim = 3
        shape = (2, n, n)
        device = torch.device("cpu")
        _version = 0

    data = FakeData()

    calls = []

    class FakeExtension:
        pass

    extension = FakeExtension()

    def fake_make_policy(data_arg, factor_cols_arg, project_tail_arg):
        factor_cols_arg[0] = n - 32
        factor_cols_arg[1] = n - 16
        project_tail_arg[0] = 0
        project_tail_arg[1] = 1
        calls.append(("make", data_arg, factor_cols_arg, project_tail_arg))

    def fake_policy(
        data_arg,
        h_arg,
        tau_arg,
        factor_cols_arg,
        project_tail_arg,
        max_factor_cols_arg,
        any_project_tail_arg,
        min_project_factor_cols_arg,
    ):
        calls.append(
            (
                "policy",
                data_arg,
                h_arg.shape,
                h_arg.stride(),
                tau_arg.shape,
                factor_cols_arg,
                project_tail_arg,
                max_factor_cols_arg,
                any_project_tail_arg,
                min_project_factor_cols_arg,
            )
        )

    setattr(extension, make_policy_name, fake_make_policy)
    setattr(extension, policy_name, fake_policy)
    monkeypatch.setattr(candidate, loader_name, lambda _data: extension)

    h, tau = getattr(candidate, try_name)(data)
    h2, tau2 = getattr(candidate, try_name)(data)

    assert h.shape == (2, n, n)
    assert h.stride() == (n * n, 1, n)
    assert tau.shape == (2, n)
    assert h2.shape == (2, n, n)
    assert tau2.shape == (2, n)
    assert [call[0] for call in calls] == ["make", "policy", "policy"]
    assert calls[1][-5] is calls[0][2]
    assert calls[1][-4] is calls[0][3]
    assert calls[1][-3] == n - 16
    assert calls[1][-2] is True
    assert calls[1][-1] == n - 16
    assert calls[2][-5] is calls[0][2]
    assert calls[2][-4] is calls[0][3]
    assert calls[2][-3] == n - 16
    assert calls[2][-2] is True
    assert calls[2][-1] == n - 16


@pytest.mark.parametrize(
    ("n", "loader_name", "try_name", "make_policy_name", "policy_name"),
    [
        (
            512,
            "_load_qr512_blocked_cuda_extension",
            "_qr512_blocked_cuda_auto_try",
            "geqrf512_blocked_make_policy",
            "geqrf512_blocked_policy",
        ),
        (
            1024,
            "_load_qr1024_blocked_cuda_extension",
            "_qr1024_blocked_cuda_auto_try",
            "geqrf1024_blocked_make_policy",
            "geqrf1024_blocked_policy",
        ),
    ],
)
def test_candidate_blocked_auto_try_prefers_policy_metadata(
    monkeypatch,
    n,
    loader_name,
    try_name,
    make_policy_name,
    policy_name,
):
    candidate = _load_candidate_module()
    candidate._BLOCKED_AUTO_POLICY_CACHE.clear()

    class FakeData:
        is_cuda = True
        dtype = torch.float32
        ndim = 3
        shape = (2, n, n)
        device = torch.device("cpu")
        _version = 0

    data = FakeData()
    calls = []

    class FakeExtension:
        pass

    extension = FakeExtension()

    def fake_make_policy(*_args):
        raise AssertionError("metadata policy path should replace separate make-policy reductions")

    def fake_make_policy_metadata(data_arg, factor_cols_arg, project_tail_arg, metadata_arg):
        factor_cols_arg[0] = n - 40
        factor_cols_arg[1] = n - 24
        project_tail_arg[0] = 0
        project_tail_arg[1] = 1
        metadata_arg.copy_(torch.tensor([n - 24, n - 40, 1, 0, n - 24, 0], dtype=torch.int32))
        calls.append(("metadata", data_arg, factor_cols_arg, project_tail_arg, metadata_arg))

    def fake_policy(
        data_arg,
        h_arg,
        tau_arg,
        factor_cols_arg,
        project_tail_arg,
        max_factor_cols_arg,
        any_project_tail_arg,
        min_project_factor_cols_arg,
    ):
        calls.append(
            (
                "policy",
                data_arg,
                h_arg.shape,
                tau_arg.shape,
                factor_cols_arg,
                project_tail_arg,
                max_factor_cols_arg,
                any_project_tail_arg,
                min_project_factor_cols_arg,
            )
        )

    setattr(extension, make_policy_name, fake_make_policy)
    setattr(extension, f"{make_policy_name}_metadata", fake_make_policy_metadata)
    setattr(extension, policy_name, fake_policy)
    monkeypatch.setattr(candidate, loader_name, lambda _data: extension)

    h, tau = getattr(candidate, try_name)(data)
    h2, tau2 = getattr(candidate, try_name)(data)

    assert h.shape == (2, n, n)
    assert tau.shape == (2, n)
    assert h2.shape == (2, n, n)
    assert tau2.shape == (2, n)
    assert [call[0] for call in calls] == ["metadata", "policy", "policy"]
    assert calls[1][-5] is calls[0][2]
    assert calls[1][-4] is calls[0][3]
    assert calls[1][-3] == n - 24
    assert calls[1][-2] is True
    assert calls[1][-1] == n - 24
    assert calls[2][-5] is calls[0][2]
    assert calls[2][-4] is calls[0][3]
    assert calls[2][-3] == n - 24
    assert calls[2][-2] is True
    assert calls[2][-1] == n - 24


@pytest.mark.parametrize("n", [512, 1024])
def test_candidate_blocked_auto_policy_metadata_global_dense_tail_marker(n):
    candidate = _load_candidate_module()
    candidate._BLOCKED_AUTO_POLICY_CACHE.clear()
    data = torch.empty((3, n, n), dtype=torch.float32)
    dense_tail = n - candidate._dense_tail_cut(n)

    class FakeExtension:
        pass

    extension = FakeExtension()

    def fake_make_policy_metadata(_data_arg, factor_cols_arg, project_tail_arg, metadata_arg):
        factor_cols_arg.fill_(dense_tail)
        project_tail_arg.fill_(1)
        metadata_arg.copy_(torch.tensor([n, dense_tail, 1, 1, dense_tail, 1], dtype=torch.int32))

    setattr(extension, f"geqrf{n}_blocked_make_policy_metadata", fake_make_policy_metadata)

    (
        factor_cols,
        project_tail,
        max_factor_cols,
        homogeneous_policy,
        any_project_tail,
        min_project_factor_cols,
    ) = candidate._blocked_auto_policy_tensors(data, n, extension, f"geqrf{n}_blocked_make_policy")

    assert factor_cols.tolist() == [dense_tail, dense_tail, dense_tail]
    assert project_tail.tolist() == [1, 1, 1]
    assert max_factor_cols == dense_tail
    assert homogeneous_policy is True
    assert any_project_tail is True
    assert min_project_factor_cols == dense_tail


@pytest.mark.parametrize("n", [512, 1024])
def test_candidate_blocked_auto_policy_metadata_ignores_non_global_dense_tail_marker(n):
    candidate = _load_candidate_module()
    candidate._BLOCKED_AUTO_POLICY_CACHE.clear()
    data = torch.empty((3, n, n), dtype=torch.float32)
    dense_tail = n - candidate._dense_tail_cut(n)

    class FakeExtension:
        pass

    extension = FakeExtension()

    def fake_make_policy_metadata(_data_arg, factor_cols_arg, project_tail_arg, metadata_arg):
        factor_cols_arg.fill_(dense_tail)
        project_tail_arg.fill_(1)
        metadata_arg.copy_(torch.tensor([n, dense_tail, 1, 1, dense_tail, 0], dtype=torch.int32))

    setattr(extension, f"geqrf{n}_blocked_make_policy_metadata", fake_make_policy_metadata)

    (
        _factor_cols,
        _project_tail,
        max_factor_cols,
        homogeneous_policy,
        any_project_tail,
        min_project_factor_cols,
    ) = candidate._blocked_auto_policy_tensors(data, n, extension, f"geqrf{n}_blocked_make_policy")

    assert max_factor_cols == n
    assert homogeneous_policy is False
    assert any_project_tail is True
    assert min_project_factor_cols == dense_tail


@pytest.mark.parametrize(
    ("n", "loader_name", "try_name", "make_policy_name", "policy_name", "indexed_name"),
    [
        (
            512,
            "_load_qr512_blocked_cuda_extension",
            "_qr512_blocked_cuda_auto_try",
            "geqrf512_blocked_make_policy",
            "geqrf512_blocked_policy",
            "geqrf512_blocked_indexed",
        ),
        (
            1024,
            "_load_qr1024_blocked_cuda_extension",
            "_qr1024_blocked_cuda_auto_try",
            "geqrf1024_blocked_make_policy",
            "geqrf1024_blocked_policy",
            "geqrf1024_blocked_indexed",
        ),
    ],
)
def test_candidate_blocked_auto_try_groups_nonhomogeneous_policy_with_indexed_calls(
    monkeypatch,
    n,
    loader_name,
    try_name,
    make_policy_name,
    policy_name,
    indexed_name,
):
    candidate = _load_candidate_module()
    candidate._BLOCKED_AUTO_POLICY_CACHE.clear()
    monkeypatch.setenv("FAST_QR_ENABLE_BLOCKED_AUTO_GROUPS", "1")

    class FakeData:
        is_cuda = True
        dtype = torch.float32
        ndim = 3
        shape = (5, n, n)
        device = torch.device("cpu")
        _version = 0

    data = FakeData()
    calls = []
    rank = candidate._rankdef_effective_cols(n)
    clustered = candidate._clustered_effective_cols(n)
    dense_tail = n - candidate._dense_tail_cut(n)

    class FakeExtension:
        pass

    extension = FakeExtension()

    def fake_make_policy(*_args):
        raise AssertionError("metadata policy path should replace separate make-policy reductions")

    def fake_make_policy_metadata(data_arg, factor_cols_arg, project_tail_arg, metadata_arg):
        factor_cols_arg.copy_(torch.tensor([rank, clustered, rank, dense_tail, n], dtype=torch.int32))
        project_tail_arg.copy_(torch.tensor([0, 0, 1, 1, 0], dtype=torch.int32))
        metadata_arg.copy_(torch.tensor([n, clustered, 1, 0, min(rank, dense_tail), 0], dtype=torch.int32))
        calls.append(("metadata", data_arg, factor_cols_arg.clone(), project_tail_arg.clone()))

    def fake_policy(*_args):
        raise AssertionError("grouped auto policy should use indexed blocked calls")

    def fake_indexed(data_arg, h_arg, tau_arg, idx_arg, factor_cols_arg, project_tail_arg):
        calls.append(
            (
                "indexed",
                data_arg,
                idx_arg.tolist(),
                int(factor_cols_arg),
                bool(project_tail_arg),
            )
        )
        h_arg[idx_arg, :, :] = float(len(calls))
        tau_arg[idx_arg, :] = float(len(calls))

    setattr(extension, make_policy_name, fake_make_policy)
    setattr(extension, f"{make_policy_name}_metadata", fake_make_policy_metadata)
    setattr(extension, policy_name, fake_policy)
    setattr(extension, indexed_name, fake_indexed)
    monkeypatch.setattr(candidate, loader_name, lambda _data: extension)

    h, tau = getattr(candidate, try_name)(data)

    assert h.shape == (5, n, n)
    assert tau.shape == (5, n)
    assert [call[0] for call in calls] == ["metadata", "indexed", "indexed", "indexed", "indexed", "indexed"]
    assert calls[1][2:] == ([0], rank, False)
    assert calls[2][2:] == ([1], clustered, False)
    assert calls[3][2:] == ([2], rank, True)
    assert calls[4][2:] == ([3], dense_tail, True)
    assert calls[5][2:] == ([4], n, False)
    assert h[:, 0, 0].tolist() == [2.0, 3.0, 4.0, 5.0, 6.0]
    assert tau[:, 0].tolist() == [2.0, 3.0, 4.0, 5.0, 6.0]


def test_candidate_blocked_auto_policy_groups_1024_mixed_dense_tail_cut():
    candidate = _load_candidate_module()
    n = 1024
    rank = candidate._rankdef_effective_cols(n)
    clustered = candidate._clustered_effective_cols(n)
    mixed_dense = n - candidate._mixed_dense_tail_cut(n)

    factor_cols = torch.tensor([rank, clustered, rank, mixed_dense, n], dtype=torch.int32)
    project_tail = torch.tensor([0, 0, 1, 1, 0], dtype=torch.int32)

    groups = candidate._blocked_auto_policy_index_groups(factor_cols, project_tail, n)

    assert mixed_dense == 1016
    assert groups is not None
    assert [(idx.tolist(), cols, tail) for idx, cols, tail in groups] == [
        ([0], rank, False),
        ([1], clustered, False),
        ([2], rank, True),
        ([3], mixed_dense, True),
        ([4], n, False),
    ]


def test_candidate_blocked_auto_policy_keeps_n_tail_group_when_mixed_tail_cut_disabled():
    candidate = _load_candidate_module()
    n = 512
    rank = candidate._rankdef_effective_cols(n)
    dense_tail = n - candidate._dense_tail_cut(n)

    assert candidate._mixed_dense_tail_cut(n) == 0
    assert dense_tail == 480
    factor_cols = torch.tensor([rank, dense_tail, n], dtype=torch.int32)
    project_tail = torch.tensor([1, 1, 1], dtype=torch.int32)

    groups = candidate._blocked_auto_policy_index_groups(factor_cols, project_tail, n)

    assert groups is not None
    assert [(idx.tolist(), cols, tail) for idx, cols, tail in groups] == [
        ([0], rank, True),
        ([1], dense_tail, True),
        ([2], n, True),
    ]


@pytest.mark.parametrize(
    ("n", "loader_name", "try_name", "make_policy_name", "policy_name", "blocked_name"),
    [
        (
            512,
            "_load_qr512_blocked_cuda_extension",
            "_qr512_blocked_cuda_auto_try",
            "geqrf512_blocked_make_policy",
            "geqrf512_blocked_policy",
            "geqrf512_blocked",
        ),
        (
            1024,
            "_load_qr1024_blocked_cuda_extension",
            "_qr1024_blocked_cuda_auto_try",
            "geqrf1024_blocked_make_policy",
            "geqrf1024_blocked_policy",
            "geqrf1024_blocked",
        ),
    ],
)
def test_candidate_blocked_auto_try_uses_homogeneous_policy_fast_path(
    monkeypatch,
    n,
    loader_name,
    try_name,
    make_policy_name,
    policy_name,
    blocked_name,
):
    candidate = _load_candidate_module()
    candidate._BLOCKED_AUTO_POLICY_CACHE.clear()

    class FakeData:
        is_cuda = True
        dtype = torch.float32
        ndim = 3
        shape = (2, n, n)
        device = torch.device("cpu")
        _version = 0

    data = FakeData()
    calls = []

    class FakeExtension:
        pass

    extension = FakeExtension()

    def fake_make_policy(data_arg, factor_cols_arg, project_tail_arg):
        factor_cols_arg.fill_(n - 32)
        project_tail_arg.zero_()
        calls.append(("make", data_arg, factor_cols_arg, project_tail_arg))

    def fake_policy(*_args):
        raise AssertionError("homogeneous policy should use regular blocked kernel")

    def fake_blocked(data_arg, h_arg, tau_arg, factor_cols_arg, project_tail_arg):
        calls.append(("blocked", data_arg, h_arg.shape, tau_arg.shape, factor_cols_arg, project_tail_arg))

    setattr(extension, make_policy_name, fake_make_policy)
    setattr(extension, policy_name, fake_policy)
    setattr(extension, blocked_name, fake_blocked)
    monkeypatch.setattr(candidate, loader_name, lambda _data: extension)

    h, tau = getattr(candidate, try_name)(data)
    h2, tau2 = getattr(candidate, try_name)(data)

    assert h.shape == (2, n, n)
    assert tau.shape == (2, n)
    assert h2.shape == (2, n, n)
    assert tau2.shape == (2, n)
    assert [call[0] for call in calls] == ["make", "blocked", "blocked"]
    assert calls[1][-2:] == (n - 32, False)
    assert calls[2][-2:] == (n - 32, False)


def test_candidate_shared_blocked_into_helpers_use_generic_large_path(monkeypatch):
    candidate = _load_candidate_module()
    calls = []
    data = SimpleNamespace(shape=(8, 2048, 2048))
    h = object()
    tau = object()
    idx = object()

    monkeypatch.setattr(candidate, "_generic_blocked_cuda_route_enabled", lambda _data, n: n == 2048)

    def fake_try_into(data_arg, h_arg, tau_arg, idx_arg, n, factor_cols=None, project_tail=False):
        calls.append((data_arg, h_arg, tau_arg, idx_arg, n, factor_cols, project_tail))
        return True

    monkeypatch.setattr(candidate, "_generic_blocked_cuda_try_into", fake_try_into)

    assert candidate._blocked_cuda_factor_cols_into(data, h, tau, idx, 1536)
    assert candidate._blocked_cuda_tail_project_into(data, h, tau, idx, 1536)
    assert candidate._blocked_cuda_full_into(data, h, tau, idx)
    assert calls == [
        (data, h, tau, idx, 2048, 1536, False),
        (data, h, tau, idx, 2048, 1536, True),
        (data, h, tau, idx, 2048, None, False),
    ]


def test_candidate_trusted_route_dispatch_avoids_rechecking_guards(monkeypatch):
    candidate = _load_candidate_module()
    data = generate_input(batch=2, n=16, cond=0, seed=789, case="rankdef")

    def fail_guard(*_args, **_kwargs):
        raise AssertionError("trusted dispatch reran a guard")

    monkeypatch.setattr(candidate, "_tail_columns_are_exact_zero", fail_guard)
    h, tau = candidate._dispatch_route("qr512_rankdef_fast", data)
    good, message = check_implementation(data, (h, tau))
    assert good, message


def test_candidate_cached_mixed_plan_avoids_recomputing_masks(monkeypatch):
    candidate = _load_candidate_module()
    candidate._ROUTE_CACHE.clear()
    data = generate_input(batch=16, n=512, cond=2, seed=32530, case="mixed")
    plan = candidate._mixed_structured_plan(data, cond=2)
    monkeypatch.setattr(candidate, "_cacheable_route_shape", lambda _batch, _n: True)
    monkeypatch.setattr(candidate, "_compute_route_plan", lambda _data: ("qr512_mixed_fast", plan))

    route, plan = candidate._route_plan_for_data(data)
    assert route == "qr512_mixed_fast"
    assert plan is not None
    assert int(plan["fallback_idx"].numel()) > 0

    def fail_plan(*_args, **_kwargs):
        raise AssertionError("cached mixed plan was not reused")

    monkeypatch.setattr(candidate, "_mixed_structured_plan", fail_plan)
    route2, plan2 = candidate._route_plan_for_data(data)
    assert route2 == route
    assert plan2 is plan

    h, tau = candidate.custom_kernel(data)
    good, message = check_implementation(data, (h, tau))
    assert good, message


def test_candidate_exposes_final_dispatch_surface():
    candidate = _load_candidate_module()
    expected = [
        "allocate_column_major_H",
        "copy_A_to_column_major_H",
        "zero_tau_tail",
        "classify_512_sampled",
        "classify_1024_sampled",
        "qr32_fast",
        "qr176_fast",
        "qr352_fast",
        "qr512_cuda_fast",
        "qr512_dense_fast",
        "qr512_mixed_fast",
        "qr512_rankdef_fast",
        "qr512_clustered_fast",
        "qr1024_cuda_fast",
        "qr1024_dense_fast",
        "qr1024_mixed_fast",
        "qr1024_rankdef_fast",
        "qr1024_clustered_fast",
        "qr1024_nearrank_fast",
        "qr2048_fast",
        "qr2048_dense_fast",
        "qr2048_rankdef_fast",
        "qr2048_mixed_fast",
        "qr2048_blocked_cuda_fast",
        "qr2048_blocked_cuda_auto_fast",
        "qr4096_fast",
        "qr4096_dense_fast",
        "qr4096_blocked_cuda_fast",
        "qr4096_blocked_cuda_auto_fast",
    ]
    for name in expected:
        assert hasattr(candidate, name)


def test_candidate_one_cta_cuda_reductions_use_warp_shuffle():
    candidate = _load_candidate_module()
    sources = [
        (candidate._QR176_CUDA_SOURCE, "block_sum_256", "__global__ void __launch_bounds__(256) geqrf176_kernel("),
        (candidate._QR352_CUDA_SOURCE, "block_sum_352", "struct HouseholderParams352"),
        (candidate._QR512_CUDA_SOURCE, "block_sum_512", "struct HouseholderParams512"),
        (candidate._QR1024_CUDA_SOURCE, "block_sum_1024", "struct HouseholderParams1024"),
    ]

    for source, function_name, end_marker in sources:
        assert f"__device__ __forceinline__ float {function_name}(" in source
        body = source.split(f"__device__ __forceinline__ float {function_name}(", 1)[1].split(end_marker, 1)[0]
        assert "__shfl_down_sync" in body
        assert "const int warp_count = (blockDim.x + 31) >> 5;" in body
        assert "for (int stride = blockDim.x >> 1" not in body


def test_candidate_cuda_hot_leaf_helpers_are_forceinline():
    candidate = _load_candidate_module()

    assert "__device__ __forceinline__ float warp_sum(float value)" in candidate._QR32_CUDA_SOURCE
    assert "constexpr int QR32_WARPS_PER_CTA = 1;" in candidate._QR32_CUDA_SOURCE
    assert "__global__ void __launch_bounds__(32 * QR32_WARPS_PER_CTA) geqrf32_kernel(" in candidate._QR32_CUDA_SOURCE
    assert "const int b = blockIdx.x * QR32_WARPS_PER_CTA + warp_slot;" in candidate._QR32_CUDA_SOURCE
    assert "__shared__ float a[QR32_WARPS_PER_CTA][32][32];" in candidate._QR32_CUDA_SOURCE
    assert "a[warp_slot][col][row] = data[" in candidate._QR32_CUDA_SOURCE
    assert "const float x = a[warp_slot][k][lane];" in candidate._QR32_CUDA_SOURCE
    assert "const float contrib = (lane >= k) ? (v * a[warp_slot][j][lane]) : 0.0f;" in candidate._QR32_CUDA_SOURCE
    assert "a[warp_slot][j][lane] -= tau_k * v * dot;" in candidate._QR32_CUDA_SOURCE
    assert "h[b * h_s0 + row * h_s1 + col * h_s2] = a[warp_slot][col][row];" in candidate._QR32_CUDA_SOURCE
    assert "a[tid][k]" not in candidate._QR32_CUDA_SOURCE
    assert "a[tid][j]" not in candidate._QR32_CUDA_SOURCE
    assert "__device__ __forceinline__ float block_sum_256(float value, float* scratch)" in candidate._QR176_CUDA_SOURCE
    assert "__device__ __forceinline__ void block_sum_tile_256(" in candidate._QR176_CUDA_SOURCE
    assert "__global__ void __launch_bounds__(256) geqrf176_kernel(" in candidate._QR176_CUDA_SOURCE
    assert "a[col * N + row] = data[" in candidate._QR176_CUDA_SOURCE
    assert "const float x = a[k * N + row];" in candidate._QR176_CUDA_SOURCE
    assert "__shared__ float dot_shared[UPDATE_COL_TILE];" in candidate._QR176_CUDA_SOURCE
    assert "chunk_col_start += UPDATE_COL_TILE" in candidate._QR176_CUDA_SOURCE
    assert "float dot_parts[UPDATE_COL_TILE];" in candidate._QR176_CUDA_SOURCE
    assert "dot_parts[cc] += v * a[col * N + row];" in candidate._QR176_CUDA_SOURCE
    assert "block_sum_tile_256(dot_parts, chunk_width, scratch, dot_shared);" in candidate._QR176_CUDA_SOURCE
    assert "block_sum_256(cc < chunk_width ? dot_parts[cc] : 0.0f, scratch)" not in candidate._QR176_CUDA_SOURCE
    assert "a[col * N + row] -= tau_k * v * dot_shared[cc];" in candidate._QR176_CUDA_SOURCE
    assert "h[b * h_s0 + row * h_s1 + col * h_s2] = a[col * N + row];" in candidate._QR176_CUDA_SOURCE
    assert "a[row * N + k]" not in candidate._QR176_CUDA_SOURCE
    assert "a[row * N + col]" not in candidate._QR176_CUDA_SOURCE
    assert "constexpr int UPDATE_COL_TILE = 1;" in candidate._QR352_CUDA_SOURCE
    assert "__device__ __forceinline__ void block_sum_tile_352(" in candidate._QR352_CUDA_SOURCE
    assert "hh_apply_reflector_tile_352(" in candidate._QR352_CUDA_SOURCE
    assert "float dot_parts[UPDATE_COL_TILE];" in candidate._QR352_CUDA_SOURCE
    assert "block_sum_tile_352(dot_parts, col_end - col_start, scratch, dot_shared);" in candidate._QR352_CUDA_SOURCE
    assert "block_sum_352(col < col_end ? dot_parts[cc] : 0.0f, scratch)" not in candidate._QR352_CUDA_SOURCE
    assert "for (int col_start = k + 1; col_start < factor_update_end; col_start += UPDATE_COL_TILE)" in candidate._QR352_CUDA_SOURCE
    assert "h[b * h_s0 + row * h_s1 + col * h_s2] -= tau_k * v * dot_shared[cc];" in candidate._QR352_CUDA_SOURCE

    one_cta_source = candidate._qr512_cuda_source()
    for token in [
        "__device__ __forceinline__ float update_operand_512(float value)",
        "__device__ __forceinline__ float block_sum_512(float value, float* scratch)",
        "__device__ __forceinline__ float hh_norm_kernel_512(",
        "__device__ __forceinline__ HouseholderParams512 hh_generate_reflector_512(",
        "__device__ __forceinline__ void hh_normalize_reflector_512(",
        "__device__ __forceinline__ void hh_apply_single_reflector_512(",
        "__device__ __forceinline__ void hh_apply_single_reflector_to_vector_512(",
    ]:
        assert token in one_cta_source


def test_candidate_one_cta_cuda_reduction_scratch_is_warp_count(monkeypatch):
    candidate = _load_candidate_module()

    assert "__shared__ float scratch[8 * UPDATE_COL_TILE];" in candidate._QR176_CUDA_SOURCE
    assert "__shared__ float scratch[8 * UPDATE_COL_TILE];" in candidate._QR352_CUDA_SOURCE

    monkeypatch.setenv("FAST_QR_QR176_THREADS_PER_CTA", "128")
    monkeypatch.setenv("FAST_QR_QR352_THREADS_PER_CTA", "128")
    monkeypatch.setenv("FAST_QR_QR512_THREADS_PER_CTA", "512")
    monkeypatch.setenv("FAST_QR_QR1024_THREADS_PER_CTA", "1024")

    assert "__shared__ float scratch[32];" in candidate._qr176_cuda_source()
    assert "__global__ void __launch_bounds__(128) geqrf176_kernel(" in candidate._qr176_cuda_source()
    assert "__shared__ float scratch[32];" in candidate._qr352_cuda_source()
    assert "__global__ void __launch_bounds__(128) geqrf352_kernel(" in candidate._qr352_cuda_source()
    assert "__shared__ float scratch[64];" in candidate._qr512_cuda_source()
    assert "__shared__ float scratch[512];" not in candidate._qr512_cuda_source()
    assert "__shared__ float scratch[128];" in candidate._qr1024_cuda_source()
    assert "__shared__ float scratch[1024];" not in candidate._qr1024_cuda_source()


def test_candidate_qr32_cuda_path_is_guarded_cpu_fallback(monkeypatch):
    candidate = _load_candidate_module()
    monkeypatch.setenv("FAST_QR_DISABLE_QR32_CUDA", "1")
    data = generate_input(batch=20, n=32, cond=1, seed=53124)
    h, tau = candidate.qr32_fast(data.clone())
    good, message = check_implementation(data, (h, tau))
    assert good, message
    assert h.shape == data.shape
    assert tau.shape == (20, 32)
    assert candidate._QR32_CUDA_EXTENSION is None


def test_candidate_qr32_cuda_execution_failure_falls_back_when_optional(monkeypatch):
    candidate = _load_candidate_module()
    fallback_output = ("fallback-h", "fallback-tau")
    captured = {}

    class BrokenExtension:
        def geqrf32(self, data, h, tau):
            captured["data"] = data
            captured["h_shape"] = tuple(h.shape)
            captured["h_stride"] = tuple(h.stride())
            captured["tau_shape"] = tuple(tau.shape)
            raise RuntimeError("launch failed")

    fake_data = SimpleNamespace(
        is_cuda=True,
        dtype=torch.float32,
        ndim=3,
        shape=(20, 32, 32),
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(candidate, "_load_qr32_cuda_extension", lambda: BrokenExtension())
    monkeypatch.setattr(candidate, "_geqrf_fallback", lambda data: fallback_output)

    assert candidate._qr32_cuda_fast(fake_data) is fallback_output
    assert captured == {
        "data": fake_data,
        "h_shape": (20, 32, 32),
        "h_stride": (1024, 1, 32),
        "tau_shape": (20, 32),
    }
    assert candidate._QR32_CUDA_EXTENSION_FAILED is True
    assert "execution failed: RuntimeError: launch failed" in candidate._QR32_CUDA_EXTENSION_ERROR


def test_candidate_qr32_cuda_execution_failure_raises_when_required(monkeypatch):
    candidate = _load_candidate_module()
    monkeypatch.setenv("FAST_QR_REQUIRE_QR32_CUDA", "1")

    class BrokenExtension:
        def geqrf32(self, data, h, tau):
            raise RuntimeError("launch failed")

    fake_data = SimpleNamespace(
        is_cuda=True,
        dtype=torch.float32,
        ndim=3,
        shape=(20, 32, 32),
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(candidate, "_load_qr32_cuda_extension", lambda: BrokenExtension())

    with pytest.raises(RuntimeError, match="qr32 CUDA extension execution failed: RuntimeError: launch failed"):
        candidate._qr32_cuda_fast(fake_data)
    assert candidate._QR32_CUDA_EXTENSION_FAILED is True
    assert "execution failed: RuntimeError: launch failed" in candidate._QR32_CUDA_EXTENSION_ERROR


def test_candidate_qr176_cuda_path_is_guarded_cpu_fallback(monkeypatch):
    candidate = _load_candidate_module()
    monkeypatch.setenv("FAST_QR_DISABLE_QR176_CUDA", "1")
    data = generate_input(batch=40, n=176, cond=1, seed=3321)
    h, tau = candidate.qr176_fast(data.clone())
    good, message = check_implementation(data, (h, tau))
    assert good, message
    assert h.shape == data.shape
    assert tau.shape == (40, 176)
    assert candidate._QR176_CUDA_EXTENSION is None


def test_candidate_qr352_cuda_path_is_guarded_cpu_fallback(monkeypatch):
    candidate = _load_candidate_module()
    monkeypatch.setenv("FAST_QR_DISABLE_QR352_CUDA", "1")
    data = generate_input(batch=1, n=352, cond=1, seed=1200)
    h, tau = candidate.qr352_fast(data.clone())
    good, message = check_implementation(data, (h, tau))
    assert good, message
    assert h.shape == data.shape
    assert tau.shape == (1, 352)
    assert candidate._QR352_CUDA_EXTENSION is None


def test_candidate_qr512_cuda_path_is_guarded_cpu_fallback(monkeypatch):
    candidate = _load_candidate_module()
    monkeypatch.setenv("FAST_QR_DISABLE_QR512_CUDA", "1")
    data = generate_input(batch=1, n=512, cond=2, seed=5120)
    h, tau = candidate.qr512_cuda_fast(data.clone())
    good, message = check_implementation(data, (h, tau))
    assert good, message
    assert h.shape == data.shape
    assert tau.shape == (1, 512)
    assert candidate._QR512_CUDA_EXTENSION is None


def test_candidate_qr1024_cuda_path_is_guarded_cpu_fallback(monkeypatch):
    candidate = _load_candidate_module()
    monkeypatch.setenv("FAST_QR_DISABLE_QR1024_CUDA", "1")
    data = generate_input(batch=1, n=1024, cond=2, seed=10240)
    h, tau = candidate.qr1024_cuda_fast(data.clone())
    good, message = check_implementation(data, (h, tau))
    assert good, message
    assert h.shape == data.shape
    assert tau.shape == (1, 1024)
    assert candidate._QR1024_CUDA_EXTENSION is None


def test_candidate_qr32_cuda_extra_flags_are_configurable(monkeypatch):
    candidate = _load_candidate_module()
    monkeypatch.delenv("FAST_QR_QR32_EXTRA_CUDA_CFLAGS", raising=False)
    monkeypatch.delenv("FAST_QR_QR32_WARPS_PER_CTA", raising=False)
    monkeypatch.delenv("FAST_QR_QR32_THREADS_PER_CTA", raising=False)
    assert candidate._qr32_cuda_extra_cuda_cflags() == ["-O3", "--use_fast_math"]
    assert candidate._qr32_cuda_warps_per_cta() == 1
    assert candidate._qr32_cuda_threads_per_cta() == 32
    default_source = candidate._qr32_cuda_source()
    assert "constexpr int QR32_WARPS_PER_CTA = 1;" in default_source
    assert "__global__ void __launch_bounds__(32) geqrf32_kernel(" in default_source
    assert "constexpr int block = 32 * QR32_WARPS_PER_CTA;" in default_source
    assert "const int64_t grid = (batch + QR32_WARPS_PER_CTA - 1) / QR32_WARPS_PER_CTA;" in default_source
    default_key = candidate._qr32_cuda_extension_build_key()
    default_name = candidate._qr32_cuda_extension_name()
    assert len(default_key) == 12
    assert default_name == f"fast_qr32_cuda_ext_v2_{default_key}"

    monkeypatch.setenv("FAST_QR_QR32_WARPS_PER_CTA", "8")
    assert candidate._qr32_cuda_warps_per_cta() == 8
    assert candidate._qr32_cuda_threads_per_cta() == 256
    assert "constexpr int QR32_WARPS_PER_CTA = 8;" in candidate._qr32_cuda_source()
    assert "__global__ void __launch_bounds__(256) geqrf32_kernel(" in candidate._qr32_cuda_source()
    assert candidate._qr32_cuda_extension_build_key() != default_key
    monkeypatch.delenv("FAST_QR_QR32_WARPS_PER_CTA")

    monkeypatch.setenv("FAST_QR_QR32_THREADS_PER_CTA", "64")
    assert candidate._qr32_cuda_warps_per_cta() == 2
    assert candidate._qr32_cuda_threads_per_cta() == 64
    assert "constexpr int QR32_WARPS_PER_CTA = 2;" in candidate._qr32_cuda_source()
    monkeypatch.delenv("FAST_QR_QR32_THREADS_PER_CTA")

    monkeypatch.setenv("FAST_QR_QR32_EXTRA_CUDA_CFLAGS", "-arch=sm_100 --maxrregcount=64")
    assert candidate._qr32_cuda_extra_cuda_cflags() == [
        "-O3",
        "--use_fast_math",
        "-arch=sm_100",
        "--maxrregcount=64",
    ]
    assert candidate._qr32_cuda_extension_build_key() != default_key
    assert candidate._qr32_cuda_extension_name().startswith("fast_qr32_cuda_ext_v2_")

    monkeypatch.setattr(candidate.torch.cuda, "is_available", lambda: False)
    sentinel = object()
    candidate._QR32_CUDA_EXTENSION = sentinel
    candidate._QR32_CUDA_EXTENSION_STATE = candidate._qr32_cuda_loader_state()
    assert candidate._load_qr32_cuda_extension() is sentinel
    monkeypatch.setenv("FAST_QR_QR32_WARPS_PER_CTA", "8")
    assert candidate._load_qr32_cuda_extension() is None
    assert candidate._QR32_CUDA_EXTENSION is None
    assert candidate._QR32_CUDA_EXTENSION_FAILED is True
    assert candidate._QR32_CUDA_EXTENSION_FAILED_STATE == candidate._qr32_cuda_loader_state()
    assert "requires CUDA" in candidate._QR32_CUDA_EXTENSION_ERROR


def test_candidate_qr176_cuda_extra_flags_are_configurable(monkeypatch):
    candidate = _load_candidate_module()
    monkeypatch.delenv("FAST_QR_QR176_EXTRA_CUDA_CFLAGS", raising=False)
    assert candidate._qr176_cuda_extra_cuda_cflags() == ["-O3", "--use_fast_math"]
    assert candidate._qr176_cuda_threads_per_cta() == 256
    assert candidate._qr176_cuda_update_col_tile() == 8
    default_source = candidate._qr176_cuda_source()
    assert "constexpr int block = 256;" in default_source
    assert "constexpr int UPDATE_COL_TILE = 8;" in default_source
    assert "__global__ void __launch_bounds__(256) geqrf176_kernel(" in default_source
    assert "__shared__ float scratch[64];" in default_source
    default_key = candidate._qr176_cuda_extension_build_key()
    default_name = candidate._qr176_cuda_extension_name()
    assert len(default_key) == 12
    assert default_name == f"fast_qr176_cuda_ext_v1_{default_key}"

    monkeypatch.setenv("FAST_QR_QR176_EXTRA_CUDA_CFLAGS", "-arch=sm_100 --maxrregcount=96")
    monkeypatch.setenv("FAST_QR_QR176_WARPS_PER_CTA", "4")
    monkeypatch.setenv("FAST_QR_QR176_UPDATE_COL_TILE", "8")
    assert candidate._qr176_cuda_extra_cuda_cflags() == [
        "-O3",
        "--use_fast_math",
        "-arch=sm_100",
        "--maxrregcount=96",
    ]
    assert candidate._qr176_cuda_threads_per_cta() == 128
    assert candidate._qr176_cuda_update_col_tile() == 8
    tuned_source = candidate._qr176_cuda_source()
    assert "constexpr int block = 128;" in tuned_source
    assert "constexpr int UPDATE_COL_TILE = 8;" in tuned_source
    assert "__global__ void __launch_bounds__(128) geqrf176_kernel(" in tuned_source
    assert "__shared__ float scratch[32];" in tuned_source
    assert candidate._qr176_cuda_extension_build_key() != default_key
    assert candidate._qr176_cuda_extension_name().startswith("fast_qr176_cuda_ext_v1_")

    sentinel = object()
    candidate._QR176_CUDA_EXTENSION = sentinel
    candidate._QR176_CUDA_EXTENSION_STATE = candidate._qr176_cuda_loader_state()
    assert candidate._load_qr176_cuda_extension() is sentinel
    monkeypatch.setenv("FAST_QR_QR176_THREADS_PER_CTA", "256")
    assert candidate._load_qr176_cuda_extension() is None
    assert candidate._QR176_CUDA_EXTENSION is None


def test_candidate_qr352_cuda_extra_flags_are_configurable(monkeypatch):
    candidate = _load_candidate_module()
    monkeypatch.delenv("FAST_QR_QR352_EXTRA_CUDA_CFLAGS", raising=False)
    assert candidate._qr352_cuda_extra_cuda_cflags() == ["-O3", "--use_fast_math"]
    assert candidate._qr352_cuda_threads_per_cta() == 256
    assert candidate._qr352_cuda_update_col_tile() == 8
    assert candidate._qr352_cuda_panel_b() == 32
    assert candidate._qr352_cuda_update_mode() == "reflectors"
    assert candidate._qr352_cuda_precision_mode() == "fp32"
    assert candidate._qr352_cuda_panel_refresh_mode() == "none"
    assert candidate._qr352_cuda_r_maintenance_mode() == "none"
    default_source = candidate._qr352_cuda_source()
    assert "constexpr int block = 256;" in default_source
    assert "constexpr int UPDATE_COL_TILE = 8;" in default_source
    assert "constexpr int PANEL_B = 32;" in default_source
    assert "constexpr int USE_COMPACT_WY_UPDATE = 0;" in default_source
    assert "constexpr int USE_TF32_INPUT_UPDATE = 0;" in default_source
    assert "constexpr int USE_FP16_INPUT_UPDATE = 0;" in default_source
    default_key = candidate._qr352_cuda_extension_build_key()
    default_name = candidate._qr352_cuda_extension_name()
    assert len(default_key) == 12
    assert default_name == f"fast_qr352_cuda_ext_v1_{default_key}"

    monkeypatch.setenv("FAST_QR_QR352_EXTRA_CUDA_CFLAGS", "-arch=sm_100 --maxrregcount=112")
    monkeypatch.setenv("FAST_QR_QR352_WARPS_PER_CTA", "4")
    monkeypatch.setenv("FAST_QR_QR352_UPDATE_COL_TILE", "8")
    monkeypatch.setenv("FAST_QR_QR352_PANEL_B", "64")
    monkeypatch.setenv("FAST_QR_QR352_UPDATE_MODE", "compact-wy")
    monkeypatch.setenv("FAST_QR_QR352_PRECISION_MODE", "tf32")
    monkeypatch.setenv("FAST_QR_QR352_PANEL_REFRESH_MODE", "prefix")
    monkeypatch.setenv("FAST_QR_QR352_R_MAINTENANCE_MODE", "panel-prefix")
    assert candidate._qr352_cuda_extra_cuda_cflags() == [
        "-O3",
        "--use_fast_math",
        "-arch=sm_100",
        "--maxrregcount=112",
    ]
    assert candidate._qr352_cuda_threads_per_cta() == 128
    assert candidate._qr352_cuda_update_col_tile() == 8
    assert candidate._qr352_cuda_panel_b() == 64
    assert candidate._qr352_cuda_update_mode() == "compact-wy"
    assert candidate._qr352_cuda_precision_mode() == "tf32-input"
    assert candidate._qr352_cuda_panel_refresh_mode() == "prefix"
    assert candidate._qr352_cuda_r_maintenance_mode() == "panel-prefix"
    tuned_source = candidate._qr352_cuda_source()
    assert "constexpr int block = 128;" in tuned_source
    assert "__global__ void __launch_bounds__(128) geqrf352_kernel(" in tuned_source
    assert "constexpr int UPDATE_COL_TILE = 8;" in tuned_source
    assert "constexpr int PANEL_B = 64;" in tuned_source
    assert "constexpr int USE_COMPACT_WY_UPDATE = 1;" in tuned_source
    assert "constexpr int USE_TF32_INPUT_UPDATE = 1;" in tuned_source
    assert "constexpr int USE_PANEL_REFRESH_PREFIX = 1;" in tuned_source
    assert "constexpr int USE_R_MAINTENANCE_PANEL_PREFIX = 1;" in tuned_source
    assert candidate._qr352_cuda_extension_build_key() != default_key
    assert candidate._qr352_cuda_extension_name().startswith("fast_qr352_cuda_ext_v1_")

    sentinel = object()
    candidate._QR352_CUDA_EXTENSION = sentinel
    candidate._QR352_CUDA_EXTENSION_STATE = candidate._qr352_cuda_loader_state()
    assert candidate._load_qr352_cuda_extension() is sentinel
    monkeypatch.setenv("FAST_QR_QR352_PANEL_B", "96")
    assert candidate._load_qr352_cuda_extension() is None
    assert candidate._QR352_CUDA_EXTENSION is None


def test_candidate_qr512_cuda_extra_flags_are_configurable(monkeypatch):
    candidate = _load_candidate_module()
    monkeypatch.delenv("FAST_QR_QR512_EXTRA_CUDA_CFLAGS", raising=False)
    assert candidate._qr512_cuda_extra_cuda_cflags() == ["-O3", "--use_fast_math"]
    assert candidate._qr512_cuda_update_col_tile() == 4
    assert "constexpr int UPDATE_COL_TILE = 4;" in candidate._qr512_cuda_source()
    default_key = candidate._qr512_cuda_extension_build_key()
    default_name = candidate._qr512_cuda_extension_name()
    assert len(default_key) == 12
    assert default_name == f"fast_qr512_cuda_ext_v2_{default_key}"

    monkeypatch.setenv("FAST_QR_QR512_EXTRA_CUDA_CFLAGS", "-arch=sm_100 --maxrregcount=128")
    monkeypatch.setenv("FAST_QR_QR512_UPDATE_COL_TILE", "8")
    assert candidate._qr512_cuda_extra_cuda_cflags() == [
        "-O3",
        "--use_fast_math",
        "-arch=sm_100",
        "--maxrregcount=128",
    ]
    assert candidate._qr512_cuda_update_col_tile() == 8
    assert "constexpr int UPDATE_COL_TILE = 8;" in candidate._qr512_cuda_source()
    assert candidate._qr512_cuda_extension_build_key() != default_key
    assert candidate._qr512_cuda_extension_name().startswith("fast_qr512_cuda_ext_v2_")


def test_candidate_qr1024_cuda_extra_flags_are_configurable(monkeypatch):
    candidate = _load_candidate_module()
    monkeypatch.delenv("FAST_QR_QR1024_EXTRA_CUDA_CFLAGS", raising=False)
    assert candidate._qr1024_cuda_extra_cuda_cflags() == ["-O3", "--use_fast_math"]
    assert candidate._qr1024_cuda_update_col_tile() == 4
    assert "constexpr int UPDATE_COL_TILE = 4;" in candidate._qr1024_cuda_source()
    default_key = candidate._qr1024_cuda_extension_build_key()
    default_name = candidate._qr1024_cuda_extension_name()
    assert len(default_key) == 12
    assert default_name == f"fast_qr1024_cuda_ext_v2_{default_key}"

    monkeypatch.setenv("FAST_QR_QR1024_EXTRA_CUDA_CFLAGS", "-arch=sm_100 --maxrregcount=160")
    monkeypatch.setenv("FAST_QR_QR1024_UPDATE_COL_TILE", "8")
    assert candidate._qr1024_cuda_extra_cuda_cflags() == [
        "-O3",
        "--use_fast_math",
        "-arch=sm_100",
        "--maxrregcount=160",
    ]
    assert candidate._qr1024_cuda_update_col_tile() == 8
    assert "constexpr int UPDATE_COL_TILE = 8;" in candidate._qr1024_cuda_source()
    assert candidate._qr1024_cuda_extension_build_key() != default_key
    assert candidate._qr1024_cuda_extension_name().startswith("fast_qr1024_cuda_ext_v2_")


def test_candidate_qr32_strict_mode_raises_on_missing_cuda(monkeypatch):
    candidate = _load_candidate_module()
    monkeypatch.setenv("FAST_QR_REQUIRE_QR32_CUDA", "1")
    monkeypatch.setenv("FAST_QR_DISABLE_QR32_CUDA", "1")
    data = generate_input(batch=20, n=32, cond=1, seed=53124)
    with pytest.raises(RuntimeError, match="qr32 CUDA extension"):
        candidate.qr32_fast(data.clone())
    assert "requires CUDA input" in candidate._QR32_CUDA_EXTENSION_ERROR


def test_candidate_qr176_strict_mode_raises_on_missing_cuda(monkeypatch):
    candidate = _load_candidate_module()
    monkeypatch.setenv("FAST_QR_REQUIRE_QR176_CUDA", "1")
    monkeypatch.setenv("FAST_QR_DISABLE_QR176_CUDA", "1")
    data = generate_input(batch=40, n=176, cond=1, seed=3321)
    with pytest.raises(RuntimeError, match="qr176 CUDA extension"):
        candidate.qr176_fast(data.clone())
    assert "requires CUDA input" in candidate._QR176_CUDA_EXTENSION_ERROR


def test_candidate_qr352_strict_mode_raises_on_missing_cuda(monkeypatch):
    candidate = _load_candidate_module()
    monkeypatch.setenv("FAST_QR_REQUIRE_QR352_CUDA", "1")
    monkeypatch.setenv("FAST_QR_DISABLE_QR352_CUDA", "1")
    data = generate_input(batch=1, n=352, cond=1, seed=1200)
    with pytest.raises(RuntimeError, match="qr352 CUDA extension"):
        candidate.qr352_fast(data.clone())
    assert "requires CUDA input" in candidate._QR352_CUDA_EXTENSION_ERROR


def test_candidate_qr512_strict_mode_raises_on_missing_cuda(monkeypatch):
    candidate = _load_candidate_module()
    monkeypatch.setenv("FAST_QR_REQUIRE_QR512_CUDA", "1")
    monkeypatch.setenv("FAST_QR_DISABLE_QR512_CUDA", "1")
    data = generate_input(batch=1, n=512, cond=2, seed=5120)
    with pytest.raises(RuntimeError, match="qr512 CUDA extension"):
        candidate.qr512_cuda_fast(data.clone())
    assert "requires CUDA input" in candidate._QR512_CUDA_EXTENSION_ERROR


def test_candidate_qr1024_strict_mode_raises_on_missing_cuda(monkeypatch):
    candidate = _load_candidate_module()
    monkeypatch.setenv("FAST_QR_REQUIRE_QR1024_CUDA", "1")
    monkeypatch.setenv("FAST_QR_DISABLE_QR1024_CUDA", "1")
    data = generate_input(batch=1, n=1024, cond=2, seed=10240)
    with pytest.raises(RuntimeError, match="qr1024 CUDA extension"):
        candidate.qr1024_cuda_fast(data.clone())
    assert "requires CUDA input" in candidate._QR1024_CUDA_EXTENSION_ERROR


def test_accelerator_preflight_allows_cpu_fallback_and_flags_strict_failure():
    candidate = _load_candidate_module()
    fallback = qr32_preflight(candidate, allow_fallback=True)
    assert fallback["ok"], fallback
    assert fallback["extension_loaded"] is False
    assert fallback["extra_cuda_cflags"] == ["-O3", "--use_fast_math"]
    assert fallback["extension_name"].startswith("fast_qr32_cuda_ext_v2_")
    assert fallback["threads_per_cta"] == 32
    assert fallback["warps_per_cta"] == 1
    assert len(fallback["extension_build_key"]) == 12
    assert "torch_cuda_arch_list" in fallback
    assert "extra_cuda_cflags_env" in fallback

    strict_candidate = _load_candidate_module()
    strict = qr32_preflight(strict_candidate, allow_fallback=False)
    if not torch.cuda.is_available():
        assert not strict["ok"], strict
        assert "requires CUDA" in (strict["extension_error"] or strict["message"])

    fallback176 = qr176_preflight(_load_candidate_module(), allow_fallback=True)
    assert fallback176["ok"], fallback176
    assert fallback176["accelerator"] == "qr176_cuda"
    assert fallback176["extension_loaded"] is False
    assert fallback176["extra_cuda_cflags"] == ["-O3", "--use_fast_math"]
    assert fallback176["extension_name"].startswith("fast_qr176_cuda_ext_v1_")
    assert fallback176["threads_per_cta"] == 256
    assert len(fallback176["extension_build_key"]) == 12

    strict176 = qr176_preflight(_load_candidate_module(), allow_fallback=False)
    if not torch.cuda.is_available():
        assert not strict176["ok"], strict176
        assert "requires CUDA" in (strict176["extension_error"] or strict176["message"])

    fallback352 = qr352_preflight(_load_candidate_module(), allow_fallback=True)
    assert fallback352["ok"], fallback352
    assert fallback352["accelerator"] == "qr352_cuda"
    assert fallback352["extension_loaded"] is False
    assert fallback352["extra_cuda_cflags"] == ["-O3", "--use_fast_math"]
    assert fallback352["extension_name"].startswith("fast_qr352_cuda_ext_v1_")
    assert fallback352["threads_per_cta"] == 256
    assert fallback352["panel_b"] == 32
    assert fallback352["update_mode"] == "reflectors"
    assert fallback352["precision_mode"] == "fp32"
    assert fallback352["panel_refresh_mode"] == "none"
    assert fallback352["r_maintenance_mode"] == "none"
    assert len(fallback352["extension_build_key"]) == 12

    strict352 = qr352_preflight(_load_candidate_module(), allow_fallback=False)
    if not torch.cuda.is_available():
        assert not strict352["ok"], strict352
        assert "requires CUDA" in (strict352["extension_error"] or strict352["message"])

    fallback512 = qr512_preflight(_load_candidate_module(), allow_fallback=True)
    assert fallback512["ok"], fallback512
    assert fallback512["accelerator"] == "qr512_cuda"
    assert fallback512["extension_loaded"] is False
    assert fallback512["extra_cuda_cflags"] == ["-O3", "--use_fast_math"]
    assert fallback512["extension_name"].startswith("fast_qr512_cuda_ext_v2_")
    assert fallback512["threads_per_cta"] == 256
    assert fallback512["panel_b"] == 32
    assert fallback512["update_col_tile"] == 4
    assert fallback512["update_mode"] == "reflectors"
    assert fallback512["precision_mode"] == "fp32"
    assert fallback512["panel_refresh_mode"] == "none"
    assert fallback512["r_maintenance_mode"] == "none"
    assert len(fallback512["extension_build_key"]) == 12

    strict512 = qr512_preflight(_load_candidate_module(), allow_fallback=False)
    if not torch.cuda.is_available():
        assert not strict512["ok"], strict512
        assert "requires CUDA" in (strict512["extension_error"] or strict512["message"])

    fallback1024 = qr1024_preflight(_load_candidate_module(), allow_fallback=True)
    assert fallback1024["ok"], fallback1024
    assert fallback1024["accelerator"] == "qr1024_cuda"
    assert fallback1024["extension_loaded"] is False
    assert fallback1024["extra_cuda_cflags"] == ["-O3", "--use_fast_math"]
    assert fallback1024["extension_name"].startswith("fast_qr1024_cuda_ext_v2_")
    assert fallback1024["threads_per_cta"] == 256
    assert fallback1024["panel_b"] == 32
    assert fallback1024["update_col_tile"] == 4
    assert fallback1024["update_mode"] == "reflectors"
    assert fallback1024["precision_mode"] == "fp32"
    assert fallback1024["panel_refresh_mode"] == "none"
    assert fallback1024["r_maintenance_mode"] == "none"
    assert len(fallback1024["extension_build_key"]) == 12

    strict1024 = qr1024_preflight(_load_candidate_module(), allow_fallback=False)
    if not torch.cuda.is_available():
        assert not strict1024["ok"], strict1024
        assert "requires CUDA" in (strict1024["extension_error"] or strict1024["message"])


def test_accelerator_preflight_can_generate_current_candidate_config_matrix():
    args = SimpleNamespace(
        config_jsonl=None,
        config=[],
        large_kernel_plan_shape_label="qr512",
        large_kernel_plan_mode="current-candidate",
        large_kernel_plan_max_configs=8,
        large_kernel_plan_env_prefix=None,
    )
    configs = load_preflight_configs(args)
    assert len(configs) == 8
    assert {tuple(sorted(row["env"])) for row in configs} == {
        (
            "FAST_QR_ENABLE_QR512_BLOCKED_CUDA",
            "FAST_QR_QR512_BLOCKED_AUTO_GROUPS",
            "FAST_QR_QR512_COMPACT_WY_TILE_COLS",
            "FAST_QR_QR512_CTAS_PER_MATRIX",
            "FAST_QR_QR512_CTA_SCHEDULE",
            "FAST_QR_QR512_PANEL_B",
            "FAST_QR_QR512_PANEL_REFRESH_MODE",
            "FAST_QR_QR512_POLICY_FULL_SCAN",
            "FAST_QR_QR512_PRECISION_MODE",
            "FAST_QR_QR512_R_MAINTENANCE_MODE",
            "FAST_QR_QR512_STRUCTURED_BEFORE_CUDA",
            "FAST_QR_QR512_SYNC_FREE_AUTO_POLICY",
            "FAST_QR_QR512_TAIL_CUT",
            "FAST_QR_QR512_TAIL_FORCE",
            "FAST_QR_QR512_TAIL_THRESHOLD",
            "FAST_QR_QR512_TILE_N",
            "FAST_QR_QR512_UPDATE_MODE",
            "FAST_QR_QR512_WARPS_PER_CTA",
        ),
    }
    assert {row["env"]["FAST_QR_QR512_PANEL_B"] for row in configs} <= {"16", "32", "48", "64"}
    assert {row["env"]["FAST_QR_QR512_PANEL_REFRESH_MODE"] for row in configs} <= {"none", "prefix"}
    assert {row["env"]["FAST_QR_QR512_PRECISION_MODE"] for row in configs} <= {"fp32", "tf32", "fp16-input"}
    assert {row["env"]["FAST_QR_QR512_R_MAINTENANCE_MODE"] for row in configs} <= {"none", "panel-prefix"}
    assert {row["env"]["FAST_QR_QR512_TILE_N"] for row in configs} <= {"64", "128"}
    assert {row["env"]["FAST_QR_QR512_UPDATE_MODE"] for row in configs} <= {"reflectors", "compact-wy"}
    assert {row["env"]["FAST_QR_QR512_COMPACT_WY_TILE_COLS"] for row in configs} <= {"2", "4", "8"}
    assert {row["env"]["FAST_QR_QR512_TAIL_CUT"] for row in configs} <= {"0", "16", "24", "32"}
    assert {row["env"]["FAST_QR_QR512_TAIL_FORCE"] for row in configs} <= {"0", "1"}
    assert {row["env"]["FAST_QR_QR512_TAIL_FORCE"] for row in configs} >= {"0", "1"}
    assert {row["env"]["FAST_QR_QR512_TAIL_THRESHOLD"] for row in configs} <= {"0.0", "0.03"}
    assert {row["env"]["FAST_QR_QR512_WARPS_PER_CTA"] for row in configs} <= {"4", "8", "16", "32"}
    assert {row["env"]["FAST_QR_QR512_CTAS_PER_MATRIX"] for row in configs} <= {"1", "2"}
    assert {row["env"]["FAST_QR_QR512_CTA_SCHEDULE"] for row in configs} <= {"fixed", "frontload"}
    assert {row["env"]["FAST_QR_QR512_BLOCKED_AUTO_GROUPS"] for row in configs} <= {"0", "1"}
    assert {row["env"]["FAST_QR_QR512_POLICY_FULL_SCAN"] for row in configs} <= {"0", "1"}
    assert {row["env"]["FAST_QR_QR512_STRUCTURED_BEFORE_CUDA"] for row in configs} == {"0", "1"}
    assert selected_accelerator_names("auto", "qr512") == ["qr512_blocked_cuda_auto"]
    assert selected_accelerator_names("auto", "qr2048") == ["qr2048_blocked_cuda_auto"]
    assert selected_accelerator_names("all", "qr512")[:2] == ["qr32_cuda", "qr176_cuda"]
    assert "qr512_blocked_cuda_auto" in selected_accelerator_names("all", "qr512")


def test_accelerator_preflight_applies_config_env_to_compile_info():
    rows = run_preflight_matrix(
        ROOT / "submissions/candidate.py",
        [
            {
                "name": "qr512_warps4",
                "env": {
                    "FAST_QR_QR512_PANEL_B": "32",
                    "FAST_QR_QR512_PANEL_REFRESH_MODE": "prefix",
                    "FAST_QR_QR512_PRECISION_MODE": "tf32",
                    "FAST_QR_QR512_R_MAINTENANCE_MODE": "panel-prefix",
                    "FAST_QR_QR512_UPDATE_MODE": "compact-wy",
                    "FAST_QR_QR512_WARPS_PER_CTA": "4",
                    "FAST_QR_QR512_STRUCTURED_BEFORE_CUDA": "0",
                },
            }
        ],
        ["qr512_cuda"],
        allow_fallback=True,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["ok"], row
    assert row["accelerator"] == "qr512_cuda"
    assert row["config_name"] == "qr512_warps4"
    assert row["config_env"]["FAST_QR_QR512_PANEL_B"] == "32"
    assert row["config_env"]["FAST_QR_QR512_PANEL_REFRESH_MODE"] == "prefix"
    assert row["config_env"]["FAST_QR_QR512_PRECISION_MODE"] == "tf32"
    assert row["config_env"]["FAST_QR_QR512_R_MAINTENANCE_MODE"] == "panel-prefix"
    assert row["config_env"]["FAST_QR_QR512_UPDATE_MODE"] == "compact-wy"
    assert row["config_env"]["FAST_QR_QR512_WARPS_PER_CTA"] == "4"
    assert row["config_env_keys"] == [
        "FAST_QR_QR512_PANEL_B",
        "FAST_QR_QR512_PANEL_REFRESH_MODE",
        "FAST_QR_QR512_PRECISION_MODE",
        "FAST_QR_QR512_R_MAINTENANCE_MODE",
        "FAST_QR_QR512_STRUCTURED_BEFORE_CUDA",
        "FAST_QR_QR512_UPDATE_MODE",
        "FAST_QR_QR512_WARPS_PER_CTA",
    ]
    assert row["threads_per_cta"] == 128
    assert row["panel_b"] == 32
    assert row["update_mode"] == "compact-wy"
    assert row["precision_mode"] == "tf32-input"
    assert row["panel_refresh_mode"] == "prefix"
    assert row["r_maintenance_mode"] == "panel-prefix"
    assert row["extension_name"].startswith("fast_qr512_cuda_ext_v2_")
    assert len(row["extension_build_key"]) == 12


def test_accelerator_preflight_auto_alias_reuses_blocked_compile_info(monkeypatch):
    candidate = _load_candidate_module()
    monkeypatch.setenv("FAST_QR_QR512_PANEL_B", "48")
    monkeypatch.setenv("FAST_QR_QR512_TILE_N", "64")
    blocked = accelerator_compile_info(candidate, "qr512_blocked_cuda")
    auto = accelerator_compile_info(candidate, "qr512_blocked_cuda_auto")
    assert auto["compile_info_accelerator"] == "qr512_blocked_cuda"
    assert auto["panel_b"] == blocked["panel_b"] == 48
    assert auto["tile_n"] == blocked["tile_n"] == 64
    assert auto["extension_build_key"] == blocked["extension_build_key"]
    assert auto["extension_name"] == blocked["extension_name"]


def test_accelerator_preflight_family_cases_cover_qr512_profiles():
    rows = run_preflight_matrix(
        ROOT / "submissions/candidate.py",
        [
            {
                "name": "qr512_warps4",
                "env": {
                    "FAST_QR_QR512_PANEL_B": "32",
                    "FAST_QR_QR512_WARPS_PER_CTA": "4",
                },
            }
        ],
        ["qr512_cuda"],
        allow_fallback=True,
        family_cases=True,
    )
    expected_cases = {name for name, _ in ACCELERATOR_FAMILY_SPECS["qr512_cuda"]}
    assert {row["preflight_case"] for row in rows} == expected_cases
    assert {row["family_cases"] for row in rows} == {True}
    assert {row["spec"]["n"] for row in rows} == {512}
    assert {row["config_name"] for row in rows} == {"qr512_warps4"}
    assert all(row["ok"] for row in rows)


def test_candidate_identity_q_upper_path_uses_column_major_h():
    candidate = _load_candidate_module()
    data = generate_input(batch=1, n=32, cond=0, seed=456, case="upper")
    h, tau = candidate.custom_kernel(data)
    good, message = check_implementation(data, (h, tau))
    assert good, message
    assert h.stride() == (32 * 32, 1, 32)
    assert torch.equal(tau, torch.zeros_like(tau))


def test_candidate_column_major_allocator_layout():
    candidate = _load_candidate_module()
    data = torch.empty((3, 16, 16), dtype=torch.float32)
    h = candidate.allocate_column_major_H(3, 16, data)
    assert h.shape == (3, 16, 16)
    assert h.dtype == torch.float32
    assert h.stride() == (16 * 16, 1, 16)


def test_candidate_output_workspace_cache_is_per_input(monkeypatch):
    candidate = _load_candidate_module()
    candidate._OUTPUT_WORKSPACE_CACHE.clear()
    monkeypatch.setenv("FAST_QR_OUTPUT_WORKSPACE_CACHE", "1")

    data_a = torch.empty((3, 16, 16), dtype=torch.float32)
    data_b = torch.empty((3, 16, 16), dtype=torch.float32)

    h_a = candidate.allocate_column_major_H(3, 16, data_a)
    h_a.fill_(1.0)
    h_a_again = candidate.allocate_column_major_H(3, 16, data_a)
    h_b = candidate.allocate_column_major_H(3, 16, data_b)

    assert h_a_again.data_ptr() == h_a.data_ptr()
    assert h_b.data_ptr() != h_a.data_ptr()
    assert h_a_again.stride() == (16 * 16, 1, 16)

    tau_a = candidate.allocate_tau(3, 16, data_a)
    tau_a.fill_(7.0)
    tau_a_zeroed = candidate.allocate_tau(3, 16, data_a, zero=True)
    tau_b = candidate.allocate_tau(3, 16, data_b)

    assert tau_a_zeroed.data_ptr() == tau_a.data_ptr()
    assert torch.equal(tau_a_zeroed, torch.zeros_like(tau_a_zeroed))
    assert tau_b.data_ptr() != tau_a.data_ptr()

    policy_a, project_a = candidate.allocate_blocked_policy_workspace(data_a, 16)
    policy_a.fill_(5)
    project_a.fill_(1)
    policy_a_again, project_a_again = candidate.allocate_blocked_policy_workspace(data_a, 16)
    policy_b, project_b = candidate.allocate_blocked_policy_workspace(data_b, 16)

    assert policy_a_again.data_ptr() == policy_a.data_ptr()
    assert project_a_again.data_ptr() == project_a.data_ptr()
    assert policy_b.data_ptr() != policy_a.data_ptr()
    assert project_b.data_ptr() != project_a.data_ptr()
    assert policy_a_again.dtype == torch.int32
    assert project_a_again.dtype == torch.int32

    monkeypatch.setenv("FAST_QR_OUTPUT_WORKSPACE_CACHE", "0")
    h_fresh = candidate.allocate_column_major_H(3, 16, data_a)
    assert h_fresh.data_ptr() != h_a.data_ptr()


def test_candidate_sampled_classifier_basic_profiles():
    candidate = _load_candidate_module()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(123)
    dense = torch.randn((4, 16, 16), generator=gen)
    assert candidate.classify_512_sampled(dense) == "dense"

    rankdef = dense.clone()
    rankdef[:, :, 12:] = 0.0
    assert candidate.classify_512_sampled(rankdef) == "rankdef"
    assert candidate.classify_1024_sampled(rankdef) == "rankdef"

    clustered = dense.clone()
    clustered[:, :, 8:] *= 1.0e-7
    assert candidate.classify_512_sampled(clustered) == "clustered"
    assert candidate.classify_1024_sampled(clustered) == "clustered"

    mixed = dense.clone()
    mixed[0, :, 12:] = 0.0
    mixed[1, :, 8:] *= 1.0e-7
    assert candidate.classify_512_sampled(mixed) == "mixed"


def test_candidate_sampled_classifier_detects_public_mixed_profiles():
    candidate = _load_candidate_module()

    public_512_mixed = generate_input(batch=640, n=512, cond=2, seed=770001, case="mixed")
    public_1024_mixed = generate_input(batch=60, n=1024, cond=2, seed=770002, case="mixed")

    assert candidate.classify_512_sampled(public_512_mixed) == "mixed"
    assert candidate.classify_1024_sampled(public_1024_mixed) == "mixed"


def test_candidate_sampled_classifier_uses_tiny_entry_budget(monkeypatch):
    candidate = _load_candidate_module()
    original = candidate._sample_entries
    sampled_entry_counts = []

    def record_sample(data, matrix_idx, row_idx, col_idx):
        sampled_entry_counts.append(int(matrix_idx.numel() * row_idx.numel() * col_idx.numel()))
        return original(data, matrix_idx, row_idx, col_idx)

    monkeypatch.setattr(candidate, "_sample_entries", record_sample)
    data_512 = generate_input(batch=640, n=512, cond=2, seed=770001, case="mixed")
    data_1024 = generate_input(batch=60, n=1024, cond=2, seed=770002, case="mixed")

    assert candidate.classify_512_sampled(data_512) == "mixed"
    assert candidate.classify_1024_sampled(data_1024) == "mixed"
    assert sampled_entry_counts
    assert max(sampled_entry_counts) <= 16 * 32 * 4
    assert sum(sampled_entry_counts) < 18000


def test_candidate_dense_routes_use_sampled_classifier_before_full_plan(monkeypatch):
    candidate = _load_candidate_module()

    def fail_plan(*_args, **_kwargs):
        raise AssertionError("dense routing should not run the full mixed-structure plan")

    def fail_full_tail_scan(*_args, **_kwargs):
        raise AssertionError("dense routing should use the sampled tail guard")

    monkeypatch.setattr(candidate, "_mixed_structured_plan", fail_plan)
    monkeypatch.setattr(candidate, "_tail_columns_are_tiny_relative", fail_full_tail_scan)
    data_512 = generate_input(batch=640, n=512, cond=2, seed=1029)
    data_1024 = generate_input(batch=60, n=1024, cond=2, seed=75342)

    assert candidate._route_for_data(data_512) == "qr512_dense_fast"
    assert candidate._route_for_data(data_1024) == "qr1024_dense_fast"


def test_candidate_sample_indices_cover_large_batches():
    candidate = _load_candidate_module()
    idx = candidate._sample_indices(640, torch.device("cpu"))
    assert len(idx) == 16
    assert int(idx[0]) == 0
    assert int(idx[-1]) == 639


def test_candidate_sample_index_cache_reuses_device_tensors():
    candidate = _load_candidate_module()
    candidate._SAMPLE_INDEX_CACHE.clear()
    device = torch.device("cpu")

    idx_a = candidate._sample_indices(640, device)
    idx_b = candidate._sample_indices(640, device)
    rows_a = candidate._sample_row_indices(512, device)
    rows_b = candidate._sample_row_indices(512, device)
    cols_a = candidate._long_index_tensor((0, 256, 511), device)
    cols_b = candidate._long_index_tensor((0, 256, 511), device)
    all_a = candidate._all_indices(640, device)
    all_b = candidate._all_indices(640, device)

    assert idx_a is idx_b
    assert rows_a is rows_b
    assert cols_a is cols_b
    assert all_a is all_b
    assert len(candidate._SAMPLE_INDEX_CACHE) == 4


def test_candidate_embedded_rectangular_geqrf_passes_rankdef_case():
    candidate = _load_candidate_module()
    n = 16
    rank = 12
    data = generate_input(batch=3, n=n, cond=0, seed=789, case="rankdef")
    h, tau = candidate._embedded_rectangular_geqrf(data, rank)
    good, message = check_implementation(data, (h, tau))
    assert good, message
    assert h.stride() == (n * n, 1, n)
    assert torch.equal(tau[:, rank:], torch.zeros_like(tau[:, rank:]))


def test_candidate_embedded_rectangular_geqrf_passes_clustered_case():
    candidate = _load_candidate_module()
    n = 32
    rank = candidate._clustered_effective_cols(n)
    assert rank == 18
    data = generate_input(batch=3, n=n, cond=0, seed=789, case="clustered")
    assert candidate._tail_columns_are_tiny_relative(data, rank)
    dense = generate_input(batch=3, n=n, cond=1, seed=789)
    assert not candidate._tail_columns_are_tiny_relative(dense, rank)
    h, tau = candidate._embedded_rectangular_geqrf(data, rank)
    good, message = check_implementation(data, (h, tau))
    assert good, message
    assert h.stride() == (n * n, 1, n)
    assert torch.equal(tau[:, rank:], torch.zeros_like(tau[:, rank:]))


def test_candidate_structured_embeddings_overwrite_prefilled_h(monkeypatch):
    candidate = _load_candidate_module()

    def nan_column_major_h(batch, n, data):
        h = torch.empty_strided(
            (batch, n, n),
            stride=(n * n, 1, n),
            device=data.device,
            dtype=torch.float32,
        )
        h.fill_(float("nan"))
        return h

    monkeypatch.setattr(candidate, "allocate_column_major_H", nan_column_major_h)

    rankdef = generate_input(batch=2, n=16, cond=0, seed=789, case="rankdef")
    h, tau = candidate._embedded_rectangular_geqrf(rankdef, candidate._rankdef_effective_cols(16))
    assert torch.isfinite(h).all()
    assert torch.isfinite(tau).all()
    good, message = check_implementation(rankdef, (h, tau))
    assert good, message

    nearrank = generate_input(batch=2, n=16, cond=0, seed=4330, case="nearrank")
    h, tau = candidate._embedded_geqrf_with_tail_projection(nearrank, candidate._rankdef_effective_cols(16))
    assert torch.isfinite(h).all()
    assert torch.isfinite(tau).all()
    good, message = check_implementation(nearrank, (h, tau))
    assert good, message

    mixed = generate_input(batch=16, n=16, cond=2, seed=5, case="mixed")
    h, tau = candidate._mixed_structured_fast(mixed, cond=2)
    assert torch.isfinite(h).all()
    assert torch.isfinite(tau).all()
    good, message = check_implementation(mixed, (h, tau))
    assert good, message


def test_classifier_experiment_emits_one_label_per_matrix():
    data = generate_input(batch=4, n=16, cond=0, seed=12, case="rankdef")
    result = classify_features(data)
    assert sum(result["label_counts"].values()) == data.shape[0]
    assert result["label_counts"] == {"rankdef-like": data.shape[0]}


def test_candidate_projected_tail_upper_matches_global_triangular_mask():
    candidate = _load_candidate_module()
    projected_tail = torch.randn((2, 16, 4), dtype=torch.float32)
    rank = 12
    rows = torch.arange(16).view(1, 16, 1)
    cols = torch.arange(rank, 16).view(1, 1, 4)
    expected = torch.where(rows <= cols, projected_tail, torch.zeros_like(projected_tail))

    assert torch.equal(candidate._projected_tail_upper(projected_tail, rank), expected)


def test_candidate_nearrank_tail_projection_passes_case():
    candidate = _load_candidate_module()
    n = 32
    rank = candidate._rankdef_effective_cols(n)
    assert rank == 24
    data = generate_input(batch=3, n=n, cond=0, seed=4330, case="nearrank")
    assert candidate._tail_matches_head_columns(data, rank)
    assert candidate.classify_1024_sampled(data) == "nearrank"

    dense = generate_input(batch=3, n=n, cond=1, seed=4330)
    assert not candidate._tail_matches_head_columns(dense, rank)
    assert not bool(candidate._nearrank_sample_mask(dense, rank).any().item())

    h, tau = candidate._embedded_nearrank_geqrf_with_tail_projection(data, rank)
    good, message = check_implementation(data, (h, tau))
    assert good, message
    assert h.stride() == (n * n, 1, n)
    assert torch.equal(tau[:, rank:], torch.zeros_like(tau[:, rank:]))


@pytest.mark.parametrize(
    ("n", "cond", "cut", "max_factor_scaled"),
    [(512, 2, 32, 18.0), (1024, 2, 64, 20.0)],
)
def test_candidate_dense_tail_projection_passes_scaled_dense_cases(n, cond, cut, max_factor_scaled):
    candidate = _load_candidate_module()
    rank = n - cut
    data = generate_input(batch=2, n=n, cond=cond, seed=1029)
    assert candidate._dense_tail_cut(n) == cut
    assert candidate._tail_columns_are_tiny_relative(data, rank, candidate._dense_tail_threshold(n))

    h, tau = candidate._dense_tail_projection_or_fallback(data)
    good, message = check_implementation(data, (h, tau))
    assert good, message
    assert diagnose(data, h, tau)["factor_scaled_max"] < max_factor_scaled
    assert h.stride() == (n * n, 1, n)
    assert torch.equal(tau[:, rank:], torch.zeros_like(tau[:, rank:]))

    unscaled = generate_input(batch=2, n=n, cond=0, seed=1029)
    assert not candidate._tail_columns_are_tiny_relative(unscaled, rank, candidate._dense_tail_threshold(n))
    assert candidate._dense_tail_route_or_fallback(unscaled, f"qr{n}_dense_fast") == "torch.geqrf"


def test_candidate_tail_policy_env_overrides(monkeypatch):
    candidate = _load_candidate_module()

    assert candidate._dense_tail_cut(512) == 32
    assert candidate._dense_tail_cut(1024) == 64
    assert candidate._dense_tail_cut(2048) == 64
    assert candidate._dense_tail_cut(4096) == 128
    monkeypatch.setenv("FAST_QR_QR512_TAIL_CUT", "20")
    assert candidate._dense_tail_cut(512) == 20
    monkeypatch.setenv("FAST_QR_QR512_TAIL_THRESHOLD", "0.0625")
    assert candidate._dense_tail_threshold(512) == pytest.approx(0.0625)

    monkeypatch.setenv("FAST_QR_DENSE_TAIL_CUT_512", "12")
    assert candidate._dense_tail_cut(512) == 12
    monkeypatch.setenv("FAST_QR_DENSE_TAIL_THRESHOLD_512", "0.125")
    assert candidate._dense_tail_threshold(512) == pytest.approx(0.125)

    assert candidate._mixed_dense_tail_cut(1024) == 8
    monkeypatch.setenv("FAST_QR_MIXED_DENSE_TAIL_CUT_1024", "4")
    assert candidate._mixed_dense_tail_cut(1024) == 4
    monkeypatch.setenv("FAST_QR_MIXED_DENSE_TAIL_THRESHOLD_1024", "0.03125")
    assert candidate._mixed_dense_tail_threshold(1024) == pytest.approx(0.03125)

    monkeypatch.setenv("FAST_QR_DENSE_TAIL_CUT_512", "512")
    with pytest.raises(ValueError, match="smaller than n=512"):
        candidate._dense_tail_cut(512)

    tune_names = {row["name"] for row in DEFAULT_TAIL_TUNE_CONFIGS}
    assert {"dense1024_cut96", "mixed1024_cut12", "dense4096_cut256"}.issubset(tune_names)


def test_candidate_route_ablation_flags_disable_selected_guards(monkeypatch):
    candidate = _load_candidate_module()
    data = generate_input(batch=2, n=512, cond=2, seed=1029)

    assert candidate._structured_routes_enabled()
    assert candidate._dense_tail_routes_enabled()
    assert candidate._dense_tail_route_or_fallback(data, "qr512_dense_fast") == "qr512_dense_fast"

    monkeypatch.setenv("FAST_QR_DISABLE_STRUCTURED_ROUTES", "1")
    assert not candidate._structured_routes_enabled()
    assert candidate._dense_tail_routes_enabled()
    assert candidate._dense_tail_route_or_fallback(data, "qr512_dense_fast") == "qr512_dense_fast"

    monkeypatch.delenv("FAST_QR_DISABLE_STRUCTURED_ROUTES")
    monkeypatch.setenv("FAST_QR_DISABLE_DENSE_TAIL", "1")
    assert candidate._structured_routes_enabled()
    assert not candidate._dense_tail_routes_enabled()
    assert candidate._dense_tail_route_or_fallback(data, "qr512_dense_fast") == "torch.geqrf"

    monkeypatch.delenv("FAST_QR_DISABLE_DENSE_TAIL")
    monkeypatch.setenv("FAST_QR_DISABLE_DATA_DEPENDENT_ROUTES", "1")
    assert not candidate._structured_routes_enabled()
    assert not candidate._dense_tail_routes_enabled()
    assert candidate._dense_tail_route_or_fallback(data, "qr512_dense_fast") == "torch.geqrf"


def test_candidate_classified_dense_tail_route_still_checks_sampled_threshold(monkeypatch):
    candidate = _load_candidate_module()
    fake_data = SimpleNamespace(shape=(640, 512, 512))

    monkeypatch.setattr(candidate, "_tail_columns_are_tiny_relative_sampled", lambda *_args: False)
    assert candidate._classified_dense_tail_route_or_fallback(fake_data, "qr512_dense_fast") == "torch.geqrf"

    monkeypatch.setattr(candidate, "_tail_columns_are_tiny_relative_sampled", lambda *_args: True)
    assert candidate._classified_dense_tail_route_or_fallback(fake_data, "qr512_dense_fast") == "qr512_dense_fast"

    monkeypatch.setattr(candidate, "_tail_columns_are_tiny_relative_sampled", lambda *_args: False)
    monkeypatch.setenv("FAST_QR_DENSE_TAIL_FORCE_512", "1")
    assert candidate._classified_dense_tail_route_or_fallback(fake_data, "qr512_dense_fast") == "qr512_dense_fast"


@pytest.mark.parametrize(
    ("n", "batch", "blocked_route_name", "expected_route"),
    [
        (512, 640, "_qr512_blocked_cuda_route_enabled", "qr512_dense_fast"),
        (1024, 60, "_qr1024_blocked_cuda_route_enabled", "qr1024_dense_fast"),
        (2048, 8, "_qr2048_blocked_cuda_route_enabled", "qr2048_dense_fast"),
        (4096, 2, "_qr4096_blocked_cuda_route_enabled", "qr4096_dense_fast"),
    ],
)
def test_candidate_dense_tail_route_preempts_full_blocked_cuda_when_forced(
    monkeypatch,
    n,
    batch,
    blocked_route_name,
    expected_route,
):
    candidate = _load_candidate_module()
    fake_data = SimpleNamespace(shape=(batch, n, n))

    monkeypatch.setenv("FAST_QR_DISABLE_STRUCTURED_ROUTES", "1")
    monkeypatch.setenv(f"FAST_QR_DENSE_TAIL_FORCE_{n}", "1")
    monkeypatch.setattr(candidate, blocked_route_name, lambda _data: True)
    if n == 512:
        monkeypatch.setattr(candidate, "_qr512_cuda_route_enabled", lambda _data: True)
    if n == 1024:
        monkeypatch.setattr(candidate, "_qr1024_cuda_route_enabled", lambda _data: True)

    route, plan = candidate._compute_route_plan(fake_data)

    assert route == expected_route
    assert plan is None


@pytest.mark.parametrize(
    ("n", "fast_name", "blocked_route_name", "auto_name"),
    [
        (2048, "qr2048_fast", "_qr2048_blocked_cuda_route_enabled", "_qr2048_blocked_cuda_auto_fast"),
        (4096, "qr4096_fast", "_qr4096_blocked_cuda_route_enabled", "_qr4096_blocked_cuda_auto_fast"),
    ],
)
def test_large_unique_shape_fast_path_skips_default_dense_tail_guard_before_blocked_auto(
    monkeypatch,
    n,
    fast_name,
    blocked_route_name,
    auto_name,
):
    candidate = _load_candidate_module()
    fake_data = SimpleNamespace(shape=(8 if n == 2048 else 2, n, n))

    monkeypatch.setattr(candidate, blocked_route_name, lambda _data: True)
    monkeypatch.setattr(candidate, "_blocked_auto_policy_enabled", lambda _data, size: size == n)
    monkeypatch.setattr(
        candidate,
        "_dense_tail_route_or_fallback",
        lambda *_args: (_ for _ in ()).throw(AssertionError("default large blocked auto should bypass tail guard")),
    )
    monkeypatch.setattr(candidate, auto_name, lambda _data: f"qr{n}_auto")

    assert getattr(candidate, fast_name)(fake_data) == f"qr{n}_auto"


@pytest.mark.parametrize(
    ("n", "fast_name", "blocked_route_name", "auto_name"),
    [
        (2048, "qr2048_fast", "_qr2048_blocked_cuda_route_enabled", "_qr2048_blocked_cuda_auto_fast"),
        (4096, "qr4096_fast", "_qr4096_blocked_cuda_route_enabled", "_qr4096_blocked_cuda_auto_fast"),
    ],
)
def test_large_unique_shape_explicit_dense_tail_policy_still_preempts_blocked_auto(
    monkeypatch,
    n,
    fast_name,
    blocked_route_name,
    auto_name,
):
    candidate = _load_candidate_module()
    fake_data = SimpleNamespace(shape=(8 if n == 2048 else 2, n, n))

    monkeypatch.setenv(f"FAST_QR_DENSE_TAIL_FORCE_{n}", "1")
    monkeypatch.setattr(candidate, blocked_route_name, lambda _data: True)
    monkeypatch.setattr(candidate, "_blocked_auto_policy_enabled", lambda _data, size: size == n)
    monkeypatch.setattr(candidate, auto_name, lambda _data: (_ for _ in ()).throw(AssertionError("auto bypassed explicit tail policy")))
    monkeypatch.setattr(candidate, "_dense_tail_route_or_fallback", lambda _data, route: route)
    monkeypatch.setattr(candidate, "_dense_tail_projection_assumed", lambda _data: f"qr{n}_tail")

    assert getattr(candidate, fast_name)(fake_data) == f"qr{n}_tail"


@pytest.mark.parametrize(
    ("n", "blocked_route_name", "expected_route"),
    [
        (2048, "_qr2048_blocked_cuda_route_enabled", "qr2048_blocked_cuda_auto_fast"),
        (4096, "_qr4096_blocked_cuda_route_enabled", "qr4096_blocked_cuda_auto_fast"),
    ],
)
def test_large_route_plan_skips_default_dense_tail_guard_before_blocked_auto(
    monkeypatch,
    n,
    blocked_route_name,
    expected_route,
):
    candidate = _load_candidate_module()
    fake_data = SimpleNamespace(shape=(8 if n == 2048 else 2, n, n))

    monkeypatch.setattr(candidate, blocked_route_name, lambda _data: True)
    monkeypatch.setattr(candidate, "_blocked_auto_policy_enabled", lambda _data, size: size == n)
    monkeypatch.setattr(
        candidate,
        "_dense_tail_route_or_fallback",
        lambda *_args: (_ for _ in ()).throw(AssertionError("default large route should bypass tail guard")),
    )

    route, plan = candidate._compute_route_plan(fake_data)

    assert route == expected_route
    assert plan is None


@pytest.mark.parametrize(
    ("n", "blocked_route_name", "expected_route"),
    [
        (2048, "_qr2048_blocked_cuda_route_enabled", "qr2048_dense_fast"),
        (4096, "_qr4096_blocked_cuda_route_enabled", "qr4096_dense_fast"),
    ],
)
def test_large_route_plan_explicit_dense_tail_policy_preempts_blocked_auto(
    monkeypatch,
    n,
    blocked_route_name,
    expected_route,
):
    candidate = _load_candidate_module()
    fake_data = SimpleNamespace(shape=(8 if n == 2048 else 2, n, n))

    monkeypatch.setenv("FAST_QR_DISABLE_STRUCTURED_ROUTES", "1")
    monkeypatch.setenv(f"FAST_QR_DENSE_TAIL_FORCE_{n}", "1")
    monkeypatch.setattr(candidate, blocked_route_name, lambda _data: True)
    monkeypatch.setattr(candidate, "_blocked_auto_policy_enabled", lambda _data, size: size == n)
    monkeypatch.setattr(candidate, "_dense_tail_route_or_fallback", lambda _data, route: route)

    route, plan = candidate._compute_route_plan(fake_data)

    assert route == expected_route
    assert plan is None


@pytest.mark.parametrize(
    ("route", "n", "blocked_route_name", "tail_project_name"),
    [
        ("qr512_dense_fast", 512, "_qr512_blocked_cuda_route_enabled", "_qr512_blocked_cuda_tail_project_fast"),
        ("qr1024_dense_fast", 1024, "_qr1024_blocked_cuda_route_enabled", "_qr1024_blocked_cuda_tail_project_fast"),
        ("qr2048_dense_fast", 2048, "_qr2048_blocked_cuda_route_enabled", "_qr2048_blocked_cuda_tail_project_fast"),
        ("qr4096_dense_fast", 4096, "_qr4096_blocked_cuda_route_enabled", "_qr4096_blocked_cuda_tail_project_fast"),
    ],
)
def test_candidate_dense_tail_dispatch_uses_blocked_tail_projection_not_full_blocked(
    monkeypatch,
    route,
    n,
    blocked_route_name,
    tail_project_name,
):
    candidate = _load_candidate_module()
    fake_data = SimpleNamespace(shape=(1, n, n))
    expected_rank = n - candidate._dense_tail_cut(n)
    calls = []

    def tail_project(_data, rank):
        calls.append(("tail", int(rank)))
        return "tail_projected"

    def fail_full_blocked(_data):
        raise AssertionError("full blocked CUDA route should not preempt dense tail projection")

    monkeypatch.setenv(f"FAST_QR_DENSE_TAIL_FORCE_{n}", "1")
    monkeypatch.setattr(candidate, blocked_route_name, lambda _data: True)
    monkeypatch.setattr(candidate, tail_project_name, tail_project)
    monkeypatch.setattr(candidate, f"_qr{n}_blocked_cuda_try", fail_full_blocked, raising=False)

    assert candidate._dispatch_route(route, fake_data) == "tail_projected"
    assert calls == [("tail", expected_rank)]


def test_candidate_nearrank_dispatch_uses_blocked_tail_projection_not_full_blocked(monkeypatch):
    candidate = _load_candidate_module()
    fake_data = SimpleNamespace(shape=(1, 1024, 1024))
    expected_rank = candidate._rankdef_effective_cols(1024)
    calls = []

    def tail_project(_data, rank):
        calls.append(("tail", int(rank)))
        return "nearrank_tail_projected"

    def fail_full_blocked(_data):
        raise AssertionError("full blocked CUDA route should not preempt nearrank tail projection")

    monkeypatch.setattr(candidate, "_qr1024_blocked_cuda_route_enabled", lambda _data: True)
    monkeypatch.setattr(candidate, "_qr1024_blocked_cuda_tail_project_fast", tail_project)
    monkeypatch.setattr(candidate, "_qr1024_blocked_cuda_try", fail_full_blocked)

    assert candidate._dispatch_route("qr1024_nearrank_fast", fake_data) == "nearrank_tail_projected"
    assert calls == [("tail", expected_rank)]


def test_candidate_nearrank_direct_wrapper_accepts_scaled_tail_match(monkeypatch):
    candidate = _load_candidate_module()
    fake_data = SimpleNamespace(shape=(1, 1024, 1024))
    expected_rank = candidate._rankdef_effective_cols(1024)
    calls = []

    def embedded(_data, rank):
        calls.append(("embedded", int(rank)))
        return "scaled_nearrank_tail_projected"

    def fail_fallback(_data):
        raise AssertionError("scaled nearrank wrapper should not fall back to full geqrf")

    monkeypatch.setattr(candidate, "_tail_matches_head_columns", lambda _data, _rank: False)
    monkeypatch.setattr(
        candidate,
        "_batch_tail_matches_scaled_head_columns",
        lambda _data, _rank, _cond: torch.tensor([True]),
    )
    monkeypatch.setattr(candidate, "_qr1024_blocked_cuda_route_enabled", lambda _data: False)
    monkeypatch.setattr(candidate, "_embedded_nearrank_geqrf_with_tail_projection", embedded)
    monkeypatch.setattr(candidate, "_geqrf_fallback", fail_fallback)

    assert candidate.qr1024_nearrank_fast(fake_data) == "scaled_nearrank_tail_projected"
    assert calls == [("embedded", expected_rank)]


def test_candidate_mixed_dispatch_keeps_structured_plan_before_generic_blocked_cuda(monkeypatch):
    candidate = _load_candidate_module()
    fake_data = SimpleNamespace(shape=(1, 512, 512))

    def fail_full_blocked(_data):
        raise AssertionError("full blocked CUDA route should not preempt mixed structured plan")

    monkeypatch.setattr(candidate, "_qr512_blocked_cuda_route_enabled", lambda _data: True)
    monkeypatch.setattr(candidate, "_qr512_blocked_cuda_try", fail_full_blocked)
    monkeypatch.setattr(candidate, "_mixed_structured_fast_from_plan", lambda _data, _plan: "mixed_structured")

    assert candidate._dispatch_route("qr512_mixed_fast", fake_data, {"sentinel": True}) == "mixed_structured"


@pytest.mark.parametrize(
    ("n", "blocked_route_name", "factor_name", "tail_name", "full_name"),
    [
        (
            512,
            "_qr512_blocked_cuda_route_enabled",
            "_qr512_blocked_cuda_factor_cols_fast",
            "_qr512_blocked_cuda_tail_project_fast",
            "_qr512_blocked_cuda_fast",
        ),
        (
            1024,
            "_qr1024_blocked_cuda_route_enabled",
            "_qr1024_blocked_cuda_factor_cols_fast",
            "_qr1024_blocked_cuda_tail_project_fast",
            "_qr1024_blocked_cuda_fast",
        ),
    ],
)
def test_candidate_mixed_structured_fast_uses_blocked_group_paths_when_enabled(
    monkeypatch,
    n,
    blocked_route_name,
    factor_name,
    tail_name,
    full_name,
):
    candidate = _load_candidate_module()
    data = torch.empty((5, n, n), dtype=torch.float32)
    rank = candidate._rankdef_effective_cols(n)
    clustered_cols = candidate._clustered_effective_cols(n)
    tiny_rank = max(1, n - 8)
    plan = {
        "rank": rank,
        "clustered_cols": clustered_cols,
        "mixed_tail_rank": tiny_rank,
        "rankdef_idx": torch.tensor([0], dtype=torch.long),
        "clustered_idx": torch.tensor([1], dtype=torch.long),
        "scaled_nearrank_idx": torch.tensor([2], dtype=torch.long),
        "tiny_dense_idx": torch.tensor([3], dtype=torch.long),
        "fallback_idx": torch.tensor([4], dtype=torch.long),
    }
    calls = []

    def output_for(label, expected_kind):
        def run(subset, factor_cols=None):
            calls.append((expected_kind, int(subset.shape[0]), None if factor_cols is None else int(factor_cols)))
            h = candidate.allocate_column_major_H(subset.shape[0], subset.shape[-1], subset)
            h.fill_(float(label))
            tau = torch.full((subset.shape[0], subset.shape[-1]), float(label), dtype=torch.float32)
            return h, tau

        return run

    def fail_rectangular_scatter(*_args, **_kwargs):
        raise AssertionError("mixed structured path should use blocked group wrapper, not direct torch.geqrf scatter")

    monkeypatch.setattr(candidate, blocked_route_name, lambda _data: True)
    monkeypatch.setattr(candidate, factor_name, output_for(1.0, "factor"))
    monkeypatch.setattr(candidate, tail_name, output_for(2.0, "tail"))
    monkeypatch.setattr(candidate, full_name, output_for(3.0, "full"))
    monkeypatch.setattr(candidate, "_scatter_rectangular_geqrf_indices", fail_rectangular_scatter)

    h, tau = candidate._mixed_structured_fast_from_plan(data, plan)

    assert h[:, 0, 0].tolist() == [1.0, 1.0, 2.0, 2.0, 3.0]
    assert tau[:, 0].tolist() == [1.0, 1.0, 2.0, 2.0, 3.0]
    assert calls == [
        ("factor", 1, rank),
        ("factor", 1, clustered_cols),
        ("tail", 1, rank),
        ("tail", 1, tiny_rank),
        ("full", 1, None),
    ]


def test_candidate_mixed_structured_fast_uses_indexed_blocked_writes(monkeypatch):
    candidate = _load_candidate_module()
    n = 512
    data = torch.empty((5, n, n), dtype=torch.float32)
    rank = candidate._rankdef_effective_cols(n)
    clustered_cols = candidate._clustered_effective_cols(n)
    tiny_rank = n - 8
    plan = {
        "rank": rank,
        "clustered_cols": clustered_cols,
        "mixed_tail_rank": tiny_rank,
        "rankdef_idx": torch.tensor([0], dtype=torch.long),
        "clustered_idx": torch.tensor([1], dtype=torch.long),
        "scaled_nearrank_idx": torch.tensor([2], dtype=torch.long),
        "tiny_dense_idx": torch.tensor([3], dtype=torch.long),
        "fallback_idx": torch.tensor([4], dtype=torch.long),
    }
    calls = []

    def indexed_factor(_data, h, tau, idx, factor_cols):
        calls.append(("factor", idx.tolist(), int(factor_cols)))
        h[idx, :, :] = float(len(calls))
        tau[idx, :] = float(len(calls))
        return True

    def indexed_tail(_data, h, tau, idx, factor_cols):
        calls.append(("tail", idx.tolist(), int(factor_cols)))
        h[idx, :, :] = float(len(calls))
        tau[idx, :] = float(len(calls))
        return True

    def indexed_full(_data, h, tau, idx):
        calls.append(("full", idx.tolist(), None))
        h[idx, :, :] = float(len(calls))
        tau[idx, :] = float(len(calls))
        return True

    def fail_scatter(*_args, **_kwargs):
        raise AssertionError("indexed blocked group path should not materialize/scatter subsets")

    monkeypatch.setattr(candidate, "_blocked_cuda_factor_cols_into", indexed_factor)
    monkeypatch.setattr(candidate, "_blocked_cuda_tail_project_into", indexed_tail)
    monkeypatch.setattr(candidate, "_blocked_cuda_full_into", indexed_full)
    monkeypatch.setattr(candidate, "_scatter_group_output_indices", fail_scatter)

    h, tau = candidate._mixed_structured_fast_from_plan(data, plan)

    assert h[:, 0, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert tau[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert calls == [
        ("factor", [0], rank),
        ("factor", [1], clustered_cols),
        ("tail", [2], rank),
        ("tail", [3], tiny_rank),
        ("full", [4], None),
    ]


def test_candidate_mixed_structured_routing_passes_per_matrix_profiles():
    candidate = _load_candidate_module()
    data = generate_input(batch=12, n=32, cond=2, seed=5, case="mixed")
    rank = candidate._rankdef_effective_cols(data.shape[-1])
    structured = (
        candidate._batch_tail_columns_are_exact_zero(data, rank)
        | candidate._batch_tail_columns_are_tiny_relative(data, candidate._clustered_effective_cols(data.shape[-1]))
        | candidate._batch_tail_matches_scaled_head_columns(data, rank, cond=2)
    )
    assert bool(structured.any().item())
    assert candidate._has_structured_mixed_subset(data, cond=2)

    h, tau = candidate._mixed_structured_fast(data, cond=2)
    good, message = check_implementation(data, (h, tau))
    assert good, message
    assert h.stride() == (32 * 32, 1, 32)
    assert torch.isfinite(tau).all()


def test_candidate_mixed_plan_exact_checks_only_candidate_subgroups(monkeypatch):
    candidate = _load_candidate_module()
    data = generate_input(batch=640, n=512, cond=2, seed=770001, case="mixed")
    calls = []

    exact_zero = candidate._batch_tail_columns_are_exact_zero
    exact_tiny = candidate._batch_tail_columns_are_tiny_relative
    exact_scaled = candidate._batch_tail_matches_scaled_head_columns

    def record_zero(subset, start):
        calls.append(("zero", int(subset.shape[0]), int(start)))
        return exact_zero(subset, start)

    def record_tiny(subset, start, threshold=1.0e-4):
        calls.append(("tiny", int(subset.shape[0]), int(start)))
        return exact_tiny(subset, start, threshold)

    def record_scaled(subset, rank, cond, threshold=1.0e-3):
        calls.append(("scaled", int(subset.shape[0]), int(rank)))
        return exact_scaled(subset, rank, cond, threshold)

    monkeypatch.setattr(candidate, "_batch_tail_columns_are_exact_zero", record_zero)
    monkeypatch.setattr(candidate, "_batch_tail_columns_are_tiny_relative", record_tiny)
    monkeypatch.setattr(candidate, "_batch_tail_matches_scaled_head_columns", record_scaled)

    plan = candidate._mixed_structured_plan(data, cond=2)

    assert calls
    assert max(size for _name, size, _arg in calls) < data.shape[0] // 4
    assert int(plan["rankdef_idx"].numel()) > 0
    assert int(plan["clustered_idx"].numel()) > 0
    assert int(plan["scaled_nearrank_idx"].numel()) > 0
    assert int(plan["fallback_idx"].numel()) > 0


def test_candidate_mixed_plan_can_trust_sampled_guards_without_exact_checks(monkeypatch):
    candidate = _load_candidate_module()
    data = generate_input(batch=640, n=512, cond=2, seed=770001, case="mixed")

    def fail_exact(*_args, **_kwargs):
        raise AssertionError("trusted sampled guard mode should not run full exact tail checks")

    monkeypatch.setenv("FAST_QR_TRUST_SAMPLED_STRUCTURED_GUARDS", "1")
    monkeypatch.setattr(candidate, "_batch_tail_columns_are_exact_zero", fail_exact)
    monkeypatch.setattr(candidate, "_batch_tail_columns_are_tiny_relative", fail_exact)
    monkeypatch.setattr(candidate, "_batch_tail_matches_scaled_head_columns", fail_exact)
    monkeypatch.setattr(candidate, "_candidate_mask_from_indices", fail_exact)

    plan = candidate._mixed_structured_plan(data, cond=2)

    assert plan["trusted_sampled_guards"] is True
    assert set(plan["exact_check_counts"].values()) == {0}
    assert int(plan["rankdef_idx"].numel()) == plan["candidate_counts"]["rankdef"]
    assert int(plan["clustered_idx"].numel()) == plan["candidate_counts"]["clustered"]
    assert int(plan["scaled_nearrank_idx"].numel()) == plan["candidate_counts"]["scaled_nearrank"]
    assert int(plan["tiny_dense_idx"].numel()) == plan["candidate_counts"]["tiny_dense_tail"]
    assert int(plan["rankdef_idx"].numel()) > 0
    assert int(plan["clustered_idx"].numel()) > 0
    assert int(plan["scaled_nearrank_idx"].numel()) > 0
    assert int(plan["fallback_idx"].numel()) > 0


def test_candidate_trusted_sampled_guards_skip_homogeneous_exact_route_checks(monkeypatch):
    candidate = _load_candidate_module()
    fake512 = SimpleNamespace(is_cuda=False, shape=(640, 512, 512))
    fake1024 = SimpleNamespace(is_cuda=False, shape=(60, 1024, 1024))

    def fail_exact(*_args, **_kwargs):
        raise AssertionError("trusted sampled guard route should not run full exact checks")

    monkeypatch.setenv("FAST_QR_TRUST_SAMPLED_STRUCTURED_GUARDS", "1")
    monkeypatch.setenv("FAST_QR_QR512_STRUCTURED_BEFORE_CUDA", "1")
    monkeypatch.setenv("FAST_QR_QR1024_STRUCTURED_BEFORE_CUDA", "1")
    monkeypatch.setenv("FAST_QR_DENSE_TAIL_CUT_512", "0")
    monkeypatch.setenv("FAST_QR_DENSE_TAIL_CUT_1024", "0")
    monkeypatch.setattr(candidate, "classify_512_sampled", lambda _data: "rankdef")
    monkeypatch.setattr(candidate, "classify_1024_sampled", lambda _data: "nearrank")
    monkeypatch.setattr(candidate, "_batch_tail_columns_are_exact_zero", fail_exact)
    monkeypatch.setattr(candidate, "_batch_tail_columns_are_tiny_relative", fail_exact)
    monkeypatch.setattr(candidate, "_tail_matches_head_columns", fail_exact)
    monkeypatch.setattr(candidate, "_batch_tail_matches_scaled_head_columns", fail_exact)

    assert candidate._compute_route_plan(fake512) == ("qr512_rankdef_fast", None)
    assert candidate._compute_route_plan(fake1024) == ("qr1024_nearrank_fast", None)


@pytest.mark.parametrize(
    ("spec", "expected_route"),
    [
        ({"batch": 16, "n": 512, "cond": 0, "seed": 32525, "case": "rankdef"}, "qr512_rankdef_fast"),
        ({"batch": 16, "n": 512, "cond": 0, "seed": 32526, "case": "clustered"}, "qr512_clustered_fast"),
        ({"batch": 16, "n": 512, "cond": 2, "seed": 32530, "case": "mixed"}, "qr512_mixed_fast"),
        ({"batch": 4, "n": 1024, "cond": 0, "seed": 4329, "case": "rankdef"}, "qr1024_rankdef_fast"),
        ({"batch": 4, "n": 1024, "cond": 0, "seed": 4330, "case": "nearrank"}, "qr1024_nearrank_fast"),
        ({"batch": 4, "n": 1024, "cond": 0, "seed": 4331, "case": "clustered"}, "qr1024_clustered_fast"),
        ({"batch": 4, "n": 1024, "cond": 2, "seed": 4332, "case": "mixed"}, "qr1024_mixed_fast"),
        ({"batch": 2, "n": 2048, "cond": 2, "seed": 224466, "case": "dense"}, "qr2048_dense_fast"),
        ({"batch": 2, "n": 2048, "cond": 0, "seed": 224467, "case": "rankdef"}, "qr2048_rankdef_fast"),
        ({"batch": 2, "n": 2048, "cond": 2, "seed": 224468, "case": "mixed"}, "qr2048_mixed_fast"),
        ({"batch": 1, "n": 4096, "cond": 1, "seed": 75342}, "qr4096_dense_fast"),
    ],
)
def test_candidate_routes_public_structured_correctness_batches(spec, expected_route):
    candidate = _load_candidate_module()
    data = generate_input(**spec)

    route, plan = candidate._route_plan_for_data(data)

    assert route == expected_route
    if route.endswith("_mixed_fast"):
        assert plan is not None


@pytest.mark.parametrize(("n", "batch", "seed", "cut"), [(1024, 12, 2, 8)])
def test_candidate_mixed_tiny_dense_tail_group_passes_public_shapes(n, batch, seed, cut):
    candidate = _load_candidate_module()
    data = generate_input(batch=batch, n=n, cond=2, seed=seed, case="mixed")
    rank = n - cut
    mask = candidate._batch_tail_columns_are_tiny_relative(
        data,
        rank,
        candidate._mixed_dense_tail_threshold(n),
    )
    assert candidate._mixed_dense_tail_cut(n) == cut
    assert bool(mask.any().item())

    h, tau = candidate._mixed_structured_fast(data, cond=2)
    good, message = check_implementation(data, (h, tau))
    assert good, message
    assert h.stride() == (n * n, 1, n)
    assert torch.isfinite(tau).all()


def test_candidate_mixed_512_dense_tail_skip_is_disabled():
    candidate = _load_candidate_module()
    assert candidate._mixed_dense_tail_cut(512) == 0
    assert candidate._mixed_dense_tail_threshold(512) == 0.0


def test_candidate_mixed_detection_includes_1024_tiny_dense_tail_group():
    candidate = _load_candidate_module()
    scaled = generate_input(batch=1, n=1024, cond=2, seed=101)
    unscaled = generate_input(batch=1, n=1024, cond=0, seed=102)
    data = torch.cat([scaled, unscaled], dim=0)
    cut = candidate._mixed_dense_tail_cut(1024)
    rank = 1024 - cut
    mask = candidate._batch_tail_columns_are_tiny_relative(
        data,
        rank,
        candidate._mixed_dense_tail_threshold(1024),
    )

    assert cut == 8
    assert mask.tolist() == [True, False]
    assert candidate._has_structured_mixed_subset(data, cond=2)

    h, tau = candidate.qr1024_fast(data)
    good, message = check_implementation(data, (h, tau))
    assert good, message
    assert h.stride() == (1024 * 1024, 1, 1024)
    assert torch.isfinite(tau).all()


def test_candidate_policy_reports_public_benchmark_routes():
    rows = policy_rows(ROOT / "submissions/candidate.py", ROOT / "cases/public_benchmarks.txt")
    assert len(rows) == 12
    assert rows[0]["dispatch"] == "qr32_fast"
    assert rows[0]["primary"] == "inline_cuda_compact_householder_or_fallback"
    assert rows[0]["cuda_kernel"] == "geqrf32_kernel"
    assert rows[0]["shape_collision"] is False
    assert rows[0]["submission_entrypoint"] == "custom_kernel(data)"
    assert rows[0]["case_metadata_available"] is False
    assert rows[0]["case_metadata_passed_to_submission"] is False
    assert rows[0]["case_info_source"] == "data.shape"
    assert rows[0]["case_selection_info_sources"] == ["data.shape"]
    assert rows[0]["dispatch_info_sources"] == ["data.shape"]
    assert rows[0]["shape_only_case_selection"] is True
    assert rows[0]["shape_only_dispatch"] is True
    assert rows[0]["uses_tensor_values_for_dispatch"] is False
    assert rows[0]["uses_tensor_values_for_case_selection"] is False
    assert rows[0]["classifier_needed_for_current_candidate"] is False
    assert rows[0]["classifier_decision_rule"] == "not_applicable_shape_unique"
    assert rows[0]["column_major_h"] == "conditional"
    assert rows[0]["h_layout"] == "column_major_when_cuda_extension_available_else_torch.geqrf_default"
    assert rows[1]["primary"] == "inline_cuda_compact_householder_or_fallback"
    assert rows[1]["cuda_kernel"] == "geqrf176_kernel"
    assert rows[1]["column_major_h"] == "conditional"
    assert rows[1]["h_layout"] == "column_major_when_cuda_extension_available_else_torch.geqrf_default"
    assert rows[2]["primary"] == "inline_cuda_compact_householder_or_fallback"
    assert rows[2]["cuda_kernel"] == "geqrf352_kernel"
    assert rows[2]["column_major_h"] == "conditional"
    assert rows[2]["h_layout"] == "column_major_when_cuda_extension_available_else_torch.geqrf_default"
    assert rows[3]["dispatch"] == "qr512_fast"
    assert rows[3]["cuda_route"] == "qr512_cuda_fast"
    assert rows[3]["cuda_kernel"] == "geqrf512_kernel"
    assert rows[3]["blocked_cuda_route"] == "qr512_blocked_cuda_auto_fast"
    assert rows[3]["blocked_cuda_base_route"] == "qr512_blocked_cuda_fast"
    assert rows[3]["cuda_primary"] == "inline_cuda_compact_householder"
    assert rows[3]["cuda_disable_env"] == "FAST_QR_DISABLE_QR512_CUDA"
    assert rows[3]["cuda_route_bypasses_classifier"] is True
    assert rows[3]["shape_collision"] is True
    assert rows[3]["requires_tensor_guard_for_case_specific_path"] is True
    assert rows[3]["classifier_needed_for_case_specific_path"] is True
    assert rows[3]["case_info_source"] == "tensor_values"
    assert rows[3]["case_selection_info_sources"] == ["data.shape", "tensor_values"]
    assert rows[3]["dispatch_info_sources"] == ["data.shape"]
    assert rows[3]["shape_only_case_selection"] is False
    assert rows[3]["shape_only_dispatch"] is True
    assert rows[3]["uses_tensor_values_for_dispatch"] is False
    assert rows[3]["uses_tensor_values_for_case_selection"] is True
    assert rows[3]["classifier_needed_for_current_candidate"] is False
    assert rows[3]["classifier_on_current_hot_path"] is False
    assert "FAST_QR_QR512_STRUCTURED_BEFORE_CUDA=1" in rows[3]["classifier_decision_rule"]
    assert rows[3]["column_major_h"] == "conditional"
    assert rows[3]["h_layout"] == "column_major_when_fast_path_applies_else_torch.geqrf_default"
    assert rows[3]["shape_collision_cases"] == ["dense", "mixed", "rankdef", "clustered"]
    assert rows[3]["dense_tail"] == {"cut": 32, "rank": 480, "threshold": 3.0e-2, "force": False}
    assert rows[3]["required_cuda_kernel"] == "qr512_blocked_householder_r_maintenance"
    assert rows[3]["required_repair_modes"] == ["panel_refresh_mode=prefix", "r_maintenance_mode=panel-prefix"]
    assert rows[3]["candidate_config_shape_label"] == "qr512"
    assert rows[3]["candidate_config_env_prefix"] == "FAST_QR_QR512"
    assert rows[3]["candidate_config_benchmark_indices"] == "3,7,9,10"
    assert rows[3]["candidate_config_correctness_indices"] == "3,6,7,8,9,10,11,19"
    assert "one-CTA QR512 CUDA path" in rows[3]["required_cuda_reason"]
    assert rows[4]["dispatch"] == "qr1024_fast"
    assert rows[4]["cuda_route"] == "qr1024_cuda_fast"
    assert rows[4]["cuda_kernel"] == "geqrf1024_kernel"
    assert rows[4]["blocked_cuda_route"] == "qr1024_blocked_cuda_auto_fast"
    assert rows[4]["blocked_cuda_base_route"] == "qr1024_blocked_cuda_fast"
    assert rows[4]["cuda_primary"] == "inline_cuda_compact_householder"
    assert rows[4]["cuda_disable_env"] == "FAST_QR_DISABLE_QR1024_CUDA"
    assert rows[4]["cuda_route_bypasses_classifier"] is True
    assert rows[4]["shape_collision_cases"] == ["dense", "mixed", "nearrank"]
    assert rows[4]["classifier_needed_for_case_specific_path"] is True
    assert rows[4]["classifier_needed_for_current_candidate"] is False
    assert rows[4]["classifier_on_current_hot_path"] is False
    assert rows[4]["dispatch_info_sources"] == ["data.shape"]
    assert rows[4]["required_cuda_kernel"] == "qr1024_blocked_householder_r_maintenance"
    assert rows[4]["required_repair_modes"] == ["panel_refresh_mode=prefix", "r_maintenance_mode=panel-prefix"]
    assert rows[4]["candidate_config_shape_label"] == "qr1024"
    assert rows[4]["candidate_config_env_prefix"] == "FAST_QR_QR1024"
    assert rows[4]["candidate_config_benchmark_indices"] == "4,8,11"
    assert rows[4]["candidate_config_correctness_indices"] == "4,12,13,14,15,20"
    assert "one-CTA QR1024" in rows[4]["required_cuda_reason"]
    assert rows[7]["primary"] == "per_matrix_mixed_structured_fast"
    assert rows[7]["required_cuda_kernel"] == "qr512_blocked_householder_r_maintenance"
    assert rows[7]["column_major_h"] is True
    assert rows[7]["h_layout"] == "column_major"
    assert rows[7]["mixed_dense_tail"]["cut"] == 0
    assert "scaled_nearrank_cols=384" in rows[7]["per_matrix_groups"]
    assert "fallback=torch.geqrf" in rows[7]["per_matrix_groups"]
    assert rows[8]["primary"] == "per_matrix_mixed_structured_fast"
    assert rows[8]["shape_collision_cases"] == ["dense", "mixed", "nearrank"]
    assert rows[8]["mixed_dense_tail"] == {"cut": 8, "rank": 1016, "threshold": 2.0e-2}
    assert "tiny_dense_tail_cut=8" in rows[8]["per_matrix_groups"]
    assert rows[5]["case_info_source"] == "data.shape"
    assert rows[5]["shape_only_case_selection"] is True
    assert rows[5]["uses_tensor_values_for_case_selection"] is False
    assert rows[5]["uses_tensor_values_for_dispatch"] is True
    assert rows[5]["dispatch_info_sources"] == ["data.shape", "tensor_values"]
    assert rows[5]["classifier_needed_for_current_candidate"] is False
    assert rows[5]["blocked_cuda_route"] == "qr2048_blocked_cuda_auto_fast"
    assert rows[5]["blocked_cuda_base_route"] == "qr2048_blocked_cuda_fast"
    assert rows[5]["required_cuda_kernel"] == "qr2048_multi_cta_blocked_householder"
    assert rows[5]["candidate_config_shape_label"] == "qr2048"
    assert rows[5]["candidate_config_env_prefix"] == "FAST_QR_QR2048"
    assert rows[5]["candidate_config_benchmark_indices"] == "5"
    assert rows[5]["candidate_config_correctness_indices"] == "16,21"
    assert "one-CTA QR is not a viable final path" in rows[5]["required_cuda_reason"]
    assert rows[6]["required_cuda_kernel"] == "qr4096_multi_cta_blocked_householder"
    assert rows[6]["blocked_cuda_route"] == "qr4096_blocked_cuda_auto_fast"
    assert rows[6]["blocked_cuda_base_route"] == "qr4096_blocked_cuda_fast"
    assert rows[6]["candidate_config_shape_label"] == "qr4096"
    assert rows[6]["candidate_config_env_prefix"] == "FAST_QR_QR4096"
    assert rows[6]["candidate_config_benchmark_indices"] == "6"
    assert rows[6]["candidate_config_correctness_indices"] == "5,18"
    assert rows[9]["active_cols"] == 384
    assert rows[10]["active_cols"] == 258
    assert rows[11]["primary"] == "nearrank_tail_projection"
    assert rows[11]["required_cuda_kernel"] == "qr1024_blocked_householder_r_maintenance"
    assert rows[11]["active_cols"] == 768


def test_candidate_policy_marks_classifier_hot_path_when_structured_first(monkeypatch):
    monkeypatch.setenv("FAST_QR_QR512_STRUCTURED_BEFORE_CUDA", "1")
    rows = policy_rows(ROOT / "submissions/candidate.py", ROOT / "cases/public_benchmarks.txt")

    assert rows[3]["cuda_route_bypasses_classifier"] is False
    assert rows[3]["requires_tensor_guard_for_case_specific_path"] is True
    assert rows[3]["classifier_needed_for_case_specific_path"] is True
    assert rows[3]["classifier_on_current_hot_path"] is True
    assert rows[3]["classifier_needed_for_current_candidate"] is True
    assert rows[3]["dispatch_info_sources"] == ["data.shape", "tensor_values"]
    assert rows[3]["shape_only_dispatch"] is False
    assert rows[3]["uses_tensor_values_for_dispatch"] is True
    assert "FAST_QR_DISABLE_STRUCTURED_ROUTES=1" in rows[3]["classifier_decision_rule"]

    assert rows[4]["cuda_route_bypasses_classifier"] is True
    assert rows[4]["classifier_needed_for_case_specific_path"] is True
    assert rows[4]["classifier_on_current_hot_path"] is False


def test_implementation_status_marks_nonfinal_routes():
    rows = readiness_rows(ROOT / "submissions/candidate.py", ROOT / "cases/public_benchmarks.txt")
    assert len(rows) == 12
    assert rows[0]["implementation_kind"] == "custom_cuda_optional_fallback"
    assert rows[0]["readiness"] == "partial_cuda_needs_b200_validation"
    assert rows[0]["has_custom_cuda"] is True
    assert rows[1]["implementation_kind"] == "custom_cuda_optional_fallback"
    assert rows[1]["readiness"] == "partial_cuda_needs_b200_validation"
    assert rows[1]["has_custom_cuda"] is True
    assert rows[2]["implementation_kind"] == "custom_cuda_optional_fallback"
    assert rows[2]["readiness"] == "partial_cuda_needs_b200_validation"
    assert rows[2]["has_custom_cuda"] is True
    assert rows[3]["implementation_kind"] == "custom_cuda_optional_fallback"
    assert rows[3]["readiness"] == "partial_cuda_needs_b200_validation"
    assert rows[3]["has_custom_cuda"] is True
    assert rows[3]["priority"] == "highest"
    assert rows[3]["required_cuda_kernel"] == "qr512_blocked_householder_r_maintenance"
    assert "panel_refresh_mode=prefix" in rows[3]["next_work"]
    assert rows[7]["implementation_kind"] == "custom_cuda_optional_fallback"
    assert rows[7]["required_cuda_kernel"] == "qr512_blocked_householder_r_maintenance"
    assert rows[11]["implementation_kind"] == "custom_cuda_optional_fallback"
    assert rows[11]["readiness"] == "partial_cuda_needs_b200_validation"
    assert rows[11]["required_cuda_kernel"] == "qr1024_blocked_householder_r_maintenance"

    summary = summarize_readiness(rows)
    assert summary["ready_for_final_submission"] is False
    assert summary["num_cases"] == 12
    assert summary["num_final_kernel_required"] == 12
    assert summary["num_custom_cuda_partial"] == 12
    assert summary["num_torch_geqrf_fallback"] == 0
    assert summary["num_torch_composite_experiment"] == 0
    assert summary["next_priority_cases"][0]["case_index"] == 3
    large_rows = {row["case_index"]: row for row in rows}
    assert "qr512_blocked_householder_r_maintenance" in large_rows[3]["next_work"]
    assert "qr1024_blocked_householder_r_maintenance" in large_rows[4]["next_work"]
    assert "qr2048_multi_cta_blocked_householder" in large_rows[5]["next_work"]
    assert "qr4096_multi_cta_blocked_householder" in large_rows[6]["next_work"]


def test_candidate_rankdef_zero_tail_guard():
    candidate = _load_candidate_module()
    data = generate_input(batch=2, n=16, cond=0, seed=321, case="rankdef")
    assert candidate._tail_columns_are_exact_zero(data, 12)
    dense = generate_input(batch=2, n=16, cond=1, seed=321)
    assert not candidate._tail_columns_are_exact_zero(dense, 12)
    assert not candidate._tail_columns_are_tiny_relative(dense, 12)


def test_official_eval_case_writer_strips_comments(tmp_path):
    src = tmp_path / "cases_with_comments.txt"
    dst = tmp_path / "cases.txt"
    src.write_text(
        "# comment\n"
        "batch: 2; n: 8; cond: 1; seed: 11 # inline\n"
        "\n"
        "batch: 1; n: 8; cond: 0; seed: 12; case: upper\n"
    )
    write_official_case_file(src, dst)
    assert dst.read_text() == (
        "batch: 2; n: 8; cond: 1; seed: 11\n"
        "batch: 1; n: 8; cond: 0; seed: 12; case: upper\n"
    )


def test_classifier_seed_sweep_row_and_summary_on_structured_case():
    candidate = _load_candidate_module()
    spec = {"batch": 3, "n": 512, "cond": 0, "seed": 789, "case": "rankdef"}
    row = run_classifier_case(candidate, spec, include_plan=True)

    assert row["ok"]
    assert row["sampled_class"] == "rankdef"
    assert row["expected_sampled_class"] == "rankdef"
    assert row["classifier_ok"] is True
    assert row["expected_route"] == "qr512_rankdef_fast"
    assert row["route_ok"] is True
    assert row["case_metadata_passed_to_submission"] is False
    assert row["uses_tensor_values_for_case_selection"] is True
    assert row["sampled_class_wall_us"] >= 0.0

    summary = summarize_classifier_sweep([{**row, "popcorn_seed": None}, {**row, "popcorn_seed": 1}])
    assert summary["ok"]
    assert summary["num_rows"] == 2
    assert summary["num_classifier_mismatch"] == 0
    assert summary["num_public_seed_rows"] == 1
    assert summary["num_popcorn_seed_rows"] == 1


def test_submit_popcorn_helpers_stage_single_file(tmp_path):
    source = ROOT / "submissions/candidate.py"
    staged = stage_submission(source, tmp_path)

    assert staged.name == "submission.py"
    assert staged.read_text() == source.read_text()
    assert selected_popcorn_modes("both") == ["test", "leaderboard"]
    assert selected_popcorn_modes("test") == ["test"]

    cmd = build_popcorn_command(
        staged,
        leaderboard="qr_v2",
        gpu="B200",
        mode="leaderboard",
        popcorn_bin="popcorn",
        extra_args=["--foo", "bar"],
    )
    assert cmd[:8] == ["popcorn", "submit", "--leaderboard", "qr_v2", "--gpu", "B200", "--mode", "leaderboard"]
    assert cmd[8] == str(staged)
    assert cmd[-2:] == ["--foo", "bar"]


def test_validate_submission_accepts_self_contained_candidate(tmp_path):
    result = validate_submission(ROOT / "submissions/candidate.py", stage_dir=tmp_path / "stage")
    assert result["ok"], result
    assert result["static_ok"], result
    assert result["secret_ok"], result
    assert result["secret_findings"] == []
    assert result["import_ok"], result
    assert result["staged_submission"]["path"].endswith("submission.py")
    assert {row["root"] for row in result["imports"]} >= {"os", "weakref", "torch", "task"}


def test_validate_submission_rejects_same_directory_helper(tmp_path):
    (tmp_path / "helper.py").write_text("def run(data):\n    return data\n")
    submission = tmp_path / "candidate.py"
    submission.write_text(
        "from task import input_t, output_t\n"
        "from helper import run\n\n"
        "def custom_kernel(data: input_t) -> output_t:\n"
        "    return run(data)\n"
    )
    result = validate_submission(submission, stage_dir=tmp_path / "stage")
    assert not result["ok"], result
    assert any(error["check"] == "import.same_dir" for error in result["errors"])


def test_validate_submission_rejects_repo_helper_import(tmp_path):
    submission = tmp_path / "candidate.py"
    submission.write_text(
        "from qr_common import ROOT\n\n"
        "def custom_kernel(data):\n"
        "    return data\n"
    )
    result = validate_submission(submission, stage_dir=tmp_path / "stage")
    assert not result["ok"], result
    assert any(error["check"] == "import.repo_local" for error in result["errors"])


def test_validate_submission_rejects_secret_in_submission_without_exposing_value(tmp_path):
    secret = "sk-" + "a" * 48
    submission = tmp_path / "candidate.py"
    submission.write_text(
        "def custom_kernel(data):\n"
        f"    token = {secret!r}\n"
        "    return data\n"
    )
    result = validate_submission(submission, stage_dir=tmp_path / "stage")
    assert not result["ok"], result
    assert not result["secret_ok"], result
    assert result["secret_findings"][0]["rule"] == "openai_api_key"
    assert any(error["check"] == "secret.openai_api_key" for error in result["errors"])
    assert secret not in json.dumps(result)


def test_sweep_diagnostics_match_tracking_schema():
    spec = _small_case("dense")
    custom_kernel = load_submission(ROOT / "submissions/candidate.py")
    row = diagnostic_row(custom_kernel, spec)
    assert row["diagnostic_passed"], row
    assert row["factor_scaled_max"] < 20.0
    assert row["orth_scaled_max"] < 100.0
    assert "tri_scaled_max" in row
    assert load_config('{"algorithm":"torch.geqrf"}') == {"algorithm": "torch.geqrf"}


def test_seed_sweep_helpers_on_small_case():
    assert parse_int_list("1,2, 3") == [1, 2, 3]
    assert parse_popcorn_seed_tokens("public,0,2") == [None, 0, 2]
    custom_kernel = load_submission(ROOT / "submissions/candidate.py")
    row = run_case(custom_kernel, {"batch": 2, "n": 8, "cond": 1, "seed": 11}, diagnose_output=True)
    assert row["ok"], row
    assert row["factor_scaled_max"] < 20.0


def test_seed_sweep_margin_gates_on_small_case():
    custom_kernel = load_submission(ROOT / "submissions/candidate.py")
    spec = {"batch": 2, "n": 8, "cond": 1, "seed": 11}
    row = run_case(custom_kernel, spec, diagnose_output=False, max_factor_scaled=20.0, max_orth_scaled=100.0)
    assert row["ok"], row
    assert row["margin_ok"], row
    assert row["factor_margin_ok"], row
    assert row["orth_margin_ok"], row

    tight = run_case(custom_kernel, spec, diagnose_output=False, max_factor_scaled=0.0)
    assert tight["ok"], tight
    assert not tight["margin_ok"], tight
    assert not tight["factor_margin_ok"], tight


def test_quantization_seed_sweep_on_small_case():
    spec = {"batch": 2, "n": 8, "cond": 0, "seed": 11, "case": "upper"}
    rows = [
        run_quantization_case(
            spec,
            experiment,
            max_factor_scaled=20.0,
            max_orth_scaled=100.0,
        )
        for experiment in ["fp16-nearby", "tf32-input-nearby"]
    ]
    for row in rows:
        assert row["ok"], row
        assert row["margin_ok"], row
        assert row["factor_scaled_max"] < 20.0
        assert row["orth_scaled_max"] < 100.0
        assert row["worst_factor_matrix"] >= 0
        assert row["wall_us"] > 0.0
    rows[0]["popcorn_seed"] = None
    rows[1]["popcorn_seed"] = 1
    summary = summarize_quantization_sweep(rows)
    assert summary["ok"], summary
    assert summary["num_rows"] == 2
    assert summary["num_public_seed_rows"] == 1
    assert summary["num_popcorn_seed_rows"] == 1


def test_mixed_seed_sweep_on_small_case():
    candidate = _load_candidate_module()
    spec = {"batch": 2, "n": 8, "cond": 2, "seed": 11, "case": "mixed"}
    row = run_mixed_seed_case(
        candidate,
        spec,
        max_factor_scaled=20.0,
        max_orth_scaled=100.0,
    )
    assert row["ok"], row
    assert row["margin_ok"], row
    assert row["route_ok"], row
    assert row["factor_scaled_max"] < 20.0
    assert row["orth_scaled_max"] < 100.0
    assert row["kernel_wall_us"] > 0.0
    row["case_source"] = "inline"
    row["popcorn_seed"] = None
    mutated = {**row, "popcorn_seed": 1}
    summary = summarize_mixed_seed_sweep([row, mutated])
    assert summary["ok"], summary
    assert summary["num_rows"] == 2
    assert summary["num_public_seed_rows"] == 1
    assert summary["num_popcorn_seed_rows"] == 1


def test_tail_policy_sweep_candidate_cut_on_small_case():
    candidate = _load_candidate_module()
    spec = {"batch": 2, "n": 8, "cond": 1, "seed": 11}
    data = generate_input(**spec)

    assert parse_cut_tokens("candidate,0,2") == ["candidate", 0, 2]
    row = run_policy_cut(
        candidate,
        data,
        spec,
        "candidate",
        diagnose_output=True,
        max_factor_scaled=20.0,
        max_orth_scaled=100.0,
    )

    assert row["ok"], row
    assert row["margin_ok"], row
    assert row["candidate_policy_cut"] == 0
    assert row["tail_cut"] == 0
    assert row["cut_source"] == "candidate"
    assert row["strategy"] == "candidate_custom_kernel"
    assert row["diagnostics"]["factor_scaled_max"] < 20.0


def test_tail_policy_tune_plan_and_summary(tmp_path):
    config = parse_inline_config("probe:FAST_QR_DENSE_TAIL_CUT_512=12,FAST_QR_DENSE_TAIL_CUT_1024=64")
    assert config == {
        "name": "probe",
        "env": {
            "FAST_QR_DENSE_TAIL_CUT_512": "12",
            "FAST_QR_DENSE_TAIL_CUT_1024": "64",
        },
    }
    args = SimpleNamespace(
        submission="submissions/candidate.py",
        cases="cases/public_benchmarks.txt",
        correctness_indices="3",
        popcorn_seeds="0",
        max_factor_scaled=18.0,
        max_orth_scaled=80.0,
        repeats=2,
        official_stopping=True,
        max_time_ns=30e9,
        skip_correctness=False,
        skip_benchmark=False,
    )

    steps = command_plan(args, tmp_path, config, 0)
    assert [step["step"] for step in steps] == ["correctness", "benchmark"]
    assert "tools/tail_policy_sweep.py" in steps[0]["cmd"]
    assert "tools/bench_local.py" in steps[1]["cmd"]
    assert "--leaderboard-warmup" in steps[1]["cmd"]

    _write_jsonl(
        tmp_path / "000_probe_tail_policy_sweep.jsonl",
        [
            {
                "ok": True,
                "margin_ok": True,
                "diagnostics": {"factor_scaled_max": 2.0, "orth_scaled_max": 1.0},
            },
            {"summary": True, "ok": True, "num_rows": 1, "num_failed": 0},
        ],
    )
    _write_jsonl(
        tmp_path / "000_probe_benchmark.jsonl",
        [
            {"ok": True, "spec": {"batch": 1, "n": 8, "cond": 1, "seed": 1}, "mean_us": 10.0},
            {"geomean_us": 10.0, "num_cases": 1},
        ],
    )

    summary = summarize_run(tmp_path, [config])
    assert summary["best"]["name"] == "probe"
    assert summary["best"]["correctness"]["max_factor_scaled"] == pytest.approx(2.0)
    assert summary["best"]["benchmark"]["geomean_us"] == pytest.approx(10.0)

    assert should_skip_benchmark_after_correctness(correctness_failed=True, benchmark_failed_configs=False)
    assert not should_skip_benchmark_after_correctness(correctness_failed=True, benchmark_failed_configs=True)
    assert not should_skip_benchmark_after_correctness(correctness_failed=False, benchmark_failed_configs=False)

    failed_config = {"name": "failed_probe", "env": {"FAST_QR_DENSE_TAIL_CUT_512": "128"}}
    _write_jsonl(
        tmp_path / "001_failed_probe_tail_policy_sweep.jsonl",
        [
            {
                "ok": True,
                "margin_ok": False,
                "diagnostics": {"factor_scaled_max": 22.0, "orth_scaled_max": 1.0},
            },
            {"summary": True, "ok": False, "num_rows": 1, "num_failed": 1},
        ],
    )
    summary = summarize_run(tmp_path, [config, failed_config])
    failed_row = next(row for row in summary["results"] if row["name"] == "failed_probe")
    assert failed_row["correctness"]["num_failed"] == 1
    assert failed_row["correctness"]["max_factor_scaled"] == pytest.approx(22.0)
    assert failed_row["benchmark"] is None


def test_candidate_config_tune_grid_plan_and_summary(tmp_path):
    grid_args = SimpleNamespace(
        no_default=True,
        config_jsonl=None,
        config=["manual:FAST_QR_QR512_PANEL_B=24"],
        shape_label="qr512",
        env_prefix=None,
        panel_widths="16,32",
        precision_modes="fp32,tf32",
        tile_ms="",
        tile_ns="",
        warps_per_cta="",
        ctas_per_matrix="",
        cluster_sizes="",
        tail_cuts="0",
        panel_refreshes="",
        structured_before_cuda="",
        collect_resource_metrics=False,
        resource_cflags_env=None,
    )
    configs = load_candidate_tune_configs(grid_args)
    assert configs[0] == {"name": "manual", "env": {"FAST_QR_QR512_PANEL_B": "24"}}
    assert len(configs) == 5
    generated = configs[1]
    assert generated["env"]["FAST_QR_QR512_PANEL_B"] == "16"
    assert generated["env"]["FAST_QR_QR512_PRECISION_MODE"] == "fp32"
    assert generated["env"]["FAST_QR_QR512_TAIL_CUT"] == "0"

    direct_grid = candidate_grid_configs(
        SimpleNamespace(
            shape_label="qr1024",
            env_prefix="FAST_QR_CUSTOM",
            panel_widths="64",
            precision_modes="tf32",
            tile_ms="128",
            tile_ns="",
            warps_per_cta="4",
            ctas_per_matrix="2",
            cta_schedules="frontload",
            sync_free_auto_policy="0",
            auto_policy_groups="0",
            cluster_sizes="",
            tail_cuts="8",
            panel_refreshes="2",
            structured_before_cuda="1",
        )
    )
    assert direct_grid == [
        {
            "name": "qr1024__panel_b_64__precision_mode_tf32__tile_m_128__warps_per_cta_4__ctas_per_matrix_2__cta_schedule_frontload__sync_free_auto_policy_0__blocked_auto_groups_0__tail_cut_8__panel_refresh_2__structured_before_cuda_1",
            "env": {
                "FAST_QR_CUSTOM_PANEL_B": "64",
                "FAST_QR_CUSTOM_PRECISION_MODE": "tf32",
                "FAST_QR_CUSTOM_TILE_M": "128",
                "FAST_QR_CUSTOM_WARPS_PER_CTA": "4",
                "FAST_QR_CUSTOM_CTAS_PER_MATRIX": "2",
                "FAST_QR_CUSTOM_CTA_SCHEDULE": "frontload",
                "FAST_QR_CUSTOM_SYNC_FREE_AUTO_POLICY": "0",
                "FAST_QR_CUSTOM_BLOCKED_AUTO_GROUPS": "0",
                "FAST_QR_CUSTOM_TAIL_CUT": "8",
                "FAST_QR_CUSTOM_PANEL_REFRESH": "2",
                "FAST_QR_CUSTOM_STRUCTURED_BEFORE_CUDA": "1",
            },
        }
    ]

    resource_grid_args = SimpleNamespace(**{**grid_args.__dict__, "collect_resource_metrics": True})
    resource_config = with_resource_metric_flags(resource_grid_args, {"name": "probe", "env": {"FAST_QR_QR512_PANEL_B": "32"}})
    assert resource_config["env"]["FAST_QR_QR512_EXTRA_CUDA_CFLAGS"] == "--ptxas-options=-v"

    custom_resource_args = SimpleNamespace(
        **{**grid_args.__dict__, "collect_resource_metrics": True, "resource_cflags_env": "FAST_QR_CUSTOM_FLAGS"}
    )
    custom_resource_config = with_resource_metric_flags(
        custom_resource_args,
        {"name": "probe", "env": {"FAST_QR_CUSTOM_FLAGS": "-arch=sm_100"}},
    )
    assert custom_resource_config["env"]["FAST_QR_CUSTOM_FLAGS"] == "-arch=sm_100 --ptxas-options=-v"

    plan_args = SimpleNamespace(
        submission="submissions/candidate.py",
        correctness_cases="cases/public_tests.txt",
        correctness_indices="19,20,21",
        popcorn_seeds="public,1",
        max_factor_scaled=18.0,
        max_orth_scaled=80.0,
        skip_diagnostics=False,
        skip_correctness=False,
        skip_benchmark=False,
        benchmark_cases="cases/public_benchmarks.txt",
        benchmark_indices="3,7,9,10",
        repeats=2,
        benchmark_popcorn_seed=7,
        official_stopping=True,
        max_time_ns=30e9,
    )
    steps = candidate_config_command_plan(plan_args, tmp_path, generated, 0)
    assert [step["step"] for step in steps] == ["correctness", "benchmark"]
    assert "tools/seed_sweep.py" in steps[0]["cmd"]
    assert "--diagnose" in steps[0]["cmd"]
    assert "tools/bench_local.py" in steps[1]["cmd"]
    indices_pos = steps[1]["cmd"].index("--indices")
    assert steps[1]["cmd"][indices_pos + 1] == "3,7,9,10"
    assert "--leaderboard-warmup" in steps[1]["cmd"]
    seed_pos = steps[1]["cmd"].index("--popcorn-seed")
    assert steps[1]["cmd"][seed_pos + 1] == "7"

    _write_jsonl(
        Path(steps[0]["out"]),
        [
            {
                "ok": True,
                "margin_ok": True,
                "case_index": 19,
                "popcorn_seed": None,
                "diagnostics": {"factor_scaled_max": 2.0, "orth_scaled_max": 1.0},
            },
            {
                "ok": True,
                "margin_ok": True,
                "case_index": 19,
                "popcorn_seed": 1,
                "diagnostics": {"factor_scaled_max": 3.0, "orth_scaled_max": 1.0},
            },
        ],
    )
    _write_jsonl(
        Path(steps[1]["out"]),
        [
            {"ok": True, "spec": {"batch": 640, "n": 512, "cond": 2, "seed": 1029}, "mean_us": 8.0},
            {"geomean_us": 8.0, "num_cases": 1},
        ],
    )
    (tmp_path / "run.log").write_text(
        "\n".join(
            [
                f"$ python tools/seed_sweep.py --out {steps[0]['out']}",
                "ptxas info    : Function properties for geqrf512_kernel",
                "    8 bytes stack frame, 16 bytes spill stores, 0 bytes spill loads",
                "ptxas info    : Used 48 registers, 2048 bytes smem, 400 bytes cmem[0]",
            ]
        )
        + "\n"
    )
    resources = parse_ptxas_resource_metrics((tmp_path / "run.log").read_text())
    assert resources == [
        {
            "function": "geqrf512_kernel",
            "registers_per_thread": 48,
            "stack_frame_bytes": 8,
            "spill_store_bytes": 16,
            "spill_load_bytes": 0,
            "smem_bytes": 2048,
            "cmem_bytes": 400,
        }
    ]
    resource_summary = resource_metrics_by_config(
        tmp_path,
        [{**generated, "env": {**generated["env"], "FAST_QR_QR512_WARPS_PER_CTA": "4"}}],
    )[generated["name"]]
    assert resource_summary["available"] is True
    assert resource_summary["max_registers_per_thread"] == 48
    assert resource_summary["max_smem_bytes"] == 2048
    assert resource_summary["max_spill_store_bytes"] == 16
    assert resource_summary["entries"][0]["occupancy_estimate"]["threads_per_cta"] == 128
    assert resource_summary["min_estimated_occupancy"] is not None

    summary = summarize_candidate_config_run(tmp_path, [generated])
    assert summary["objective"] == "minimize_geomean_us"
    assert summary["num_configs_with_inert_env"] == 0
    assert summary["num_configs_with_cuda_route_bypassed_env"] == 1
    assert summary["num_configs_with_resource_metrics"] == 1
    assert summary["best"]["name"] == generated["name"]
    assert "FAST_QR_QR512_PANEL_B" in summary["best"]["env_consumption"]["candidate_consumed_env_keys"]
    assert "FAST_QR_QR512_PRECISION_MODE" in summary["best"]["env_consumption"]["candidate_consumed_env_keys"]
    assert "FAST_QR_QR512_TAIL_CUT" in summary["best"]["env_consumption"]["candidate_consumed_env_keys"]
    assert summary["best"]["env_consumption"]["inert_env_keys"] == []
    assert "FAST_QR_QR512_TAIL_CUT" in summary["best"]["env_consumption"]["cuda_route_bypassed_env_keys"]
    assert summary["best"]["correctness"]["num_failed"] == 0
    assert summary["best"]["correctness"]["max_factor_scaled"] == pytest.approx(3.0)
    assert summary["best"]["benchmark"]["geomean_us"] == pytest.approx(8.0)
    assert summary["best"]["resource_metrics"]["max_registers_per_thread"] == 48

    consumed_config = {
        "name": "thread_probe",
        "env": {
            "FAST_QR_QR512_PANEL_B": "32",
            "FAST_QR_QR512_TAIL_CUT": "24",
            "FAST_QR_QR512_WARPS_PER_CTA": "8",
            "FAST_QR_QR512_EXTRA_CUDA_CFLAGS": "--ptxas-options=-v",
            "FAST_QR_OCCUPANCY_REGISTERS_PER_SM": "65536",
        },
    }
    consumed_summary = summarize_candidate_config_run(tmp_path, [consumed_config])["results"][0]["env_consumption"]
    assert "FAST_QR_QR512_WARPS_PER_CTA" in consumed_summary["candidate_consumed_env_keys"]
    assert "FAST_QR_QR512_EXTRA_CUDA_CFLAGS" in consumed_summary["candidate_consumed_env_keys"]
    assert "FAST_QR_QR512_PANEL_B" in consumed_summary["candidate_consumed_env_keys"]
    assert "FAST_QR_QR512_TAIL_CUT" in consumed_summary["candidate_consumed_env_keys"]
    assert "FAST_QR_OCCUPANCY_REGISTERS_PER_SM" in consumed_summary["tuner_consumed_env_keys"]
    assert consumed_summary["cuda_route_bypassed_env_keys"] == ["FAST_QR_QR512_TAIL_CUT"]
    assert consumed_summary["inert_env_keys"] == []

    structured_first_config = {
        "name": "structured_first_tail_probe",
        "env": {
            "FAST_QR_QR512_STRUCTURED_BEFORE_CUDA": "1",
            "FAST_QR_QR512_TAIL_CUT": "24",
        },
    }
    structured_first_consumption = summarize_candidate_config_run(tmp_path, [structured_first_config])["results"][0][
        "env_consumption"
    ]
    assert "FAST_QR_QR512_STRUCTURED_BEFORE_CUDA" in structured_first_consumption["candidate_consumed_env_keys"]
    assert "FAST_QR_QR512_TAIL_CUT" in structured_first_consumption["candidate_consumed_env_keys"]
    assert structured_first_consumption["cuda_route_bypassed_env_keys"] == []

    disabled_cuda_config = {
        "name": "tail_policy_probe",
        "env": {
            "FAST_QR_DISABLE_QR512_CUDA": "1",
            "FAST_QR_QR512_TAIL_CUT": "24",
        },
    }
    disabled_consumption = summarize_candidate_config_run(tmp_path, [disabled_cuda_config])["results"][0][
        "env_consumption"
    ]
    assert "FAST_QR_QR512_TAIL_CUT" in disabled_consumption["candidate_consumed_env_keys"]
    assert disabled_consumption["cuda_route_bypassed_env_keys"] == []


def test_large_kernel_plan_generates_tuner_compatible_configs(tmp_path):
    configs512 = generate_large_kernel_configs("qr512", max_configs=4)
    assert len(configs512) == 4
    assert configs512[0]["name"].startswith("qr512__")
    assert configs512[0]["mode"] == "future-blocked"
    assert configs512[0]["effective_only"] is False
    assert configs512[0]["env"]["FAST_QR_QR512_PANEL_B"] in {"16", "32", "48", "64"}
    assert configs512[0]["env"]["FAST_QR_QR512_PANEL_REFRESH_MODE"] in {"none", "prefix"}
    assert configs512[0]["env"]["FAST_QR_QR512_PRECISION_MODE"] in {"fp32", "tf32", "fp16-input"}
    assert configs512[0]["env"]["FAST_QR_QR512_R_MAINTENANCE_MODE"] in {"none", "panel-prefix"}
    assert configs512[0]["env"]["FAST_QR_QR512_UPDATE_MODE"] in {"reflectors", "compact-wy"}
    assert configs512[0]["env"]["FAST_QR_QR512_COMPACT_WY_TILE_COLS"] in {"2", "4", "8"}
    assert configs512[0]["env"]["FAST_QR_QR512_WARPS_PER_CTA"] in {"4", "8", "16", "32"}
    assert configs512[0]["env"]["FAST_QR_QR512_CTA_SCHEDULE"] in {"fixed", "frontload"}
    assert configs512[0]["env"]["FAST_QR_QR512_TAIL_CUT"] in {"0", "16", "24", "32"}
    assert configs512[0]["env"]["FAST_QR_QR512_TAIL_FORCE"] in {"0", "1"}

    current512 = generate_large_kernel_configs("qr512", max_configs=8, mode="current-candidate")
    assert len(current512) == 8
    assert current512[0]["mode"] == "current-candidate"
    assert current512[0]["effective_only"] is True
    assert current512[0]["seed_name"] == "b200_default_sync_free_compact_wy_frontload_2cta"
    assert current512[0]["env"]["FAST_QR_QR512_PANEL_B"] == "32"
    assert current512[0]["env"]["FAST_QR_QR512_TILE_N"] == "128"
    assert current512[0]["env"]["FAST_QR_QR512_PANEL_REFRESH_MODE"] == "prefix"
    assert current512[0]["env"]["FAST_QR_QR512_R_MAINTENANCE_MODE"] == "panel-prefix"
    assert current512[0]["env"]["FAST_QR_QR512_UPDATE_MODE"] == "compact-wy"
    assert current512[0]["env"]["FAST_QR_QR512_CTAS_PER_MATRIX"] == "2"
    assert current512[0]["env"]["FAST_QR_QR512_CTA_SCHEDULE"] == "frontload"
    assert current512[0]["env"]["FAST_QR_QR512_SYNC_FREE_AUTO_POLICY"] == "1"
    assert current512[0]["env"]["FAST_QR_QR512_BLOCKED_AUTO_GROUPS"] == "1"
    assert current512[0]["env"]["FAST_QR_QR512_POLICY_FULL_SCAN"] == "1"
    assert current512[0]["env"]["FAST_QR_QR512_STRUCTURED_BEFORE_CUDA"] == "0"
    assert current512[0]["env"]["FAST_QR_QR512_TAIL_CUT"] == "32"
    assert current512[0]["env"]["FAST_QR_QR512_TAIL_FORCE"] == "0"
    assert current512[1]["seed_name"] == "sync_free_repair"
    assert current512[1]["env"]["FAST_QR_QR512_SYNC_FREE_AUTO_POLICY"] == "1"
    assert current512[1]["env"]["FAST_QR_QR512_TAIL_FORCE"] == "1"
    assert current512[2]["seed_name"] == "sparse_policy_repair"
    assert current512[2]["env"]["FAST_QR_QR512_POLICY_FULL_SCAN"] == "0"
    assert any(row.get("seed_name") == "structured_first_repair" for row in current512)
    assert {tuple(sorted(row["env"])) for row in current512} == {
        (
            "FAST_QR_ENABLE_QR512_BLOCKED_CUDA",
            "FAST_QR_QR512_BLOCKED_AUTO_GROUPS",
            "FAST_QR_QR512_COMPACT_WY_TILE_COLS",
            "FAST_QR_QR512_CTAS_PER_MATRIX",
            "FAST_QR_QR512_CTA_SCHEDULE",
            "FAST_QR_QR512_PANEL_B",
            "FAST_QR_QR512_PANEL_REFRESH_MODE",
            "FAST_QR_QR512_POLICY_FULL_SCAN",
            "FAST_QR_QR512_PRECISION_MODE",
            "FAST_QR_QR512_R_MAINTENANCE_MODE",
            "FAST_QR_QR512_STRUCTURED_BEFORE_CUDA",
            "FAST_QR_QR512_SYNC_FREE_AUTO_POLICY",
            "FAST_QR_QR512_TAIL_CUT",
            "FAST_QR_QR512_TAIL_FORCE",
            "FAST_QR_QR512_TAIL_THRESHOLD",
            "FAST_QR_QR512_TILE_N",
            "FAST_QR_QR512_UPDATE_MODE",
            "FAST_QR_QR512_WARPS_PER_CTA",
        ),
    }
    assert {row["env"]["FAST_QR_QR512_PANEL_B"] for row in current512} <= {"16", "32", "48", "64"}
    assert {row["env"]["FAST_QR_QR512_PANEL_REFRESH_MODE"] for row in current512} <= {"none", "prefix"}
    assert {row["env"]["FAST_QR_QR512_PRECISION_MODE"] for row in current512} <= {"fp32", "tf32", "fp16-input"}
    assert {row["env"]["FAST_QR_QR512_R_MAINTENANCE_MODE"] for row in current512} <= {"none", "panel-prefix"}
    assert {row["env"]["FAST_QR_QR512_TILE_N"] for row in current512} <= {"64", "128"}
    assert {row["env"]["FAST_QR_QR512_UPDATE_MODE"] for row in current512} <= {"reflectors", "compact-wy"}
    assert {row["env"]["FAST_QR_QR512_COMPACT_WY_TILE_COLS"] for row in current512} <= {"2", "4", "8"}
    assert {row["env"]["FAST_QR_QR512_TAIL_CUT"] for row in current512} <= {"0", "16", "24", "32"}
    assert {row["env"]["FAST_QR_QR512_TAIL_FORCE"] for row in current512} <= {"0", "1"}
    assert {row["env"]["FAST_QR_QR512_TAIL_FORCE"] for row in current512} >= {"0", "1"}
    assert {row["env"]["FAST_QR_QR512_TAIL_THRESHOLD"] for row in current512} <= {"0.0", "0.03"}
    assert {row["env"]["FAST_QR_QR512_WARPS_PER_CTA"] for row in current512} <= {"4", "8", "16", "32"}
    assert {row["env"]["FAST_QR_QR512_CTAS_PER_MATRIX"] for row in current512} <= {"1", "2"}
    assert {row["env"]["FAST_QR_QR512_CTA_SCHEDULE"] for row in current512} <= {"fixed", "frontload"}
    assert {row["env"]["FAST_QR_QR512_SYNC_FREE_AUTO_POLICY"] for row in current512} <= {"0", "1"}
    assert {row["env"]["FAST_QR_QR512_BLOCKED_AUTO_GROUPS"] for row in current512} <= {"0", "1"}
    assert {row["env"]["FAST_QR_QR512_POLICY_FULL_SCAN"] for row in current512} <= {"0", "1"}
    assert {row["env"]["FAST_QR_QR512_STRUCTURED_BEFORE_CUDA"] for row in current512} == {"0", "1"}

    repair_only512 = generate_large_kernel_configs(
        "qr512",
        max_configs=8,
        mode="current-candidate",
        axis_overrides={
            "panel_refresh_modes": ["prefix"],
            "r_maintenance_modes": ["panel-prefix"],
        },
    )
    assert {row["env"]["FAST_QR_QR512_PANEL_REFRESH_MODE"] for row in repair_only512} == {"prefix"}
    assert {row["env"]["FAST_QR_QR512_R_MAINTENANCE_MODE"] for row in repair_only512} == {"panel-prefix"}
    assert repair_only512[0]["seed_name"] == "b200_default_sync_free_compact_wy_frontload_2cta"
    with pytest.raises(ValueError):
        generate_large_kernel_configs("qr512", axis_overrides={"not_an_axis": ["1"]})

    cmd512 = large_kernel_tune_command("qr512", "results/qr512_kernel_configs.jsonl", repeats=2)
    assert cmd512[cmd512.index("--benchmark-indices") + 1] == "3,7,9,10"
    assert cmd512[cmd512.index("--correctness-indices") + 1] == "3,6,7,8,9,10,11,19"

    configs1024 = generate_large_kernel_configs("qr1024", max_configs=4)
    assert len(configs1024) == 4
    assert configs1024[0]["name"].startswith("qr1024__")
    assert configs1024[0]["env"]["FAST_QR_QR1024_PANEL_B"] in {"32", "48", "64", "96"}
    assert configs1024[0]["env"]["FAST_QR_QR1024_PANEL_REFRESH_MODE"] in {"none", "prefix"}
    assert configs1024[0]["env"]["FAST_QR_QR1024_PRECISION_MODE"] in {"fp32", "tf32", "fp16-input"}
    assert configs1024[0]["env"]["FAST_QR_QR1024_R_MAINTENANCE_MODE"] in {"none", "panel-prefix"}
    assert configs1024[0]["env"]["FAST_QR_QR1024_UPDATE_MODE"] in {"reflectors", "compact-wy"}
    assert configs1024[0]["env"]["FAST_QR_QR1024_COMPACT_WY_TILE_COLS"] in {"2", "4", "8"}
    assert configs1024[0]["env"]["FAST_QR_QR1024_WARPS_PER_CTA"] in {"8", "16", "32"}
    assert configs1024[0]["env"]["FAST_QR_QR1024_CTA_SCHEDULE"] in {"fixed", "frontload"}
    assert configs1024[0]["env"]["FAST_QR_QR1024_TAIL_CUT"] in {"0", "8", "64", "128"}
    assert configs1024[0]["env"]["FAST_QR_QR1024_TAIL_FORCE"] in {"0", "1"}

    current1024 = generate_large_kernel_configs("qr1024", max_configs=8, mode="current-candidate")
    assert len(current1024) == 8
    assert current1024[0]["seed_name"] == "b200_default_sync_free_compact_wy_frontload_2cta"
    assert current1024[0]["env"]["FAST_QR_QR1024_PANEL_B"] == "32"
    assert current1024[0]["env"]["FAST_QR_QR1024_TILE_N"] == "128"
    assert current1024[0]["env"]["FAST_QR_QR1024_PANEL_REFRESH_MODE"] == "prefix"
    assert current1024[0]["env"]["FAST_QR_QR1024_R_MAINTENANCE_MODE"] == "panel-prefix"
    assert current1024[0]["env"]["FAST_QR_QR1024_UPDATE_MODE"] == "compact-wy"
    assert current1024[0]["env"]["FAST_QR_QR1024_CTAS_PER_MATRIX"] == "2"
    assert current1024[0]["env"]["FAST_QR_QR1024_CTA_SCHEDULE"] == "frontload"
    assert current1024[0]["env"]["FAST_QR_QR1024_SYNC_FREE_AUTO_POLICY"] == "1"
    assert current1024[0]["env"]["FAST_QR_QR1024_POLICY_FULL_SCAN"] == "1"
    assert current1024[0]["env"]["FAST_QR_QR1024_TAIL_CUT"] == "64"
    assert current1024[0]["env"]["FAST_QR_QR1024_TAIL_FORCE"] == "0"
    assert any(row.get("seed_name") == "sparse_policy_repair" for row in current1024)
    assert any(row.get("seed_name") == "mixed_safe_tail_repair" for row in current1024)
    assert {row["env"]["FAST_QR_QR1024_TAIL_FORCE"] for row in current1024} >= {"0", "1"}
    assert {tuple(sorted(row["env"])) for row in current1024} == {
        (
            "FAST_QR_ENABLE_QR1024_BLOCKED_CUDA",
            "FAST_QR_QR1024_BLOCKED_AUTO_GROUPS",
            "FAST_QR_QR1024_COMPACT_WY_TILE_COLS",
            "FAST_QR_QR1024_CTAS_PER_MATRIX",
            "FAST_QR_QR1024_CTA_SCHEDULE",
            "FAST_QR_QR1024_PANEL_B",
            "FAST_QR_QR1024_PANEL_REFRESH_MODE",
            "FAST_QR_QR1024_POLICY_FULL_SCAN",
            "FAST_QR_QR1024_PRECISION_MODE",
            "FAST_QR_QR1024_R_MAINTENANCE_MODE",
            "FAST_QR_QR1024_STRUCTURED_BEFORE_CUDA",
            "FAST_QR_QR1024_SYNC_FREE_AUTO_POLICY",
            "FAST_QR_QR1024_TAIL_CUT",
            "FAST_QR_QR1024_TAIL_FORCE",
            "FAST_QR_QR1024_TAIL_THRESHOLD",
            "FAST_QR_QR1024_TILE_N",
            "FAST_QR_QR1024_UPDATE_MODE",
            "FAST_QR_QR1024_WARPS_PER_CTA",
        ),
    }
    assert {row["env"]["FAST_QR_QR1024_PANEL_B"] for row in current1024} <= {"32", "48", "64", "96"}
    assert {row["env"]["FAST_QR_QR1024_PANEL_REFRESH_MODE"] for row in current1024} <= {"none", "prefix"}
    assert {row["env"]["FAST_QR_QR1024_PRECISION_MODE"] for row in current1024} <= {"fp32", "tf32", "fp16-input"}
    assert {row["env"]["FAST_QR_QR1024_R_MAINTENANCE_MODE"] for row in current1024} <= {"none", "panel-prefix"}
    assert {row["env"]["FAST_QR_QR1024_TILE_N"] for row in current1024} <= {"128", "256"}
    assert {row["env"]["FAST_QR_QR1024_UPDATE_MODE"] for row in current1024} <= {"reflectors", "compact-wy"}
    assert {row["env"]["FAST_QR_QR1024_COMPACT_WY_TILE_COLS"] for row in current1024} <= {"2", "4", "8"}
    assert {row["env"]["FAST_QR_QR1024_TAIL_CUT"] for row in current1024} <= {"0", "8", "64", "128"}
    assert {row["env"]["FAST_QR_QR1024_TAIL_FORCE"] for row in current1024} <= {"0", "1"}
    assert {row["env"]["FAST_QR_QR1024_TAIL_THRESHOLD"] for row in current1024} <= {"0.0", "0.03"}
    assert {row["env"]["FAST_QR_QR1024_WARPS_PER_CTA"] for row in current1024} <= {"8", "16", "32"}
    assert {row["env"]["FAST_QR_QR1024_CTAS_PER_MATRIX"] for row in current1024} <= {"1", "2", "4"}
    assert {row["env"]["FAST_QR_QR1024_CTA_SCHEDULE"] for row in current1024} <= {"fixed", "frontload"}
    assert {row["env"]["FAST_QR_QR1024_SYNC_FREE_AUTO_POLICY"] for row in current1024} <= {"0", "1"}
    assert {row["env"]["FAST_QR_QR1024_BLOCKED_AUTO_GROUPS"] for row in current1024} <= {"0", "1"}
    assert {row["env"]["FAST_QR_QR1024_POLICY_FULL_SCAN"] for row in current1024} <= {"0", "1"}
    assert {row["env"]["FAST_QR_QR1024_STRUCTURED_BEFORE_CUDA"] for row in current1024} == {"0", "1"}

    cmd1024 = large_kernel_tune_command("qr1024", "results/qr1024_kernel_configs.jsonl", repeats=2)
    assert cmd1024[cmd1024.index("--benchmark-indices") + 1] == "4,8,11"
    assert cmd1024[cmd1024.index("--correctness-indices") + 1] == "4,12,13,14,15,20"

    configs2048 = generate_large_kernel_configs("qr2048", max_configs=5)
    assert len(configs2048) == 5
    assert configs2048[0]["name"].startswith("qr2048__")
    assert configs2048[0]["env"]["FAST_QR_QR2048_PANEL_B"] in {"32", "64", "96"}
    assert configs2048[0]["env"]["FAST_QR_QR2048_COMPACT_WY_TILE_COLS"] in {"1", "2", "4"}
    assert configs2048[0]["env"]["FAST_QR_QR2048_CTAS_PER_MATRIX"] in {"4", "8"}
    assert configs2048[0]["env"]["FAST_QR_QR2048_CTA_SCHEDULE"] in {"fixed", "frontload", "all-tiles"}
    assert configs2048[0]["env"]["FAST_QR_QR2048_TAIL_CUT"] in {"0", "64"}
    assert configs2048[0]["env"]["FAST_QR_QR2048_TAIL_THRESHOLD"] in {"0.0", "0.1", "0.2"}
    assert configs2048[0]["env"]["FAST_QR_QR2048_TAIL_FORCE"] in {"0", "1"}

    current2048 = generate_large_kernel_configs("qr2048", max_configs=8, mode="current-candidate")
    assert len(current2048) == 8
    assert current2048[0]["seed_name"] == "b200_default_compact_wy_all_tiles_8cta"
    assert current2048[0]["env"]["FAST_QR_QR2048_PANEL_B"] == "64"
    assert current2048[0]["env"]["FAST_QR_QR2048_UPDATE_MODE"] == "compact-wy"
    assert current2048[0]["env"]["FAST_QR_QR2048_TILE_N"] == "128"
    assert current2048[0]["env"]["FAST_QR_QR2048_COMPACT_WY_TILE_COLS"] == "2"
    assert current2048[0]["env"]["FAST_QR_QR2048_CTAS_PER_MATRIX"] == "8"
    assert current2048[0]["env"]["FAST_QR_QR2048_CTA_SCHEDULE"] == "all-tiles"
    assert current2048[0]["env"]["FAST_QR_QR2048_SYNC_FREE_AUTO_POLICY"] == "1"
    assert current2048[0]["env"]["FAST_QR_QR2048_TAIL_CUT"] == "64"
    assert current2048[0]["env"]["FAST_QR_QR2048_TAIL_THRESHOLD"] == "0.2"
    assert {tuple(sorted(row["env"])) for row in current2048} == {
        (
            "FAST_QR_ENABLE_QR2048_BLOCKED_CUDA",
            "FAST_QR_QR2048_BLOCKED_AUTO_GROUPS",
            "FAST_QR_QR2048_COMPACT_WY_TILE_COLS",
            "FAST_QR_QR2048_CTAS_PER_MATRIX",
            "FAST_QR_QR2048_CTA_SCHEDULE",
            "FAST_QR_QR2048_PANEL_B",
            "FAST_QR_QR2048_PANEL_REFRESH_MODE",
            "FAST_QR_QR2048_POLICY_FULL_SCAN",
            "FAST_QR_QR2048_PRECISION_MODE",
            "FAST_QR_QR2048_R_MAINTENANCE_MODE",
            "FAST_QR_QR2048_SYNC_FREE_AUTO_POLICY",
            "FAST_QR_QR2048_TAIL_CUT",
            "FAST_QR_QR2048_TAIL_FORCE",
            "FAST_QR_QR2048_TAIL_THRESHOLD",
            "FAST_QR_QR2048_TILE_N",
            "FAST_QR_QR2048_UPDATE_MODE",
            "FAST_QR_QR2048_WARPS_PER_CTA",
        ),
    }
    assert {row["env"]["FAST_QR_ENABLE_QR2048_BLOCKED_CUDA"] for row in current2048} == {"1"}
    assert {row["env"]["FAST_QR_QR2048_PANEL_B"] for row in current2048} <= {"32", "64", "96"}
    assert {row["env"]["FAST_QR_QR2048_PANEL_REFRESH_MODE"] for row in current2048} <= {"none", "prefix"}
    assert {row["env"]["FAST_QR_QR2048_PRECISION_MODE"] for row in current2048} <= {"fp32", "tf32", "fp16-input"}
    assert {row["env"]["FAST_QR_QR2048_R_MAINTENANCE_MODE"] for row in current2048} <= {"none", "panel-prefix"}
    assert {row["env"]["FAST_QR_QR2048_TILE_N"] for row in current2048} <= {"128", "256"}
    assert {row["env"]["FAST_QR_QR2048_UPDATE_MODE"] for row in current2048} <= {"reflectors", "compact-wy"}
    assert {row["env"]["FAST_QR_QR2048_COMPACT_WY_TILE_COLS"] for row in current2048} <= {"1", "2", "4"}
    assert {row["env"]["FAST_QR_QR2048_WARPS_PER_CTA"] for row in current2048} <= {"4", "8"}
    assert {row["env"]["FAST_QR_QR2048_CTAS_PER_MATRIX"] for row in current2048} <= {"4", "8"}
    assert {row["env"]["FAST_QR_QR2048_CTA_SCHEDULE"] for row in current2048} <= {
        "fixed",
        "frontload",
        "all-tiles",
    }
    assert {row["env"]["FAST_QR_QR2048_SYNC_FREE_AUTO_POLICY"] for row in current2048} <= {"0", "1"}
    assert {row["env"]["FAST_QR_QR2048_BLOCKED_AUTO_GROUPS"] for row in current2048} <= {"0", "1"}
    assert {row["env"]["FAST_QR_QR2048_POLICY_FULL_SCAN"] for row in current2048} <= {"0", "1"}
    assert {row["env"]["FAST_QR_QR2048_TAIL_CUT"] for row in current2048} <= {"0", "64"}
    assert {row["env"]["FAST_QR_QR2048_TAIL_FORCE"] for row in current2048} <= {"0", "1"}
    assert {row["env"]["FAST_QR_QR2048_TAIL_FORCE"] for row in current2048} >= {"0", "1"}
    assert {row["env"]["FAST_QR_QR2048_TAIL_THRESHOLD"] for row in current2048} <= {"0.0", "0.1", "0.2"}

    config_path = tmp_path / "qr2048_large_configs.jsonl"
    _write_jsonl(config_path, [{"name": row["name"], "env": row["env"]} for row in configs2048])
    loader_args = SimpleNamespace(
        no_default=True,
        config_jsonl=str(config_path),
        config=[],
        shape_label="qr2048",
        env_prefix=None,
        panel_widths="",
        precision_modes="",
        tile_ms="",
        tile_ns="",
        warps_per_cta="",
        ctas_per_matrix="",
        cluster_sizes="",
        tail_cuts="",
        panel_refreshes="",
        structured_before_cuda="",
    )
    loaded = load_candidate_tune_configs(loader_args)
    assert loaded == [{"name": row["name"], "env": row["env"]} for row in configs2048]

    cmd = large_kernel_tune_command("qr2048", str(config_path), repeats=2, official_stopping=True)
    assert "tools/tune_candidate_configs.py" in cmd
    assert "--config-jsonl" in cmd
    assert str(config_path) in cmd
    assert "--benchmark-indices" in cmd
    assert cmd[cmd.index("--benchmark-indices") + 1] == "5"
    assert "--correctness-indices" in cmd
    assert cmd[cmd.index("--correctness-indices") + 1] == "16,21"
    assert "--official-stopping" in cmd

    configs4096 = generate_large_kernel_configs("qr4096", max_configs=3, env_prefix="FAST_QR_BIG")
    assert len(configs4096) == 3
    assert configs4096[0]["env"]["FAST_QR_BIG_PANEL_B"] in {"64", "96"}
    assert configs4096[0]["env"]["FAST_QR_BIG_COMPACT_WY_TILE_COLS"] in {"1", "2", "4"}
    assert configs4096[0]["env"]["FAST_QR_BIG_CTAS_PER_MATRIX"] in {"8", "16"}
    assert configs4096[0]["env"]["FAST_QR_BIG_CTA_SCHEDULE"] in {"fixed", "frontload", "all-tiles"}
    assert configs4096[0]["env"]["FAST_QR_BIG_TAIL_CUT"] in {"0", "128"}
    assert configs4096[0]["env"]["FAST_QR_BIG_TAIL_THRESHOLD"] in {"0.0", "0.1", "0.2"}
    assert configs4096[0]["env"]["FAST_QR_BIG_TAIL_FORCE"] in {"0", "1"}

    current4096 = generate_large_kernel_configs("qr4096", max_configs=8, env_prefix="FAST_QR_BIG", mode="current-candidate")
    assert len(current4096) == 8
    assert current4096[0]["seed_name"] == "b200_default_compact_wy_all_tiles_16cta"
    assert current4096[0]["env"]["FAST_QR_QR4096_PANEL_B"] == "64"
    assert current4096[0]["env"]["FAST_QR_QR4096_UPDATE_MODE"] == "compact-wy"
    assert current4096[0]["env"]["FAST_QR_QR4096_TILE_N"] == "256"
    assert current4096[0]["env"]["FAST_QR_QR4096_COMPACT_WY_TILE_COLS"] == "2"
    assert current4096[0]["env"]["FAST_QR_QR4096_CTAS_PER_MATRIX"] == "16"
    assert current4096[0]["env"]["FAST_QR_QR4096_CTA_SCHEDULE"] == "all-tiles"
    assert current4096[0]["env"]["FAST_QR_QR4096_SYNC_FREE_AUTO_POLICY"] == "1"
    assert current4096[0]["env"]["FAST_QR_QR4096_TAIL_CUT"] == "128"
    assert current4096[0]["env"]["FAST_QR_QR4096_TAIL_THRESHOLD"] == "0.2"
    assert {tuple(sorted(row["env"])) for row in current4096} == {
        (
            "FAST_QR_ENABLE_QR4096_BLOCKED_CUDA",
            "FAST_QR_QR4096_BLOCKED_AUTO_GROUPS",
            "FAST_QR_QR4096_COMPACT_WY_TILE_COLS",
            "FAST_QR_QR4096_CTAS_PER_MATRIX",
            "FAST_QR_QR4096_CTA_SCHEDULE",
            "FAST_QR_QR4096_PANEL_B",
            "FAST_QR_QR4096_PANEL_REFRESH_MODE",
            "FAST_QR_QR4096_POLICY_FULL_SCAN",
            "FAST_QR_QR4096_PRECISION_MODE",
            "FAST_QR_QR4096_R_MAINTENANCE_MODE",
            "FAST_QR_QR4096_SYNC_FREE_AUTO_POLICY",
            "FAST_QR_QR4096_TAIL_CUT",
            "FAST_QR_QR4096_TAIL_FORCE",
            "FAST_QR_QR4096_TAIL_THRESHOLD",
            "FAST_QR_QR4096_TILE_N",
            "FAST_QR_QR4096_UPDATE_MODE",
            "FAST_QR_QR4096_WARPS_PER_CTA",
        ),
    }
    assert {row["env"]["FAST_QR_ENABLE_QR4096_BLOCKED_CUDA"] for row in current4096} == {"1"}
    assert {row["env"]["FAST_QR_QR4096_PANEL_B"] for row in current4096} <= {"64", "96"}
    assert {row["env"]["FAST_QR_QR4096_PANEL_REFRESH_MODE"] for row in current4096} <= {"none", "prefix"}
    assert {row["env"]["FAST_QR_QR4096_PRECISION_MODE"] for row in current4096} <= {"fp32", "tf32", "fp16-input"}
    assert {row["env"]["FAST_QR_QR4096_R_MAINTENANCE_MODE"] for row in current4096} <= {"none", "panel-prefix"}
    assert {row["env"]["FAST_QR_QR4096_TILE_N"] for row in current4096} <= {"256", "512"}
    assert {row["env"]["FAST_QR_QR4096_UPDATE_MODE"] for row in current4096} <= {"reflectors", "compact-wy"}
    assert {row["env"]["FAST_QR_QR4096_COMPACT_WY_TILE_COLS"] for row in current4096} <= {"1", "2", "4"}
    assert {row["env"]["FAST_QR_QR4096_WARPS_PER_CTA"] for row in current4096} <= {"8"}
    assert {row["env"]["FAST_QR_QR4096_CTAS_PER_MATRIX"] for row in current4096} <= {"8", "16"}
    assert {row["env"]["FAST_QR_QR4096_CTA_SCHEDULE"] for row in current4096} <= {
        "fixed",
        "frontload",
        "all-tiles",
    }
    assert {row["env"]["FAST_QR_QR4096_SYNC_FREE_AUTO_POLICY"] for row in current4096} <= {"0", "1"}
    assert {row["env"]["FAST_QR_QR4096_BLOCKED_AUTO_GROUPS"] for row in current4096} <= {"0", "1"}
    assert {row["env"]["FAST_QR_QR4096_POLICY_FULL_SCAN"] for row in current4096} <= {"0", "1"}
    assert {row["env"]["FAST_QR_QR4096_TAIL_CUT"] for row in current4096} <= {"0", "128"}
    assert {row["env"]["FAST_QR_QR4096_TAIL_FORCE"] for row in current4096} <= {"0", "1"}
    assert {row["env"]["FAST_QR_QR4096_TAIL_FORCE"] for row in current4096} >= {"0", "1"}
    assert {row["env"]["FAST_QR_QR4096_TAIL_THRESHOLD"] for row in current4096} <= {"0.0", "0.1", "0.2"}


def test_large_kernel_plan_cli_out_is_quiet_and_print_command_is_command_only(tmp_path):
    out_path = tmp_path / "qr512_configs.jsonl"
    completed = subprocess.run(
        [
            sys.executable,
            "tools/large_kernel_plan.py",
            "--shape-label",
            "qr512",
            "--mode",
            "current-candidate",
            "--max-configs",
            "2",
            "--out",
            str(out_path),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert completed.stdout == ""
    assert len(out_path.read_text().splitlines()) == 2

    command_out = tmp_path / "qr512_command_configs.jsonl"
    command_completed = subprocess.run(
        [
            sys.executable,
            "tools/large_kernel_plan.py",
            "--shape-label",
            "qr512",
            "--mode",
            "current-candidate",
            "--max-configs",
            "2",
            "--out",
            str(command_out),
            "--print-command",
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines = command_completed.stdout.strip().splitlines()
    assert len(lines) == 1
    assert "tools/tune_candidate_configs.py" in lines[0]
    assert str(command_out) in lines[0]
