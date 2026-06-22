import json
from pathlib import Path
import subprocess
import sys
import uuid


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from qr_common import (  # noqa: E402
    apply_popcorn_seed,
    batch_count,
    combine_seed,
    file_provenance,
    load_cases,
    parse_case,
    repo_provenance,
)
from spec_utils import render_case_file, specs_from_task_yml  # noqa: E402
from spec_utils import benchmark_shape_collisions, custom_kernel_interface, format_custom_kernel_interface  # noqa: E402
from spec_utils import evaluator_benchmark_contract  # noqa: E402
from validate_b200_suite import EXPECTED_DEFAULT_STEPS  # noqa: E402


def test_public_benchmark_cases_are_current_shape_set():
    cases = load_cases(ROOT / "cases/public_benchmarks.txt")
    assert len(cases) == 12
    assert cases[0] == {"batch": 20, "n": 32, "cond": 1, "seed": 43214}
    assert cases[-1] == {"batch": 60, "n": 1024, "cond": 0, "seed": 770005, "case": "nearrank"}
    assert {case.get("case", "dense") for case in cases} >= {"dense", "mixed", "rankdef", "clustered", "nearrank"}
    assert cases == specs_from_task_yml("benchmarks")


def test_public_benchmark_shape_collisions_require_data_dependent_dispatch():
    collisions = benchmark_shape_collisions(specs_from_task_yml("benchmarks"))
    assert collisions == [
        {
            "batch": 640,
            "n": 512,
            "indexes": [3, 7, 9, 10],
            "cases": ["dense", "mixed", "rankdef", "clustered"],
            "specs": [
                "batch: 640; n: 512; cond: 2; seed: 1029",
                "batch: 640; n: 512; cond: 2; seed: 770001; case: mixed",
                "batch: 640; n: 512; cond: 0; seed: 770003; case: rankdef",
                "batch: 640; n: 512; cond: 0; seed: 770004; case: clustered",
            ],
        },
        {
            "batch": 60,
            "n": 1024,
            "indexes": [4, 8, 11],
            "cases": ["dense", "mixed", "nearrank"],
            "specs": [
                "batch: 60; n: 1024; cond: 2; seed: 75342",
                "batch: 60; n: 1024; cond: 2; seed: 770002; case: mixed",
                "batch: 60; n: 1024; cond: 0; seed: 770005; case: nearrank",
            ],
        },
    ]


def test_official_submission_interface_only_passes_data():
    interface = custom_kernel_interface(ROOT / "official/submission.py")
    assert format_custom_kernel_interface(interface) == "custom_kernel(data)"
    assert interface["required_positional_args"] == 1
    assert interface["vararg"] is None
    assert interface["kwarg"] is None


def test_public_correctness_tests_are_encoded():
    cases = load_cases(ROOT / "cases/public_tests.txt")
    assert len(cases) == 22
    assert cases[19]["case"] == "mixed"
    assert cases[21] == {"batch": 2, "n": 2048, "cond": 2, "seed": 224468, "case": "mixed"}
    assert cases == specs_from_task_yml("tests")


def test_generated_case_files_are_in_sync_with_task_yml():
    assert (ROOT / "cases/public_tests.txt").read_text() == render_case_file(specs_from_task_yml("tests"))
    assert (ROOT / "cases/public_benchmarks.txt").read_text() == render_case_file(
        specs_from_task_yml("benchmarks")
    )


def test_print_spec_reports_interface_and_shape_collisions():
    completed = subprocess.run(
        [sys.executable, "tools/print_spec.py"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert "submission_interface: custom_kernel(data)" in completed.stdout
    assert "case_metadata_passed_to_submission: false" in completed.stdout
    assert "submission_info_sources: data.shape, data.dtype, data.device, tensor_values" in completed.stdout
    assert "benchmark_imports_submission_in_worker: true" in completed.stdout
    assert "benchmark_calls_custom_kernel_before_timing: true" in completed.stdout
    assert "benchmark_calls_custom_kernel_inside_timed_loop: true" in completed.stdout
    assert "benchmark_rechecks_timed_outputs_when_requested: true" in completed.stdout
    assert "benchmark_clears_l2_inside_timed_loop: true" in completed.stdout
    assert "benchmark_shape_collisions: 2" in completed.stdout
    assert "shape_only_case_selection_sufficient: false" in completed.stdout
    assert "batch: 640; n: 512 -> cases: dense, mixed, rankdef, clustered" in completed.stdout
    assert "batch: 60; n: 1024 -> cases: dense, mixed, nearrank" in completed.stdout


def test_eval_contract_warms_submission_before_timed_loop():
    contract = evaluator_benchmark_contract(ROOT / "official/eval.py")
    assert contract["submission_imported_inside_benchmark_worker"] is True
    assert contract["custom_kernel_called_before_timing"] is True
    assert contract["custom_kernel_called_inside_timed_loop"] is True
    assert contract["timed_outputs_rechecked_when_requested"] is True
    assert contract["l2_cache_cleared_inside_timed_loop"] is True
    assert contract["first_pre_timing_custom_kernel_line"] < contract["first_timed_custom_kernel_line"]


def test_route_trace_records_dispatch_information_sources():
    trace_source = (ROOT / "tools/trace_candidate_routes.py").read_text()
    validate_source = (ROOT / "tools/validate_b200_suite.py").read_text()
    analyze_source = (ROOT / "tools/analyze_b200_results.py").read_text()
    readme = (ROOT / "README.md").read_text()

    for source in [trace_source, validate_source, analyze_source, readme]:
        assert "case_metadata_passed_to_submission" in source
        assert "case_selection_info_sources" in source
        assert "shape_only_case_selection" in source
        assert "uses_tensor_values_for_dispatch" in source
        assert "uses_tensor_values_for_case_selection" in source
        assert "route_decision_source" in source
        assert "dispatch_info_sources" in source

    assert "shape_requires_tensor_route" in trace_source
    assert "n in (512, 1024)" in trace_source


def test_dev_robustness_cases_cover_non_public_profiles():
    cases = load_cases(ROOT / "cases/dev_robustness.txt")
    assert len(cases) == 11
    assert {"rowscale", "nearcollinear", "band"}.issubset({case.get("case") for case in cases})


def test_expected_repo_entrypoints_exist():
    for path in [
        "official/reference.py",
        "official/task.py",
        "official/eval.py",
        "official/submission.py",
        "official/task.yml",
        "official/utils.py",
        "official/UPSTREAM_COMMIT",
        "official/FETCHED_AT",
        "submissions/baseline_geqrf.py",
        "submissions/candidate.py",
        "tools/bench_local.py",
        "tools/analyze_b200_results.py",
        "tools/audit_secrets.py",
        "tools/benchmark_guards.py",
        "tools/blocked_qr_reference.py",
        "tools/blocked_qr_sweep.py",
        "tools/candidate_policy.py",
        "tools/classifier_seed_sweep.py",
        "tools/check_cases.py",
        "tools/check_b200_env.py",
        "tools/check_one.py",
        "tools/diagnose.py",
        "tools/experiments.py",
        "tools/fetch_official.py",
        "tools/implementation_status.py",
        "tools/large_kernel_plan.py",
        "tools/mixed_seed_sweep.py",
        "tools/preflight_accelerators.py",
        "tools/print_spec.py",
        "tools/quantization_seed_sweep.py",
        "tools/run_b200_suite.py",
        "tools/run_local_checks.py",
        "tools/run_official_eval.py",
        "tools/seed_sweep.py",
        "tools/spec_utils.py",
        "tools/submit_popcorn.py",
        "tools/sweep.py",
        "tools/summarize_suite.py",
        "tools/sync_cases_from_task_yml.py",
        "tools/tail_policy_sweep.py",
        "tools/trace_candidate_routes.py",
        "tools/tune_candidate_configs.py",
        "tools/tune_tail_policy.py",
        "tools/validate_b200_suite.py",
        "tools/validate_local_checks.py",
        "tools/validate_submission.py",
        "results/runs.jsonl",
    ]:
        assert (ROOT / path).is_file()


def test_gitignore_keeps_generated_result_exports_untracked():
    ignored = subprocess.run(
        [
            "git",
            "check-ignore",
            "results/example_suite/run.log",
            "results/example_suite/suite_summary.md",
            "results/example_suite.tgz",
            "results/baseline.jsonl",
        ],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert ignored.returncode == 0, ignored.stderr

    tracked_placeholders = subprocess.run(
        ["git", "check-ignore", "results/.gitkeep", "results/runs.jsonl"],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert tracked_placeholders.returncode == 1, tracked_placeholders.stdout


def test_readme_explains_tracked_empty_runs_jsonl():
    readme = (ROOT / "README.md").read_text()
    assert "results/runs.jsonl" in readme
    assert "intentionally committed as an empty JSONL placeholder" in readme
    assert "normal for it to contain no rows before the first sweep" in readme


def test_b200_suite_records_policy_and_margin_preflight():
    source = (ROOT / "tools/run_b200_suite.py").read_text()
    env_source = (ROOT / "tools/qr_common.py").read_text()
    assert "suite_provenance" in source
    assert "file_provenance" in source
    assert "candidate_policy_public.jsonl" in source
    assert "candidate_implementation_status.jsonl" in source
    assert "candidate_route_trace_public.jsonl" in source
    assert "candidate_guard_overhead_public.jsonl" in source
    assert "classifier_seed_sweep.jsonl" in source
    assert "quantization_seed_sweep.jsonl" in source
    assert "mixed_seed_sweep.jsonl" in source
    assert "secret_audit.jsonl" in source
    assert "runtime_preflight.jsonl" in source
    assert "suite_validation" in source
    assert "--require-final-kernels" in source
    assert "validate_completed_export" in source
    assert "tools/validate_b200_suite.py" in source
    assert "submission_validation.jsonl" in source
    assert "tools/validate_submission.py" in source
    assert "candidate_ablation_no_structured_routes_public.jsonl" in source
    assert "candidate_ablation_no_data_dependent_routes_public.jsonl" in source
    assert "candidate_ablation_no_qr512_qr1024_cuda_public.jsonl" in source
    assert "candidate_public_tests.jsonl" in source
    assert "candidate_public_benchmark_correctness.jsonl" in source
    assert "candidate_dev_robustness.jsonl" in source
    assert "candidate_tail_policy_sweep.jsonl" in source
    assert "accelerator_preflight.jsonl" in source
    assert "seed_sweep_margin.jsonl" in source
    assert "candidate_public.jsonl" in source
    assert "candidate_official_style.jsonl" in source
    assert "--benchmark-max-factor-scaled" in source
    assert "--benchmark-max-orth-scaled" in source
    assert "--dev-max-factor-scaled" in source
    assert "--dev-max-orth-scaled" in source
    assert "suite_summary.json" in source
    assert "suite_summary.md" in source
    assert "suite_analysis.json" in source
    assert "suite_analysis.md" in source
    assert "tools/analyze_b200_results.py" in source
    assert "tools/implementation_status.py" in source
    assert "tools/submit_popcorn.py" in source
    assert "--include-popcorn-test" in source
    assert "--popcorn-timeout-s" in source
    assert "tools/tune_tail_policy.py" in source
    assert "--include-tail-policy-tune" in source
    assert "--tail-tune-config" in source
    assert "tools/tune_candidate_configs.py" in source
    assert "--include-candidate-config-tune" in source
    assert "tools/blocked_qr_sweep.py" in source
    assert "--include-blocked-qr-sweep" in source
    assert "--blocked-qr-sweep-r-maintenance-modes" in source
    assert "--blocked-qr-sweep-panel-refresh-modes" in source
    assert "candidate_config_accelerator_preflight.jsonl" in source
    assert "--skip-candidate-config-accelerator-preflight" in source
    assert "--candidate-config-tune-panel-widths" in source
    assert "--candidate-config-tune-precision-modes" in source
    assert "--candidate-config-tune-ctas-per-matrix" in source
    assert "--candidate-config-tune-benchmark-indices" in source
    assert "--candidate-config-tune-collect-resource-metrics" in source
    assert "--candidate-config-tune-resource-cflags-env" in source
    assert "--allow-failed-configs" in source
    assert source.index('"tools/analyze_b200_results.py"') < source.index('"tools/validate_b200_suite.py"')
    assert source.index('tar_path = make_tarball(suite_dir)', source.index('"event": "suite_finish"')) > source.index(
        '"event": "suite_finish"'
    )
    assert source.index("validate_completed_export(suite_dir, require_final_kernels=args.require_final_kernels)") > source.index("tar_path = make_tarball(suite_dir)")
    assert source.index("validate_completed_export(suite_dir, require_final_kernels=args.require_final_kernels)") < source.index("print(f\"\\nDONE")
    assert "--skip-submission-validation" in source
    assert "--skip-route-trace" in source
    assert "--skip-guard-benchmark" in source
    assert "--skip-route-ablations" in source
    assert "--skip-secret-audit" in source
    assert "--skip-runtime-preflight" in source
    assert "--skip-suite-validation" in source
    assert "--ablation-repeats" in source
    assert "--guard-repeats" in source
    assert "--guard-warmup" in source
    assert "--skip-candidate-tests" in source
    assert "--skip-benchmark-correctness" in source
    assert "--skip-dev-robustness" in source
    assert "--skip-quantization-sweep" in source
    assert "--skip-mixed-seed-sweep" in source
    assert "--skip-classifier-sweep" in source
    assert "--skip-tail-policy-sweep" in source
    assert "--skip-accelerator-preflight" in source
    assert "--allow-accelerator-fallback" in source
    assert "--skip-candidate-public" in source
    assert "--skip-candidate-official-style" in source
    assert "--max-factor-scaled" in source
    assert "--max-orth-scaled" in source
    assert "--tail-policy-indices" in source
    assert "--classifier-sweep-indices" in source
    assert "--classifier-sweep-popcorn-seeds" in source
    assert "--quantization-sweep-indices" in source
    assert "--quantization-sweep-popcorn-seeds" in source
    assert "--quantization-sweep-experiments" in source
    assert "--mixed-sweep-benchmark-indices" in source
    assert "--mixed-sweep-test-indices" in source
    assert "--mixed-sweep-popcorn-seeds" in source
    assert "--tail-policy-popcorn-seeds" in source
    assert '"public,1,2,3"' in source
    assert '"public,0,1,2,3"' in source
    assert "--tail-policy-cuts" in source
    assert "--tail-policy-max-factor-scaled" in source
    assert "--tail-policy-max-orth-scaled" in source
    assert "--experiment-indices" in source
    assert "--experiment-tail-cuts" in source
    assert '"1,2,3,4,5,6,7,8,9,10,11"' in source
    assert "tracked_runtime_env" in source
    assert '"FAST_QR_DENSE_TAIL_CUT_512"' in env_source
    assert '"FAST_QR_MIXED_DENSE_TAIL_CUT_1024"' in env_source
    assert "--dry-run" in source
    assert "--dry-run-json" in source
    assert "runtime_estimate" in source
    assert "estimated B200 wall time" in source


def test_b200_suite_dry_run_prints_plan_without_creating_results_dir():
    suite_name = f"dry_run_test_{uuid.uuid4().hex}"
    suite_dir = ROOT / "results" / suite_name
    assert not suite_dir.exists()

    completed = subprocess.run(
        [
            sys.executable,
            "tools/run_b200_suite.py",
            "--dry-run",
            "--dry-run-json",
            "--suite-name",
            suite_name,
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    plan = json.loads(completed.stdout)
    assert plan["dry_run"] is True
    assert plan["suite_dir"].endswith(f"results/{suite_name}")
    assert plan["will_validate_suite"] is True
    assert plan["will_validate_completed_export"] is True
    assert plan["validation_blockers"] == []
    assert plan["workload"]["num_gpu_heavy_steps"] > 0
    assert plan["workload"]["num_benchmark_timing_steps"] == 6
    assert plan["workload"]["num_route_ablation_timing_steps"] == 6
    assert plan["workload"]["num_verifier_experiment_steps"] == 11
    assert plan["workload"]["num_tail_policy_tuning_steps"] == 0
    assert plan["workload"]["num_candidate_config_tuning_steps"] == 0
    assert plan["workload"]["step_counts_by_category"]["correctness"] >= 4
    assert "candidate_official_style" in plan["workload"]["gpu_heavy_steps"]
    estimate = plan["workload"]["runtime_estimate"]
    assert estimate["rough"] is True
    assert estimate["low_minutes"] > 0
    assert estimate["high_minutes"] > estimate["low_minutes"]
    assert estimate["by_category"]["official_style_timing"]["high_minutes"] > 0
    assert estimate["by_category"]["verifier_experiments"]["high_minutes"] > 0
    assert estimate["slowest_steps"]
    assert not suite_dir.exists()

    step_names = [row["step"] for row in plan["steps"]]
    assert step_names == EXPECTED_DEFAULT_STEPS
    assert step_names[:4] == ["print_spec", "sync_cases_check", "secret_audit", "runtime_preflight"]
    assert "pytest" in step_names
    assert "tail_policy_tune" not in step_names
    assert "candidate_config_tune" not in step_names
    assert "suite_summary" in step_names
    assert "suite_analysis" in step_names
    assert "suite_validation" in step_names
    assert step_names.index("suite_summary") < step_names.index("suite_analysis") < step_names.index("suite_validation")

    skipped = subprocess.run(
        [
            sys.executable,
            "tools/run_b200_suite.py",
            "--dry-run",
            "--dry-run-json",
            "--suite-name",
            f"{suite_name}_skip_pytest",
            "--skip-pytest",
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    skipped_plan = json.loads(skipped.stdout)
    assert skipped_plan["will_validate_suite"] is False
    assert skipped_plan["will_validate_completed_export"] is False
    assert skipped_plan["validation_blockers"] == ["--skip-pytest"]
    assert skipped_plan["workload"]["num_gpu_heavy_steps"] == plan["workload"]["num_gpu_heavy_steps"]
    assert skipped_plan["workload"]["runtime_estimate"]["high_minutes"] < estimate["high_minutes"]

    skipped_runtime = subprocess.run(
        [
            sys.executable,
            "tools/run_b200_suite.py",
            "--dry-run",
            "--dry-run-json",
            "--suite-name",
            f"{suite_name}_skip_runtime",
            "--skip-runtime-preflight",
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    skipped_runtime_plan = json.loads(skipped_runtime.stdout)
    assert skipped_runtime_plan["will_validate_suite"] is False
    assert skipped_runtime_plan["validation_blockers"] == ["--skip-runtime-preflight"]

    popcorn = subprocess.run(
        [
            sys.executable,
            "tools/run_b200_suite.py",
            "--dry-run",
            "--dry-run-json",
            "--suite-name",
            f"{suite_name}_popcorn",
            "--include-popcorn-test",
            "--popcorn-timeout-s",
            "30",
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    popcorn_plan = json.loads(popcorn.stdout)
    popcorn_steps = [row["step"] for row in popcorn_plan["steps"]]
    assert "popcorn_test" in popcorn_steps
    assert popcorn_plan["workload"]["num_official_remote_steps"] == 1
    assert popcorn_plan["workload"]["runtime_estimate"]["by_category"]["official_remote"]["high_minutes"] == 0.5
    popcorn_step = next(row for row in popcorn_plan["steps"] if row["step"] == "popcorn_test")
    assert "tools/submit_popcorn.py" in popcorn_step["cmd"]
    assert "--mode" in popcorn_step["cmd"]
    assert "test" in popcorn_step["cmd"]
    assert "--timeout-s" in popcorn_step["cmd"]

    tail_tune = subprocess.run(
        [
            sys.executable,
            "tools/run_b200_suite.py",
            "--dry-run",
            "--dry-run-json",
            "--suite-name",
            f"{suite_name}_tail_tune",
            "--include-tail-policy-tune",
            "--tail-tune-config",
            "probe:FAST_QR_DENSE_TAIL_CUT_512=20",
            "--tail-tune-official-stopping",
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    tail_tune_plan = json.loads(tail_tune.stdout)
    tail_tune_steps = [row["step"] for row in tail_tune_plan["steps"]]
    assert "tail_policy_tune" in tail_tune_steps
    assert tail_tune_plan["workload"]["num_tail_policy_tuning_steps"] == 1
    assert tail_tune_plan["workload"]["runtime_estimate"]["by_category"]["tail_policy_tuning"]["high_minutes"] > 0
    tune_step = next(row for row in tail_tune_plan["steps"] if row["step"] == "tail_policy_tune")
    assert "tools/tune_tail_policy.py" in tune_step["cmd"]
    assert "--allow-failed-configs" in tune_step["cmd"]
    assert "--config" in tune_step["cmd"]
    assert "probe:FAST_QR_DENSE_TAIL_CUT_512=20" in tune_step["cmd"]
    assert "--official-stopping" in tune_step["cmd"]

    config_tune = subprocess.run(
        [
            sys.executable,
            "tools/run_b200_suite.py",
            "--dry-run",
            "--dry-run-json",
            "--suite-name",
            f"{suite_name}_candidate_config_tune",
            "--include-candidate-config-tune",
            "--candidate-config-tune-shape-label",
            "qr512",
            "--candidate-config-tune-panel-widths",
            "16,32",
            "--candidate-config-tune-precision-modes",
            "fp32,tf32",
            "--candidate-config-tune-benchmark-indices",
            "3,7,9,10",
            "--candidate-config-tune-config",
            "probe:FAST_QR_QR512_PANEL_B=24",
            "--candidate-config-tune-collect-resource-metrics",
            "--candidate-config-tune-resource-cflags-env",
            "FAST_QR_QR512_EXTRA_CUDA_CFLAGS",
            "--candidate-config-tune-official-stopping",
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    config_tune_plan = json.loads(config_tune.stdout)
    config_tune_steps = [row["step"] for row in config_tune_plan["steps"]]
    assert "candidate_config_tune" in config_tune_steps
    assert config_tune_plan["workload"]["num_candidate_config_tuning_steps"] == 1
    assert config_tune_plan["workload"]["runtime_estimate"]["by_category"]["candidate_config_tuning"]["high_minutes"] > 0
    config_step = next(row for row in config_tune_plan["steps"] if row["step"] == "candidate_config_tune")
    assert "tools/tune_candidate_configs.py" in config_step["cmd"]
    assert "--allow-failed-configs" in config_step["cmd"]
    assert "--shape-label" in config_step["cmd"]
    assert "qr512" in config_step["cmd"]
    assert "--panel-widths" in config_step["cmd"]
    assert "16,32" in config_step["cmd"]
    assert "--precision-modes" in config_step["cmd"]
    assert "fp32,tf32" in config_step["cmd"]
    assert "--benchmark-indices" in config_step["cmd"]
    assert "3,7,9,10" in config_step["cmd"]
    assert "probe:FAST_QR_QR512_PANEL_B=24" in config_step["cmd"]
    assert "--collect-resource-metrics" in config_step["cmd"]
    assert "--resource-cflags-env" in config_step["cmd"]
    assert "FAST_QR_QR512_EXTRA_CUDA_CFLAGS" in config_step["cmd"]
    assert "--official-stopping" in config_step["cmd"]

    skipped_secret = subprocess.run(
        [
            sys.executable,
            "tools/run_b200_suite.py",
            "--dry-run",
            "--dry-run-json",
            "--suite-name",
            f"{suite_name}_skip_secret",
            "--skip-secret-audit",
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    skipped_secret_plan = json.loads(skipped_secret.stdout)
    assert skipped_secret_plan["will_validate_suite"] is False
    assert skipped_secret_plan["validation_blockers"] == ["--skip-secret-audit"]


def test_b200_suite_rejects_existing_suite_dir_before_running():
    suite_name = f"b200_collision_test_{uuid.uuid4().hex}"
    suite_dir = ROOT / "results" / suite_name
    suite_dir.mkdir()
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "tools/run_b200_suite.py",
                "--suite-name",
                suite_name,
            ],
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    finally:
        suite_dir.rmdir()

    assert completed.returncode == 2
    assert "result directory already exists" in completed.stderr
    assert "choose a different --suite-name" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_local_checks_dry_run_is_non_cuda_and_writes_no_results_dir():
    suite_name = f"local_dry_run_test_{uuid.uuid4().hex}"
    suite_dir = ROOT / "results" / suite_name
    assert not suite_dir.exists()

    completed = subprocess.run(
        [
            sys.executable,
            "tools/run_local_checks.py",
            "--dry-run",
            "--dry-run-json",
            "--suite-name",
            suite_name,
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    plan = json.loads(completed.stdout)
    assert plan["dry_run"] is True
    assert plan["suite_dir"].endswith(f"results/{suite_name}")
    assert not suite_dir.exists()

    step_names = [row["step"] for row in plan["steps"]]
    assert step_names == [
        "print_spec",
        "sync_cases_check",
        "secret_audit",
        "runtime_preflight_allow_failure",
        "submission_validation",
        "candidate_policy",
        "candidate_implementation_status",
        "b200_suite_dry_run",
        "b200_next_required_dry_run",
        "pytest",
    ]
    command_text = "\n".join(" ".join(row["cmd"]) for row in plan["steps"])
    assert "tools/bench_local.py" not in command_text
    assert "tools/check_cases.py" not in command_text
    assert "--allow-failure" in command_text

    source = (ROOT / "tools/run_local_checks.py").read_text()
    assert "validate_completed_export(suite_dir)" in source
    assert "local export validation: PASS" in source
    assert source.index("validate_completed_export(suite_dir)") > source.index("tar_path = make_tarball(suite_dir)")


def test_local_checks_rejects_existing_suite_dir_before_running():
    suite_name = f"local_collision_test_{uuid.uuid4().hex}"
    suite_dir = ROOT / "results" / suite_name
    suite_dir.mkdir()
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "tools/run_local_checks.py",
                "--suite-name",
                suite_name,
            ],
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    finally:
        suite_dir.rmdir()

    assert completed.returncode == 2
    assert "result directory already exists" in completed.stderr
    assert "choose a different --suite-name" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_provenance_helpers_record_source_hashes_and_frozen_upstream():
    candidate = file_provenance(ROOT / "submissions/candidate.py")
    assert candidate["path"] == "submissions/candidate.py"
    assert candidate["bytes"] > 0
    assert len(candidate["sha256"]) == 64

    repo = repo_provenance()
    assert len(repo["git_hash"]) >= 4
    assert len(repo["git_full_hash"]) >= len(repo["git_hash"])
    assert isinstance(repo["git_dirty"], bool)
    assert repo["official_upstream_commit"] == (ROOT / "official/UPSTREAM_COMMIT").read_text().strip()
    assert isinstance(repo["git_status_porcelain"], list)


def test_candidate_contains_guarded_small_cuda_probes():
    source = (ROOT / "submissions/candidate.py").read_text()
    assert "FAST_QR_DISABLE_QR32_CUDA" in source
    assert "FAST_QR_DISABLE_QR176_CUDA" in source
    assert "FAST_QR_DISABLE_QR352_CUDA" in source
    assert "FAST_QR_DISABLE_QR512_CUDA" in source
    assert "FAST_QR_DISABLE_QR1024_CUDA" in source
    assert "FAST_QR_DISABLE_STRUCTURED_ROUTES" in source
    assert "FAST_QR_DISABLE_DENSE_TAIL" in source
    assert "FAST_QR_DISABLE_DATA_DEPENDENT_ROUTES" in source
    assert "FAST_QR_DENSE_TAIL_CUT_512" in source
    assert "FAST_QR_MIXED_DENSE_TAIL_CUT_1024" in source
    assert "geqrf32_kernel" in source
    assert "geqrf176_kernel" in source
    assert "geqrf352_kernel" in source
    assert "geqrf512" in source
    assert "geqrf1024" in source
    assert "load_inline" in source


def test_case_parser_ignores_comments_and_formats_numbers():
    assert parse_case("batch: 2; n: 4096; cond: 1; seed: 7 # comment") == {
        "batch": 2,
        "n": 4096,
        "cond": 1,
        "seed": 7,
    }
    assert parse_case("batch: 16; n: 512; cond: 0; seed: 1; case: mixed")["case"] == "mixed"


def test_benchmark_batch_count_matches_official_formula():
    assert batch_count({"batch": 20, "n": 32}) == 50
    assert batch_count({"batch": 640, "n": 512}) == 1
    assert batch_count({"batch": 60, "n": 1024}) == 1
    assert batch_count({"batch": 2, "n": 4096}) == 2


def test_popcorn_seed_combination_matches_official_formula():
    cases = [{"batch": 1, "n": 8, "cond": 1, "seed": 53124}, {"batch": 1, "n": 8, "cond": 1}]
    seeded = apply_popcorn_seed(cases, 123)
    assert seeded[0]["seed"] == combine_seed(53124, 123)
    assert "seed" not in seeded[1]
    assert cases[0]["seed"] == 53124
