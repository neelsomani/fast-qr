from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from qr_common import apply_popcorn_seed, batch_count, combine_seed, load_cases, parse_case  # noqa: E402
from spec_utils import render_case_file, specs_from_task_yml  # noqa: E402


def test_public_benchmark_cases_are_current_shape_set():
    cases = load_cases(ROOT / "cases/public_benchmarks.txt")
    assert len(cases) == 12
    assert cases[0] == {"batch": 20, "n": 32, "cond": 1, "seed": 43214}
    assert cases[-1] == {"batch": 60, "n": 1024, "cond": 0, "seed": 770005, "case": "nearrank"}
    assert {case.get("case", "dense") for case in cases} >= {"dense", "mixed", "rankdef", "clustered", "nearrank"}
    assert cases == specs_from_task_yml("benchmarks")


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
        "tools/check_one.py",
        "tools/diagnose.py",
        "tools/experiments.py",
        "tools/fetch_official.py",
        "tools/print_spec.py",
        "tools/run_b200_suite.py",
        "tools/run_official_eval.py",
        "tools/spec_utils.py",
        "tools/sweep.py",
        "tools/sync_cases_from_task_yml.py",
        "results/runs.jsonl",
    ]:
        assert (ROOT / path).is_file()


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
