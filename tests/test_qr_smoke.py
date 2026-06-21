from pathlib import Path
import sys

import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from qr_common import ensure_official_on_path, load_submission  # noqa: E402

ensure_official_on_path()
from reference import check_implementation, generate_input  # noqa: E402
from diagnose import diagnose  # noqa: E402
from experiments import (  # noqa: E402
    experiment_column_major,
    experiment_identity_q,
    experiment_r_projection,
    experiment_tail_delete,
    parse_experiments,
)
from run_official_eval import write_official_case_file  # noqa: E402
from sweep import diagnostic_row, load_config  # noqa: E402


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


def test_sweep_diagnostics_match_tracking_schema():
    spec = _small_case("dense")
    custom_kernel = load_submission(ROOT / "submissions/candidate.py")
    row = diagnostic_row(custom_kernel, spec)
    assert row["diagnostic_passed"], row
    assert row["factor_scaled_max"] < 20.0
    assert row["orth_scaled_max"] < 100.0
    assert "tri_scaled_max" in row
    assert load_config('{"algorithm":"torch.geqrf"}') == {"algorithm": "torch.geqrf"}
