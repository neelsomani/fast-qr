# Fast QR B200 Lab

Local lab for the GPU Mode `qr_v2` batched square QR challenge on NVIDIA B200.
The repo freezes the official evaluator files, encodes the public tests and benchmarks, and provides local tools for correctness, diagnostics, experiments, and result tracking.
Use `official/reference.py`, `official/eval.py`, and `official/task.yml` as the source of truth; do not infer tolerances or benchmark cases from prose.

## Layout

```text
official/      Frozen upstream qr_v2 files and shared utils.py
cases/         Public test, public benchmark, and extra robustness cases
submissions/   Baseline torch.geqrf implementation and editable candidate
tools/         Local benchmark, correctness, diagnostic, experiment, and sweep tools
tests/         Repo-level scaffold tests that do not require CUDA
results/       Local JSONL output directory
```

Edit `submissions/candidate.py` for experiments. `submissions/baseline_geqrf.py` is the correctness-first baseline.
`official/UPSTREAM_COMMIT` and `official/FETCHED_AT` record the upstream snapshot used for the frozen files.

## B200 Install

These commands assume Ubuntu or another Linux distro with a B200 installed.
B200 is compute capability 10.0, so future compiled CUDA extensions should use `TORCH_CUDA_ARCH_LIST=10.0` or equivalent native `sm_100` flags. CUDA 12.8 added Blackwell compiler support, including `SM_100`; use a recent NVIDIA data-center driver. CUDA 12.8 GA requires driver `>=570.26` on Linux, CUDA 12.8 Update 1 requires `>=570.124.06`, and CUDA 13.x requires `>=580`.

1. Install system packages. Use Python 3.11 or newer; many current Ubuntu images already ship Python 3.12.

```bash
sudo apt-get update
sudo apt-get install -y git curl python3 python3-venv python3-pip
python3 - <<'PY'
import sys
assert sys.version_info >= (3, 11), sys.version
print(sys.version)
PY
```

2. Clone or enter this repo.

```bash
cd /path/to/fast-qr
```

3. Create a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

4. Install PyTorch with CUDA 12.8 wheels, then the local tooling dependencies.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-dev.txt
```

If the PyTorch install selector has moved beyond CUDA 12.8, choose the newest CUDA wheel that supports B200 on your driver.

5. Verify the B200 is visible to PyTorch.

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())
print(torch.cuda.get_device_capability())
PY
```

Expected capability for B200 is `(10, 0)`.

6. Install and register Popcorn for official remote tests.

```bash
curl -fsSL https://raw.githubusercontent.com/gpu-mode/popcorn-cli/main/install.sh | bash
popcorn register discord
```

## Run Tests

Run the repo scaffold tests. These only validate file layout and case encoding.

```bash
python tools/print_spec.py
python tools/sync_cases_from_task_yml.py --check
python -m pytest
```

Run one public correctness test locally.

```bash
python tools/check_one.py \
  --submission submissions/candidate.py \
  --cases cases/public_tests.txt \
  --index 0
```

Run every public correctness test locally.

```bash
for i in $(seq 0 21); do
  python tools/check_one.py --submission submissions/candidate.py --cases cases/public_tests.txt --index "$i"
done
```

Public test indexes are the line numbers in `cases/public_tests.txt`, zero-based:

```text
0  batch: 20; n: 32; cond: 1; seed: 53124
1  batch: 40; n: 176; cond: 1; seed: 3321
2  batch: 40; n: 352; cond: 1; seed: 1200
3  batch: 16; n: 512; cond: 2; seed: 32523
4  batch: 4; n: 1024; cond: 2; seed: 4327
5  batch: 1; n: 4096; cond: 1; seed: 75342
6  batch: 16; n: 512; cond: 4; seed: 32524; case: dense
7  batch: 16; n: 512; cond: 0; seed: 32525; case: rankdef
8  batch: 16; n: 512; cond: 0; seed: 32526; case: clustered
9  batch: 16; n: 512; cond: 0; seed: 32527; case: band
10 batch: 16; n: 512; cond: 0; seed: 32528; case: rowscale
11 batch: 16; n: 512; cond: 0; seed: 32529; case: nearcollinear
12 batch: 4; n: 1024; cond: 4; seed: 4328; case: dense
13 batch: 4; n: 1024; cond: 0; seed: 4329; case: rankdef
14 batch: 4; n: 1024; cond: 0; seed: 4330; case: nearrank
15 batch: 4; n: 1024; cond: 0; seed: 4331; case: clustered
16 batch: 2; n: 2048; cond: 2; seed: 224466; case: dense
17 batch: 2; n: 2048; cond: 0; seed: 224467; case: rankdef
18 batch: 1; n: 4096; cond: 0; seed: 75343; case: upper
19 batch: 16; n: 512; cond: 2; seed: 32530; case: mixed
20 batch: 4; n: 1024; cond: 2; seed: 4332; case: mixed
21 batch: 2; n: 2048; cond: 2; seed: 224468; case: mixed
```

Run every extra dev robustness test.

```bash
for i in $(seq 0 10); do
  python tools/check_one.py --submission submissions/candidate.py --cases cases/dev_robustness.txt --index "$i"
done
```

Run detailed diagnostics for a failing case.

```bash
python tools/check_one.py \
  --submission submissions/candidate.py \
  --cases cases/public_tests.txt \
  --index 19 \
  --diagnose
```

Run the official remote test.

```bash
popcorn submit --leaderboard qr_v2 --gpu B200 --mode test submissions/candidate.py
```

## Run Benchmarks

For the first B200 pass, you can run the whole collection suite with one command:

```bash
python tools/run_b200_suite.py
```

It writes a timestamped directory under `results/`, appends JSONL files for each stage, writes `run.log` and `manifest.jsonl`, and creates a `.tgz` archive next to the directory. Useful switches:

```bash
python tools/run_b200_suite.py --include-local-official-eval
python tools/run_b200_suite.py --skip-official-style
python tools/run_b200_suite.py --popcorn-seed 123
python tools/run_b200_suite.py --skip-smoke --skip-baseline-public --skip-experiments --skip-official-style
```

Run the public benchmark set locally with timed-output rechecks.

```bash
python tools/bench_local.py \
  --submission submissions/candidate.py \
  --cases cases/public_benchmarks.txt \
  --repeats 3 \
  --recheck \
  --record-env \
  --out results/candidate_public_benchmarks.jsonl
```

Use `--popcorn-seed` on `check_one.py`, `diagnose.py`, `experiments.py`, `bench_local.py`, or `sweep.py` to apply the same seed-combination rule as the official evaluator.

Record a tracked sweep to `results/runs.jsonl`. By default each passing case appends timing and environment metadata; pass diagnostic flags only when you want expensive residual checks.

```bash
python tools/sweep.py \
  --submission submissions/candidate.py \
  --cases cases/public_benchmarks.txt \
  --repeats 3 \
  --recheck \
  --label baseline-geqrf \
  --config-json '{"algorithm":"torch.geqrf"}'
```

Run the normal cheap verifier experiments on one mixed public test. This excludes negative controls such as `identity-q`.

```bash
python tools/experiments.py \
  --submission submissions/candidate.py \
  --cases cases/public_tests.txt \
  --index 19 \
  --experiments all \
  --out results/experiments.jsonl
```

Useful individual experiments:

```bash
python tools/experiments.py --case 'batch: 2; n: 8; cond: 1; seed: 11' --experiments r-projection
python tools/experiments.py --case 'batch: 1; n: 4096; cond: 0; seed: 75343; case: upper' --experiments identity-q
python tools/experiments.py --cases cases/public_tests.txt --index 0 --experiments tf32-input-nearby
python tools/experiments.py --cases cases/public_tests.txt --index 19 --experiments all-with-controls
```

Submit to the official leaderboard after local and official test mode pass.

```bash
popcorn submit --leaderboard qr_v2 --gpu B200 --mode leaderboard submissions/candidate.py
```

## Refresh Official Files

The `official/` files are frozen from:

- https://raw.githubusercontent.com/gpu-mode/reference-kernels/main/problems/linalg/qr_v2/submission.py
- https://raw.githubusercontent.com/gpu-mode/reference-kernels/main/problems/linalg/qr_v2/reference.py
- https://raw.githubusercontent.com/gpu-mode/reference-kernels/main/problems/linalg/qr_v2/task.py
- https://raw.githubusercontent.com/gpu-mode/reference-kernels/main/problems/linalg/qr_v2/eval.py
- https://raw.githubusercontent.com/gpu-mode/reference-kernels/main/problems/linalg/qr_v2/task.yml
- https://raw.githubusercontent.com/gpu-mode/reference-kernels/main/problems/pmpp_v2/utils.py

Refresh intentionally, then rerun `python -m pytest` and at least the baseline public tests.

The helper below resolves the upstream SHA first and fetches all official files from that exact commit:

```bash
python tools/fetch_official.py
```

After refreshing, regenerate case files from `official/task.yml`:

```bash
python tools/sync_cases_from_task_yml.py
python tools/sync_cases_from_task_yml.py --check
```

Run the frozen official evaluator locally on the B200 box with:

```bash
python tools/run_official_eval.py \
  --submission submissions/candidate.py \
  --cases cases/public_tests.txt \
  --mode test
```

`run_official_eval.py` strips comments from the case file before invoking frozen `official/eval.py`, so local dev case files can include comments. It copies the selected submission as `submission.py`, matching Popcorn's one-file submission model; keep final official-validation submissions self-contained.

For final local timing comparisons, use official-style adaptive stopping and optional leaderboard warmup:

```bash
python tools/bench_local.py \
  --submission submissions/candidate.py \
  --cases cases/public_benchmarks.txt \
  --repeats 1000 \
  --official-stopping \
  --max-time-ns 30000000000 \
  --leaderboard-warmup \
  --recheck
```

Sweep diagnostics are expensive for `n=2048` and `n=4096`, so they are opt-in:

```bash
python tools/sweep.py \
  --submission submissions/candidate.py \
  --cases cases/public_benchmarks.txt \
  --repeats 1000 \
  --official-stopping \
  --leaderboard-warmup \
  --diagnose-cases 3,7,9,10
```

## References

- NVIDIA CUDA GPU compute capability table: https://developer.nvidia.com/cuda/gpus
- NVIDIA CUDA 12.8 release notes: https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/
- NVIDIA CUDA toolkit and driver compatibility: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
- PyTorch local install selector: https://pytorch.org/get-started/locally/
- Popcorn QR B200 guide: https://raw.githubusercontent.com/gpu-mode/popcorn-cli/main/docs/linalg-qr-b200.md
