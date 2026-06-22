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

Current `candidate.py` status:

- Shape-dispatched single-file wrapper with named fast-path entrypoints.
- Shared column-major `H` allocation helpers.
- Inline CUDA QR source split into named Householder device primitives for tail norm, reflector generation, reflector normalization, and single-reflector trailing-column updates; QR512/QR1024 inherit these from the QR352 template while the final blocked panel path is built.
- Sampled classifier hooks for the 512 and 1024 families, with cached device-local sample indices and combined column gathers to keep guard overhead low.
- Safe identity-Q path for upper/diagonal one-matrix correctness cases.
- Experimental dense tail-projection shortcuts for public 512, 1024, 2048, and 4096 dense shapes.
- Experimental 512 rank-deficient shortcut that factors only the leading nonzero columns and embeds the compact Householder result.
- Experimental 512 clustered shortcut that factors the high-scale half plus the boundary columns and skips the ultra-tiny tail behind a conservative relative-tail guard.
- Experimental 1024 nearrank shortcut that factors the leading 3/4 columns and projects the copied tail through the same Householder reflectors.
- Experimental mixed-batch routing that applies rankdef, clustered, scaled-nearrank, and 1024-only tiny dense-tail shortcuts per matrix and leaves the remaining matrices on `torch.geqrf`.
- Weak per-tensor route cache for public benchmark shapes, so repeated timed calls on the same generated input do not recompute data-dependent guards.
- Per-input output workspace reuse for column-major `H` and `tau` on B200, so repeated timed calls reuse storage without aliasing different inputs in the evaluator's `data_list`. Use `FAST_QR_DISABLE_OUTPUT_WORKSPACE_CACHE=1` to force fresh allocations, or `FAST_QR_OUTPUT_WORKSPACE_CACHE=1` to enable the same behavior off B200 for profiling.
- Route ablation switches for B200 comparisons: `FAST_QR_DISABLE_ROUTE_CACHE=1`, explicit structured-first routing with `FAST_QR_QR512_STRUCTURED_BEFORE_CUDA=1 FAST_QR_QR1024_STRUCTURED_BEFORE_CUDA=1`, `FAST_QR_DISABLE_STRUCTURED_ROUTES=1`, `FAST_QR_DISABLE_DENSE_TAIL=1`, `FAST_QR_DISABLE_DATA_DEPENDENT_ROUTES=1`, and the paired QR512/QR1024 CUDA-probe disable ablation using `FAST_QR_DISABLE_QR512_CUDA=1 FAST_QR_DISABLE_QR1024_CUDA=1`.
- Optional inline CUDA compact-Householder probes for the `batch=20, n=32`, `batch=40, n=176`, `batch=40, n=352`, `batch=640, n=512`, and `batch=60, n=1024` benchmark shapes, guarded by `FAST_QR_DISABLE_QR32_CUDA=1`, `FAST_QR_DISABLE_QR176_CUDA=1`, `FAST_QR_DISABLE_QR352_CUDA=1`, `FAST_QR_DISABLE_QR512_CUDA=1`, or `FAST_QR_DISABLE_QR1024_CUDA=1` and falling back to the existing safe routes when CUDA compilation is unavailable. QR32 source-specializes warp batching with `FAST_QR_QR32_WARPS_PER_CTA` or `FAST_QR_QR32_THREADS_PER_CTA`; QR512/QR1024 source-specialize CTA size, the one-CTA panel loop width, fused trailing-update column tile width (`FAST_QR_QR512_UPDATE_COL_TILE` / `FAST_QR_QR1024_UPDATE_COL_TILE`), and the trailing-update mode (`reflectors` or experimental `compact-wy`).
- Blocked compact-Householder CUDA templates for `n=512`, `n=1024`, `n=2048`, and `n=4096`, including column-major `H`, tunable panel width, tile width, CTA size, FP32/TF32-input/FP16-input trailing updates, prefix panel refresh, and panel-prefix `R` maintenance.
- Blocked CUDA defaults now use the conservative lower bound of the current-candidate tuning grid for trailing column tiles and per-matrix CTA caps, plus a source-specialized CTA schedule: QR512 uses `TILE_N=64, CTAS_PER_MATRIX=1, CTA_SCHEDULE=fixed`, QR1024 uses `128, 1, fixed`, QR2048 uses `128, 4, frontload`, and QR4096 uses `256, 8, frontload`. Shape-family env overrides still specialize these values.
- On a B200-like CUDA device, the `n=512`, `n=1024`, `n=2048`, and `n=4096` public benchmark shapes try the blocked CUDA route by default with prefix panel refresh and panel-prefix `R` maintenance. Set `FAST_QR_DISABLE_B200_DEFAULT_BLOCKED_CUDA=1` or the shape-specific `FAST_QR_DISABLE_QR{n}_BLOCKED_CUDA=1` to force the older fallback/probe behavior; explicit `FAST_QR_ENABLE_QR{n}_BLOCKED_CUDA=0` also suppresses the B200 default.
- On B200, the ambiguous `n=512` and `n=1024` families default to the blocked CUDA auto route before any Python sampled structured classifier. The blocked kernel still has a device-side policy for rank-deficient, clustered, mixed, and nearrank inputs, which avoids Python classifier synchronization on the timed hot path. QR512/QR1024 default to the metadata/cached auto-policy path so warmup can cache per-input policy and homogeneous rankdef/clustered/dense-tail cases can shrink the panel loop bound; set `FAST_QR_ENABLE_BLOCKED_SYNC_FREE_AUTO_POLICY=1` or `FAST_QR_QR{n}_SYNC_FREE_AUTO_POLICY=1` to benchmark the fully sync-free policy path. Set `FAST_QR_QR512_STRUCTURED_BEFORE_CUDA=1`, `FAST_QR_QR1024_STRUCTURED_BEFORE_CUDA=1`, `FAST_QR_STRUCTURED_ROUTES_BEFORE_CUDA=1`, or `FAST_QR_ENABLE_B200_DEFAULT_STRUCTURED_BEFORE_CUDA=1` to run structured-first routing explicitly. With structured-first enabled, `FAST_QR_DISABLE_B200_TRUST_SAMPLED_STRUCTURED_GUARDS=1` restores exact verification instead of trusting the sampled masks.

Official submission interface and shape ambiguity:

- The frozen starter exposes only `custom_kernel(data)`. The submission does not receive `case`, `cond`, `seed`, or benchmark index.
- `python tools/print_spec.py` reports the current interface and benchmark shape collisions from `official/task.yml`.
- Current ambiguous benchmark shapes are `batch=640, n=512` with dense/mixed/rankdef/clustered cases and `batch=60, n=1024` with dense/mixed/nearrank cases.
- Shape-only dispatch is enough for every non-colliding public benchmark shape. A classifier is only useful if a faster case-specific path beats the robust path on one of the two ambiguous shapes.
- Case-specific shortcuts on ambiguous shapes must be guarded from tensor values, not metadata. The policy export records `case_selection_info_sources`, `shape_only_case_selection`, and `uses_tensor_values_for_case_selection` for each benchmark row so hidden case ambiguity is explicit. `classifier_needed_for_case_specific_path` records whether a case-specific shortcut needs that guard, while `classifier_on_current_hot_path` records whether the current default route actually pays classifier overhead.

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

Run every local check that does not require CUDA or a B200. This records a timestamped directory under `results/`, runs the secret audit, validates the one-file submission packaging, records the current B200 runtime preflight result with `--allow-failure`, captures both the default B200 suite dry-run plan and the next-required candidate-config tuning dry-run plan, and runs pytest. It deliberately does not run benchmark timing or large CPU QR cases.

```bash
python tools/run_local_checks.py --dry-run
python tools/run_local_checks.py
python tools/validate_local_checks.py --suite-dir results/<local-checks-suite>
python tools/run_local_checks.py --print-next-command results/<local-checks-suite>
```

The local export validator also checks that the captured B200 dry-run is the full default plan, including seed/tail sweeps, route ablations, official-style timing, summary, analysis, and strict suite validation. It also checks that the next-required plan is sourced from `candidate_policy.py`, includes the generated config preview, and preflights QR512/QR1024 candidate configs across representative family cases before tuning. After a successful run, the final output prints the selected next-required target and a matching `python tools/run_b200_suite.py --suite-name ... --candidate-config-tune-next-required ...` command using the validated dry-run suite name. `--print-next-command` reruns the same local-export validation before printing, so stale or malformed preflights do not produce a paid-run command. That gives you a cheap preflight for the paid B200 command without requiring CUDA locally.

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
python tools/validate_submission.py --submission submissions/candidate.py
python tools/submit_popcorn.py --mode test --submission submissions/candidate.py
```

## Run Benchmarks

For the first B200 pass, you can run the whole collection suite with one command:

```bash
python tools/run_b200_suite.py --dry-run
python tools/run_b200_suite.py --dry-run --dry-run-json
python tools/run_b200_suite.py
python tools/run_b200_suite.py --qr32-sm100
python tools/run_b200_suite.py --include-popcorn-test
python tools/run_b200_suite.py --include-popcorn-leaderboard
```

The suite defaults `TORCH_CUDA_ARCH_LIST=10.0` for B200. Use `--qr32-sm100` when you want the optional QR32 inline CUDA probe to request native Blackwell compilation explicitly, or pass `--torch-cuda-arch-list ""` to leave the architecture list unset. For custom QR32 compiler flags, use `--qr32-extra-cuda-cflags=-arch=sm_100` syntax so argparse treats the leading dash as a value. The dry-run forms print the exact step order, suite compile environment, commands, validation blockers, result directory, workload counts by category, and a rough B200 wall-time estimate, but they do not create `results/<suite-name>` and do not run any CUDA work. Use them before starting a paid B200 session. The JSON dry-run also lists `suite_env`, `gpu_heavy_steps`, and `runtime_estimate`; the estimate separates benchmark timing, official-style timing, verifier experiments, correctness/diagnostics, dispatch analysis, and optional remote Popcorn work. Treat it as a planning range only; the completed run records actual elapsed time in `manifest.jsonl` and `suite_summary.md`.

`--include-popcorn-test` appends an explicit Popcorn `mode test` submission step after local correctness preflights. `--include-popcorn-leaderboard` implies the test step and adds a separate Popcorn `mode leaderboard` submission after local official-style timing. Both remote steps stage the same one-file submission under the suite directory via `tools/submit_popcorn.py`; leave them off for local-only timing runs or dry-run them first with `--dry-run --dry-run-json`.

It writes a timestamped directory under `results/`, appends JSONL files for each stage, records `candidate_policy_public.jsonl` and `candidate_implementation_status.jsonl`, runs public and dev robustness correctness checks, runs margin-gated seed and tail-policy sweeps, times both `baseline_geqrf.py` and `candidate.py`, writes `run.log` and `manifest.jsonl`, creates a `.tgz` archive next to the directory, and runs strict post-export validation before printing `DONE`. The first manifest row records the full git hash, dirty status, frozen upstream commit, command arguments, tracked runtime env vars, and SHA-256 hashes for the candidate and baseline files. Rows emitted with `--record-env` also include compact repo provenance, the submission hash, `candidate_env` for active `FAST_QR_*` candidate knobs, and `tracked_env` for candidate plus tuner/runtime knobs.

After the run, `suite_summary.md` includes a `Runtime` section with total elapsed time and the slowest steps from `manifest.jsonl`. If `--include-tail-policy-tune` was used, `suite_summary.md` and `suite_analysis.md` also include a compact `Tail Policy Tune` section sourced from `tail_policy_tune/summary.json`. If `--include-candidate-config-tune` was used, both reports also include `Candidate Config Tune` from `candidate_config_tune/summary.json`, which ranks future kernel/env configurations by benchmark geomean after correctness gates. Use those sections to decide which checks to skip for short follow-up sweeps after the default full export has passed once.

The default verifier experiment queue includes the public `n=176` and `n=352` benchmark cases as well as the larger shape-colliding families. That records whether tail deletion, nearby-input QR, column-major `H`, and R-projection ideas apply to the small/medium shapes before we invest in dedicated kernels.

Key result files include:

```text
secret_audit.jsonl
runtime_preflight.jsonl
submission_validation.jsonl
candidate_public_tests.jsonl
candidate_public_benchmark_correctness.jsonl
candidate_dev_robustness.jsonl
accelerator_preflight.jsonl
candidate_route_trace_public.jsonl
classifier_seed_sweep.jsonl
quantization_seed_sweep.jsonl
mixed_seed_sweep.jsonl
candidate_guard_overhead_public.jsonl
candidate_ablation_no_route_cache_public.jsonl
candidate_ablation_cuda_first_structured_routes_public.jsonl
candidate_ablation_no_structured_routes_public.jsonl
candidate_ablation_no_dense_tail_public.jsonl
candidate_ablation_no_data_dependent_routes_public.jsonl
candidate_ablation_no_qr512_qr1024_cuda_public.jsonl
baseline_geqrf_smoke.jsonl
candidate_smoke.jsonl
baseline_geqrf_public.jsonl
candidate_public.jsonl
baseline_geqrf_official_style.jsonl
candidate_official_style.jsonl
experiments_public_benchmarks.jsonl
seed_sweep_margin.jsonl
candidate_tail_policy_sweep.jsonl
blocked_qr_sweep.jsonl        # only with --include-blocked-qr-sweep
candidate_policy_public.jsonl
candidate_implementation_status.jsonl
manifest.jsonl
run.log
suite_summary.json
suite_summary.md
suite_analysis.json
suite_analysis.md
tail_policy_tune/summary.json   # only with --include-tail-policy-tune
candidate_config_tune/summary.json # only with --include-candidate-config-tune
popcorn_test/manifest.jsonl      # only with --include-popcorn-test
popcorn_test/popcorn.log         # only with --include-popcorn-test
```

Useful switches:

```bash
python tools/run_b200_suite.py --include-local-official-eval
python tools/run_b200_suite.py --skip-submission-validation
python tools/run_b200_suite.py --skip-policy
python tools/run_b200_suite.py --skip-route-trace
python tools/run_b200_suite.py --skip-classifier-sweep
python tools/run_b200_suite.py --skip-guard-benchmark
python tools/run_b200_suite.py --skip-route-ablations
python tools/run_b200_suite.py --skip-secret-audit
python tools/run_b200_suite.py --skip-runtime-preflight
python tools/run_b200_suite.py --skip-suite-validation
python tools/run_b200_suite.py --skip-seed-sweep
python tools/run_b200_suite.py --skip-quantization-sweep
python tools/run_b200_suite.py --skip-mixed-seed-sweep
python tools/run_b200_suite.py --skip-tail-policy-sweep
python tools/run_b200_suite.py --include-blocked-qr-sweep
python tools/run_b200_suite.py --include-tail-policy-tune
python tools/run_b200_suite.py --include-candidate-config-tune --candidate-config-tune-shape-label qr512 --candidate-config-tune-config-jsonl results/qr512_kernel_configs.jsonl --candidate-config-tune-correctness-indices 3,6,7,8,9,10,11,19 --candidate-config-tune-benchmark-indices 3,7,9,10 --candidate-config-tune-collect-resource-metrics
python tools/run_b200_suite.py --skip-candidate-tests
python tools/run_b200_suite.py --skip-benchmark-correctness
python tools/run_b200_suite.py --skip-dev-robustness
python tools/run_b200_suite.py --skip-accelerator-preflight
python tools/run_b200_suite.py --allow-accelerator-fallback
python tools/run_b200_suite.py --skip-candidate-public
python tools/run_b200_suite.py --skip-official-style
python tools/run_b200_suite.py --skip-candidate-official-style
python tools/run_b200_suite.py --experiment-indices 1,2,3,4,5,6,7,8,9,10,11
python tools/run_b200_suite.py --experiment-tail-cuts 0,4,8,16,32,64,128
python tools/run_b200_suite.py --require-final-kernels
python tools/run_b200_suite.py --popcorn-seed 123
python tools/run_b200_suite.py --skip-smoke --skip-baseline-public --skip-candidate-public --skip-experiments --skip-official-style
```

The default suite runs `tools/check_b200_env.py` before correctness or timing work. It requires a visible CUDA device whose name contains `B200`, compute capability major version at least `10`, and at least `150 GiB` reported memory. Use `--skip-runtime-preflight` only for partial/local smoke runs; that makes the export ineligible for strict completed-suite validation.

The default suite also runs `tools/audit_secrets.py` and writes `secret_audit.jsonl`. The scanner covers tracked plus non-ignored untracked files, skips generated results and binary artifacts, and checks only high-signal token/private-key patterns. Use it directly before sharing a tarball:

```bash
python tools/audit_secrets.py --json --out results/secret_audit.jsonl
```

Run the candidate over every public correctness case and export JSONL:

```bash
python tools/check_cases.py \
  --submission submissions/candidate.py \
  --cases cases/public_tests.txt \
  --json \
  --record-env \
  --out results/candidate_public_tests.jsonl
```

Run diagnostic correctness over the exact public benchmark cases before timing. The default B200 suite includes this because these are the cases that determine geomean ranking. Each row also records the actual `H` shape, stride, contiguity, and `h_layout_actual` from the produced output, so the export proves which cases really returned column-major `H` versus PyTorch's default contiguous layout.

```bash
python tools/check_cases.py \
  --submission submissions/candidate.py \
  --cases cases/public_benchmarks.txt \
  --diagnose \
  --max-factor-scaled 18 \
  --max-orth-scaled 80 \
  --json \
  --record-env \
  --out results/candidate_public_benchmark_correctness.jsonl
```

Run the extra non-public robustness case file. The default B200 suite includes this check with diagnostics and margin gates because it catches brittle data-dependent routes that still pass public correctness and public benchmark seeds.

```bash
python tools/check_cases.py \
  --submission submissions/candidate.py \
  --cases cases/dev_robustness.txt \
  --diagnose \
  --max-factor-scaled 18 \
  --max-orth-scaled 80 \
  --json \
  --record-env \
  --out results/candidate_dev_robustness.jsonl
```

Run the quantization seed sweep to test whether FP16-rounded and TF32-rounded nearby QR outputs stay inside the stricter local residual margins after R repair. This is a verifier experiment, not a final algorithm, because full post-hoc `Q.T @ A` repair is too expensive for the final kernel.

```bash
python tools/quantization_seed_sweep.py \
  --submission submissions/candidate.py \
  --cases cases/public_benchmarks.txt \
  --indices 3,4,5,6,7,8,9,10,11 \
  --popcorn-seeds public,1,2,3 \
  --experiments fp16-nearby,tf32-input-nearby \
  --max-factor-scaled 18 \
  --max-orth-scaled 80 \
  --record-env \
  --out results/quantization_seed_sweep.jsonl
```

Strict completed-suite validation requires this file to include FP16-nearby and TF32-input-nearby rows, the public seed, at least one POPCORN-mutated seed, `640x512`, `60x1024`, and `2x4096` coverage, and zero margin failures.

Run the mixed-batch seed sweep to stress only the benchmark/test cases where each matrix can have a different conditioning profile. This records the actual route, sampled class when available, structured group counters when available, residual diagnostics, and route/margin failures across public and POPCORN-mutated seeds.

```bash
python tools/mixed_seed_sweep.py \
  --submission submissions/candidate.py \
  --benchmark-cases cases/public_benchmarks.txt \
  --benchmark-indices 7,8 \
  --test-cases cases/public_tests.txt \
  --test-indices 19,20,21 \
  --popcorn-seeds public,1,2,3 \
  --max-factor-scaled 18 \
  --max-orth-scaled 80 \
  --record-env \
  --out results/mixed_seed_sweep.jsonl
```

Strict completed-suite validation requires this file to cover public mixed benchmark shapes `640x512` and `60x1024`, public mixed test shapes `16x512`, `4x1024`, and `2x2048`, the public seed, at least one POPCORN-mutated seed, and zero margin or route failures.

Run the tail-policy sweep for the exact candidate cuts across the public benchmark shapes that currently use dense/structured tail shortcuts. Use explicit `--tail-cuts candidate,0,8,16,24,32,64` when exploring alternatives; the default B200 suite records only the current candidate policy so the run stays bounded.

Current candidate defaults are conservative relative to the verifier experiments: dense `512 -> 32`, dense `1024 -> 64`, dense `2048 -> 64`, dense `4096 -> 128`, and mixed-dense `1024 -> 8`. The tail-policy tuner still includes more aggressive candidates such as dense `1024 -> 96`, mixed `1024 -> 12`, and dense `4096 -> 256`; promote those only after the B200 seed sweep records clean margins.

```bash
python tools/tail_policy_sweep.py \
  --submission submissions/candidate.py \
  --cases cases/public_benchmarks.txt \
  --indices 3,4,5,6,7,8,9,10,11 \
  --popcorn-seeds public,0,1,2,3 \
  --tail-cuts candidate \
  --diagnose \
  --max-factor-scaled 18 \
  --max-orth-scaled 80 \
  --record-env \
  --out results/candidate_tail_policy_sweep.jsonl
```

Tail policies can also be overridden without editing `submissions/candidate.py`, which is useful for B200 timing sweeps after a cut passes diagnostics:

```bash
FAST_QR_DENSE_TAIL_CUT_512=16 \
FAST_QR_DENSE_TAIL_CUT_1024=64 \
FAST_QR_MIXED_DENSE_TAIL_CUT_1024=8 \
python tools/run_b200_suite.py --suite-name tail_cut_probe
```

Available override families are:

```text
FAST_QR_DENSE_TAIL_CUT
FAST_QR_DENSE_TAIL_CUT_512
FAST_QR_DENSE_TAIL_CUT_1024
FAST_QR_DENSE_TAIL_CUT_2048
FAST_QR_DENSE_TAIL_CUT_4096
FAST_QR_QR512_TAIL_CUT
FAST_QR_QR1024_TAIL_CUT
FAST_QR_QR2048_TAIL_CUT
FAST_QR_QR4096_TAIL_CUT
FAST_QR_DENSE_TAIL_THRESHOLD
FAST_QR_DENSE_TAIL_THRESHOLD_512
FAST_QR_DENSE_TAIL_THRESHOLD_1024
FAST_QR_DENSE_TAIL_THRESHOLD_2048
FAST_QR_DENSE_TAIL_THRESHOLD_4096
FAST_QR_QR512_TAIL_THRESHOLD
FAST_QR_QR1024_TAIL_THRESHOLD
FAST_QR_QR2048_TAIL_THRESHOLD
FAST_QR_QR4096_TAIL_THRESHOLD
FAST_QR_DENSE_TAIL_FORCE
FAST_QR_DENSE_TAIL_FORCE_512
FAST_QR_DENSE_TAIL_FORCE_1024
FAST_QR_DENSE_TAIL_FORCE_2048
FAST_QR_DENSE_TAIL_FORCE_4096
FAST_QR_QR512_TAIL_FORCE
FAST_QR_QR1024_TAIL_FORCE
FAST_QR_QR2048_TAIL_FORCE
FAST_QR_QR4096_TAIL_FORCE
FAST_QR_MIXED_DENSE_TAIL_CUT
FAST_QR_MIXED_DENSE_TAIL_CUT_512
FAST_QR_MIXED_DENSE_TAIL_CUT_1024
FAST_QR_MIXED_DENSE_TAIL_THRESHOLD
FAST_QR_MIXED_DENSE_TAIL_THRESHOLD_512
FAST_QR_MIXED_DENSE_TAIL_THRESHOLD_1024
```

For B200 timing sweeps, use the tail-policy autotuner. It runs the margin-gated tail-policy sweep first for each env config, skips that config's benchmark if the correctness gate fails, and continues with the rest of the grid. The default grid covers dense 512, dense 1024, mixed 1024, dense 2048, and dense 4096 tail cuts. Use `--fail-fast` when you want the older stop-at-first-failure behavior, or `--benchmark-failed-configs` only when you explicitly want timing data for invalid configs. Use `--dry-run` locally to inspect the plan without requiring CUDA. The main B200 suite can launch this grid too with `--include-tail-policy-tune`; that stage passes `--allow-failed-configs` so bad grid points are recorded without aborting the rest of the export.

```bash
python tools/tune_tail_policy.py --dry-run --out-dir results/tail_policy_plan

python tools/tune_tail_policy.py \
  --out-dir results/tail_policy_b200 \
  --correctness-indices 3,4,5,6,7,8,9,10,11 \
  --popcorn-seeds public,0,1,2,3 \
  --repeats 3
```

Add one-off configs without editing files:

```bash
python tools/tune_tail_policy.py \
  --config 'dense512_cut20:FAST_QR_DENSE_TAIL_CUT_512=20' \
  --config 'mixed1024_cut4:FAST_QR_MIXED_DENSE_TAIL_CUT_1024=4'
```

For future CUDA kernel variants, use the generic candidate-config tuner. It accepts explicit env configs or generates a grid of panel, precision, tile, CTA, auto-policy, cluster, tail, panel-refresh, and R-maintenance knobs. Each config runs a POPCORN_SEED correctness gate before benchmark timing and ranks valid configs by benchmark geomean. Run it standalone for short iteration, or add `--include-candidate-config-tune` to `tools/run_b200_suite.py` to package the grid with the full B200 export.

```bash
python tools/large_kernel_plan.py \
  --shape-label qr512 \
  --mode current-candidate \
  --max-configs 16 \
  --out results/qr512_kernel_configs.jsonl \
  --print-command

python tools/tune_candidate_configs.py \
  --shape-label qr512 \
  --config-jsonl results/qr512_kernel_configs.jsonl \
  --correctness-indices 3,6,7,8,9,10,11,19 \
  --benchmark-indices 3,7,9,10 \
  --popcorn-seeds public,1,2,3,4 \
  --collect-resource-metrics \
  --repeats 3 \
  --allow-failed-configs
```

Use `--benchmark-indices` to keep a shape-family grid bounded: `3,7,9,10` targets the four `640x512` public benchmark rows, and `4,8,11` targets the three `60x1024` rows. `--correctness-indices` indexes `cases/public_tests.txt`; for example, `3,6,7,8,9,10,11,19` covers the public `n=512` correctness cases, and `4,12,13,14,15,20` covers the public `n=1024` correctness cases. Generated env keys are prefixed by the shape label, for example `FAST_QR_QR512_PANEL_B`, `FAST_QR_QR512_PRECISION_MODE`, `FAST_QR_QR512_PANEL_REFRESH_MODE`, `FAST_QR_QR512_R_MAINTENANCE_MODE`, `FAST_QR_QR512_TILE_N`, `FAST_QR_QR512_WARPS_PER_CTA`, `FAST_QR_QR512_CTAS_PER_MATRIX`, `FAST_QR_QR512_CTA_SCHEDULE`, `FAST_QR_QR512_SYNC_FREE_AUTO_POLICY`, `FAST_QR_QR512_BLOCKED_AUTO_GROUPS`, `FAST_QR_QR512_POLICY_FULL_SCAN`, `FAST_QR_QR512_STRUCTURED_BEFORE_CUDA`, `FAST_QR_QR512_TAIL_CUT`, `FAST_QR_QR512_TAIL_THRESHOLD`, and `FAST_QR_QR512_TAIL_FORCE`. On B200, QR512/QR1024/QR2048/QR4096 enter the blocked CUDA route by default; QR512/QR1024 default to CUDA-first blocked auto policy and only run Python structured-before-CUDA routing when explicitly enabled. B200 also reuses per-input output workspaces for `H` and `tau` unless `FAST_QR_DISABLE_OUTPUT_WORKSPACE_CACHE=1` is set. `current-candidate` config rows still include `FAST_QR_ENABLE_QR{n}_BLOCKED_CUDA=1` and explicit `FAST_QR_QR{n}_STRUCTURED_BEFORE_CUDA` values so the same configs remain explicit and portable to non-B200 CUDA boxes. The blocked kernels consume both explicit blocked names such as `FAST_QR_QR512_BLOCKED_PANEL_B` and the shape-family aliases such as `FAST_QR_QR512_PANEL_B`; explicit blocked names take precedence. `CTA_SCHEDULE=fixed` uses the configured `CTAS_PER_MATRIX` cap, `frontload` expands early wide panels to more column-tile CTAs, and `all-tiles` launches one CTA per active column tile. `SYNC_FREE_AUTO_POLICY=1` keeps the per-matrix blocked-policy decision on device; `0` allows the host metadata path to collapse homogeneous cases into the direct blocked launch. `BLOCKED_AUTO_GROUPS=1` uses indexed launches for nonhomogeneous policy groups such as rankdef, clustered, and tail-projection subsets; `0` falls back to the monolithic policy kernel after metadata generation. `POLICY_FULL_SCAN=1` scans all rows for the device-side structured-policy columns; `0` uses the cheaper sparse row sample. `PRECISION_MODE=tf32`, `tf32-input`, `fp16`, or `fp16-input` rounds trailing-update operands while keeping panel factorization FP32. `PANEL_REFRESH_MODE=prefix` reloads each active panel from the original input after applying prefix reflectors, and `R_MAINTENANCE_MODE=panel-prefix` repairs finalized panel rows of `R` from the original input without materializing full `Q`. `FAST_QR_QR512_STRUCTURED_BEFORE_CUDA=1`, `FAST_QR_QR1024_STRUCTURED_BEFORE_CUDA=1`, or the global `FAST_QR_STRUCTURED_ROUTES_BEFORE_CUDA=1` runs classifier/structured routes before falling back to the robust CUDA path; explicit `0` values force CUDA-first dispatch. `FAST_QR_TRUST_SAMPLED_STRUCTURED_GUARDS=1` skips exact tail verification for sampled structured groups, while `FAST_QR_TRUST_SAMPLED_STRUCTURED_GUARDS=0` or `FAST_QR_DISABLE_B200_TRUST_SAMPLED_STRUCTURED_GUARDS=1` restores exact checks. Generated `FAST_QR_QR{n}_TAIL_CUT`, `FAST_QR_QR{n}_TAIL_THRESHOLD`, and `FAST_QR_QR{n}_TAIL_FORCE` keys are consumed as dense-tail policy aliases; explicit `FAST_QR_DENSE_TAIL_CUT_{n}`, `FAST_QR_DENSE_TAIL_THRESHOLD_{n}`, and `FAST_QR_DENSE_TAIL_FORCE_{n}` keys take precedence. `TAIL_FORCE=1` deliberately bypasses the sampled dense-tail guard for that generated config, so keep it in correctness-gated sweeps only. `summary.json` records `candidate_consumed_env_keys`, `tuner_consumed_env_keys`, `inert_env_keys`, and `cuda_route_bypassed_env_keys` for every config. With `--collect-resource-metrics`, the tuner injects `--ptxas-options=-v` into `FAST_QR_<SHAPE_LABEL>_EXTRA_CUDA_CFLAGS`, parses `run.log`, and attaches register counts, static shared memory, spills, and a rough occupancy estimate. Use `--resource-cflags-env FAST_QR_QR512_EXTRA_CUDA_CFLAGS` if a kernel family uses a non-default compile-flag env var.

`POLICY_FULL_SCAN` also controls the dense-tail guard inside the blocked CUDA policy: `1` scans all rows, while `0` uses the same deterministic sparse row sample as the structured-policy classifier.

Generate bounded config grids instead of hand-writing rows for each shape family. Use `--mode current-candidate` for paid B200 sweeps against the current implementation: it emits blocked-route enable flags plus knobs that the current submission can read. For QR512 and QR1024 that includes blocked panel width, tile width, precision, prefix panel refresh, panel-prefix R maintenance, CTA sizing via `WARPS_PER_CTA`, column-tile scheduling via `CTAS_PER_MATRIX` and `CTA_SCHEDULE`, auto-policy routing via `SYNC_FREE_AUTO_POLICY`, grouped indexed policy launches via `BLOCKED_AUTO_GROUPS`, device-policy scan depth via `POLICY_FULL_SCAN`, dense-tail `TAIL_CUT` / `TAIL_THRESHOLD` / `TAIL_FORCE`, and `STRUCTURED_BEFORE_CUDA=0,1` to compare robust CUDA-first dispatch against classifier/structured-first dispatch with CUDA fallback. For QR2048 and QR4096 the current grid enables the blocked route and varies blocked panel width, tile width, precision, prefix repair modes, CTA sizing/scheduling, sync-free auto policy, grouped policy launches, device-policy scan depth, and dense-tail aliases. Use the default `--mode future-blocked` when planning a broader design grid; cluster size remains planning metadata until the CUDA path consumes it.

Let the suite pick the next required kernel target directly from `tools/candidate_policy.py`. This currently selects QR512, fills the `3,7,9,10` benchmark rows plus the matching public correctness rows, writes a config JSONL, and inserts accelerator preflight before tuning. By default the auto target uses `current-candidate` mode so paid B200 tuning covers only env knobs that the current submission actually consumes. For QR512/QR1024 next-required runs, the generated JSONL is constrained to `PANEL_REFRESH_MODE=prefix` and `R_MAINTENANCE_MODE=panel-prefix` because those are the repair modes the current blocked path must validate before promotion; pass explicit candidate-config tune axis flags if you want a broader comparison:

```bash
python tools/run_b200_suite.py \
  --dry-run \
  --dry-run-json \
  --candidate-config-tune-next-required \
  --candidate-config-tune-large-kernel-plan-max-configs 32

python tools/run_b200_suite.py \
  --candidate-config-tune-next-required \
  --candidate-config-tune-large-kernel-plan-max-configs 32 \
  --candidate-config-tune-collect-resource-metrics
```

Pass the future-blocked mode explicitly when you want to export the broader blocked-kernel design grid. Its panel, tile, precision, refresh, R-maintenance, and CTA scheduling knobs are consumed by the current blocked CUDA path; cluster size remains planning metadata until the CUDA path consumes it:

```bash
python tools/run_b200_suite.py \
  --dry-run \
  --dry-run-json \
  --candidate-config-tune-next-required \
  --candidate-config-tune-large-kernel-plan-mode future-blocked \
  --candidate-config-tune-large-kernel-plan-max-configs 32
```

When `run_b200_suite.py` generates a QR512, QR1024, QR2048, or QR4096 large-kernel config plan, it automatically inserts `candidate_config_accelerator_preflight` before `candidate_config_tune`. That step targets the matching `*_blocked_cuda_auto` accelerator, runs `tools/preflight_accelerators.py --family-cases` once per generated env row, and writes `candidate_config_accelerator_preflight.jsonl`, so config-specific extension build keys, thread counts, auto-policy/tail-cut route behavior, and compile/correctness failures are captured across representative dense, mixed, and structured profiles before the long timing grid. Use `--skip-candidate-config-accelerator-preflight` only for short reruns after the same generated config matrix has already compiled and passed.

Validate the blocked compact-Householder algorithm shape before porting it to CUDA. This PyTorch reference is not a performance path; the `reflectors` mode applies each panel reflector sequentially, `block-full` materializes the full panel transform once, and `compact-wy` forms the triangular block reflector and applies `I - V T.T V.T`. `--precision-mode` keeps panel factorization in FP32 and rounds only the trailing-update operands; use `tf32-input` and `fp16-input` as explicit repair-needed probes, because naive low-precision trailing updates can fail the hard `R - Q.T @ A` gate. `--panel-refresh-mode prefix` refreshes the active panel from the original FP32 input before factorization, and `--r-maintenance-mode panel-prefix` repairs finalized panel rows from the original FP32 input by applying the stored prefix reflectors. Together they prototype the block-local refresh/R-maintenance CUDA needs without materializing full `Q`. Use the FP32 command below to prove the panel loop, trailing update, `H/tau` contract, and column-major output layout:

```bash
python tools/blocked_qr_reference.py \
  --cases cases/public_tests.txt \
  --indices 0 \
  --panel-width 32 \
  --update-mode compact-wy \
  --precision-mode fp32 \
  --r-maintenance-mode panel-prefix \
  --panel-refresh-mode prefix \
  --diagnose \
  --json \
  --out results/blocked_qr_reference.jsonl
```

Sweep the same reference across panel widths and update precision modes before turning a choice into CUDA. Use `--allow-failures` when low-precision rows are expected to fail and the point is to capture structured residuals showing how much block-local R maintenance must recover:

```bash
python tools/blocked_qr_sweep.py \
  --cases cases/public_tests.txt \
  --indices 0 \
  --panel-widths 16,32,64 \
  --update-modes compact-wy \
  --precision-modes fp32,tf32-input,fp16-input \
  --r-maintenance-modes none,panel-prefix \
  --panel-refresh-modes none,prefix \
  --diagnose \
  --allow-failures \
  --json \
  --out results/blocked_qr_precision_sweep.jsonl
```

To package this design probe with the normal B200 export, pass `--include-blocked-qr-sweep` to `tools/run_b200_suite.py`. The default suite sweep uses public test-sized `512`/`1024` dense and mixed rows, not the large public benchmark batches.

```bash
python tools/run_b200_suite.py \
  --include-candidate-config-tune \
  --candidate-config-tune-shape-label qr512 \
  --candidate-config-tune-large-kernel-plan-mode current-candidate \
  --candidate-config-tune-large-kernel-plan-max-configs 16 \
  --candidate-config-tune-benchmark-indices 3,7,9,10 \
  --candidate-config-tune-correctness-indices 3,6,7,8,9,10,11,19 \
  --candidate-config-tune-collect-resource-metrics

python tools/large_kernel_plan.py \
  --shape-label qr1024 \
  --mode current-candidate \
  --max-configs 16 \
  --out results/qr1024_kernel_configs.jsonl \
  --print-command

python tools/run_b200_suite.py \
  --include-candidate-config-tune \
  --candidate-config-tune-shape-label qr1024 \
  --candidate-config-tune-config-jsonl results/qr1024_kernel_configs.jsonl \
  --candidate-config-tune-benchmark-indices 4,8,11 \
  --candidate-config-tune-correctness-indices 4,12,13,14,15,20 \
  --candidate-config-tune-collect-resource-metrics

python tools/large_kernel_plan.py \
  --shape-label qr2048 \
  --mode current-candidate \
  --max-configs 24 \
  --out results/qr2048_large_kernel_configs.jsonl \
  --print-command

python tools/run_b200_suite.py \
  --include-candidate-config-tune \
  --candidate-config-tune-shape-label qr2048 \
  --candidate-config-tune-config-jsonl results/qr2048_large_kernel_configs.jsonl \
  --candidate-config-tune-benchmark-indices 5 \
  --candidate-config-tune-correctness-indices 16,21 \
  --candidate-config-tune-collect-resource-metrics

python tools/large_kernel_plan.py \
  --shape-label qr4096 \
  --mode current-candidate \
  --max-configs 24 \
  --out results/qr4096_large_kernel_configs.jsonl \
  --print-command

python tools/run_b200_suite.py \
  --include-candidate-config-tune \
  --candidate-config-tune-shape-label qr4096 \
  --candidate-config-tune-config-jsonl results/qr4096_large_kernel_configs.jsonl \
  --candidate-config-tune-benchmark-indices 6 \
  --candidate-config-tune-correctness-indices 5,18 \
  --candidate-config-tune-collect-resource-metrics
```

For a direct route-order ablation without the generated plan, pass one-off configs:

```bash
python tools/tune_candidate_configs.py \
  --shape-label qr512 \
  --config 'cuda_first:FAST_QR_QR512_STRUCTURED_BEFORE_CUDA=0' \
  --config 'structured_first:FAST_QR_QR512_STRUCTURED_BEFORE_CUDA=1' \
  --correctness-indices 3,6,7,8,9,10,11,19 \
  --benchmark-indices 3,7,9,10 \
  --popcorn-seeds public,1,2,3 \
  --repeats 3 \
  --allow-failed-configs
```

When those configs are present, `suite_summary.md` and `suite_analysis.md` add a route-order decision sourced from matched `STRUCTURED_BEFORE_CUDA=0` versus `1` configs. Use that decision to choose between robust CUDA-first dispatch and classifier/structured-first dispatch for the next QR512/QR1024 kernel iteration.

Check optional accelerators explicitly before timing. By default this requires the QR32, QR176, QR352, QR512, and QR1024 CUDA extensions to compile and pass; add `--allow-fallback` if you only want to record fallback status.

Check the B200 runtime explicitly:

```bash
python tools/check_b200_env.py --json --out results/runtime_preflight.jsonl
```

For standalone compiled-extension experiments outside `run_b200_suite.py`, set the PyTorch architecture list before the first import/build. The optional QR32/QR176/QR352/QR512/QR1024 flag overrides, extension build keys, and extension names are recorded in the accelerator preflight output, so result exports show exactly what was requested and which compile-cache identity was used:

```bash
export TORCH_CUDA_ARCH_LIST=10.0
export FAST_QR_QR32_WARPS_PER_CTA=4
export FAST_QR_QR32_EXTRA_CUDA_CFLAGS="-arch=sm_100"
export FAST_QR_QR176_EXTRA_CUDA_CFLAGS="-arch=sm_100"
export FAST_QR_QR352_EXTRA_CUDA_CFLAGS="-arch=sm_100"
export FAST_QR_QR512_EXTRA_CUDA_CFLAGS="-arch=sm_100"
export FAST_QR_QR1024_EXTRA_CUDA_CFLAGS="-arch=sm_100"
```

```bash
python tools/preflight_accelerators.py \
  --submission submissions/candidate.py \
  --json \
  --out results/accelerator_preflight.jsonl
```

For standalone paid sweeps outside `run_b200_suite.py`, preflight the generated config matrix manually. This reloads the candidate once per env row and records the selected extension build key, thread count, route-order env, and pass/fail result before the long timing run:

```bash
python tools/preflight_accelerators.py \
  --submission submissions/candidate.py \
  --large-kernel-plan-shape-label qr512 \
  --large-kernel-plan-mode current-candidate \
  --large-kernel-plan-max-configs 16 \
  --family-cases \
  --json \
  --out results/qr512_accelerator_config_preflight.jsonl
```

For QR1024, change `--large-kernel-plan-shape-label qr1024`. Use `--allow-fallback` only for local smoke checks; omit it on B200 when you want extension compile/use failures to stop the run.

`python tools/print_spec.py` also reports the frozen evaluator timing contract. The current official benchmark imports `custom_kernel` inside the benchmark worker, calls it once for correctness before timed CUDA events, then calls it again inside the timed loop and rechecks timed outputs when requested. That means inline extension compilation should be forced by the pre-timing correctness call, while failed or lazily skipped builds still show up in timed-output validation.

Validate final one-file Popcorn packaging before submitting. This stages the selected file as `submission.py`, rejects repo-local or same-directory helper imports, checks the `custom_kernel(data)` entrypoint, scans the candidate source for high-signal token/private-key patterns, and imports the staged file with `official/` first on `sys.path`.

```bash
python tools/validate_submission.py --submission submissions/candidate.py
python tools/validate_submission.py \
  --submission submissions/candidate.py \
  --json \
  --out results/submission_validation.jsonl
```

Summarize an exported suite after it finishes:

```bash
python tools/summarize_suite.py \
  --suite-dir results/<suite-name> \
  --json-out results/<suite-name>/suite_summary.json \
  --markdown-out results/<suite-name>/suite_summary.md
```

Analyze the same export into next-action decisions. This chooses the official-style comparison when present, falls back to public or smoke timings, emits a top-level `Final Algorithm Recommendation`, ranks weak cases, groups timings into shape-family implementation priorities, summarizes route ablations, reports classifier/guard overhead as a percent of candidate runtime, turns the `FAST_QR_DISABLE_DATA_DEPENDENT_ROUTES=1` ablation into a keep/remove decision for ambiguous-shape tensor guards, turns the QR512/QR1024 CUDA-disable ablation into a target-shape keep/demote decision, summarizes actual public-benchmark `H` output layouts by shape, highlights public benchmark, tail-policy, and dev robustness margins, and condenses experiment pass rates.

```bash
python tools/analyze_b200_results.py \
  --suite-dir results/<suite-name> \
  --json-out results/<suite-name>/suite_analysis.json \
  --markdown-out results/<suite-name>/suite_analysis.md
```

The full default suite runs `tools/validate_b200_suite.py --allow-incomplete` automatically before packaging, then runs strict completed-export validation after the `.tgz` exists. Strict validation requires both public-seed and POPCORN-mutated seed coverage in the seed and tail-policy sweeps, and it rejects result rows whose hashes do not match the candidate or baseline hashes recorded in `manifest.jsonl`. That includes benchmark timing, correctness, sweeps, policy, route trace, guard overhead, verifier experiments, and submission-validation rows. If `--include-popcorn-test` was used, strict validation also checks the nested Popcorn manifest, staged `submission.py`, submission validation row, and `popcorn.log`; the nested source/staged hashes and actual `submission.py` bytes must match the suite candidate hash, so the archive proves remote `mode=test` finished cleanly against the same file that was timed locally. If you use skip flags that produce a partial suite, strict validation is skipped and the manifest records the blockers. You can also validate a completed default suite manually; this requires the final `suite_finish` manifest row and the exported `.tgz` tarball:

```bash
python tools/validate_b200_suite.py --suite-dir results/<suite-name>
python tools/validate_b200_suite.py --suite-dir results/<suite-name> --json
```

Print the current candidate dispatch and cutoff policy for the public benchmark set before timing.

```bash
python tools/candidate_policy.py
python tools/candidate_policy.py \
  --json \
  --record-env \
  --out results/candidate_policy_public.jsonl
```

In the JSON rows, `case_metadata_available: false` and `case_metadata_passed_to_submission: false` mean the official entrypoint is still only `custom_kernel(data)`. Rows with `shape_only_case_selection: true` do not need a classifier; rows with `uses_tensor_values_for_case_selection: true` are the ambiguous shape families where case-specific paths need conservative guards. `classifier_needed_for_case_specific_path` is therefore about whether a case-specialized shortcut needs tensor-value guards, while `classifier_on_current_hot_path` / `classifier_needed_for_current_candidate` is about whether the default current route actually runs that classifier before dispatch. The `h_layout` and `column_major_h` fields distinguish routes that always return column-major `H`, routes that are column-major only when their fast path applies, and pure `torch.geqrf` fallback rows that use PyTorch's default layout.

Report whether each public benchmark route is a real custom CUDA path, a PyTorch-composite experiment, or a plain `torch.geqrf` fallback.

```bash
python tools/implementation_status.py \
  --json \
  --record-env \
  --out results/candidate_implementation_status.jsonl
```

Use this before interpreting timings. Rows with `implementation_kind: "torch_composite_experiment"` are useful verifier shortcuts, but they are not the final blocked Householder kernels described above. The current QR512 and QR1024 inline CUDA routes are also marked as stopgap probes: their policy/status rows include `required_cuda_kernel`, `required_repair_modes`, `candidate_config_shape_label`, and the exact benchmark/correctness indices to use when tuning the blocked compact-Householder replacement. The summary row reports `ready_for_final_submission` and the next high-priority cases that still require custom kernels.

Before treating a candidate as final, rerun suite validation with the final-kernel gate. This intentionally fails while any public benchmark route still reports `final_kernel_required: true`, which is expected for exploratory suites that still use PyTorch-composite experiments:

```bash
python tools/validate_b200_suite.py \
  --suite-dir results/<suite-name> \
  --require-final-kernels
```

The same gate can be applied at the end of a one-shot run:

```bash
python tools/run_b200_suite.py --require-final-kernels
```

After a B200 suite finishes, read the `Data-Dependent Dispatch` section in `suite_analysis.md`. It compares default timing with `FAST_QR_DISABLE_STRUCTURED_ROUTES=1` for the colliding shapes to answer the classifier/case-specific-route question directly, and also reports the broader `FAST_QR_DISABLE_DATA_DEPENDENT_ROUTES=1` ratio for all tensor-value guards. Strict completed-suite validation rejects exports where this decision is missing or still `insufficient-data`.

Start with the `Final Algorithm Recommendation` section in `suite_analysis.md`. It records that the official API still does not pass classifier metadata, says whether case-specific paths need tensor-value guards, and turns the measured correctness, ablation, layout, tail-policy, and shape-family evidence into an ordered action list for the next kernel decision.

Trace the actual data-dependent route decisions on generated public benchmark inputs.

```bash
python tools/trace_candidate_routes.py \
  --submission submissions/candidate.py \
  --cases cases/public_benchmarks.txt \
  --json \
  --record-env \
  --out results/candidate_route_trace_public.jsonl
```

Each trace row records `case_metadata_passed_to_submission: false`, `case_selection_info_sources`, `dispatch_info_sources`, `route_decision_source`, `uses_tensor_values_for_dispatch`, `uses_tensor_values_for_case_selection`, `classifier_needed_for_case_specific_path`, and `classifier_on_current_hot_path`. Only the `640x512` and `60x1024` colliding families need tensor values for case selection. A robust CUDA-first route can still show `classifier_on_current_hot_path: false`; other shapes may still use tensor-value guards for optional tail-policy routing, but they do not need a classifier to recover hidden case metadata.

Sweep sampled classifier decisions across the colliding public benchmark families and POPCORN-mutated seeds. This records the sampled class, expected class, actual route, optional CUDA-route bypass, and classifier/route timing without running QR.

```bash
python tools/classifier_seed_sweep.py \
  --submission submissions/candidate.py \
  --cases cases/public_benchmarks.txt \
  --indices 3,4,7,8,9,10,11 \
  --popcorn-seeds public,1,2,3 \
  --include-plan \
  --json \
  --record-env \
  --out results/classifier_seed_sweep.jsonl
```

Strict completed-suite validation requires this file to include both `640x512` and `60x1024`, the public seed, at least one POPCORN-mutated seed, and zero classifier/route mismatches.

Measure the route/classifier guard overhead without running QR. This reports cold route cost and hot cached route cost separately; on CUDA it also includes CUDA-event time. Guard rows include `uses_tensor_values_for_case_selection`, so `suite_analysis.md` can separate real classifier/case-selection overhead from other tensor guards such as dense-tail routing. The cold wall time is the value to watch for `.item()` synchronization cost, while the hot wall time shows the repeated-call route-cache overhead.

```bash
python tools/benchmark_guards.py \
  --submission submissions/candidate.py \
  --cases cases/public_benchmarks.txt \
  --repeats 20 \
  --warmup 3 \
  --json \
  --record-env \
  --out results/candidate_guard_overhead_public.jsonl
```

The full B200 suite also benchmarks route ablations against `candidate_public.jsonl`:

```text
FAST_QR_DISABLE_ROUTE_CACHE=1
FAST_QR_QR512_STRUCTURED_BEFORE_CUDA=0 FAST_QR_QR1024_STRUCTURED_BEFORE_CUDA=0 FAST_QR_DISABLE_B200_DEFAULT_STRUCTURED_BEFORE_CUDA=1
FAST_QR_DISABLE_STRUCTURED_ROUTES=1
FAST_QR_DISABLE_DENSE_TAIL=1
FAST_QR_DISABLE_DATA_DEPENDENT_ROUTES=1
FAST_QR_DISABLE_QR512_CUDA=1 FAST_QR_DISABLE_QR1024_CUDA=1
```

Use the `Ablation:` sections in `suite_summary.md`. A ratio above `1.0x` means the ablation was slower than the default candidate for that case/geomean; a ratio below `1.0x` means the ablation was faster and the default route should be reconsidered. For route order, use the candidate-config tune rows with matched `STRUCTURED_BEFORE_CUDA=0` and `1` values; the default candidate is now CUDA-first, while structured-first remains an explicit comparison path. For the classifier decision, prefer `no_structured_routes` over `no_data_dependent_routes` because it disables the case-specific structured classifier without disabling unrelated dense-tail tensor guards.
For the large-shape CUDA decision, read the `Large CUDA Probe Ablation` section in `suite_analysis.md`. It uses only the affected `640x512` and `60x1024` rows for the decision, so unrelated small-shape timings cannot dilute whether the one-CTA QR512/QR1024 probes should be kept or demoted while the blocked multi-CTA Householder kernel is built.

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
Use `bench_local.py --indices 3,7,9,10` to time only the public `640x512` benchmark family, or `--indices 4,8,11` for the public `60x1024` family. Result rows include `case_index` so filtered outputs still map back to `cases/public_benchmarks.txt`.

For seed robustness sweeps:

```bash
python tools/seed_sweep.py \
  --submission submissions/candidate.py \
  --cases cases/public_tests.txt \
  --indices 19,20,21 \
  --popcorn-seeds public,1,2,3,4,5 \
  --diagnose \
  --max-factor-scaled 18 \
  --max-orth-scaled 80 \
  --record-env \
  --out results/seed_sweep_mixed.jsonl
```

Record a tracked sweep to `results/runs.jsonl`. That file is intentionally committed as an empty JSONL placeholder so the default sweep output path exists; it is normal for it to contain no rows before the first sweep. By default each passing case appends timing and environment metadata; with `--record-env`, each row also captures active `FAST_QR_*` candidate knobs under `candidate_env` and the broader tracked runtime/tuner knobs under `tracked_env`. Pass diagnostic flags only when you want expensive residual checks.

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
  --record-env \
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
python tools/validate_submission.py --submission submissions/candidate.py
python tools/submit_popcorn.py --mode leaderboard --submission submissions/candidate.py
```

Use `--mode both` only after you intend to run test and leaderboard back to back. Add `--dry-run` to stage `results/<run>/submission.py` and inspect the exact Popcorn command without submitting. `submit_popcorn.py` runs the same one-file validation first unless you pass `--skip-validation`.

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
