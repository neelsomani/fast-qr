from __future__ import annotations

import argparse
import json
import shlex

from run_b200_suite import candidate_config_required_targets, candidate_config_tune_command_for_target


def build_rows(args: argparse.Namespace) -> list[dict]:
    targets = candidate_config_required_targets(args)
    rows = []
    for index, target in enumerate(targets, start=1):
        suite_name = f"{args.suite_prefix}_{index}_{target['shape_label']}"
        command = candidate_config_tune_command_for_target(
            target,
            suite_name=suite_name,
            max_configs=args.max_configs,
            mode=args.mode,
            python=args.python,
        )
        rows.append(
            {
                "index": index,
                "shape_label": target["shape_label"],
                "required_cuda_kernel": target.get("required_cuda_kernel"),
                "benchmark_indices": target.get("benchmark_indices", ""),
                "correctness_indices": target.get("correctness_indices", ""),
                "source_case_indices": target.get("source_case_indices", []),
                "required_repair_modes": target.get("required_repair_modes", []),
                "suite_name": suite_name,
                "command": command,
                "shell_command": shlex.join(command),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print B200 candidate-config tuning commands for every required CUDA shape family."
    )
    parser.add_argument("--submission", default="submissions/candidate.py")
    parser.add_argument("--candidate-config-tune-benchmark-cases", default="cases/public_benchmarks.txt")
    parser.add_argument("--suite-prefix", default="b200_required")
    parser.add_argument("--max-configs", type=int, default=8)
    parser.add_argument("--mode", default="current-candidate", choices=("current-candidate", "future-blocked"))
    parser.add_argument("--python", default="python")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows = build_rows(args)
    if args.json:
        print(json.dumps({"num_targets": len(rows), "targets": rows}, sort_keys=True))
        return 0

    print(f"required B200 tuning targets: {len(rows)}")
    for row in rows:
        print(
            f"{row['index']}. {row['shape_label']} / {row['required_cuda_kernel']} "
            f"(bench: {row['benchmark_indices']}; correctness: {row['correctness_indices']})"
        )
        print(row["shell_command"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
