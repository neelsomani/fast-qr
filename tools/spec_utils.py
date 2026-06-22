from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import yaml

from qr_common import ROOT, format_case


def load_task_yml(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path is not None else ROOT / "official" / "task.yml"
    return yaml.safe_load(path.read_text()) or {}


def task_section(doc: dict[str, Any], section: str) -> Any:
    if section in doc:
        return doc[section]
    config = doc.get("config")
    if isinstance(config, dict) and section in config:
        return config[section]
    raise KeyError(f"{section!r} not found in task.yml")


def specs_from_task_yml(section: str, path: str | Path | None = None) -> list[dict[str, Any]]:
    specs = task_section(load_task_yml(path), section)
    if not isinstance(specs, list):
        raise TypeError(f"task.yml section {section!r} must be a list")
    return [dict(spec) for spec in specs]


def case_label(spec: dict[str, Any]) -> str:
    return str(spec.get("case", "dense"))


def benchmark_shape_collisions(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int], list[tuple[int, dict[str, Any]]]] = {}
    for index, spec in enumerate(specs):
        key = (int(spec["batch"]), int(spec["n"]))
        groups.setdefault(key, []).append((index, spec))

    collisions = []
    for (batch, n), items in groups.items():
        labels = []
        for _, spec in items:
            label = case_label(spec)
            if label not in labels:
                labels.append(label)
        if len(labels) <= 1:
            continue
        collisions.append(
            {
                "batch": batch,
                "n": n,
                "indexes": [index for index, _ in items],
                "cases": labels,
                "specs": [format_case(spec) for _, spec in items],
            }
        )
    return sorted(collisions, key=lambda row: (row["n"], row["batch"]))


def render_case_file(specs: list[dict[str, Any]]) -> str:
    return "\n".join(format_case(spec) for spec in specs) + "\n"


def ranking_by(path: str | Path | None = None) -> str:
    doc = load_task_yml(path)
    return str(doc.get("ranking_by") or doc.get("config", {}).get("ranking_by", "unknown"))


def task_name(path: str | Path | None = None) -> str:
    path = Path(path) if path is not None else ROOT / "official" / "task.yml"
    doc = load_task_yml(path)
    if "name" in doc:
        return str(doc["name"])
    for line in path.read_text().splitlines():
        stripped = line.strip()
        match = re.match(r"#\s*name:\s*([A-Za-z0-9_-]+)", stripped)
        if match:
            return match.group(1)
    return "unknown"


def reference_tolerance_factors(path: str | Path | None = None) -> dict[str, float]:
    path = Path(path) if path is not None else ROOT / "official" / "reference.py"
    tree = ast.parse(path.read_text())
    values: dict[str, float] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name not in {"_FACTOR_RTOL_FACTOR", "_ORTH_RTOL_FACTOR"}:
            continue
        value = ast.literal_eval(node.value)
        values[name] = float(value)
    return {
        "factor_rtol_factor": values["_FACTOR_RTOL_FACTOR"],
        "orth_rtol_factor": values["_ORTH_RTOL_FACTOR"],
    }


def custom_kernel_interface(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path is not None else ROOT / "official" / "submission.py"
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name != "custom_kernel":
            continue
        args = node.args
        positional = [arg.arg for arg in [*args.posonlyargs, *args.args]]
        return {
            "name": "custom_kernel",
            "line": node.lineno,
            "positional_args": positional,
            "required_positional_args": len(positional) - len(args.defaults),
            "vararg": args.vararg.arg if args.vararg else None,
            "kwarg": args.kwarg.arg if args.kwarg else None,
        }
    raise ValueError(f"custom_kernel function not found in {path}")


def format_custom_kernel_interface(interface: dict[str, Any]) -> str:
    args = list(interface.get("positional_args") or [])
    if interface.get("vararg"):
        args.append(f"*{interface['vararg']}")
    if interface.get("kwarg"):
        args.append(f"**{interface['kwarg']}")
    return f"{interface.get('name', 'custom_kernel')}({', '.join(args)})"


def _call_line_numbers(node: ast.AST, name: str) -> list[int]:
    lines = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == name:
            lines.append(child.lineno)
    return sorted(lines)


def _assignment_line(function: ast.FunctionDef, target_name: str) -> int | None:
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == target_name:
                return node.lineno
    return None


def _has_import_from(function: ast.FunctionDef, module: str, name: str) -> bool:
    for node in ast.walk(function):
        if not isinstance(node, ast.ImportFrom) or node.module != module:
            continue
        if any(alias.name == name for alias in node.names):
            return True
    return False


def _has_recheck_block(function: ast.FunctionDef) -> bool:
    for node in ast.walk(function):
        if not isinstance(node, ast.If):
            continue
        if isinstance(node.test, ast.Name) and node.test.id == "recheck":
            if _call_line_numbers(node, "check_implementation"):
                return True
    return False


def evaluator_benchmark_contract(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path is not None else ROOT / "official" / "eval.py"
    tree = ast.parse(path.read_text())
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_run_single_benchmark"
        ),
        None,
    )
    if function is None:
        raise ValueError(f"_run_single_benchmark function not found in {path}")

    custom_kernel_calls = _call_line_numbers(function, "custom_kernel")
    first_timed_event = min(_call_line_numbers(function, "clear_l2_cache") or [10**9])
    bm_start_line = _assignment_line(function, "bm_start_time")
    if bm_start_line is not None:
        first_timed_event = min(first_timed_event, bm_start_line)

    pre_timing_calls = [line for line in custom_kernel_calls if line < first_timed_event]
    timed_calls = [line for line in custom_kernel_calls if line > first_timed_event]
    return {
        "submission_imported_inside_benchmark_worker": _has_import_from(function, "submission", "custom_kernel"),
        "custom_kernel_called_before_timing": bool(pre_timing_calls),
        "custom_kernel_called_inside_timed_loop": bool(timed_calls),
        "timed_outputs_rechecked_when_requested": _has_recheck_block(function),
        "l2_cache_cleared_inside_timed_loop": bool(_call_line_numbers(function, "clear_l2_cache")),
        "first_pre_timing_custom_kernel_line": pre_timing_calls[0] if pre_timing_calls else None,
        "first_timed_custom_kernel_line": timed_calls[0] if timed_calls else None,
    }
