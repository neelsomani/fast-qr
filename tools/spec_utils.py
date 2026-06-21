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
