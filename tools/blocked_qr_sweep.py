from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blocked_qr_reference import (
    PANEL_REFRESH_MODES,
    PRECISION_MODES,
    R_MAINTENANCE_MODES,
    UPDATE_MODES,
    run_case,
    selected_specs,
)
from qr_common import ROOT, append_jsonl, apply_popcorn_seed


def _csv_tokens(raw: str) -> list[str]:
    return [piece.strip() for piece in raw.split(",") if piece.strip()]


def parse_panel_widths(raw: str) -> list[int]:
    widths = [int(token) for token in _csv_tokens(raw)]
    if not widths:
        raise ValueError("at least one panel width is required")
    for width in widths:
        if width <= 0:
            raise ValueError(f"panel widths must be positive, got {width}")
    return widths


def parse_choices(raw: str, choices: tuple[str, ...], label: str) -> list[str]:
    values = _csv_tokens(raw)
    if not values:
        raise ValueError(f"at least one {label} is required")
    unknown = [value for value in values if value not in choices]
    if unknown:
        raise ValueError(f"unknown {label}: {', '.join(unknown)}; expected one of: {', '.join(choices)}")
    return values


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_precision = []
    for precision_mode in sorted({str(row["precision_mode"]) for row in rows}):
        group = [row for row in rows if row["precision_mode"] == precision_mode]
        by_precision.append(
            {
                "precision_mode": precision_mode,
                "num_rows": len(group),
                "num_failed": sum(1 for row in group if not row["ok"]),
                "max_factor_scaled": max(
                    (row["factor_scaled_max"] for row in group if "factor_scaled_max" in row),
                    default=None,
                ),
                "max_orth_scaled": max(
                    (row["orth_scaled_max"] for row in group if "orth_scaled_max" in row),
                    default=None,
                ),
            }
        )

    return {
        "summary": True,
        "ok": all(row["ok"] for row in rows),
        "num_rows": len(rows),
        "num_failed": sum(1 for row in rows if not row["ok"]),
        "panel_widths": sorted({int(row["panel_width"]) for row in rows}),
        "update_modes": sorted({str(row["update_mode"]) for row in rows}),
        "precision_modes": sorted({str(row["precision_mode"]) for row in rows}),
        "r_maintenance_modes": sorted({str(row["r_maintenance_mode"]) for row in rows}),
        "panel_refresh_modes": sorted({str(row["panel_refresh_mode"]) for row in rows}),
        "by_precision_mode": by_precision,
    }


def sweep(
    selected: list[tuple[int, dict[str, Any]]],
    panel_widths: list[int],
    update_modes: list[str],
    precision_modes: list[str],
    r_maintenance_modes: list[str],
    panel_refresh_modes: list[str],
    *,
    popcorn_seed: int | None = None,
    diagnose_output: bool = False,
) -> list[dict[str, Any]]:
    specs = apply_popcorn_seed([spec for _, spec in selected], popcorn_seed)
    rows = []
    for (case_index, _), spec in zip(selected, specs):
        for panel_width in panel_widths:
            for update_mode in update_modes:
                for precision_mode in precision_modes:
                    for r_maintenance_mode in r_maintenance_modes:
                        for panel_refresh_mode in panel_refresh_modes:
                            row = run_case(
                                spec,
                                panel_width=panel_width,
                                column_major_h=True,
                                update_mode=update_mode,
                                precision_mode=precision_mode,
                                r_maintenance_mode=r_maintenance_mode,
                                panel_refresh_mode=panel_refresh_mode,
                                diagnose_output=diagnose_output,
                            )
                            row["case_index"] = case_index
                            row["popcorn_seed"] = popcorn_seed
                            rows.append(row)
    rows.append(summarize([row for row in rows if not row.get("summary")]))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep blocked compact-Householder reference update and precision modes.",
        allow_abbrev=False,
    )
    parser.add_argument("--cases", default="cases/public_tests.txt")
    parser.add_argument("--case", default=None)
    parser.add_argument("--indices", default="0", help="Comma-separated case indexes. Defaults to first public test.")
    parser.add_argument("--panel-widths", default="16,32")
    parser.add_argument("--update-modes", default="compact-wy")
    parser.add_argument("--precision-modes", default="fp32,tf32-input")
    parser.add_argument("--r-maintenance-modes", default="none")
    parser.add_argument("--panel-refresh-modes", default="none")
    parser.add_argument("--popcorn-seed", type=int, default=None)
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    selected = selected_specs(args)
    panel_widths = parse_panel_widths(args.panel_widths)
    update_modes = parse_choices(args.update_modes, UPDATE_MODES, "update mode")
    precision_modes = parse_choices(args.precision_modes, PRECISION_MODES, "precision mode")
    r_maintenance_modes = parse_choices(args.r_maintenance_modes, R_MAINTENANCE_MODES, "R maintenance mode")
    panel_refresh_modes = parse_choices(args.panel_refresh_modes, PANEL_REFRESH_MODES, "panel refresh mode")
    rows = sweep(
        selected,
        panel_widths,
        update_modes,
        precision_modes,
        r_maintenance_modes,
        panel_refresh_modes,
        popcorn_seed=args.popcorn_seed,
        diagnose_output=args.diagnose,
    )

    if args.json:
        for row in rows:
            print(json.dumps(row, sort_keys=True), flush=True)
    else:
        for row in rows:
            if row.get("summary"):
                print(json.dumps(row, sort_keys=True))
                continue
            status = "PASS" if row["ok"] else "FAIL"
            print(
                f"{row['case_index']}: {status} {row['case_text']} panel={row['panel_width']} "
                f"update={row['update_mode']} precision={row['precision_mode']} "
                f"r={row['r_maintenance_mode']} refresh={row['panel_refresh_mode']}"
            )

    if args.out:
        append_jsonl(ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out), rows)

    summary = rows[-1]
    if summary["ok"] or args.allow_failures:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
