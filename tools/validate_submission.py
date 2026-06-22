from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from qr_common import OFFICIAL_DIR, ROOT, append_jsonl, file_provenance
from audit_secrets import scan_text as scan_secret_text


LOCAL_IMPORT_ROOTS = {
    "candidate",
    "baseline_geqrf",
    "diagnose",
    "eval",
    "experiments",
    "official",
    "qr_common",
    "reference",
    "spec_utils",
    "submission",
    "submissions",
    "tools",
    "utils",
}

DEFAULT_ALLOWED_IMPORT_ROOTS = {
    "task",
    "torch",
    "triton",
    "typing_extensions",
}


def stdlib_roots() -> set[str]:
    roots = set(getattr(sys, "stdlib_module_names", set()))
    roots.update({"__future__"})
    return roots


def root_module(module: str | None) -> str:
    return (module or "").split(".", 1)[0]


def import_rows(tree: ast.AST) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                rows.append(
                    {
                        "kind": "import",
                        "module": alias.name,
                        "root": root_module(alias.name),
                        "line": node.lineno,
                        "level": 0,
                    }
                )
        elif isinstance(node, ast.ImportFrom):
            rows.append(
                {
                    "kind": "from",
                    "module": node.module or "",
                    "root": root_module(node.module),
                    "line": node.lineno,
                    "level": node.level,
                    "names": [alias.name for alias in node.names],
                }
            )
    return rows


def local_module_path(submission_dir: Path, root: str) -> Path | None:
    if not root:
        return None
    py_file = submission_dir / f"{root}.py"
    package = submission_dir / root / "__init__.py"
    if py_file.is_file():
        return py_file
    if package.is_file():
        return package
    return None


def custom_kernel_signature(tree: ast.AST) -> dict[str, Any]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "custom_kernel":
            args = node.args
            positional = len(args.posonlyargs) + len(args.args)
            required_defaults = len(args.defaults)
            return {
                "found": True,
                "line": node.lineno,
                "positional_args": positional,
                "required_positional_args": positional - required_defaults,
                "vararg": args.vararg.arg if args.vararg else None,
                "kwarg": args.kwarg.arg if args.kwarg else None,
            }
    return {"found": False}


def static_validate_submission(
    source: Path,
    allowed_import_roots: set[str] | None = None,
) -> dict[str, Any]:
    allowed_roots = set(DEFAULT_ALLOWED_IMPORT_ROOTS)
    if allowed_import_roots:
        allowed_roots.update(allowed_import_roots)
    allowed_roots.update(stdlib_roots())

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        return {
            "ok": False,
            "errors": [{"check": "source.read", "message": f"{type(exc).__name__}: {exc}"}],
            "warnings": warnings,
            "imports": [],
            "entrypoint": {"found": False},
        }

    try:
        tree = ast.parse(text, filename=str(source))
    except SyntaxError as exc:
        return {
            "ok": False,
            "errors": [
                {
                    "check": "source.syntax",
                    "line": exc.lineno,
                    "message": f"SyntaxError: {exc.msg}",
                }
            ],
            "warnings": warnings,
            "imports": [],
            "entrypoint": {"found": False},
        }

    imports = import_rows(tree)
    source_dir = source.parent
    for row in imports:
        root = row["root"]
        line = row["line"]
        if row.get("level", 0):
            errors.append(
                {
                    "check": "import.relative",
                    "line": line,
                    "message": "relative imports are not valid for a one-file Popcorn submission",
                }
            )
            continue
        if root in LOCAL_IMPORT_ROOTS and root != "task":
            errors.append(
                {
                    "check": "import.repo_local",
                    "line": line,
                    "module": row["module"],
                    "message": f"submission imports repo-local module {row['module']!r}",
                }
            )
            continue
        local_path = local_module_path(source_dir, root)
        if local_path is not None and root != "task":
            errors.append(
                {
                    "check": "import.same_dir",
                    "line": line,
                    "module": row["module"],
                    "message": f"submission imports same-directory helper {local_path.name!r}",
                }
            )
            continue
        if root and root not in allowed_roots:
            errors.append(
                {
                    "check": "import.allowlist",
                    "line": line,
                    "module": row["module"],
                    "message": f"module {row['module']!r} is not in the final-submission import allowlist",
                }
            )

    entrypoint = custom_kernel_signature(tree)
    if not entrypoint["found"]:
        errors.append({"check": "entrypoint.exists", "message": "custom_kernel(data) is missing"})
    elif entrypoint["required_positional_args"] > 1:
        errors.append(
            {
                "check": "entrypoint.signature",
                "line": entrypoint["line"],
                "message": "custom_kernel must be callable with exactly one positional data argument",
            }
        )
    elif entrypoint["positional_args"] < 1 and entrypoint["vararg"] is None:
        errors.append(
            {
                "check": "entrypoint.signature",
                "line": entrypoint["line"],
                "message": "custom_kernel must accept the data argument",
            }
        )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "imports": imports,
        "entrypoint": entrypoint,
    }


def stage_submission(source: Path, stage_dir: Path) -> Path:
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged = stage_dir / "submission.py"
    shutil.copy2(source, staged)
    return staged


def filtered_import_path(stage_dir: Path, original_dir: Path) -> list[str]:
    blocked = {
        "",
        str(ROOT),
        str(ROOT / "tools"),
        str(ROOT / "submissions"),
        str(original_dir),
    }
    blocked_resolved = set()
    for item in blocked:
        if item:
            try:
                blocked_resolved.add(str(Path(item).resolve()))
            except OSError:
                pass

    out = [str(OFFICIAL_DIR), str(stage_dir)]
    for item in sys.path:
        if item in blocked:
            continue
        try:
            resolved = str(Path(item).resolve())
        except OSError:
            resolved = item
        if resolved in blocked_resolved:
            continue
        if item not in out:
            out.append(item)
    return out


def import_staged_submission(staged: Path, original_dir: Path) -> tuple[bool, str | None]:
    old_path = list(sys.path)
    module_name = f"fast_qr_validation_{abs(hash(staged))}"
    try:
        sys.path = filtered_import_path(staged.parent, original_dir)
        spec = importlib.util.spec_from_file_location(module_name, staged)
        if spec is None or spec.loader is None:
            return False, f"could not create import spec for {staged}"
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        custom_kernel = getattr(module, "custom_kernel", None)
        if not callable(custom_kernel):
            return False, "staged submission does not expose callable custom_kernel"
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        sys.path = old_path
        sys.modules.pop(module_name, None)


def validate_submission(
    source: str | Path,
    stage_dir: str | Path | None = None,
    allowed_import_roots: set[str] | None = None,
    skip_import: bool = False,
) -> dict[str, Any]:
    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    source_path = source_path.resolve()

    row: dict[str, Any] = {
        "event": "submission_validation",
        "time": datetime.now().isoformat(),
        "source": str(source_path),
        "ok": False,
        "errors": [],
        "warnings": [],
    }
    if not source_path.is_file():
        row["errors"] = [{"check": "source.exists", "message": f"submission file does not exist: {source_path}"}]
        return row

    row["source_submission"] = file_provenance(source_path)
    static = static_validate_submission(source_path, allowed_import_roots=allowed_import_roots)
    try:
        secret_text = source_path.read_text(encoding="utf-8")
        secret_findings = scan_secret_text(source_path, secret_text)
    except OSError as exc:
        secret_findings = [
            {
                "check": "secret.scan",
                "message": f"{type(exc).__name__}: {exc}",
            }
        ]
    row.update(
        {
            "static_ok": static["ok"],
            "secret_ok": not secret_findings,
            "secret_findings": secret_findings,
            "errors": [
                *static["errors"],
                *[
                    {
                        "check": f"secret.{finding.get('rule', 'unknown')}",
                        "line": finding.get("line"),
                        "message": f"potential secret in submission: {finding.get('match_preview', '<masked>')}",
                    }
                    for finding in secret_findings
                ],
            ],
            "warnings": list(static["warnings"]),
            "imports": static["imports"],
            "entrypoint": static["entrypoint"],
        }
    )

    if row["errors"]:
        row["ok"] = False
        return row

    if skip_import:
        row["import_ok"] = None
        row["ok"] = True
        return row

    if stage_dir is None:
        with tempfile.TemporaryDirectory(prefix="fast_qr_submission_") as tmp:
            return validate_submission(
                source_path,
                stage_dir=tmp,
                allowed_import_roots=allowed_import_roots,
                skip_import=skip_import,
            )

    stage_path = Path(stage_dir)
    if not stage_path.is_absolute():
        stage_path = ROOT / stage_path
    staged = stage_submission(source_path, stage_path)
    row["staged_submission"] = file_provenance(staged)
    import_ok, message = import_staged_submission(staged, source_path.parent)
    row["import_ok"] = import_ok
    if message:
        row["errors"].append({"check": "staged.import", "message": message})
    row["ok"] = bool(static["ok"] and import_ok)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate that a QR v2 submission is self-contained enough for one-file Popcorn staging.",
        allow_abbrev=False,
    )
    parser.add_argument("--submission", default="submissions/candidate.py")
    parser.add_argument("--stage-dir", default=None)
    parser.add_argument("--allow-import", action="append", default=[], help="Extra import root to allow.")
    parser.add_argument("--skip-import", action="store_true", help="Only run static validation; do not import staged file.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", default=None, help="Append the validation row to this JSONL file.")
    args = parser.parse_args()

    result = validate_submission(
        args.submission,
        stage_dir=args.stage_dir,
        allowed_import_roots=set(args.allow_import),
        skip_import=args.skip_import,
    )
    if args.out:
        append_jsonl(ROOT / args.out if not Path(args.out).is_absolute() else args.out, result)

    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        status = "PASS" if result["ok"] else "FAIL"
        print(f"{status}: {result['source']}")
        for error in result["errors"]:
            line = f":{error['line']}" if "line" in error else ""
            module = f" [{error['module']}]" if "module" in error else ""
            print(f"- {error['check']}{line}{module}: {error['message']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
