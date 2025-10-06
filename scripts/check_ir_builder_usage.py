#!/usr/bin/env python3
# scripts/check_ir_builder_usage.py

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys

DEFAULT_TARGETS = ("jax2onnx/converter", "jax2onnx/plugins")


def _attr_chain(node: ast.AST) -> tuple[str, ...] | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.insert(0, current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.insert(0, current.id)
        return tuple(parts)
    return None


def _check_file(pyfile: Path) -> list[str]:
    try:
        src = pyfile.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"{pyfile}: unable to read file ({exc})"]

    try:
        tree = ast.parse(src, filename=str(pyfile))
    except SyntaxError as exc:
        lineno = exc.lineno or 0
        return [f"{pyfile}:{lineno}: SyntaxError: {exc.msg}"]

    issues: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            attr_path = _attr_chain(node.func)
            if attr_path:
                base_parts = attr_path[:-1]
                tail = attr_path[-1]
                if "builder" in base_parts and (
                    tail == "initializer" or tail.startswith("add_initializer_from")
                ):
                    if not any(kw.arg == "name" for kw in node.keywords if kw.arg):
                        issues.append(
                            f"{pyfile}:{node.lineno}: builder initializer missing name keyword"
                        )
            for kw in node.keywords:
                if kw.arg != "_outputs":
                    continue
                value = kw.value
                if isinstance(value, ast.Constant):
                    if isinstance(value.value, str):
                        issues.append(
                            f"{pyfile}:{value.lineno}: _outputs must be a list/tuple, not a string"
                        )
                    continue
                if isinstance(
                    value,
                    (
                        ast.List,
                        ast.Tuple,
                        ast.Dict,
                        ast.Name,
                        ast.Call,
                        ast.Attribute,
                        ast.Subscript,
                    ),
                ):
                    continue
                issues.append(
                    f"{pyfile}:{value.lineno}: unexpected _outputs expression ({type(value).__name__})"
                )
    return issues


def _iter_pyfiles(targets: list[str]) -> list[Path]:
    pyfiles: list[Path] = []
    for target in targets:
        path = Path(target)
        if not path.exists():
            continue
        pyfiles.extend(sorted(path.rglob("*.py")))
    return pyfiles


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check IR builder usage conventions.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=DEFAULT_TARGETS,
        help="Directories or files to scan (defaults to converter+plugins)",
    )
    args = parser.parse_args(argv)

    pyfiles = _iter_pyfiles(list(dict.fromkeys(args.paths)))
    issues: list[str] = []
    for pyfile in pyfiles:
        if pyfile.name.endswith("_pb2.py"):
            continue
        issues.extend(_check_file(pyfile))

    if issues:
        print("IR builder usage issues detected:", file=sys.stderr)
        for issue in issues:
            print(issue, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
