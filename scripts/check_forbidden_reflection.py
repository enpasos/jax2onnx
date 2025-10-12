#!/usr/bin/env python3
# scripts/check_forbidden_reflection.py

from __future__ import annotations

import argparse
import ast
import pathlib
import sys
from typing import Iterable

BANNED_GETATTR_NAMES = {
    "node",
    "_nodes",
    "nodes",
    "input",
    "inputs",
    "output",
    "outputs",
    "initializer",
    "initializers",
}

BANNED_ISINSTANCE_TYPES = {
    "ir.Graph",
    "Graph",
    "onnx_ir.Graph",
}


def discover_files(paths: Iterable[str]) -> list[pathlib.Path]:
    candidates: list[pathlib.Path] = []
    for path_str in paths:
        path = pathlib.Path(path_str)
        if path.is_dir():
            candidates.extend(path.glob("**/*.py"))
        elif path.is_file():
            candidates.append(path)
    return candidates


def resolve_type_repr(node: ast.expr) -> str | None:
    if isinstance(node, ast.Attribute):
        prefix = resolve_type_repr(node.value)
        if prefix:
            return f"{prefix}.{node.attr}"
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def check_file(path: pathlib.Path) -> list[str]:
    try:
        source = path.read_text()
    except OSError as exc:
        return [f"{path}:0: unable to read file ({exc})"]

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno}: syntax error while parsing: {exc.msg}"]

    violations: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Name):
                if node.func.id == "getattr":
                    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                        attr_name = node.args[1].value
                        if (
                            isinstance(attr_name, str)
                            and attr_name in BANNED_GETATTR_NAMES
                        ):
                            violations.append(
                                f"{path}:{node.lineno}: forbidden getattr(..., '{attr_name}') usage"
                            )
                elif node.func.id == "isinstance":
                    if len(node.args) == 2:
                        type_repr = resolve_type_repr(node.args[1])
                        if type_repr in BANNED_ISINSTANCE_TYPES:
                            violations.append(
                                f"{path}:{node.lineno}: forbidden isinstance(..., {type_repr}) usage"
                            )
            self.generic_visit(node)

    Visitor().visit(tree)
    return violations


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Fail on disallowed isinstance/getattr patterns."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["jax2onnx/converter"],
        help="Files or directories to inspect (default: jax2onnx/converter)",
    )
    args = parser.parse_args(argv)

    files = discover_files(args.paths)
    failures: list[str] = []
    for file in files:
        failures.extend(check_file(file))

    if failures:
        print("\n".join(sorted(failures)))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
