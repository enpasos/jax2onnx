#!/usr/bin/env python3
# scripts/audit_ir_dynamic_access.py

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


TARGET_DIRS: tuple[str, ...] = ("jax2onnx/converter", "jax2onnx/plugins")


@dataclass(frozen=True)
class Occurrence:
    path: Path
    lineno: int
    col_offset: int
    snippet: str


class _GetattrVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self._path = path
        self.occurrences: list[Occurrence] = []

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id == "getattr":
            snippet = ast.get_source_segment(self._source, node) or "getattr(...)"
            self.occurrences.append(
                Occurrence(
                    path=self._path,
                    lineno=node.lineno,
                    col_offset=node.col_offset + 1,
                    snippet=snippet.strip(),
                )
            )
        self.generic_visit(node)

    def run(self, source: str) -> Sequence[Occurrence]:
        self._source = source
        tree = ast.parse(source)
        self.visit(tree)
        return tuple(self.occurrences)


def load_allowlist(path: Path | None) -> set[tuple[Path, int]]:
    if path is None or not path.exists():
        return set()
    entries = json.loads(path.read_text())
    return {
        (Path(item["path"]), int(item["lineno"]))
        for item in entries
        if "path" in item and "lineno" in item
    }


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report dynamic getattr usage in converter/plugins. "
            "Optionally fail when new occurrences appear."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (defaults to current working directory).",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=None,
        help="Optional JSON allowlist with existing occurrences.",
    )
    parser.add_argument(
        "--fail-on-new",
        action="store_true",
        help="Exit with code 1 if occurrences outside the allowlist are found.",
    )
    return parser.parse_args(argv)


def scan_paths(root: Path, allowlist: set[tuple[Path, int]]) -> list[Occurrence]:
    occurrences: list[Occurrence] = []
    for rel_dir in TARGET_DIRS:
        directory = root / rel_dir
        for path in sorted(directory.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            visitor = _GetattrVisitor(path.relative_to(root))
            visitor.run(source)
            for occ in visitor.occurrences:
                if (occ.path, occ.lineno) not in allowlist:
                    occurrences.append(occ)
    return occurrences


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or ())
    allowlist = load_allowlist(args.allowlist)
    occurrences = scan_paths(args.root, allowlist)

    if occurrences:
        print("Detected dynamic getattr usage:")
        for occ in occurrences:
            print(f"{occ.path}:{occ.lineno}:{occ.col_offset}: {occ.snippet}")
        if args.fail_on_new:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
