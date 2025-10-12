#!/usr/bin/env python3
# scripts/check_variable_annotations.py

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, List, Sequence


DEFAULT_TARGETS: Sequence[Path] = (
    Path("jax2onnx") / "converter",
    Path("jax2onnx") / "plugins",
)
ALLOWED_NAMES = {
    "__all__",
    "__annotations__",
    "__doc__",
    "__path__",
    "__package__",
    "__version__",
}
TYPE_ALIAS_TOKENS = (
    "typing.",
    "Union[",
    "Optional[",
    "Callable[",
    "Literal[",
    "TypeVar(",
    "TypeAlias",
    "tuple[",
    "list[",
    "set[",
    "dict[",
    "frozenset[",
    "Sequence[",
    "Mapping[",
    "Iterable[",
    "Protocol",
)


@dataclass
class Violation:
    filename: Path
    line: int
    name: str


class ModuleVisitor(ast.NodeVisitor):
    def __init__(self, source: str):
        self._source = source
        self._scope: list[str] = []
        self.violations: list[Violation] = []

    def _push(self, scope: str) -> None:
        self._scope.append(scope)

    def _pop(self) -> None:
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._push("function")
        self.generic_visit(node)
        self._pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._push("async_function")
        self.generic_visit(node)
        self._pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self._push("class")
        self.generic_visit(node)
        self._pop()

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        if self._scope:
            return
        if node.type_comment:
            return
        if not self._needs_annotation(node.value):
            return
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            name = target.id
            if _should_skip_name(name):
                continue
            self.violations.append(
                Violation(
                    filename=Path(""),
                    line=target.lineno,
                    name=name,
                )
            )

    def _needs_annotation(self, value: ast.AST) -> bool:
        segment = ast.get_source_segment(self._source, value) or ""
        segment = segment.strip()
        if not segment:
            return True
        lowered = segment.lower()
        if lowered.startswith("typing.cast(") or lowered.startswith("cast("):
            return False
        for token in TYPE_ALIAS_TOKENS:
            if token.lower() in lowered:
                return False
        return True


def _should_skip_name(name: str) -> bool:
    if name in ALLOWED_NAMES:
        return True
    stripped = name.strip("_")
    if not stripped:
        return True
    if stripped[0].isupper() and not stripped.isupper():
        # assume CamelCase type alias or class sentinel
        return True
    return False


def _iter_python_files(paths: Sequence[Path]) -> Iterable[Path]:
    for root in paths:
        if root.is_file() and root.suffix == ".py":
            yield root
            continue
        for file in root.rglob("*.py"):
            if file.name == "__init__.py":
                continue
            yield file


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Verify module-level assignments have explicit type annotations."
    )
    parser.add_argument("paths", nargs="*", type=Path, default=[])
    args = parser.parse_args(argv)

    targets = args.paths or list(DEFAULT_TARGETS)

    violations: List[Violation] = []
    for file in _iter_python_files(targets):
        source = file.read_text()
        tree = ast.parse(source, filename=str(file))
        visitor = ModuleVisitor(source)
        visitor.visit(tree)
        for violation in visitor.violations:
            violations.append(
                Violation(filename=file, line=violation.line, name=violation.name)
            )

    if not violations:
        return 0

    for violation in sorted(violations, key=lambda v: (str(v.filename), v.line)):
        print(
            f"{violation.filename}:{violation.line}: "
            f"missing annotation for module-level variable '{violation.name}'"
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
