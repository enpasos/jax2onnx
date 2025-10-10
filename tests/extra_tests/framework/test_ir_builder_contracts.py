# tests/extra_tests/framework/test_ir_builder_contracts.py

from __future__ import annotations

import ast
from pathlib import Path

import pytest


ALLOWED_OUTPUT_NODE_TYPES = (
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Name,
    ast.Call,
    ast.Attribute,
    ast.Subscript,
)


def _project_root(start: Path) -> Path:
    """Walk upwards until the repository root containing 'jax2onnx' is found."""

    path = start
    while path != path.parent:
        if (path / "jax2onnx").exists():
            return path
        path = path.parent
    return start.parents[2]


def _collect_outputs_violations(pyfile: Path) -> list[tuple[int, str]]:
    src = pyfile.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src, filename=str(pyfile))
    except SyntaxError as exc:  # pragma: no cover - fail fast on syntax errors
        return [(exc.lineno or 0, f"SyntaxError: {exc.msg}")]

    issues: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg != "_outputs":
                continue
            value = kw.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                issues.append((value.lineno, "string literal used for _outputs"))
                continue
            if isinstance(value, ast.Constant) and value.value is None:
                continue
            if isinstance(value, ALLOWED_OUTPUT_NODE_TYPES):
                continue
            issues.append(
                (
                    value.lineno,
                    f"unsupported _outputs expression: {type(value).__name__}",
                )
            )
    return issues


@pytest.mark.parametrize(
    "subdir",
    [
        "jax2onnx/converter",
        "jax2onnx/plugins",
    ],
)
def test_outputs_keyword_uses_collections(subdir: str) -> None:
    root = _project_root(Path(__file__).resolve())
    target = root / subdir
    if not target.exists():  # pragma: no cover - defensive for unusual layouts
        pytest.skip(f"directory missing: {target}")

    offenders: list[tuple[Path, int, str]] = []
    for pyfile in target.rglob("*.py"):
        if pyfile.name.endswith("_pb2.py"):
            continue
        for lineno, msg in _collect_outputs_violations(pyfile):
            offenders.append((pyfile, lineno, msg))

    if offenders:
        formatted = "\n".join(
            f"- {path.relative_to(root)}:{lineno}: {msg}"
            for path, lineno, msg in sorted(offenders)
        )
        pytest.fail(
            "_outputs must be provided as a sequence (list/tuple).\n"
            "Found string literals or unsupported expressions:\n" + formatted
        )
