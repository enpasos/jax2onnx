from __future__ import annotations

import ast
from pathlib import Path
import pytest


FORBIDDEN_ROOT = "onnx"  # forbid 'onnx' and any submodule 'onnx.*'


def _project_root(start: Path) -> Path:
    """
    Walk upwards until we find a directory containing 'jax2onnx'.
    Falls back to two levels up if not found (reasonable in most layouts).
    """
    p = start
    while p != p.parent:
        if (p / "jax2onnx").exists():
            return p
        p = p.parent
    # fallback
    return start.parents[2]


def _scan_file_for_onnx_imports(pyfile: Path) -> list[tuple[int, str]]:
    """
    Parse a Python file and return [(lineno, 'import stmt')] for any import
    that references 'onnx' at top-level (import onnx / from onnx import ...).
    """
    src = pyfile.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src, filename=str(pyfile))
    except SyntaxError as e:
        # Treat invalid syntax as a test failure to avoid silent skips
        return [(e.lineno or 0, f"SyntaxError: {e.msg}")]
    hits: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                if mod == FORBIDDEN_ROOT or mod.startswith(FORBIDDEN_ROOT + "."):
                    as_part = f" as {alias.asname}" if alias.asname else ""
                    hits.append((node.lineno, f"import {mod}{as_part}"))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == FORBIDDEN_ROOT or mod.startswith(FORBIDDEN_ROOT + "."):
                names = ", ".join(
                    f"{a.name}" + (f" as {a.asname}" if a.asname else "")
                    for a in node.names
                )
                hits.append((node.lineno, f"from {mod} import {names}"))
    return hits


def _find_offenders(root: Path) -> list[tuple[Path, int, str]]:
    offenders: list[tuple[Path, int, str]] = []

    def _walk(dirpath: Path) -> None:
        if not dirpath.exists():
            return
        for py in dirpath.rglob("*.py"):
            # Skip obvious non-code files if any (optional)
            if py.name.endswith("_pb2.py"):
                continue
            hits = _scan_file_for_onnx_imports(py)
            for ln, stmt in hits:
                offenders.append((py, ln, stmt))

    _walk(root / "jax2onnx" / "converter2")
    _walk(root / "jax2onnx" / "plugins2")
    return offenders


def test_no_onnx_imports_in_converter2_and_plugins2():
    """
    Policy test: the new IR2 pipeline (converter2, plugins2) must not import the
    ONNX *protobuf* library anywhere. All protobuf operations belong outside
    converter2/plugins2 (e.g., in a top-level serde/adapter layer).
    """
    root = _project_root(Path(__file__).resolve())
    offenders = _find_offenders(root)
    if offenders:
        msg_lines = [
            f"- {path.relative_to(root)}:{lineno}: {stmt}"
            for path, lineno, stmt in sorted(offenders)
        ]
        detailed = "\n".join(msg_lines)
        pytest.fail(
            "onnx imports found in converter2/plugins2 modules.\n"
            "These packages must be IR-only and must not import 'onnx'.\n"
            + detailed
        )
