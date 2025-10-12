# tests/extra_tests/framework/test_no_onnx_in_converter_plugins.py

from __future__ import annotations

import ast
from pathlib import Path
import pytest


FORBIDDEN_ROOT = "onnx"  # forbid 'onnx' and any submodule 'onnx.*'
FORBIDDEN_ATTR_CHAINS = {
    ("onnx", "ModelProto"),
    ("onnx", "helper"),
    ("onnx", "shape_inference"),
}


def _attr_chain(node: ast.AST) -> tuple[str, ...] | None:
    """Return attribute access path (e.g. onnx.helper.make_model -> (onnx, helper, make_model))."""

    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.insert(0, current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.insert(0, current.id)
        return tuple(parts)
    return None


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


def _scan_file_for_onnx_usage(pyfile: Path) -> dict[str, list[tuple[int, str]]]:
    """Return policy violations grouped by category for a given file."""

    src = pyfile.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src, filename=str(pyfile))
    except SyntaxError as e:
        # Treat invalid syntax as a test failure to avoid silent skips
        return {
            "imports": [(e.lineno or 0, f"SyntaxError: {e.msg}")],
            "onnx_attrs": [],
            "builder_initializer": [],
        }

    hits: dict[str, list[tuple[int, str]]] = {
        "imports": [],
        "onnx_attrs": [],
        "builder_initializer": [],
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                if mod == FORBIDDEN_ROOT or mod.startswith(FORBIDDEN_ROOT + "."):
                    as_part = f" as {alias.asname}" if alias.asname else ""
                    hits["imports"].append((node.lineno, f"import {mod}{as_part}"))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == FORBIDDEN_ROOT or mod.startswith(FORBIDDEN_ROOT + "."):
                names = ", ".join(
                    f"{a.name}" + (f" as {a.asname}" if a.asname else "")
                    for a in node.names
                )
                hits["imports"].append((node.lineno, f"from {mod} import {names}"))

        attr_path = _attr_chain(node) if isinstance(node, ast.Attribute) else None
        if attr_path and attr_path[:2] in FORBIDDEN_ATTR_CHAINS:
            lineno = getattr(node, "lineno", 0)
            hits["onnx_attrs"].append((lineno, ".".join(attr_path)))

        if isinstance(node, ast.Call):
            attr_path = _attr_chain(node.func)
            if not attr_path:
                continue
            base_parts = attr_path[:-1]
            tail = attr_path[-1]
            if "builder" in base_parts and (
                tail == "initializer" or tail.startswith("add_initializer_from")
            ):
                if not any(kw.arg == "name" for kw in node.keywords if kw.arg):
                    hits["builder_initializer"].append(
                        (node.lineno, ".".join(attr_path))
                    )

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
            hits = _scan_file_for_onnx_usage(py)
            offenders.extend((py, ln, stmt) for ln, stmt in hits["imports"])
            offenders.extend(
                (py, ln, f"forbidden onnx attr access: {stmt}")
                for ln, stmt in hits["onnx_attrs"]
            )
            offenders.extend(
                (
                    py,
                    ln,
                    f"builder initializer missing name kw: {stmt}",
                )
                for ln, stmt in hits["builder_initializer"]
            )

    _walk(root / "jax2onnx" / "converter")
    _walk(root / "jax2onnx" / "plugins")
    return offenders


def test_no_onnx_imports_in_converter_and_plugins():
    """
    Policy test: the new IR2 pipeline (converter, plugins) must not import the
    ONNX *protobuf* library anywhere. All protobuf operations belong outside
    converter/plugins (e.g., in a top-level serde/adapter layer).
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
            "onnx imports found in converter/plugins modules.\n"
            "These packages must be IR-only and must not import 'onnx'.\n" + detailed
        )
