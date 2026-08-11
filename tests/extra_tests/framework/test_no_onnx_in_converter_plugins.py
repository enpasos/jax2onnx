# tests/extra_tests/framework/test_no_onnx_in_converter_plugins.py

from __future__ import annotations

import ast
from pathlib import Path

import pytest


FORBIDDEN_ROOT = "onnx"  # forbid 'onnx' and any submodule 'onnx.*'
FORBIDDEN_ONNX_IR_PREFIX = "onnx_ir._"
FORBIDDEN_ATTR_CHAINS = {
    ("onnx", "ModelProto"),
    ("onnx", "helper"),
    ("onnx", "shape_inference"),
}


def _is_forbidden_import(module: str) -> bool:
    return (
        module == FORBIDDEN_ROOT
        or module.startswith(FORBIDDEN_ROOT + ".")
        or module.startswith(FORBIDDEN_ONNX_IR_PREFIX)
    )


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
                if _is_forbidden_import(mod):
                    as_part = f" as {alias.asname}" if alias.asname else ""
                    hits["imports"].append((node.lineno, f"import {mod}{as_part}"))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            qualified_names = (f"{mod}.{alias.name}" for alias in node.names if mod)
            if _is_forbidden_import(mod) or any(
                _is_forbidden_import(name) for name in qualified_names
            ):
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


@pytest.mark.parametrize(
    "source",
    [
        "import onnx_ir._tape\n",
        "from onnx_ir._tape import Builder\n",
        "from onnx_ir import _tape\n",
        "from onnx_ir import _tape as tape\n",
        "from onnx_ir import tape, _tape\n",
    ],
    ids=[
        "direct-import",
        "direct-from-import",
        "root-from-import",
        "aliased-root-from-import",
        "mixed-root-from-import",
    ],
)
def test_scan_rejects_private_onnx_ir_imports(tmp_path: Path, source: str):
    pyfile = tmp_path / "module.py"
    pyfile.write_text(source, encoding="utf-8")

    hits = _scan_file_for_onnx_usage(pyfile)

    assert hits["imports"] == [(1, source.strip())]


@pytest.mark.parametrize(
    "source",
    [
        "import onnx_ir\n",
        "import onnx_ir as ir\n",
        "from onnx_ir import tape\n",
        "from onnx_ir import tape as _tape\n",
        "from onnx_ir.tape import Tape\n",
        "from another_package import _tape\n",
    ],
    ids=[
        "root-import",
        "aliased-root-import",
        "public-root-from-import",
        "public-root-from-import-private-alias",
        "public-module-from-import",
        "unrelated-private-module",
    ],
)
def test_scan_allows_public_onnx_ir_imports(tmp_path: Path, source: str):
    pyfile = tmp_path / "module.py"
    pyfile.write_text(source, encoding="utf-8")

    hits = _scan_file_for_onnx_usage(pyfile)

    assert hits["imports"] == []


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


def test_no_forbidden_onnx_imports_in_converter_and_plugins():
    """
    Policy test: the IR pipeline must use onnx-ir's public API and stay free of
    ONNX protobuf imports. Protobuf operations belong outside converter/plugins
    (e.g., in a top-level serde/adapter layer).
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
            "Forbidden ONNX imports found in converter/plugins modules.\n"
            "These packages must be IR-only, must not import 'onnx', and must "
            "use the public onnx_ir API rather than onnx_ir._* modules.\n" + detailed
        )
