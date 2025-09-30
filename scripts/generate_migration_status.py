#!/usr/bin/env python
"""Generate MigrationStatus.md summarizing converter2/plugin coverage."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, Set

from jax2onnx.plugins2.plugin_system import (
    EXAMPLE_REGISTRY2,
    PLUGIN_REGISTRY2,
    import_all_plugins as import_plugins2,
)

ROOT = Path(__file__).resolve().parents[1]


def canonical_context(context: str) -> str:
    remaps = {
        "examples2": "examples",
        "primitives2": "primitives",
        "extra_tests2": "extra_tests",
    }
    if context in remaps:
        return remaps[context]
    for prefix in ("examples2.", "primitives2.", "extra_tests2."):
        if context.startswith(prefix):
            base = remaps[prefix[:-1]]
            suffix = context[len(prefix) :]
            return f"{base}.{suffix}" if suffix else base
    return context


def _coerce_tests(metadata_tests) -> Set[str]:
    tests: Set[str] = set()
    if isinstance(metadata_tests, Iterable):
        for entry in metadata_tests:
            if isinstance(entry, dict):
                name = entry.get("testcase")
                if isinstance(name, str):
                    tests.add(name)
    return tests


def _iter_test_files(base: Path) -> Iterator[Path]:
    if not base.exists():
        return iter(())

    def _iter() -> Iterator[Path]:
        for path in base.rglob("test_*.py"):
            if "__pycache__" in path.parts or path.name == "__init__.py":
                continue
            yield path

    return _iter()


def _add_entry(
    store: Dict[str, Dict[str, Dict[str, Set[str]]]],
    *,
    context: str,
    component: str,
    tests: Iterable[str],
) -> None:
    canon = canonical_context(context)
    components = store.setdefault(canon, {})
    entry = components.setdefault(component, {"contexts": set(), "tests": set()})
    entry["contexts"].add(context)
    entry["tests"].update(tests)


def _add_test_dir(
    store: Dict[str, Dict[str, Dict[str, Set[str]]]],
    *,
    base_dir: Path,
    context_prefix: str,
) -> None:
    for path in _iter_test_files(base_dir):
        rel = path.relative_to(base_dir)
        sub_parts = [part for part in rel.parts[:-1] if part != "__pycache__"]
        context = (
            ".".join([context_prefix] + sub_parts) if sub_parts else context_prefix
        )
        component = rel.stem
        _add_entry(store, context=context, component=component, tests={component})


def gather() -> Dict[str, Dict[str, Dict[str, Set[str]]]]:
    import_plugins2()

    store: Dict[str, Dict[str, Dict[str, Set[str]]]] = {}

    for plugin in PLUGIN_REGISTRY2.values():
        metadata = getattr(plugin, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        context = metadata.get("context")
        component = metadata.get("component")
        if isinstance(context, str) and isinstance(component, str) and component:
            tests = _coerce_tests(metadata.get("testcases"))
            _add_entry(store, context=context, component=component, tests=tests)

    for metadata in EXAMPLE_REGISTRY2.values():
        context = metadata.get("context")
        component = metadata.get("component")
        if isinstance(context, str) and isinstance(component, str) and component:
            tests = _coerce_tests(metadata.get("testcases"))
            _add_entry(store, context=context, component=component, tests=tests)

    tests_root = ROOT / "tests"
    _add_test_dir(
        store, base_dir=tests_root / "extra_tests2", context_prefix="extra_tests2"
    )
    _add_test_dir(store, base_dir=tests_root / "examples2", context_prefix="examples2")
    _add_test_dir(
        store, base_dir=tests_root / "primitives2", context_prefix="primitives2"
    )

    return store


def format_table(store: Dict[str, Dict[str, Dict[str, Set[str]]]]) -> str:
    lines: list[str] = []
    lines.append("# Migration Status")
    lines.append("")
    lines.append(
        "This file is auto-generated. Run `poetry run python scripts/generate_migration_status.py` "
        "to refresh after adding or updating converter2/plugins2 coverage."
    )
    lines.append("")

    for canon_context in sorted(store):
        components = store[canon_context]
        lines.append(f"## {canon_context}")
        lines.append("")
        for component in sorted(components):
            entry = components[component]
            ctx_str = ", ".join(sorted(entry["contexts"]))
            tests = entry["tests"]
            tests_str = ", ".join(sorted(tests)) if tests else "-"
            lines.append(f"- **{component}**")
            lines.append(f"  - contexts: {ctx_str}")
            lines.append(f"  - tests: {tests_str}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    store = gather()
    output = format_table(store)
    out_path = ROOT / "MigrationStatus.md"
    out_path.write_text(output)


if __name__ == "__main__":
    main()
