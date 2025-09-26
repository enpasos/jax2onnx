#!/usr/bin/env python
"""Generate MigrationStatus.md summarizing legacy vs converter2 coverage."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, Set

from jax2onnx.plugins.plugin_system import PLUGIN_REGISTRY as LEGACY_REGISTRY
from jax2onnx.plugins.plugin_system import import_all_plugins as import_legacy_plugins
from jax2onnx.plugins2.plugin_system import (
    EXAMPLE_REGISTRY2,
    PLUGIN_REGISTRY2 as NEW_REGISTRY,
    import_all_plugins as import_new_plugins,
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


def add_entry(
    store: Dict[str, Dict[str, Dict[str, Dict[str, Set[str]]]]],
    context: str,
    component: str,
    bucket: str,
    tests: Iterable[str],
) -> None:
    canon = canonical_context(context)
    ctx_map = store.setdefault(
        canon,
        {
            "legacy": defaultdict(lambda: {"contexts": set(), "tests": set()}),
            "new": defaultdict(lambda: {"contexts": set(), "tests": set()}),
        },
    )
    entry = ctx_map[bucket][component]
    entry["contexts"].add(context)
    entry["tests"].update(tests)


def _iter_test_files(base: Path) -> Iterator[Path]:
    if not base.exists():
        return iter(())
    def _iter() -> Iterator[Path]:
        for path in base.rglob("test_*.py"):
            if "__pycache__" in path.parts or path.name == "__init__.py":
                continue
            yield path
    return _iter()


def _add_extra_tests(
    store: Dict[str, Dict[str, Dict[str, Dict[str, Set[str]]]]],
    *,
    base_dir: Path,
    context_prefix: str,
    bucket: str,
) -> None:
    for path in _iter_test_files(base_dir):
        rel = path.relative_to(base_dir)
        sub_parts = [part for part in rel.parts[:-1] if part != "__pycache__"]
        context = ".".join([context_prefix] + sub_parts) if sub_parts else context_prefix
        component = rel.stem
        add_entry(store, context, component, bucket, [component])


def gather() -> Dict[str, Dict[str, Dict[str, Dict[str, Set[str]]]]]:
    import_legacy_plugins()
    import_new_plugins()

    store: Dict[str, Dict[str, Dict[str, Dict[str, Set[str]]]]] = {}

    for plugin in LEGACY_REGISTRY.values():
        metadata = getattr(plugin, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        context = metadata.get("context")
        component = metadata.get("component")
        if isinstance(context, str) and isinstance(component, str) and component:
            tests = _coerce_tests(metadata.get("testcases"))
            add_entry(store, context, component, "legacy", tests)

    for metadata in EXAMPLE_REGISTRY2.values():
        context = metadata.get("context")
        component = metadata.get("component")
        if isinstance(context, str) and isinstance(component, str) and component:
            tests = _coerce_tests(metadata.get("testcases"))
            add_entry(store, context, component, "new", tests)

    for plugin in NEW_REGISTRY.values():
        metadata = getattr(plugin, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        context = metadata.get("context")
        component = metadata.get("component")
        if isinstance(context, str) and isinstance(component, str) and component:
            tests = _coerce_tests(metadata.get("testcases"))
            add_entry(store, context, component, "new", tests)

    tests_root = ROOT / "tests"
    _add_extra_tests(
        store,
        base_dir=tests_root / "extra_tests",
        context_prefix="extra_tests",
        bucket="legacy",
    )
    _add_extra_tests(
        store,
        base_dir=tests_root / "extra_tests2",
        context_prefix="extra_tests2",
        bucket="new",
    )

    return store


def format_table(store: Dict[str, Dict[str, Dict[str, Dict[str, Set[str]]]]]) -> str:
    lines: list[str] = []
    lines.append("# Migration Status")
    lines.append("")
    lines.append(
        "This file is auto-generated. Run `poetry run python scripts/generate_migration_status.py` "
        "to refresh after adding or converting plugins/examples."
    )
    lines.append("")

    for canon_context in sorted(store):
        data = store[canon_context]
        all_components: Set[str] = set(data["legacy"].keys()) | set(data["new"].keys())
        lines.append(f"## {canon_context}")
        lines.append("")
        for component in sorted(all_components):
            legacy_entry = data["legacy"].get(
                component, {"contexts": set(), "tests": set()}
            )
            new_entry = data["new"].get(component, {"contexts": set(), "tests": set()})

            legacy_tests = legacy_entry["tests"]
            new_tests = new_entry["tests"]

            if not legacy_tests:
                coverage = "-"
            elif legacy_tests.issubset(new_tests):
                coverage = "✅ complete"
            elif new_tests:
                missing = ", ".join(sorted(legacy_tests - new_tests))
                coverage = f"⚠️ partial (missing: {missing})"
            else:
                coverage = "❌ missing"

            lines.append(f"### {component}")
            lines.append(f"- Coverage: {coverage}")
            if legacy_entry["contexts"]:
                ctx_str = ", ".join(sorted(legacy_entry["contexts"]))
                lines.append(f"- Legacy contexts: {ctx_str}")
            if new_entry["contexts"]:
                ctx_str = ", ".join(sorted(new_entry["contexts"]))
                lines.append(f"- Converter2 contexts: {ctx_str}")
            if legacy_tests:
                tests_str = ", ".join(sorted(legacy_tests))
                lines.append(f"- Legacy testcases: {tests_str}")
            if new_tests:
                tests_str = ", ".join(sorted(new_tests))
                lines.append(f"- Converter2 testcases: {tests_str}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    store = gather()
    output = format_table(store)
    root = Path(__file__).resolve().parents[1]
    out_path = root / "MigrationStatus.md"
    out_path.write_text(output)


if __name__ == "__main__":
    main()
