from __future__ import annotations

from typing import Dict, Iterable

from jax2onnx.plugins.plugin_system import import_all_plugins
from jax2onnx.plugins2.plugin_system import (
    EXAMPLE_REGISTRY2,
    PLUGIN_REGISTRY2,
    import_all_plugins as import_all_plugins2,
)

_Metadata = Dict[str, object]


def _iter_metadata(registry: Dict[str, object]) -> Iterable[_Metadata]:
    for plugin in registry.values():
        metadata = getattr(plugin, "metadata", None)
        if isinstance(metadata, dict):
            yield metadata


def test_plugins2_testcases_require_use_onnx_ir():
    # Ensure both legacy and new-world registries are populated so metadata is complete.
    import_all_plugins()
    import_all_plugins2()

    metas = list(_iter_metadata(PLUGIN_REGISTRY2))
    metas.extend(EXAMPLE_REGISTRY2.values())

    violations: list[str] = []
    for meta in metas:
        component = meta.get("component", "<unknown>")
        context = meta.get("context", "<unknown>")
        testcases = meta.get("testcases")
        if not isinstance(testcases, list):
            continue
        for case in testcases:
            if not isinstance(case, dict):
                continue
            testcase = case.get("testcase", "<unknown>")
            if case.get("legacy_only", False):
                continue
            if not case.get("use_onnx_ir", False):
                violations.append(
                    f"{context}::{component}::{testcase} is missing use_onnx_ir=True"
                )

    if violations:
        formatted = "\n".join(f"- {v}" for v in violations)
        raise AssertionError(
            "All plugins2 testcases must set use_onnx_ir=True.\n"
            f"Fix the following entries:\n{formatted}"
        )
