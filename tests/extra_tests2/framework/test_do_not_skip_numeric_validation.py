from __future__ import annotations

from typing import Dict, Iterable, Tuple

from jax2onnx.plugins.plugin_system import PLUGIN_REGISTRY, import_all_plugins
from jax2onnx.plugins2.plugin_system import (
    EXAMPLE_REGISTRY2,
    PLUGIN_REGISTRY2,
    import_all_plugins as import_all_plugins2,
)

_ComponentTestcase = Tuple[str, str]
_Metadata = Dict[str, object]


def _iter_metadata_from_registry(registry: Dict[str, object]) -> Iterable[_Metadata]:
    for plugin in registry.values():
        metadata = getattr(plugin, "metadata", None)
        if isinstance(metadata, dict):
            yield metadata


def _is_plugins2_context(context: object) -> bool:
    if not isinstance(context, str):
        return False
    head = context.split(".", 1)[0]
    return head.endswith("2")


def _collect_skip_flags(metas: Iterable[_Metadata]) -> Dict[_ComponentTestcase, Dict[str, object]]:
    result: Dict[_ComponentTestcase, Dict[str, object]] = {}
    for meta in metas:
        component = meta.get("component")
        testcases = meta.get("testcases")
        context = meta.get("context") if isinstance(meta.get("context"), str) else ""
        if not isinstance(component, str) or not component:
            continue
        if not isinstance(testcases, list):
            continue
        for case in testcases:
            if not isinstance(case, dict):
                continue
            testcase = case.get("testcase")
            if not isinstance(testcase, str) or not testcase:
                continue
            skip_flag = bool(case.get("skip_numeric_validation"))
            key: _ComponentTestcase = (component, testcase)
            stored = result.get(key)
            if stored is None:
                result[key] = {"skip": skip_flag, "context": context}
            elif stored["skip"] != skip_flag:
                raise AssertionError(
                    f"Conflicting skip_numeric_validation flags for {component}::{testcase}."
                )
    return result


def test_plugins2_skip_numeric_validation_matches_legacy():
    import_all_plugins()
    legacy_metas = [
        meta
        for meta in _iter_metadata_from_registry(PLUGIN_REGISTRY)
        if not _is_plugins2_context(meta.get("context"))
    ]
    legacy_flags = _collect_skip_flags(legacy_metas)

    import_all_plugins2()
    plugins2_metas = list(_iter_metadata_from_registry(PLUGIN_REGISTRY2))
    plugins2_metas.extend(EXAMPLE_REGISTRY2.values())
    plugins2_flags = _collect_skip_flags(plugins2_metas)

    violations = []
    for key, plugins2_entry in plugins2_flags.items():
        if not plugins2_entry["skip"]:
            continue
        legacy_entry = legacy_flags.get(key)
        if legacy_entry is None:
            violations.append(
                f"{plugins2_entry['context']}::{key[0]}::{key[1]} sets skip_numeric_validation in plugins2 "
                "but no matching testcase exists in legacy plugins."
            )
            continue
        if not legacy_entry["skip"]:
            violations.append(
                f"{plugins2_entry['context']}::{key[0]}::{key[1]} sets skip_numeric_validation in plugins2 "
                f"but the legacy testcase ({legacy_entry['context']}) does not."
            )

    if violations:
        formatted = "\n".join(f"- {msg}" for msg in violations)
        raise AssertionError(
            "plugins2 testcases may only set skip_numeric_validation when the legacy counterpart does.\n"
            f"Either drop the flag in plugins2 or mirror it into the legacy testcase.\n{formatted}"
        )
