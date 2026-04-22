# tests/extra_tests/framework/test_since_equal_in_convert.py

import re

from jax2onnx.plugins.plugin_system import (
    PLUGIN_REGISTRY as IR_REGISTRY,
    import_all_plugins as import_all_ir,
)


def test_plugins_since_metadata_present():
    """Every plugin primitive must carry a concrete semver `since` marker."""

    import_all_ir()

    missing = []
    invalid = []
    for primitive, plugin in IR_REGISTRY.items():
        metadata = getattr(plugin, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        context = metadata.get("context", "")
        since = metadata.get("since")
        if not isinstance(since, str) or not since.strip():
            missing.append(f"{primitive} (context={context!r})")
            continue
        if re.fullmatch(r"\d+\.\d+\.\d+", since.strip()) is None:
            invalid.append(f"{primitive} (context={context!r}, since={since!r})")

    assert not missing, (
        "plugin primitives must record a non-empty `since` metadata entry. "
        "Missing: " + ", ".join(missing)
    )
    assert not invalid, (
        "plugin primitives must record a concrete semantic-version `since` "
        "metadata entry. Invalid: " + ", ".join(invalid)
    )
