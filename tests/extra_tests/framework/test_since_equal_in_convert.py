# tests/extra_tests/framework/test_since_equal_in_convert.py

from jax2onnx.plugins.plugin_system import (
    PLUGIN_REGISTRY as IR_REGISTRY,
    import_all_plugins as import_all_ir,
)


def test_plugins_since_metadata_present():
    """Every plugins primitive must carry a non-empty `since` marker."""

    import_all_ir()

    missing = []
    for primitive, plugin in IR_REGISTRY.items():
        metadata = getattr(plugin, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        context = metadata.get("context", "")
        since = metadata.get("since")
        if not isinstance(since, str) or not since.strip():
            missing.append(f"{primitive} (context={context!r})")

    assert not missing, (
        "plugins primitives must record a non-empty `since` metadata entry. "
        "Missing: " + ", ".join(missing)
    )
