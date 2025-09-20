import pytest

from jax2onnx.plugins.plugin_system import (
    PLUGIN_REGISTRY as LEGACY_REGISTRY,
    import_all_plugins as import_all_legacy,
)
from jax2onnx.plugins2.plugin_system import (
    PLUGIN_REGISTRY2 as IR_REGISTRY,
    import_all_plugins as import_all_ir,
)


@pytest.mark.parametrize("_", [0])
def test_since_attribute_matches_between_legacy_and_ir(_):
    """Ensure converted plugins keep the same `since` metadata."""

    import_all_legacy()
    import_all_ir()

    mismatches = []
    for primitive, legacy_plugin in LEGACY_REGISTRY.items():
        ir_plugin = IR_REGISTRY.get(primitive)
        if ir_plugin is None:
            continue  # not converted yet

        legacy_ctx = legacy_plugin.metadata.get("context")
        ir_ctx = ir_plugin.metadata.get("context")
        if not (legacy_ctx == "primitives.lax" and ir_ctx == "primitives2.lax"):
            continue

        legacy_since = legacy_plugin.metadata.get("since")
        ir_since = ir_plugin.metadata.get("since")
        if legacy_since != ir_since:
            mismatches.append(
                f"{primitive}: legacy since={legacy_since!r}, plugins2 since={ir_since!r}"
            )

    assert (
        not mismatches
    ), "Converted primitives must keep their `since` metadata.\n" + "\n".join(
        mismatches
    )
