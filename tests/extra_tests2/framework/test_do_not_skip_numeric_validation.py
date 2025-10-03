from __future__ import annotations

from typing import Dict, Iterable

from jax2onnx.plugins2.plugin_system import (
    EXAMPLE_REGISTRY2,
    PLUGIN_REGISTRY2,
    import_all_plugins as import_all_plugins2,
)

_Metadata = Dict[str, object]

# Skip flag usage is intentionally rare. Extend the allowlist only when a
# testcase is inherently stochastic (e.g., dropout) or when ORT cannot execute
# the graph yet (document the reason beside the metadata entry).
_ALLOWED_SKIP_CASES: set[tuple[str, str, str]] = {
    ("primitives2.lax", "reduce_max", "reduce_max_axes_input"),
    ("primitives2.eqx", "dropout", "eqx_dropout_training_mode"),
    ("examples2.eqx", "MlpExample", "mlp_training_mode"),
    ("examples2.eqx", "MlpExample", "mlp_batched_training_mode"),
    (
        "examples2.nnx",
        "TransformerDecoderWithSequential",
        "tiny_decoder_with_sequential_and_full_dynamic_shapes",
    ),
    ("primitives2.nn", "truncated_normal", "initializer"),
    (
        "primitives2.nn",
        "truncated_normal",
        "random_truncated_normal_positional",
    ),
    (
        "primitives2.nn",
        "truncated_normal",
        "flax_dense_like_init",
    ),
    (
        "primitives2.random",
        "random_bits",
        "random_bits_uint32",
    ),
}


def _iter_metadata_from_registry(registry: Dict[str, object]) -> Iterable[_Metadata]:
    for plugin in registry.values():
        metadata = getattr(plugin, "metadata", None)
        if isinstance(metadata, dict):
            yield metadata


def test_plugins2_skip_numeric_validation_is_constrained():
    import_all_plugins2()

    metas = list(_iter_metadata_from_registry(PLUGIN_REGISTRY2))
    metas.extend(EXAMPLE_REGISTRY2.values())

    unexpected: list[str] = []

    for meta in metas:
        context = meta.get("context", "<unknown>")
        component = meta.get("component", "<unknown>")
        testcases = meta.get("testcases")

        if not isinstance(testcases, list):
            continue

        for case in testcases:
            if not isinstance(case, dict):
                continue
            if case.get("legacy_only"):
                continue

            if case.get("skip_numeric_validation"):
                testcase = case.get("testcase", "<unnamed>")
                key = (str(context), str(component), str(testcase))
                if key not in _ALLOWED_SKIP_CASES:
                    unexpected.append("::".join(key))

    if unexpected:
        formatted = "\n".join(f"- {entry}" for entry in sorted(unexpected))
        raise AssertionError(
            "plugins2 testcases should not skip numeric validation unless they appear in the allowlist.\n"
            "Either drop the flag or update _ALLOWED_SKIP_CASES with a justification.\n"
            f"Offending entries:\n{formatted}"
        )
