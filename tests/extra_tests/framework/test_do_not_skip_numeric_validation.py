# tests/extra_tests/framework/test_do_not_skip_numeric_validation.py

from __future__ import annotations

from typing import Dict, Iterable

from jax2onnx.plugins.plugin_system import (
    EXAMPLE_REGISTRY,
    PLUGIN_REGISTRY,
    import_all_plugins as import_all_plugins,
)

_Metadata = Dict[str, object]

# Skip flag usage is intentionally rare. Extend the allowlist only when a
# testcase is inherently stochastic (e.g., dropout) or when ORT cannot execute
# the graph yet (document the reason beside the metadata entry).
_ALLOWED_SKIP_CASES: set[tuple[str, str, str]] = {
    ("primitives.lax", "reduce_max", "reduce_max_axes_input"),
    (
        "primitives.lax",
        "bitcast_convert_type",
        "bitcast_scalar_f32_to_i32",
    ),
    (
        "primitives.lax",
        "bitcast_convert_type",
        "bitcast_tensor_i32_to_f32",
    ),
    ("primitives.eqx", "dropout", "eqx_dropout_training_mode"),
    ("examples.eqx", "MlpExample", "mlp_training_mode"),
    ("examples.eqx", "MlpExample", "mlp_batched_training_mode"),
    (
        "examples.nnx",
        "TransformerDecoderWithSequential",
        "tiny_decoder_with_sequential_and_full_dynamic_shapes",
    ),
    ("primitives.nn", "truncated_normal", "initializer"),
    (
        "primitives.nn",
        "truncated_normal",
        "random_truncated_normal_positional",
    ),
    (
        "primitives.nn",
        "truncated_normal",
        "flax_dense_like_init",
    ),
    (
        "primitives.random",
        "random_bits",
        "random_bits_uint32",
    ),
    (
        "primitives.random",
        "random_normal",
        "random_normal_f32_2x3",
    ),
    (
        "primitives.random",
        "random_categorical",
        "random_categorical_logits_batch",
    ),
    (
        "primitives.random",
        "random_categorical",
        "random_categorical_logits_batch_opset23",
    ),
    (
        "primitives.random",
        "random_categorical",
        "random_categorical_logits_rank3",
    ),
    (
        "primitives.random",
        "random_categorical",
        "random_categorical_logits_rank3_opset23",
    ),
    (
        "primitives.jax_image",
        "resize",
        "resize_linear_opset9_upsample",
    ),
    (
        "primitives.lax",
        "dynamic_update_slice",
        "dus_tensorscatter_axis1_opset24",
    ),
    (
        "examples.nnx_gpt_oss",
        "FlaxSDPA",
        "gpt_oss_sdpa_flax",
    ),
    # ORT on the CI/default setup cannot execute ONNX CumProd at current
    # default opsets yet; keep structural coverage while runtime support catches up.
    ("primitives.lax", "cumprod", "cumprod_i32_axis2"),
    ("primitives.lax", "cumprod", "cumprod_f32_axism1_reverse"),
    ("primitives.jnp", "cumprod", "jnp_cumprod_axis1"),
    ("primitives.jnp", "cumprod", "jnp_cumprod_axis_none_flatten"),
    ("primitives.jnp", "cumprod", "jnp_cumprod_dtype_cast"),
}
_ALLOWED_SKIP_CONTEXTS: set[str] = {
    "examples.maxtext",  # ORT cannot execute MaxText graphs yet.
}


def _iter_metadata_from_registry(registry: Dict[str, object]) -> Iterable[_Metadata]:
    for plugin in registry.values():
        metadata = getattr(plugin, "metadata", None)
        if isinstance(metadata, dict):
            yield metadata


def test_plugins_skip_numeric_validation_is_constrained():
    import_all_plugins()

    metas = list(_iter_metadata_from_registry(PLUGIN_REGISTRY))
    metas.extend(EXAMPLE_REGISTRY.values())

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
            if case.get("skip_numeric_validation"):
                testcase = case.get("testcase", "<unnamed>")
                key = (str(context), str(component), str(testcase))
                if str(context) in _ALLOWED_SKIP_CONTEXTS:
                    continue
                if key not in _ALLOWED_SKIP_CASES:
                    unexpected.append("::".join(key))

    if unexpected:
        formatted = "\n".join(f"- {entry}" for entry in sorted(unexpected))
        raise AssertionError(
            "plugins testcases should not skip numeric validation unless they appear in the allowlist.\n"
            "Either drop the flag or update _ALLOWED_SKIP_CASES with a justification.\n"
            f"Offending entries:\n{formatted}"
        )
