from __future__ import annotations

from flax import nnx

from jax2onnx.plugins2.plugin_system import (
    construct_and_call,
    register_example,
    with_requested_dtype,
    with_rng_seed,
)


register_example(
    component="MultiHeadAttention",
    description=(
        "nnx.MultiHeadAttention exercised in several configurations, including "
        "custom attention_fn and symbolic batch variants."
    ),
    source="https://github.com/google/flax/blob/main/README.md",
    since="v0.2.0",
    context="examples2.nnx",
    children=["nnx.GeneralLinear", "nnx.dot_product_attention"],
    testcases=[
        {
            "testcase": "multihead_attention_nn",
            "callable": construct_and_call(
                nnx.MultiHeadAttention,
                num_heads=8,
                in_features=256,
                qkv_features=256,
                out_features=256,
                decode=False,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 4, 256)],
            "expected_output_shapes": [("B", 4, 256)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "multihead_attention_nnx",
            "callable": construct_and_call(
                nnx.MultiHeadAttention,
                num_heads=8,
                in_features=256,
                qkv_features=256,
                out_features=256,
                attention_fn=lambda *args, **kwargs: nnx.dot_product_attention(
                    *args, **kwargs
                ),
                decode=False,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 4, 256)],
            "expected_output_shapes": [("B", 4, 256)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "multihead_attention_2_nnx",
            "callable": construct_and_call(
                nnx.MultiHeadAttention,
                num_heads=4,
                in_features=16,
                qkv_features=16,
                out_features=16,
                decode=False,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 5, 16)],
            "expected_output_shapes": [("B", 5, 16)],
            "run_only_f32_variant": True,
        },
    ],
)
