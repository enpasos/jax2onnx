# file: jax2onnx/examples/multi_head_attention.py

import jax
from flax import nnx

from jax2onnx.plugin_system import register_example, construct_and_call


register_example(
    component="MultiHeadAttention",
    description="This is a multi-head attention module implemented by Flax/nnx that has no ONNX correspondent on the same granularity.",
    source="https://github.com/google/flax/blob/main/README.md",
    since="v0.2.0",
    context="examples.nnx",
    children=["nnx.GeneralLinear", "nnx.dot_product_attention"],
    testcases=[
        {
            "testcase": "multihead_attention_nn",
            # Lazy construction + safe init to avoid jax.random.truncated_normal in JAX 0.7.1
            "callable": construct_and_call(
                nnx.MultiHeadAttention,
                num_heads=8,
                in_features=256,
                qkv_features=256,
                out_features=256,
                kernel_init=(lambda key, shape, dtype=jax.numpy.float32:
                             jax.random.normal(key, shape, dtype) * 0.02),
                dropout_rate=0.0,
                rngs=nnx.Rngs(0),
                decode=False,
            ),
            "input_shapes": [("B", 4, 256)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "multihead_attention_nnx",
            # Lazy construction + safe init to avoid jax.random.truncated_normal in JAX 0.7.1
            "callable": construct_and_call(
                nnx.MultiHeadAttention,
                num_heads=8,
                in_features=256,
                qkv_features=256,
                out_features=256,
                kernel_init=(lambda key, shape, dtype=jax.numpy.float32:
                             jax.random.normal(key, shape, dtype) * 0.02),
                dropout_rate=0.0,
                rngs=nnx.Rngs(0),
                attention_fn=lambda *args, **kwargs: nnx.dot_product_attention(
                    *args, **kwargs
                ),
                decode=False,
            ),
            "input_shapes": [("B", 4, 256)],
            "run_only_f32_variant": True,
        },
        # ---------------------------------------------------------------
        # Symbolic-batch attention: ensure √dₖ constant is scalar (rank-0)
        {
            "testcase": "multihead_attention_2_nnx",
            # Lazy construction + safe init to avoid jax.random.truncated_normal in JAX 0.7.1
            "callable": construct_and_call(
                nnx.MultiHeadAttention,
                num_heads=4,
                in_features=16,
                qkv_features=16,
                out_features=16,
                kernel_init=(lambda key, shape, dtype=jax.numpy.float32:
                             jax.random.normal(key, shape, dtype) * 0.02),
                dropout_rate=0.0,
                rngs=nnx.Rngs(0),
                decode=False,
            ),
            # q and kv identical shapes
            "input_shapes": [("B", 5, 16)],
            "run_only_f32_variant": True,
        },
    ],
)
