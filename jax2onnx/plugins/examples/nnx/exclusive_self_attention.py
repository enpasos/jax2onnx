# jax2onnx/plugins/examples/nnx/exclusive_self_attention.py

from __future__ import annotations

from typing import Final

import jax
import jax.numpy as jnp
from flax import nnx

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import register_example


def _exclusive_self_attention(q, k, v):
    y = nnx.dot_product_attention(q, k, v)
    eps = jnp.asarray(1e-12, dtype=v.dtype)
    v_norm = v * jax.lax.rsqrt(jnp.sum(jnp.square(v), axis=-1, keepdims=True) + eps)
    return y - jnp.sum(y * v_norm, axis=-1, keepdims=True) * v_norm


EXPECT_XSA_FALLBACK: Final = EG(
    [
        (
            "MatMul -> Mul -> Softmax -> MatMul",
            {
                "counts": {
                    "MatMul": 2,
                    "Softmax": 1,
                    "ReduceSumSquare": 1,
                    "ReduceSum": 1,
                    "Div": 1,
                    "Sub": 1,
                }
            },
        ),
        "ReduceSumSquare -> Add -> Sqrt -> Div",
        "Mul -> ReduceSum -> Mul -> Sub",
    ]
)


EXPECT_XSA_NATIVE: Final = EG(
    [
        (
            "Attention",
            {
                "counts": {
                    "Attention": 1,
                    "ReduceSumSquare": 1,
                    "ReduceSum": 1,
                    "Div": 1,
                    "Sub": 1,
                },
                "must_absent": ["MatMul", "Softmax"],
            },
        ),
        "ReduceSumSquare -> Add -> Sqrt -> Div",
        "Mul -> ReduceSum -> Mul -> Sub",
    ]
)


register_example(
    component="ExclusiveSelfAttention",
    description=(
        "An XSA-style attention block that removes the component of the attention "
        "output aligned with the token's own value vector."
    ),
    source="https://arxiv.org/abs/2603.09078",
    since="0.12.4",
    context="examples.nnx",
    children=["nnx.dot_product_attention"],
    testcases=[
        {
            "testcase": "exclusive_self_attention",
            "callable": _exclusive_self_attention,
            "input_shapes": [(2, 8, 4, 16), (2, 8, 4, 16), (2, 8, 4, 16)],
            "opset_version": 21,
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_XSA_FALLBACK,
        },
        {
            "testcase": "exclusive_self_attention_opset23",
            "callable": _exclusive_self_attention,
            "input_shapes": [(2, 8, 4, 16), (2, 8, 4, 16), (2, 8, 4, 16)],
            "opset_version": 23,
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_XSA_NATIVE,
        },
    ],
)
