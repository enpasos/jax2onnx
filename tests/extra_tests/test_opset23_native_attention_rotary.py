# tests/extra_tests/test_opset23_native_attention_rotary.py

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax2onnx.user_interface import to_onnx


def _node_types(model) -> set[str]:
    return {node.op_type for node in model.graph.node}


def _dot_product_attention(q, k, v):
    return nnx.dot_product_attention(q, k, v)


def _dot_product_attention_with_mask_and_bias(q, k, v, mask, bias):
    return nnx.dot_product_attention(q, k, v, mask=mask, bias=bias)


def test_opset23_uses_attention_for_nnx_dot_product_attention():
    model = to_onnx(
        _dot_product_attention,
        [
            (2, 8, 4, 16),
            (2, 8, 4, 16),
            (2, 8, 4, 16),
        ],
        opset=23,
    )

    ops = _node_types(model)
    assert "Attention" in ops


def test_opset23_uses_attention_with_mask_and_bias():
    model = to_onnx(
        _dot_product_attention_with_mask_and_bias,
        [
            jax.ShapeDtypeStruct((2, 8, 4, 16), jnp.float32),
            jax.ShapeDtypeStruct((2, 8, 4, 16), jnp.float32),
            jax.ShapeDtypeStruct((2, 8, 4, 16), jnp.float32),
            jax.ShapeDtypeStruct((2, 4, 8, 8), jnp.bool_),
            jax.ShapeDtypeStruct((2, 4, 8, 8), jnp.float32),
        ],
        opset=23,
    )

    ops = _node_types(model)
    assert "Attention" in ops


def test_opset23_uses_rotary_embedding_for_eqx_rope():
    rope = eqx.nn.RotaryPositionalEmbedding(embedding_size=32, theta=10_000.0)

    def fn(x):
        return rope(x)

    model = to_onnx(
        fn,
        [jax.ShapeDtypeStruct((41, 32), np.float32)],
        opset=23,
    )

    ops = _node_types(model)
    assert "RotaryEmbedding" in ops
