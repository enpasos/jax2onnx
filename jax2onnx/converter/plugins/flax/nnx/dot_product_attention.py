# file: jax2onnx/converter/plugins/flax/nnx/dot_product_attention.py

import numpy as np
from jax import core
from jax.extend.core import Primitive
from onnx import helper
import contextlib
from flax import nnx
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

nnx.dot_product_attention_p = Primitive("nnx.dot_product_attention")


def get_primitive():
    return nnx.dot_product_attention_p


def _shape_dot_product_attention(q_shape, k_shape, v_shape):
    # Typically attention outputs shape equal to q_shape
    return q_shape


def _get_monkey_patch():
    def dot_product_attention(q, k, v, axis=-1):
        def dot_product_attention_abstract_eval(q, k, v, axis):
            output_shape = _shape_dot_product_attention(q.shape, k.shape, v.shape)
            return core.ShapedArray(output_shape, q.dtype)

        nnx.dot_product_attention_p.multiple_results = False
        nnx.dot_product_attention_p.def_abstract_eval(
            dot_product_attention_abstract_eval
        )
        return nnx.dot_product_attention_p.bind(q, k, v, axis=axis)

    return dot_product_attention


@contextlib.contextmanager
def temporary_patch():
    original_fn = nnx.dot_product_attention
    nnx.dot_product_attention = _get_monkey_patch()
    try:
        yield
    finally:
        nnx.dot_product_attention = original_fn


def get_handler(s: "Jaxpr2OnnxConverter"):
    def handle_dot_product_attention(node_inputs, node_outputs, params):
        q_var, k_var, v_var = node_inputs[:3]
        output_var = node_outputs[0]

        q_name = s.get_name(q_var)
        k_name = s.get_name(k_var)
        v_name = s.get_name(v_var)
        output_name = s.get_name(output_var)

        q_shape = q_var.aval.shape
        k_shape = k_var.aval.shape
        v_shape = v_var.aval.shape
        B, N, H, E = q_shape
        _, M, _, _ = k_shape

        scale_value = 1.0 / np.sqrt(E)
        scale_const_name = s.builder.get_constant_name(
            np.array(scale_value, dtype=np.float32)
        )

        # Attention scores (Einsum)
        attn_scores_name = s.get_unique_name("attn_scores")
        attn_scores_shape = (B, N, H, M)
        s.add_node(
            helper.make_node(
                "Einsum",
                inputs=[q_name, k_name],
                outputs=[attn_scores_name],
                equation="BNHE,BMHE->BNHM",
                name=s.get_unique_name("einsum_qk"),
            )
        )
        s.add_shape_info(attn_scores_name, attn_scores_shape)

        # Scaled attention scores (Mul)
        scaled_scores_name = s.get_unique_name("scaled_scores")
        s.add_node(
            helper.make_node(
                "Mul",
                inputs=[attn_scores_name, scale_const_name],
                outputs=[scaled_scores_name],
                name=s.get_unique_name("scale"),
            )
        )
        s.add_shape_info(scaled_scores_name, attn_scores_shape)

        # Attention weights (Softmax)
        attn_weights_name = s.get_unique_name("attn_weights")
        s.add_node(
            helper.make_node(
                "Softmax",
                inputs=[scaled_scores_name],
                outputs=[attn_weights_name],
                axis=params.get("softmax_axis", -1),
                name=s.get_unique_name("softmax"),
            )
        )
        s.add_shape_info(attn_weights_name, attn_scores_shape)

        # Attention output (Einsum)
        attn_output_shape = (B, N, H, v_shape[-1])
        s.add_node(
            helper.make_node(
                "Einsum",
                inputs=[attn_weights_name, v_name],
                outputs=[output_name],
                equation="BNHM,BMHE->BNHE",
                name=s.get_unique_name("einsum_weights_v"),
            )
        )
        s.add_shape_info(output_name, attn_output_shape)

    return handle_dot_product_attention


def get_metadata() -> dict:
    return {
        "jaxpr_primitive": "nnx.dot_product_attention",
        "jax_doc": "https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/attention.html#flax.nnx.dot_product_attention",
        "onnx": [
            {
                "component": "Constant",
                "doc": "https://onnx.ai/onnx/operators/onnx__Constant.html",
            },
            {
                "component": "Einsum",
                "doc": "https://onnx.ai/onnx/operators/onnx__Einsum.html",
            },
            {
                "component": "Mul",
                "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html",
            },
            {
                "component": "Softmax",
                "doc": "https://onnx.ai/onnx/operators/onnx__Softmax.html",
            },
        ],
        "since": "v0.1.0",
        "context": "plugins.nnx",
        "testcases": [
            {
                "testcase": "dot_product_attention",
                "callable": lambda q, k, v: nnx.dot_product_attention(q, k, v),
                "input_shapes": [(2, 4, 8, 32), (2, 4, 8, 32), (2, 4, 8, 32)],
            },
        ],
    }
