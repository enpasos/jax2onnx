# file: jax2onnx/plugins/flax/nnx/dot_product_attention.py

from typing import TYPE_CHECKING, Callable

import numpy as np
from flax import nnx
from jax import core, numpy as jnp
from jax.extend.core import Primitive
from onnx import TensorProto, helper
from jax.interpreters import batching

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


# Global variable to store the original function
_ORIGINAL_DOT_PRODUCT_ATTENTION_CALL: Callable | None = None


# Callable definitions for test cases
def dpa_with_mask(q, k, v, mask):
    return nnx.dot_product_attention(q, k, v, mask=mask)


def dpa_with_bias(q, k, v, bias):
    return nnx.dot_product_attention(q, k, v, bias=bias)


def dpa_with_mask_and_bias(q, k, v, mask, bias):
    return nnx.dot_product_attention(q, k, v, mask=mask, bias=bias)


nnx.dot_product_attention_p = Primitive("nnx.dot_product_attention")
nnx.dot_product_attention_p.multiple_results = False


@register_primitive(
    jaxpr_primitive=nnx.dot_product_attention_p.name,
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/attention.html#flax.nnx.dot_product_attention",
    onnx=[
        {
            "component": "Shape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Shape.html",
        },
        {
            "component": "Gather",
            "doc": "https://onnx.ai/onnx/operators/onnx__Gather.html",
        },
        {"component": "Cast", "doc": "https://onnx.ai/onnx/operators/onnx__Cast.html"},
        {"component": "Sqrt", "doc": "https://onnx.ai/onnx/operators/onnx__Sqrt.html"},
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
        {
            "component": "Einsum",
            "doc": "https://onnx.ai/onnx/operators/onnx__Einsum.html",
        },
        {
            "component": "Softmax",
            "doc": "https://onnx.ai/onnx/operators/onnx__Softmax.html",
        },
    ],
    since="v0.1.0",
    context="primitives.nnx",
    component="dot_product_attention",
    testcases=[
        {
            "testcase": "dpa_basic",
            "callable": lambda q, k, v: nnx.dot_product_attention(q, k, v),
            "input_shapes": [(2, 8, 4, 16), (2, 8, 4, 16), (2, 8, 4, 16)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
        },
        {
            "testcase": "dpa_with_tensor_mask",
            "callable": dpa_with_mask,
            "input_shapes": [(2, 8, 4, 16), (2, 8, 4, 16), (2, 8, 4, 16), (2, 4, 8, 8)],
            "input_dtypes": [np.float32, np.float32, np.float32, np.bool_],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "dpa_with_bias",
            "callable": dpa_with_bias,
            "input_shapes": [(2, 8, 4, 16), (2, 8, 4, 16), (2, 8, 4, 16), (2, 4, 8, 8)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "dpa_with_causal_mask",
            "callable": dpa_with_mask,
            "input_values": [
                np.random.randn(1, 8, 4, 16).astype(np.float32),
                np.random.randn(1, 8, 4, 16).astype(np.float32),
                np.random.randn(1, 8, 4, 16).astype(np.float32),
                np.tril(np.ones((1, 4, 8, 8), dtype=bool)),
            ],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
        },
        {
            "testcase": "dpa_with_mask_and_bias",
            "callable": dpa_with_mask_and_bias,
            "input_shapes": [
                (2, 8, 4, 16),
                (2, 8, 4, 16),
                (2, 8, 4, 16),
                (2, 4, 8, 8),
                (2, 4, 8, 8),
            ],
            "input_dtypes": [np.float32, np.float32, np.float32, np.bool_, np.float32],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
        },
    ],
)
class DotProductAttentionPlugin(PrimitiveLeafPlugin):

    @staticmethod
    def abstract_eval(q, k, v, *args, **kwargs):
        # The output shape is always the same as the query's shape.
        return core.ShapedArray(q.shape, q.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        q, k, v, *optional_inputs = node_inputs
        out_var = node_outputs[0]
        q_name, k_name, v_name = map(s.get_name, (q, k, v))
        out_name = s.get_name(out_var)
        B, T, N, H = q.aval.shape
        _, S, _, _ = k.aval.shape
        np_dtype = q.aval.dtype

        q_t = s.get_unique_name("q_T")
        s.add_node(helper.make_node("Transpose", [q_name], [q_t], perm=[0, 2, 1, 3]))
        s.add_shape_info(q_t, (B, N, T, H), np_dtype)

        k_t = s.get_unique_name("k_T")
        s.add_node(helper.make_node("Transpose", [k_name], [k_t], perm=[0, 2, 3, 1]))
        s.add_shape_info(k_t, (B, N, H, S), np_dtype)

        logits = s.get_unique_name("attn_scores")
        s.add_node(helper.make_node("MatMul", [q_t, k_t], [logits]))
        s.add_shape_info(logits, (B, N, T, S), np_dtype)

        scale_const = s.get_constant_name(np.array(1.0 / np.sqrt(H), dtype=np_dtype))
        scaled_scores = s.get_unique_name("scaled_scores")
        s.add_node(helper.make_node("Mul", [logits, scale_const], [scaled_scores]))
        s.add_shape_info(scaled_scores, (B, N, T, S), np_dtype)
        final_logits = scaled_scores

        has_mask = params.get("has_mask", False)
        has_bias = params.get("has_bias", False)

        opt_input_idx = 0
        mask_var = None
        bias_var = None

        if has_mask:
            mask_var = optional_inputs[opt_input_idx]
            opt_input_idx += 1
        if has_bias:
            bias_var = optional_inputs[opt_input_idx]

        if bias_var is not None:
            bias_name = s.get_name(bias_var)
            biased_logits = s.get_unique_name("biased_logits")
            s.add_node(
                helper.make_node("Add", [final_logits, bias_name], [biased_logits])
            )
            s.add_shape_info(biased_logits, (B, N, T, S), np_dtype)
            final_logits = biased_logits

        if mask_var is not None:
            mask_name = s.get_name(mask_var)
            mask_cond_name = mask_name

            if mask_var.aval.dtype != jnp.bool_:
                mask_bool_name = s.get_unique_name("mask_bool")
                s.add_node(
                    helper.make_node(
                        "Cast", [mask_name], [mask_bool_name], to=TensorProto.BOOL
                    )
                )
                s.add_shape_info(mask_bool_name, mask_var.aval.shape, dtype=bool)
                mask_cond_name = mask_bool_name

            large_negative_number_const = s.get_constant_name(
                np.array(-1e9, dtype=np_dtype)
            )
            masked_logits = s.get_unique_name("masked_logits")
            s.add_node(
                helper.make_node(
                    "Where",
                    [mask_cond_name, final_logits, large_negative_number_const],
                    [masked_logits],
                )
            )
            s.add_shape_info(masked_logits, (B, N, T, S), np_dtype)
            final_logits = masked_logits

        weights = s.get_unique_name("attn_weights")
        s.add_node(helper.make_node("Softmax", [final_logits], [weights], axis=-1))
        s.add_shape_info(weights, (B, N, T, S), np_dtype)

        v_t = s.get_unique_name("v_T")
        s.add_node(helper.make_node("Transpose", [v_name], [v_t], perm=[0, 2, 1, 3]))
        s.add_shape_info(v_t, (B, N, S, H), np_dtype)

        out_t = s.get_unique_name("out_T")
        s.add_node(helper.make_node("MatMul", [weights, v_t], [out_t]))
        s.add_shape_info(out_t, (B, N, T, H), np_dtype)

        s.add_node(
            helper.make_node("Transpose", [out_t], [out_name], perm=[0, 2, 1, 3])
        )
        s.add_shape_info(out_name, q.aval.shape, np_dtype)

    @staticmethod
    def get_monkey_patch():
        def patched(q, k, v, mask=None, bias=None, **kwargs):
            has_mask = mask is not None
            has_bias = bias is not None
            inputs = [q, k, v]
            if has_mask:
                inputs.append(mask)
            if has_bias:
                inputs.append(bias)
            # Pass kwargs through to the primitive binding
            return nnx.dot_product_attention_p.bind(
                *inputs, has_mask=has_mask, has_bias=has_bias, **kwargs
            )

        return patched

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [nnx],
            "patch_function": lambda _: DotProductAttentionPlugin.get_monkey_patch(),
            "target_attribute": "dot_product_attention",
        }


nnx.dot_product_attention_p.def_abstract_eval(DotProductAttentionPlugin.abstract_eval)


def dpa_batch(xs, dims, **params):
    bdim = next((d for d in dims if d is not None), None)
    if bdim is not None:
        xs = [jnp.moveaxis(x, d, 0) if d is not None else x for x, d in zip(xs, dims)]
    return nnx.dot_product_attention_p.bind(*xs, **params), 0


batching.primitive_batchers[nnx.dot_product_attention_p] = dpa_batch
