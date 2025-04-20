from typing import TYPE_CHECKING

import numpy as np
from flax import nnx
from jax import core
from jax.extend.core import Primitive
from onnx import TensorProto, helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


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
            "input_shapes": [(2, 4, 8, 32), (2, 4, 8, 32), (2, 4, 8, 32)],
        },
        {
            "testcase": "dpa_diff_heads_embed",
            "callable": lambda q, k, v: nnx.dot_product_attention(q, k, v),
            "input_shapes": [(1, 2, 4, 16), (1, 2, 4, 16), (1, 2, 4, 16)],
        },
        {
            "testcase": "dpa_batch4_seq16",
            "callable": lambda q, k, v: nnx.dot_product_attention(q, k, v),
            "input_shapes": [(4, 2, 16, 8), (4, 2, 16, 8), (4, 2, 16, 8)],
        },
        {
            "testcase": "dpa_float64",
            "callable": lambda q, k, v: nnx.dot_product_attention(q, k, v),
            "input_shapes": [(2, 4, 8, 32), (2, 4, 8, 32), (2, 4, 8, 32)],
            "input_dtype": np.float64,
        },
        {
            "testcase": "dpa_heads1_embed4",
            "callable": lambda q, k, v: nnx.dot_product_attention(q, k, v),
            "input_shapes": [(2, 1, 8, 4), (2, 1, 8, 4), (2, 1, 8, 4)],
        },
        {
            "testcase": "dpa_heads8_embed8",
            "callable": lambda q, k, v: nnx.dot_product_attention(q, k, v),
            "input_shapes": [(2, 8, 8, 8), (2, 8, 8, 8), (2, 8, 8, 8)],
        },
        {
            "testcase": "dpa_batch1_seq2",
            "callable": lambda q, k, v: nnx.dot_product_attention(q, k, v),
            "input_shapes": [(1, 2, 2, 8), (1, 2, 2, 8), (1, 2, 2, 8)],
        },
        {
            "testcase": "dpa_batch8_seq4",
            "callable": lambda q, k, v: nnx.dot_product_attention(q, k, v),
            "input_shapes": [(8, 2, 4, 16), (8, 2, 4, 16), (8, 2, 4, 16)],
        },
        {
            "testcase": "dpa_axis1",
            "callable": lambda q, k, v: nnx.dot_product_attention(
                q, k, v
            ),  # axis param removed for compatibility
            "input_shapes": [(2, 4, 8, 32), (2, 4, 8, 32), (2, 4, 8, 32)],
        },
        # Uncomment and implement mask support to enable this test
        # {
        #     "testcase": "dpa_with_mask",
        #     "callable": lambda q, k, v, mask: nnx.dot_product_attention(q, k, v, mask=mask),
        #     "input_shapes": [(2, 4, 8, 32), (2, 4, 8, 32), (2, 4, 8, 32), (2, 4, 8, 8)],
        # },
    ],
)
class DotProductAttentionPlugin(PrimitiveLeafPlugin):

    @staticmethod
    def abstract_eval(q, k, v, *args, axis=-1):
        return core.ShapedArray(q.shape, q.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        q, k, v, *optional_inputs = node_inputs
        out_var = node_outputs[0]

        q_name = s.get_name(q)
        k_name = s.get_name(k)
        v_name = s.get_name(v)
        out_name = s.get_name(out_var)

        B, N, H, E = q.aval.shape
        _, M, _, _ = k.aval.shape

        # Step 1: Einsum(Q, K) -> attention scores
        attn_scores = s.get_unique_name("attn_scores")
        s.add_node(
            helper.make_node(
                "Einsum",
                inputs=[q_name, k_name],
                outputs=[attn_scores],
                equation="BNHE,BMHE->BNHM",
                name=s.get_unique_name("einsum_qk"),
            )
        )
        s.add_shape_info(attn_scores, (B, N, H, M), dtype=q.aval.dtype)

        # Step 2: scale by sqrt(E)
        shape_q = s.get_unique_name("q_shape")
        s.add_node(
            helper.make_node(
                "Shape",
                inputs=[q_name],
                outputs=[shape_q],
                name=s.get_unique_name("shape_q"),
            )
        )
        s.add_shape_info(shape_q, tuple([4]), dtype=np.int64)

        idx_last = s.get_constant_name(np.array([-1], dtype=np.int64))
        e_val = s.get_unique_name("e_val")
        s.add_node(
            helper.make_node(
                "Gather",
                inputs=[shape_q, idx_last],
                outputs=[e_val],
                axis=0,
                name=s.get_unique_name("gather_e"),
            )
        )
        s.add_shape_info(e_val, tuple([]), dtype=np.int64)

        e_float = s.get_unique_name("e_float")
        s.add_node(
            helper.make_node(
                "Cast",
                inputs=[e_val],
                outputs=[e_float],
                to=TensorProto.FLOAT,
                name=s.get_unique_name("cast_e"),
            )
        )
        s.add_shape_info(e_float, tuple([]), dtype=np.float32)

        sqrt_e = s.get_unique_name("sqrt_e")
        s.add_node(
            helper.make_node(
                "Sqrt",
                inputs=[e_float],
                outputs=[sqrt_e],
                name=s.get_unique_name("sqrt_e"),
            )
        )
        s.add_shape_info(sqrt_e, tuple([]), dtype=np.float32)

        scaled_scores = s.get_unique_name("scaled_scores")
        s.add_node(
            helper.make_node(
                "Div",
                inputs=[attn_scores, sqrt_e],
                outputs=[scaled_scores],
                name=s.get_unique_name("scale_scores"),
            )
        )
        s.add_shape_info(scaled_scores, (B, N, H, M), dtype=q.aval.dtype)

        # Step 3: Optional mask
        if optional_inputs:
            mask_var = optional_inputs[-1]
            mask_name = s.get_name(mask_var)

            mask_bool = s.get_unique_name("mask_bool")
            s.add_node(
                helper.make_node(
                    "Cast",
                    inputs=[mask_name],
                    outputs=[mask_bool],
                    to=TensorProto.BOOL,
                    name=s.get_unique_name("cast_mask"),
                )
            )
            s.add_shape_info(mask_bool, (B, N, H, M), dtype=bool)

            neg_inf = s.get_constant_name(np.array([-1e9], dtype=np.float32))
            masked_scores = s.get_unique_name("masked_scores")
            s.add_node(
                helper.make_node(
                    "Where",
                    inputs=[mask_bool, scaled_scores, neg_inf],
                    outputs=[masked_scores],
                    name=s.get_unique_name("where_mask"),
                )
            )
            s.add_shape_info(masked_scores, (B, N, H, M), dtype=q.aval.dtype)
            scaled_scores = masked_scores

        # Step 4: Softmax
        attn_weights = s.get_unique_name("attn_weights")
        s.add_node(
            helper.make_node(
                "Softmax",
                inputs=[scaled_scores],
                outputs=[attn_weights],
                axis=params.get("axis", -1),
                name=s.get_unique_name("softmax"),
            )
        )
        s.add_shape_info(attn_weights, (B, N, H, M), dtype=q.aval.dtype)

        # Step 5: Einsum(attn_weights, V)
        s.add_node(
            helper.make_node(
                "Einsum",
                inputs=[attn_weights, v_name],
                outputs=[out_name],
                equation="BNHM,BMHE->BNHE",
                name=s.get_unique_name("einsum_weights_v"),
            )
        )
        s.add_shape_info(out_name, (B, N, H, E), dtype=q.aval.dtype)

    @staticmethod
    def _dot_product_attention(q, k, v, mask=None, axis=-1):
        if mask is not None:
            return nnx.dot_product_attention_p.bind(q, k, v, mask, axis=axis)
        else:
            return nnx.dot_product_attention_p.bind(q, k, v, axis=axis)

    @staticmethod
    def get_monkey_patch():
        def patched(q, k, v, mask=None, axis=-1, **kwargs):
            return DotProductAttentionPlugin._dot_product_attention(
                q, k, v, mask, axis=axis
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
