import numpy as np
from jax import core
from jax.extend.core import Primitive
from flax import nnx
from typing import TYPE_CHECKING
from onnx import helper, TensorProto
from jax2onnx.plugin_system import register_primitive, PrimitiveLeafPlugin

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


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
    testcases=[
        {
            "testcase": "dot_product_attention",
            "callable": lambda q, k, v: nnx.dot_product_attention(q, k, v),
            "input_shapes": [(2, 4, 8, 32), (2, 4, 8, 32), (2, 4, 8, 32)],
        },
        # not fully implemented yet
        # {
        #     "testcase": "dot_product_attention_with_mask",
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
        s.add_shape_info(attn_scores, (B, N, H, M))

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

        idx_last = s.builder.get_constant_name(np.array([-1], dtype=np.int64))
        e_dim = s.get_unique_name("e_val")
        s.add_node(
            helper.make_node(
                "Gather",
                inputs=[shape_q, idx_last],
                outputs=[e_dim],
                axis=0,
                name=s.get_unique_name("gather_e"),
            )
        )

        e_float = s.get_unique_name("e_float")
        s.add_node(
            helper.make_node(
                "Cast",
                inputs=[e_dim],
                outputs=[e_float],
                to=TensorProto.FLOAT,
                name=s.get_unique_name("cast_e"),
            )
        )

        sqrt_e = s.get_unique_name("sqrt_e")
        s.add_node(
            helper.make_node(
                "Sqrt",
                inputs=[e_float],
                outputs=[sqrt_e],
                name=s.get_unique_name("sqrt_e"),
            )
        )

        scaled_scores = s.get_unique_name("scaled_scores")
        s.add_node(
            helper.make_node(
                "Div",
                inputs=[attn_scores, sqrt_e],
                outputs=[scaled_scores],
                name=s.get_unique_name("scale_scores"),
            )
        )
        s.add_shape_info(scaled_scores, (B, N, H, M))

        # Step 3: Optional mask (Where to filter)
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

            neg_inf = s.builder.get_constant_name(np.array([-1e9], dtype=np.float32))
            masked_scores = s.get_unique_name("masked_scores")
            s.add_node(
                helper.make_node(
                    "Where",
                    inputs=[mask_bool, scaled_scores, neg_inf],
                    outputs=[masked_scores],
                    name=s.get_unique_name("where_mask"),
                )
            )
            scaled_scores = masked_scores  # use masked scores from now on

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
        s.add_shape_info(attn_weights, (B, N, H, M))

        # Step 5: Final einsum(attn_weights, V)
        s.add_node(
            helper.make_node(
                "Einsum",
                inputs=[attn_weights, v_name],
                outputs=[out_name],
                equation="BNHM,BMHE->BNHE",
                name=s.get_unique_name("einsum_weights_v"),
            )
        )
        s.add_shape_info(out_name, (B, N, H, E))

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
