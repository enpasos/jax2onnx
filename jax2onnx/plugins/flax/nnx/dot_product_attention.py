# file: jax2onnx/plugins/flax/nnx/dot_product_attention.py

import numpy as np
from jax import core
from jax.extend.core import Primitive
from onnx import helper, TensorProto
from flax import nnx
from typing import TYPE_CHECKING
from jax2onnx.plugin_system import register_primitive, PrimitiveLeafPlugin

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

# Define the JAX primitive for dot product attention.
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
            "callable": lambda q, k, v: nnx.nn.attention.dot_product_attention(q, k, v),
            "input_shapes": [(2, 4, 8, 32), (2, 4, 8, 32), (2, 4, 8, 32)],
        },
    ],
)
class DotProductAttentionPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting flax.nnx.dot_product_attention to ONNX.
    """

    @staticmethod
    def _shape_dot_product_attention(q_shape, k_shape, v_shape):
        return q_shape

    @staticmethod
    def abstract_eval(q, k, v, axis=-1):
        output_shape = DotProductAttentionPlugin._shape_dot_product_attention(
            q.shape, k.shape, v.shape
        )
        return core.ShapedArray(output_shape, q.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
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

        # Step 1: Compute raw attention scores with Einsum
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

        # Step 2: Compute sqrt(E) dynamically from q.shape[-1]
        # q_shape = Shape(q)
        q_shape_name = s.get_unique_name("q_shape")
        s.add_node(
            helper.make_node(
                "Shape",
                inputs=[q_name],
                outputs=[q_shape_name],
                name=s.get_unique_name("shape_q"),
            )
        )

        # Gather last dim
        e_index_name = s.builder.get_constant_name(np.array([-1], dtype=np.int64))
        e_dim_name = s.get_unique_name("q_last_dim")
        s.add_node(
            helper.make_node(
                "Gather",
                inputs=[q_shape_name, e_index_name],
                outputs=[e_dim_name],
                axis=0,
                name=s.get_unique_name("gather_E"),
            )
        )

        # Cast E to float
        e_dim_float_name = s.get_unique_name("e_dim_float")
        s.add_node(
            helper.make_node(
                "Cast",
                inputs=[e_dim_name],
                outputs=[e_dim_float_name],
                to=TensorProto.FLOAT,
                name=s.get_unique_name("cast_e"),
            )
        )

        # sqrt(E)
        sqrt_e_name = s.get_unique_name("sqrt_e")
        s.add_node(
            helper.make_node(
                "Sqrt",
                inputs=[e_dim_float_name],
                outputs=[sqrt_e_name],
                name=s.get_unique_name("sqrt_e"),
            )
        )

        # Step 3: Divide qk by sqrt(E)
        scaled_scores_name = s.get_unique_name("scaled_scores")
        s.add_node(
            helper.make_node(
                "Div",
                inputs=[attn_scores_name, sqrt_e_name],
                outputs=[scaled_scores_name],
                name=s.get_unique_name("scale"),
            )
        )
        s.add_shape_info(scaled_scores_name, attn_scores_shape)

        # Step 4: Softmax
        attn_weights_name = s.get_unique_name("attn_weights")
        s.add_node(
            helper.make_node(
                "Softmax",
                inputs=[scaled_scores_name],
                outputs=[attn_weights_name],
                axis=params.get("axis", -1),
                name=s.get_unique_name("softmax"),
            )
        )
        s.add_shape_info(attn_weights_name, attn_scores_shape)

        # Step 5: Weighted sum: einsum(attn_weights, v)
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

    @staticmethod
    def _dot_product_attention(q, k, v, axis=-1):
        return nnx.dot_product_attention_p.bind(q, k, v, axis=axis)

    @staticmethod
    def get_monkey_patch():
        def patched_dot_product_attention(q, k, v, axis=-1):
            return DotProductAttentionPlugin._dot_product_attention(q, k, v, axis)

        return patched_dot_product_attention

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [nnx],
            "patch_function": lambda _: DotProductAttentionPlugin.get_monkey_patch(),
            "target_attribute": "dot_product_attention",
        }


# Register abstract evaluation
nnx.dot_product_attention_p.def_abstract_eval(DotProductAttentionPlugin.abstract_eval)
