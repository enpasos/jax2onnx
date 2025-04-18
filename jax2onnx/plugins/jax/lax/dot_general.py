from typing import TYPE_CHECKING

import jax
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.dot_general_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.dot_general.html",
    onnx=[
        {
            "component": "MatMul",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="dot_general",
    testcases=[
        {
            "testcase": "dot_general",
            "callable": lambda x1, x2: jax.lax.dot_general(
                x1, x2, (((1,), (0,)), ((), ()))
            ),
            "input_shapes": [(3, 3), (3, 3)],
        }
    ],
)
class DotGeneralPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.lax.dot_general to ONNX MatMul.
    """

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        lhs_var, rhs_var = node_inputs
        out_var = node_outputs[0]

        lhs_name = s.get_name(lhs_var)
        rhs_name = s.get_name(rhs_var)
        out_name = s.get_var_name(out_var)

        out_shape = out_var.aval.shape

        ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) = params[
            "dimension_numbers"
        ]

        # This plugin currently only handles:
        #   Contract: last dim of lhs with first dim of rhs
        #   No batching
        if lhs_contract != (1,) or rhs_contract != (0,) or lhs_batch or rhs_batch:
            raise NotImplementedError(
                f"dot_general config not supported: contract={params['dimension_numbers']}"
            )

        # MatMul directly supports (N, K) @ (K, M) => (N, M)
        # So if shapes are fine, no need to reshape
        matmul_node = helper.make_node(
            "MatMul",
            inputs=[lhs_name, rhs_name],
            outputs=[out_name],
            name=s.get_unique_name("dot_general_matmul"),
        )
        s.add_node(matmul_node)
        s.add_shape_info(out_name, out_shape)
