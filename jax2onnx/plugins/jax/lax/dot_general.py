# file: jax2onnx/plugins/jax/lax/dot_general.py

from typing import TYPE_CHECKING

import numpy as np
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
        },
        {
            "testcase": "dot_general_lhs1_rhs1",
            "callable": lambda x1, x2: jax.lax.dot_general(
                x1, x2, (((1,), (1,)), ((), ()))
            ),
            "input_shapes": [(3, 3), (3, 3)],
        },
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

        if lhs_batch or rhs_batch:
            raise NotImplementedError(
                f"dot_general with batching not supported: contract={params['dimension_numbers']}"
            )

        # Standard matrix multiplication → use Gemm so ORT won't fuse into float-only FusedMatMul
        if lhs_contract == (1,) and rhs_contract == (0,):
            zero_const = s.get_constant_name(np.array(0, dtype=out_var.aval.dtype))
            gemm_node = helper.make_node(
                "Gemm",
                inputs=[lhs_name, rhs_name, zero_const],
                outputs=[out_name],
                name=s.get_unique_name("dot_general_gemm"),
                alpha=1.0,
                beta=0.0,
            )
            s.add_node(gemm_node)
            s.add_shape_info(out_name, out_shape, out_var.aval.dtype)

        # Contraction of the last dimension of both inputs → transpose then Gemm
        elif lhs_contract == (1,) and rhs_contract == (1,):
            transposed_rhs_name = s.get_unique_name("transposed_rhs")
            s.add_node(
                helper.make_node(
                    "Transpose",
                    inputs=[rhs_name],
                    outputs=[transposed_rhs_name],
                    perm=[1, 0],
                    name=s.get_unique_name("transpose_rhs"),
                )
            )
            transposed_shape = (rhs_var.aval.shape[1], rhs_var.aval.shape[0])
            s.add_shape_info(transposed_rhs_name, transposed_shape, rhs_var.aval.dtype)

            zero_const = s.get_constant_name(np.array(0, dtype=out_var.aval.dtype))
            gemm_node = helper.make_node(
                "Gemm",
                inputs=[lhs_name, transposed_rhs_name, zero_const],
                outputs=[out_name],
                name=s.get_unique_name("dot_general_gemm"),
                alpha=1.0,
                beta=0.0,
            )
            s.add_node(gemm_node)
            s.add_shape_info(out_name, out_shape, out_var.aval.dtype)
        else:
            raise NotImplementedError(
                f"dot_general config not supported: contract={params['dimension_numbers']}"
            )
