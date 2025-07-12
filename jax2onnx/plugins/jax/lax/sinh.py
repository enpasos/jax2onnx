from typing import TYPE_CHECKING

import jax
import numpy as np
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.sinh_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.sinh.html",
    onnx=[
        {"component": "Sinh", "doc": "https://onnx.ai/onnx/operators/onnx__Sinh.html"}
    ],
    since="v0.4.4",
    context="primitives.lax",
    component="sinh",
    testcases=[
        {
            "testcase": "sinh",
            "callable": lambda x: jax.lax.sinh(x),
            "input_shapes": [(3,)],
        }
    ],
)
class SinhPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.sinh to ONNX Sinh."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        (x,) = node_inputs
        (out_var,) = node_outputs

        x_name = s.get_name(x)
        out_name = s.get_name(out_var)

        # Correctly get the dtype from the input variable's abstract value (aval)
        dtype = x.aval.dtype

        if dtype == np.float32:
            # For float32, we can just emit the native ONNX Sinh operator
            node = helper.make_node("Sinh", inputs=[x_name], outputs=[out_name])
            s.add_node(node)
        else:
            # For double (and other fp types), lower to (exp(x) - exp(-x)) / 2

            # exp(x)
            exp_x_name = s.get_unique_name("exp_x")
            s.add_node(helper.make_node("Exp", inputs=[x_name], outputs=[exp_x_name]))
            s.add_shape_info(exp_x_name, x.aval.shape, dtype)

            # -x
            neg_x_name = s.get_unique_name("neg_x")
            s.add_node(helper.make_node("Neg", inputs=[x_name], outputs=[neg_x_name]))
            s.add_shape_info(neg_x_name, x.aval.shape, dtype)

            # exp(-x)
            exp_neg_x_name = s.get_unique_name("exp_neg_x")
            s.add_node(
                helper.make_node("Exp", inputs=[neg_x_name], outputs=[exp_neg_x_name])
            )
            s.add_shape_info(exp_neg_x_name, x.aval.shape, dtype)

            # exp(x) - exp(-x)
            diff_name = s.get_unique_name("diff")
            s.add_node(
                helper.make_node(
                    "Sub", inputs=[exp_x_name, exp_neg_x_name], outputs=[diff_name]
                )
            )
            s.add_shape_info(diff_name, x.aval.shape, dtype)

            # constant 0.5
            half_const_name = s.get_constant_name(np.array(0.5, dtype=dtype))

            # (exp(x) - exp(-x)) * 0.5
            node = helper.make_node(
                "Mul", inputs=[diff_name, half_const_name], outputs=[out_name]
            )
            s.add_node(node)
