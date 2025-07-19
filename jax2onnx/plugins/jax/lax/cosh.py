from typing import TYPE_CHECKING

import jax
import numpy as np
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.cosh_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.cosh.html",
    onnx=[
        {
            "component": "Cosh",
            "doc": "https://onnx.ai/onnx/operators/onnx__Cosh.html",
        }
    ],
    since="v0.4.4",
    context="primitives.lax",
    component="cosh",
    testcases=[
        {
            "testcase": "cosh",
            "callable": lambda x: jax.lax.cosh(x),
            "input_shapes": [(3,)],
        }
    ],
)
class CoshPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.cosh to ONNX Cosh."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX cosh primitive."""
        (x,) = node_inputs
        (out_var,) = node_outputs

        x_name = s.get_name(x)
        out_name = s.get_name(out_var)

        # Correctly get the dtype from the input variable's abstract value (aval)
        dtype = x.aval.dtype

        if dtype == np.float32:
            # For float32, we can just emit the native ONNX Cosh operator
            node = helper.make_node("Cosh", inputs=[x_name], outputs=[out_name])
            s.add_node(node)
        else:
            # For double (and other fp types), lower to (exp(x) + exp(-x)) / 2

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

            # exp(x) + exp(-x)
            sum_name = s.get_unique_name("sum")
            s.add_node(
                helper.make_node(
                    "Add", inputs=[exp_x_name, exp_neg_x_name], outputs=[sum_name]
                )
            )
            s.add_shape_info(sum_name, x.aval.shape, dtype)

            # constant 0.5
            half_const_name = s.get_constant_name(np.array(0.5, dtype=dtype))

            # (exp(x) + exp(-x)) * 0.5
            node = helper.make_node(
                "Mul", inputs=[sum_name, half_const_name], outputs=[out_name]
            )
            s.add_node(node)
