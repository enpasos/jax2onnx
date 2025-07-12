from typing import TYPE_CHECKING

import jax
import numpy as np
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.cos_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.cos.html",
    onnx=[
        {
            "component": "Cos",
            "doc": "https://onnx.ai/onnx/operators/onnx__Cos.html",
        }
    ],
    since="v0.4.4",
    context="primitives.lax",
    component="cos",
    testcases=[
        {
            "testcase": "cos",
            "callable": lambda x: jax.lax.cos(x),
            "input_shapes": [(3,)],
        }
    ],
)
class CosPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.cos to ONNX Cos."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX cos primitive."""
        (x,) = node_inputs
        (out_var,) = node_outputs

        x_name = s.get_name(x)
        out_name = s.get_name(out_var)
        dtype = x.aval.dtype

        if dtype == np.float64:
            # WORKAROUND: Cast f64 to f32 as the ONNX runtime lacks a f64 kernel.
            # This will result in a loss of precision.

            # Cast input to float32
            x_f32_name = s.get_unique_name("x_f32")
            s.add_node(
                helper.make_node(
                    "Cast",
                    inputs=[x_name],
                    outputs=[x_f32_name],
                    to=helper.TensorProto.FLOAT,
                )
            )
            s.add_shape_info(x_f32_name, x.aval.shape, np.float32)

            # Apply Cos on float32
            cos_f32_name = s.get_unique_name("cos_f32")
            s.add_node(
                helper.make_node("Cos", inputs=[x_f32_name], outputs=[cos_f32_name])
            )
            s.add_shape_info(cos_f32_name, x.aval.shape, np.float32)

            # Cast result back to float64
            s.add_node(
                helper.make_node(
                    "Cast",
                    inputs=[cos_f32_name],
                    outputs=[out_name],
                    to=helper.TensorProto.DOUBLE,
                )
            )
        else:
            # Standard implementation for float32
            node = helper.make_node("Cos", inputs=[x_name], outputs=[out_name])
            s.add_node(node)
