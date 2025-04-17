from typing import TYPE_CHECKING

import jax
import numpy as np
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.reduce_max_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_max.html",
    onnx=[
        {
            "component": "ReduceMax",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMax.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="reduce_max",
    testcases=[
        {
            "testcase": "reduce_max",
            "callable": lambda x: jax.lax.reduce_max(x, axes=(0,)),
            "input_shapes": [(3, 3)],
        }
    ],
)
class ReduceMaxPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.reduce_max to ONNX ReduceMax."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX reduce_max primitive."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])
        axes = params["axes"]
        axes_name = s.get_constant_name(np.array(axes, dtype=np.int64))
        node = helper.make_node(
            "ReduceMax",
            inputs=[input_name, axes_name],
            outputs=[output_name],
            name=s.get_unique_name("reduce_max"),
            keepdims=0 if not params.get("keepdims", False) else 1,
        )
        s.add_node(node)
