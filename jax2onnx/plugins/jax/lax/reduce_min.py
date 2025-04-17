from typing import TYPE_CHECKING

import jax
import numpy as np
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.reduce_min_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_min.html",
    onnx=[
        {
            "component": "ReduceMin",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMin.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="reduce_min",
    testcases=[
        {
            "testcase": "reduce_min",
            "callable": lambda x: jax.lax.reduce_min(x, axes=(0,)),
            "input_shapes": [(3, 3)],
        }
    ],
)
class ReduceMinPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.reduce_min to ONNX ReduceMin."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX reduce_min primitive."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])
        axes = params["axes"]
        axes_name = s.get_constant_name(np.array(axes, dtype=np.int64))
        node = helper.make_node(
            "ReduceMin",
            inputs=[input_name, axes_name],
            outputs=[output_name],
            name=s.get_unique_name("reduce_min"),
            keepdims=0 if not params.get("keepdims", False) else 1,
        )
        s.add_node(node)
