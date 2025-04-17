from typing import TYPE_CHECKING

import jax
import numpy as np
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.reduce_sum_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_sum.html",
    onnx=[
        {
            "component": "ReduceSum",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceSum.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="reduce_sum",
    testcases=[
        {
            "testcase": "reduce_sum",
            "callable": lambda x: jax.lax.reduce_sum(x, axes=(0,)),
            "input_shapes": [(3, 3)],
        }
    ],
)
class ReduceSumPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.reduce_sum to ONNX ReduceSum."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX reduce_sum primitive."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])
        axes = params["axes"]
        axes_name = s.get_constant_name(np.array(axes, dtype=np.int64))
        node = helper.make_node(
            "ReduceSum",
            inputs=[input_name, axes_name],
            outputs=[output_name],
            name=s.get_unique_name("reduce_sum"),
            keepdims=0 if not params.get("keepdims", False) else 1,
        )
        s.add_node(node)
