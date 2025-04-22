from typing import TYPE_CHECKING

import jax
import numpy as np
from onnx import helper

from jax2onnx.converter.dynamic_utils import encode_dims
from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.slice_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.slice.html",
    onnx=[
        {
            "component": "Slice",
            "doc": "https://onnx.ai/onnx/operators/onnx__Slice.html",
        }
    ],
    since="v0.1.0",
    context="primitives.lax",
    component="slice",
    testcases=[
        {
            "testcase": "slice_test1",
            "callable": lambda x: x[1:3],
            "input_shapes": [(5,)],
        },
        {
            "testcase": "slice_3d_none_strides",
            "callable": lambda a: a[0:2, 0:1, 0:256],
            "input_shapes": [(2, 50, 256)],
        },
    ],
)
class SlicePlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.slice to ONNX Slice."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX slice primitive."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])
        start_indices = params["start_indices"]
        limit_indices = params["limit_indices"]
        axes = list(range(len(start_indices)))
        starts_name = s.get_constant_name(encode_dims(start_indices))
        ends_name = s.get_constant_name(encode_dims(limit_indices))
        axes_name = s.get_constant_name(np.array(axes, dtype=np.int64))
        inputs_list = [input_name, starts_name, ends_name, axes_name]
        if "strides" in params and params["strides"]:
            strides = params["strides"]
            steps_name = s.get_constant_name(encode_dims(strides))
            inputs_list.append(steps_name)
        node = helper.make_node(
            "Slice",
            inputs=inputs_list,
            outputs=[output_name],
            name=s.get_unique_name("slice"),
        )
        s.add_node(node)
