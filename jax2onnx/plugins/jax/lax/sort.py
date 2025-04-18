from typing import TYPE_CHECKING

import jax
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.sort_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.sort.html",
    onnx=[
        {
            "component": "TopK",
            "doc": "https://onnx.ai/onnx/operators/onnx__TopK.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="sort",
    testcases=[
        {
            "testcase": "sort_1d",
            "callable": lambda x: jax.lax.sort(x),
            "input_shapes": [(3,)],
        },
        {
            "testcase": "sort_1d_empty",
            "callable": lambda x: jax.lax.sort(x),
            "input_shapes": [(0,)],  # Empty array
        },
        {
            "testcase": "sort_1d_single",
            "callable": lambda x: jax.lax.sort(x),
            "input_shapes": [(1,)],  # Single-element array
        },
        {
            "testcase": "sort_1d_larger",
            "callable": lambda x: jax.lax.sort(x),
            "input_shapes": [(10,)],
        },
        {
            "testcase": "sort_1d_specific_values",
            "callable": lambda x: jax.lax.sort(x),
            "input_shapes": [(5,)],
            "input_data": [([5, 2, 1, 4, 3],)],  # Specific input data
            "expected_output": [([1, 2, 3, 4, 5],)],  # Expected output
        },
    ],
)
class SortPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.sort to ONNX TopK."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX sort primitive."""
        input_name = s.get_name(node_inputs[0])
        shape_name = s.get_unique_name("sort_shape")
        value_name = s.get_var_name(node_outputs[0])
        indices_name = s.get_unique_name("sort_indices_output")
        if "axis" in params:
            # Not supported for now
            raise NotImplementedError("Sort with axis not supported yet")
        else:
            node_shape = helper.make_node(
                "Shape",
                inputs=[input_name],
                outputs=[shape_name],
                name=s.get_unique_name("shape"),
            )
            s.add_node(node_shape)
        s.add_shape_info(
            shape_name, shape=(len(node_inputs[0].aval.shape),), dtype="int64"
        )
        s.add_shape_info(indices_name, shape=node_inputs[0].aval.shape, dtype="int64")
        node_topk = helper.make_node(
            "TopK",
            inputs=[input_name, shape_name],
            outputs=[value_name, indices_name],
            name=s.get_unique_name("sort"),
            largest=0,
        )
        s.add_node(node_topk)
