import jax
from typing import TYPE_CHECKING
from onnx import helper
import jax.numpy as jnp
from jax2onnx.plugin_system import register_primitive, PrimitivePlugin

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.gather_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.gather.html",
    onnx=[
        {
            "component": "Gather",
            "doc": "https://onnx.ai/onnx/operators/onnx__Gather.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    testcases=[
        {
            "testcase": "gather",
            "callable": lambda x: jax.lax.gather(
                x,
                jnp.array([[1], [0]]),
                jax.lax.GatherDimensionNumbers(
                    offset_dims=(1,),
                    collapsed_slice_dims=(0,),
                    start_index_map=(0,),
                ),
                (1, 3),
            ),
            "input_shapes": [(3, 3)],
        }
    ],
)
class GatherPlugin(PrimitivePlugin):
    """Plugin for converting jax.lax.gather to ONNX Gather."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX gather primitive."""
        input_names = [s.get_name(inp) for inp in node_inputs]
        output_name = s.get_var_name(node_outputs[0])

        # Extract parameters from the JAX gather node
        dimension_numbers = params["dimension_numbers"]
        params["slice_sizes"]

        # Determine the axis to gather along based on dimension_numbers
        # This is a simplification; a more robust solution might be required
        # for complex gather operations.
        if len(dimension_numbers.start_index_map) == 1:
            axis = dimension_numbers.start_index_map[0]
        else:
            # Handle cases with multiple start_index_map values (more complex gather)
            # For simplicity, we assume the first start_index_map if multiple exist.
            axis = dimension_numbers.start_index_map[0]

        node = helper.make_node(
            "Gather",  # Use ONNX Gather instead of GatherElements
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("gather"),
            axis=axis,  # Add the axis parameter.
        )
        s.add_node(node)
