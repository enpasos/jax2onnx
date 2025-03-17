import jax
from typing import TYPE_CHECKING
from onnx import helper
import jax.numpy as jnp

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.gather_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_gather(node_inputs, node_outputs, params):
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

    return _handle_gather


def get_metadata() -> dict:
    """Return metadata describing this plugin and its test cases."""

    start_indices = jnp.array([[1], [0]])  # Gather rows 1 and 0
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(0,),
        start_index_map=(0,),
    )
    slice_sizes = (1, 3)

    return {
        "jaxpr_primitive": "gather",
        "jax_doc": "https://docs.jax.dev/en/latest/_autosummary/jax.lax.gather.html",
        "onnx": [
            {
                "component": "Gather",  # Updated to ONNX Gather
                "doc": "https://onnx.ai/onnx/operators/onnx__Gather.html",
            }
        ],
        "since": "v0.2.0",
        "context": "plugins.lax",
        "testcases": [
            {
                "testcase": "gather",
                "callable": lambda x: jax.lax.gather(
                    x, start_indices, dimension_numbers, slice_sizes
                ),
                "input_shapes": [(3, 3)],
            }
        ],
    }
