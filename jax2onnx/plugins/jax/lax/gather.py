# file: jax2onnx/plugins/jax/lax/gather.py

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.gather_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.gather.html",
    onnx=[
        {
            "component": "GatherND",
            "doc": "https://onnx.ai/onnx/operators/onnx__GatherND.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="gather",
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
class GatherPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.gather to ONNX GatherND."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX gather primitive using ONNX GatherND."""
        # Get input and output names
        data_var, indices_var = node_inputs
        output_var = node_outputs[0]

        data_name = s.get_name(data_var)
        indices_name = s.get_name(indices_var)
        output_name = s.get_var_name(output_var)

        # Extract parameters from the JAX gather node
        dimension_numbers = params["dimension_numbers"]
        params["slice_sizes"]

        # Extract dimension mappings from JAX gather
        start_index_map = dimension_numbers.start_index_map

        # Shape information
        indices_shape = indices_var.aval.shape
        jax_out_shape = output_var.aval.shape

        # GatherND requires indices transformed to select the starting points
        # of each slice in the operand (data tensor)

        # First, we need to cast indices to int64 as required by ONNX GatherND
        cast_indices_name = s.get_unique_name("indices_int64")
        cast_node = helper.make_node(
            "Cast",
            inputs=[indices_name],
            outputs=[cast_indices_name],
            name=s.get_unique_name("cast_indices"),
            to=helper.TensorProto.INT64,
        )
        s.add_node(cast_node)

        # Register shape and dtype information for the cast output
        s.add_shape_info(
            cast_indices_name, indices_shape, dtype=helper.TensorProto.INT64
        )

        # Then, we may need to transform the indices tensor
        if len(start_index_map) == 1 and len(indices_shape) > 1:
            # Simple case: we need to transform indices to GatherND format
            # Get a unique name for the intermediate result
            # transformed_indices_name = s.get_unique_name("transformed_indices")

            # Create a node to transform the indices
            batch_dims = 0  # JAX gather typically has no batch dims in our case

            # Create the GatherND node
            gather_node = helper.make_node(
                "GatherND",
                inputs=[data_name, cast_indices_name],
                outputs=[output_name],
                name=s.get_unique_name("gathernd"),
                batch_dims=batch_dims,
            )
            s.add_node(gather_node)

            # In some cases, the output may need to be reshaped to match JAX's expected shape
            onnx_out_shape = (
                output_var.aval.shape
            )  # We'll use the shape inferred by JAX
            s.add_shape_info(output_name, onnx_out_shape)
        else:
            # For more complex cases, we'll need additional preprocessing
            # This is a simplification handling the common case
            batch_dims = 0

            # Create the GatherND node
            gather_node = helper.make_node(
                "GatherND",
                inputs=[data_name, indices_name],
                outputs=[output_name],
                name=s.get_unique_name("gathernd"),
                batch_dims=batch_dims,
            )
            s.add_node(gather_node)

            # Register the expected output shape
            s.add_shape_info(output_name, jax_out_shape)
