from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import jax
from jax import lax
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # only for static type checkers
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


# ---------------------------------------------------------------------
# 1. primitive alias
# ---------------------------------------------------------------------
broadcast_in_dim_p = lax.broadcast_in_dim_p
# ---------------------------------------------------------------------


@register_primitive(
    jaxpr_primitive=jax.lax.broadcast_in_dim_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/jax-primitives.html",
    onnx=[
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        },
        {
            "component": "Expand",
            "doc": "https://onnx.ai/onnx/operators/onnx__Expand.html",
        },
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="broadcast_in_dim",
    testcases=[
        {
            "testcase": "broadcast_in_dim",
            "callable": lambda x: jax.lax.broadcast_in_dim(
                x, (3,), broadcast_dimensions=(0,)
            ),
            "input_shapes": [(3,)],
        },
        {
            "testcase": "broadcast_in_dim_2d_to_3d",
            "callable": lambda x: jax.lax.broadcast_in_dim(
                x, (2, 3, 4), broadcast_dimensions=(1, 2)
            ),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "broadcast_in_dim_scalar",
            "callable": lambda x: jax.lax.broadcast_in_dim(
                x, (2, 3, 4), broadcast_dimensions=()
            ),
            "input_shapes": [()],
        },
    ],
)
class BroadcastInDimPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.lax.broadcast_in_dim to ONNX.
    """

    # -----------------------------------------------------------------
    # lowering to ONNX
    # -----------------------------------------------------------------
    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX broadcast_in_dim primitive."""
        input_var = node_inputs[0]
        input_name = s.get_name(input_var)
        # --- Get dtype from input variable ---
        input_dtype = input_var.aval.dtype
        input_shape = input_var.aval.shape

        output_var = node_outputs[0]
        output_name = s.get_var_name(output_var)
        # --- Use output variable's aval for target shape and final dtype ---
        output_dtype = output_var.aval.dtype  # Should match input_dtype
        output_target_shape = output_var.aval.shape

        broadcast_dimensions = params["broadcast_dimensions"]

        # --- Create Reshape node ---
        reshape_output = s.get_unique_name("reshape_output")
        # Calculate the intermediate shape after reshape but before expand
        reshape_shape = []
        idx = 0
        for i in range(len(output_target_shape)):  # Iterate based on target rank
            if i in broadcast_dimensions:
                if idx < len(input_shape):
                    reshape_shape.append(input_shape[idx])
                else:
                    reshape_shape.append(1)  # Handle scalar broadcasting case
                idx += 1
            else:
                reshape_shape.append(1)  # Dimension to be expanded

        reshape_shape_name = s.get_constant_name(
            np.array(reshape_shape, dtype=np.int64)
        )
        node_reshape = helper.make_node(
            "Reshape",
            inputs=[input_name, reshape_shape_name],
            outputs=[reshape_output],
            name=s.get_unique_name("reshape_for_broadcast"),
        )
        s.add_node(node_reshape)
        # *** FIX: Pass the correct dtype ***
        s.add_shape_info(
            reshape_output, tuple(reshape_shape), input_dtype
        )  # Use input's dtype

        # --- Create Expand node ---
        # Use the target shape from the output variable's aval
        target_shape_name = s.get_constant_name(
            np.array(output_target_shape, dtype=np.int64)
        )
        node_expand = helper.make_node(
            "Expand",
            inputs=[reshape_output, target_shape_name],
            outputs=[output_name],
            name=s.get_unique_name("expand"),
        )
        s.add_node(node_expand)
        # *** FIX: Pass the correct dtype ***
        s.add_shape_info(
            output_name, output_target_shape, output_dtype
        )  # Use output's dtype
