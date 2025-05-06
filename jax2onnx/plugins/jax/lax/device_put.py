"""
Plugin for handling the JAX device_put primitive.

This plugin converts JAX's device_put primitive to appropriate ONNX operations.
"""

from typing import TYPE_CHECKING

import jax
import numpy as np
from jax.extend import core as extend_core

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.device_put_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.device_put.html#jax.device_put",
    onnx=[
        {
            "component": "Identity",
            "doc": "https://onnx.ai/onnx/operators/onnx__Identity.html",
        }
    ],
    since="v0.4.0",
    context="primitives.lax",
    component="device_put",
    testcases=[
        {
            "testcase": "device_put_array",
            "callable": lambda x: jax.device_put(x),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "device_put_scalar",
            "callable": lambda: jax.device_put(42),
            "input_shapes": [],
        },
    ],
)
class DevicePutPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.device_put to appropriate ONNX operations."""

    def to_onnx(
        self, converter: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params
    ):
        """
        Convert jax.lax.device_put to ONNX operations.

        For constants, creates a constant node.
        For variables, creates an Identity node.

        Arguments:
            converter: The Jaxpr2OnnxConverter instance
            node_inputs: Input variables to the primitive
            node_outputs: Output variables from the primitive
            params: Parameters for the primitive
        """
        inp = node_inputs[0]
        out = node_outputs[0]

        if isinstance(inp, extend_core.Literal):
            # Handle conversion of literal values
            val = inp.val
            np_val = np.array(val)

            # Check output type and ensure we match it
            output_aval = out.aval
            output_dtype = output_aval.dtype

            # Convert value to match expected output dtype
            if np_val.dtype != output_dtype:
                np_val = np_val.astype(output_dtype)

            # Get tensor name for the constant
            tensor_name = converter.get_unique_name("const")

            # Use add_initializer to add the constant to the ONNX graph
            # This will handle data type conversion and initialization

            data_type = converter.builder._numpy_dtype_to_onnx(np_val.dtype)
            converter.builder.add_initializer(
                tensor_name, np_val.flatten().tolist(), data_type, dims=np_val.shape
            )

            output_name = converter.get_name(out)
            node = converter.builder.create_node(
                "Identity",
                [tensor_name],
                [output_name],
                name=converter.get_unique_name("device_put"),
            )
            converter.add_node(node)
        else:
            # For non-literal inputs, simply pass through with Identity
            input_names = [converter.get_name(inp) for inp in node_inputs]
            output_names = [converter.get_name(out) for out in node_outputs]
            if not output_names:
                return

            node = converter.builder.create_node(
                "Identity",
                input_names,
                output_names,
                name=converter.get_unique_name(f"identity_{jax.lax.device_put_p.name}"),
            )
            converter.add_node(node)
