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
    jax_doc="https://jax.readthedocs.io/en/latest/jax.device_put.html",
    onnx=[
        {
            "component": "Identity",
            "doc": "https://onnx.ai/onnx/operators/onnx__Identity.html",
        }
    ],
    since="v0.4.0",
    context="primitives.jax.lax",
    component="device_put",
    testcases=[],
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

            # Convert int64 to int32 and float64 to float32 for ONNX compatibility
            if np_val.dtype == np.int64:
                np_val = np_val.astype(np.int32)
            elif np_val.dtype == np.float64:
                np_val = np_val.astype(np.float32)

            tensor_name = converter.get_unique_name("const")
            tensor = converter.builder.create_tensor(
                name=tensor_name,
                data_type=converter.builder.numpy_dtype_to_onnx(np_val.dtype),
                dims=np_val.shape,
                vals=np_val.flatten().tolist(),
            )
            converter.builder.add_initializer(tensor)

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
