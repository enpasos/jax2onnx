import jax
import numpy as np
from typing import TYPE_CHECKING
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.conv_general_dilated_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_conv(node_inputs, node_outputs, params):
        """Handle JAX conv_general_dilated primitive."""
        input_name = s.get_name(node_inputs[0])
        filter_name = s.get_name(node_inputs[1])
        output_name = s.get_var_name(node_outputs[0])
        # Extract parameters
        dimension_numbers = params["dimension_numbers"]
        window_strides = params["window_strides"]
        padding = params["padding"]
        # For simplicity, assume standard NCHW ordering and that filters have shape (O, I, H, W)
        kernel_shape = node_inputs[1].aval.shape[2:]
        node = helper.make_node(
            "Conv",
            inputs=[input_name, filter_name],
            outputs=[output_name],
            name=s.get_unique_name("conv"),
            kernel_shape=kernel_shape,
            strides=window_strides,
            dilations=params.get("rhs_dilation", (1,) * len(window_strides)),
            pads=sum(padding, ()),
        )
        s.add_node(node)

    return _handle_conv
