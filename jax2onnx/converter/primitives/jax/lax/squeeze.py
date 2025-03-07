import jax
import numpy as np
from typing import TYPE_CHECKING
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.squeeze_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_squeeze(node_inputs, node_outputs, params):
        """Handle JAX squeeze primitive."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])
        dims = params["dimensions"]
        axes = s.get_constant_name(np.array(dims, dtype=np.int64))
        node = helper.make_node(
            "Squeeze",
            inputs=[input_name, axes],
            outputs=[output_name],
            name=s.get_unique_name("squeeze"),
        )
        s.add_node(node)

    return _handle_squeeze
