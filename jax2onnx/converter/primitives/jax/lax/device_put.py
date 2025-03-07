import jax
import numpy as np
from typing import TYPE_CHECKING
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.device_put_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_device_put(node_inputs, node_outputs, params):
        """Handle JAX device_put primitive."""
        name = s.get_unique_name("const")
        val = node_inputs[0]
        actual_val = val.val
        np_val = np.array(actual_val)
        if np_val.dtype == np.int64:
            np_val = np_val.astype(np.int32)
        elif np_val.dtype == np.float64:
            np_val = np_val.astype(np.float32)
        constant_name = s.get_constant_name(np_val)
        input_names = [constant_name]
        output_name = s.get_var_name(node_outputs[0])
        node = helper.make_node(
            "Identity",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("device_put"),
        )
        s.add_node(node)

    return _handle_device_put
