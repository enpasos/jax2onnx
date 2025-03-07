import numpy as np
import jax
from typing import TYPE_CHECKING, List, Dict, Any
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.reduce_min_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_reduce_min(node_inputs, node_outputs, params):
        """Handle JAX reduce_min primitive."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])
        axes = params["axes"]
        axes_name = s.get_constant_name(np.array(axes, dtype=np.int64))
        node = helper.make_node(
            "ReduceMin",
            inputs=[input_name, axes_name],
            outputs=[output_name],
            name=s.get_unique_name("reduce_min"),
            keepdims=0 if not params.get("keepdims", False) else 1,
        )
        s.add_node(node)

    return _handle_reduce_min


def get_metadata() -> dict:
    """
    Return metadata describing the plugin.

    This could include documentation links, test cases, version information, etc.
    For now, we return an empty list.
    """
    return {}
