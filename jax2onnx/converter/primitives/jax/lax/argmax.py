import jax
import numpy as np
from typing import TYPE_CHECKING
from onnx import helper, TensorProto

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.argmax_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_argmax(node_inputs, node_outputs, params):
        """Handle JAX argmax primitive."""
        input_name = s.get_name(node_inputs[0])
        intermediate_name = s.get_unique_name("argmax_intermediate")
        output_name = s.get_var_name(node_outputs[0])
        axis = params["axes"][0]
        keepdims = 1 if params.get("keepdims", False) else 0

        node_1 = helper.make_node(
            "ArgMax",
            inputs=[input_name],
            outputs=[intermediate_name],
            name=s.get_unique_name("argmax"),
            axis=axis,
            keepdims=keepdims,
        )
        s.add_node(node_1)

        node_2 = helper.make_node(
            "Cast",
            inputs=[intermediate_name],
            outputs=[output_name],
            to=TensorProto.INT32,
        )
        s.add_node(node_2)

    return _handle_argmax
