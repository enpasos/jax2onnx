# file: jax2onnx/converter/primitives/jax/lax/neg.py

import jax
from typing import TYPE_CHECKING
from onnx import helper


if TYPE_CHECKING:
    from jax2onnx.converter.converter import JaxprToOnnx


def get_primitive():
    return jax.lax.neg_p


def get_handler(s: "JaxprToOnnx"):
    def _handle_neg(node_inputs, node_outputs, params):
        """Handle JAX neg primitive."""
        input_names = [s._get_name(inp) for inp in node_inputs]
        output_name = s._get_var_name(node_outputs[0])
        node = helper.make_node(
            "Neg",
            inputs=input_names,
            outputs=[output_name],
            name=s._get_unique_name("neg"),
        )
        s.nodes.append(node)

    return _handle_neg
