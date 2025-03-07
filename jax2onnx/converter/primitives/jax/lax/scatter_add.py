import jax
from typing import TYPE_CHECKING
from onnx import helper, TensorProto

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.scatter_add_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_scatter_add(node_inputs, node_outputs, params):
        """Handle JAX scatter_add primitive."""
        input_names = [s.get_name(inp) for inp in node_inputs]
        intermediate = s.get_unique_name("scatter_add:x")
        output_name = s.get_var_name(node_outputs[0])

        node_1 = helper.make_node(
            "Cast",
            inputs=[input_names[1]],
            outputs=[intermediate],
            to=TensorProto.INT64,
        )
        s.add_node(node_1)

        node_2 = helper.make_node(
            "ScatterND",
            inputs=[input_names[0], intermediate, input_names[2]],
            outputs=[output_name],
            name=s.get_unique_name("scatter_add"),
            reduction="add",
        )
        s.add_node(node_2)

    return _handle_scatter_add
