import jax
import numpy as np
from typing import TYPE_CHECKING
from onnx import helper, TensorProto

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.convert_element_type_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_convert_element_type(node_inputs, node_outputs, params):
        """Handle JAX convert_element_type primitive."""
        input_names = [s.get_name(inp) for inp in node_inputs]
        output_name = s.get_var_name(node_outputs[0])
        new_dtype = s.builder.numpy_dtype_to_onnx(params["new_dtype"])
        node = helper.make_node(
            "Cast",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("convert_element_type"),
            to=new_dtype,
        )
        s.add_node(node)

    return _handle_convert_element_type
