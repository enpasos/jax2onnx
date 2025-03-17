import jax
import numpy as np
from typing import TYPE_CHECKING
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.convert_element_type_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_convert_element_type(node_inputs, node_outputs, params):
        input_names = [s.get_name(inp) for inp in node_inputs]
        output_name = s.get_var_name(node_outputs[0])
        new_dtype = s.builder._numpy_dtype_to_onnx(params["new_dtype"])
        node = helper.make_node(
            "Cast",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("convert_element_type"),
            to=new_dtype,
        )
        s.add_node(node)

    return _handle_convert_element_type


def get_metadata() -> dict:
    """Return metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "convert_element_type",
        "jax_doc": "https://docs.jax.dev/en/latest/_autosummary/jax.lax.convert_element_type.html",
        "onnx": [
            {
                "component": "Cast",
                "doc": "https://onnx.ai/onnx/operators/onnx__Cast.html",
            }
        ],
        "since": "v0.2.0",
        "context": "plugins.lax",
        "testcases": [
            {
                "testcase": "convert_element_type",
                "callable": lambda x: x.astype(np.int16),
                "input_shapes": [(3,)],
            }
        ],
    }
