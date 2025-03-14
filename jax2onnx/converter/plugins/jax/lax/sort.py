import jax
from typing import TYPE_CHECKING
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.sort_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_sort(node_inputs, node_outputs, params):
        """Handle JAX sort primitive."""
        input_name = s.get_name(node_inputs[0])
        shape_name = s.get_unique_name("sort_shape")
        value_name = s.get_var_name(node_outputs[0])
        indices_name = s.get_unique_name("sort_indices_output")
        if "axis" in params:
            # Not supported for now
            raise NotImplementedError("Sort with axis not supported yet")
        else:
            node_shape = helper.make_node(
                "Shape",
                inputs=[input_name],
                outputs=[shape_name],
                name=s.get_unique_name("shape"),
            )
            s.add_node(node_shape)
        node_topk = helper.make_node(
            "TopK",
            inputs=[input_name, shape_name],
            outputs=[value_name, indices_name],
            name=s.get_unique_name("sort"),
            largest=0,
        )
        s.add_node(node_topk)

    return _handle_sort


def get_metadata() -> dict:
    """Return metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "sort",
        "jax_doc": "https://docs.jax.dev/en/latest/_autosummary/jax.lax.sort.html",
        "onnx": [
            {
                "component": "Sort",
                "doc": "https://onnx.ai/onnx/operators/onnx__Sort.html",
            }
        ],
        "since": "v0.2.0",
        "context": "plugins.lax",
        "testcases": [
            {
                "testcase": "sort",
                "callable": lambda x: jax.lax.sort(x),
                "input_shapes": [(3,)],
            }
        ],
    }
