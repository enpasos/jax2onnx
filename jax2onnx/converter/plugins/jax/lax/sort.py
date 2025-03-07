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
    """
    Return metadata describing the plugin.

    This could include documentation links, test cases, version information, etc.
    For now, we return an empty list.
    """
    return {}
