import jax
from typing import TYPE_CHECKING
from onnx import helper, TensorProto

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.argmin_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_argmin(node_inputs, node_outputs, params):
        """Handle JAX argmin primitive."""
        input_name = s.get_name(node_inputs[0])
        intermediate_name = s.get_unique_name("argmin_intermediate")
        output_name = s.get_var_name(node_outputs[0])
        axis = params["axes"][0]
        keepdims = params.get("keepdims", 0)
        node_1 = helper.make_node(
            "ArgMin",
            inputs=[input_name],
            outputs=[intermediate_name],
            name=s.get_unique_name("argmin"),
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

    return _handle_argmin


def get_metadata() -> dict:
    """Return metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "argmin",
        "jax_doc": "https://docs.jax.dev/en/latest/_autosummary/jax.lax.argmin.html",
        "onnx": [
            {
                "component": "ArgMin",
                "doc": "https://onnx.ai/onnx/operators/onnx__ArgMin.html",
            }
        ],
        "since": "v0.2.0",
        "context": "plugins.lax",
        "testcases": [
            {
                "testcase": "argmin_test1",
                "callable": lambda x: jax.lax.argmin(
                    x, axis=0, index_dtype=jax.numpy.int32
                ),
                "input_shapes": [(3, 3)],
            },
            {
                "testcase": "argmin_test2",
                "callable": lambda x: jax.lax.argmin(
                    x, axis=1, index_dtype=jax.numpy.int32
                ),
                "input_shapes": [(3, 3)],
            },
        ],
    }
