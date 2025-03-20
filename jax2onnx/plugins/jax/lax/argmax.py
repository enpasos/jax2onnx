import jax
from typing import TYPE_CHECKING
from onnx import helper, TensorProto
from jax2onnx.plugin_system import register_primitive, PrimitivePlugin
import jax.numpy as jnp

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.argmax_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.argmax.html",
    onnx=[
        {
            "component": "ArgMax",
            "doc": "https://onnx.ai/onnx/operators/onnx__ArgMax.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    testcases=[
        {
            "testcase": "argmax_test1",
            "callable": lambda x: jax.lax.argmax(x, axis=0, index_dtype=jnp.int32),
            "input_shapes": [(3, 3)],
        },
        {
            "testcase": "argmax_test2",
            "callable": lambda x: jax.lax.argmax(x, axis=1, index_dtype=jnp.int32),
            "input_shapes": [(3, 3)],
        },
    ],
)
class ArgMaxPlugin(PrimitivePlugin):
    """
    Plugin for converting jax.lax.argmax to ONNX.
    """

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX argmax primitive."""
        input_name = s.get_name(node_inputs[0])
        intermediate_name = s.get_unique_name("argmax_intermediate")
        output_name = s.get_name(node_outputs[0])  # Corrected: get_name
        axis = params["axes"][0]
        # keepdims is always False for jax.lax.argmax
        keepdims = 0  # Hardcoded: jax.lax.argmax always has keepdims=False

        node_1 = helper.make_node(
            "ArgMax",
            inputs=[input_name],
            outputs=[intermediate_name],
            name=s.get_unique_name("argmax"),
            axis=axis,
            keepdims=keepdims,
            select_last_index=int(
                params["index_dtype"] == jnp.int64
            ),  # Properly set select_last_index, convert bool to int
        )
        s.add_node(node_1)

        node_2 = helper.make_node(
            "Cast",
            inputs=[intermediate_name],
            outputs=[output_name],
            to=TensorProto.INT32,  # Cast to the correct output type (INT32)
        )
        s.add_node(node_2)
