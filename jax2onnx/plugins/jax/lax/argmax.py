from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from onnx import TensorProto, helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


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
    component="argmax",
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
class ArgMaxPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.lax.argmax to ONNX.
    """

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX argmax primitive."""
        input_name = s.get_name(node_inputs[0])
        intermediate_name = s.get_unique_name("argmax_intermediate")
        output_name = s.get_name(node_outputs[0])
        axis = params["axes"][0]
        keepdims = 0  # jax.lax.argmax always has keepdims=False

        node_1 = helper.make_node(
            "ArgMax",
            inputs=[input_name],
            outputs=[intermediate_name],
            name=s.get_unique_name("argmax"),
            axis=axis,
            keepdims=keepdims,
            select_last_index=int(params["index_dtype"] == jnp.int64),
        )
        s.add_node(node_1)

        # ✅ Ensure ONNX gets the correct dtype for the ArgMax output (int64)
        s.add_shape_info(intermediate_name, node_outputs[0].aval.shape, dtype="int64")

        node_2 = helper.make_node(
            "Cast",
            inputs=[intermediate_name],
            outputs=[output_name],
            to=TensorProto.INT32,
        )
        s.add_node(node_2)
