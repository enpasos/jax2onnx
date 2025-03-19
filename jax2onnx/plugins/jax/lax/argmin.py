import jax
from typing import TYPE_CHECKING
from onnx import helper, TensorProto
from jax2onnx.plugin_system import register_plugin, PrimitivePlugin
import jax.numpy as jnp

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


@register_plugin(
    jaxpr_primitive=jax.lax.argmin_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.argmin.html",
    onnx=[
        {
            "component": "ArgMin",
            "doc": "https://onnx.ai/onnx/operators/onnx__ArgMin.html",
        }
    ],
    since="v0.2.0",
    context="plugins.lax",
    testcases=[
        {
            "testcase": "argmin_test1",
            "callable": lambda x: jax.lax.argmin(x, axis=0, index_dtype=jnp.int32),
            "input_shapes": [(3, 3)],
        },
        {
            "testcase": "argmin_test2",
            "callable": lambda x: jax.lax.argmin(x, axis=1, index_dtype=jnp.int32),
            "input_shapes": [(3, 3)],
        },
    ],
)
class ArgMinPlugin(PrimitivePlugin):
    """
    Plugin for converting jax.lax.argmin to ONNX.
    """

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX argmin primitive."""
        input_name = s.get_name(node_inputs[0])
        intermediate_name = s.get_unique_name("argmin_intermediate")
        output_name = s.get_name(node_outputs[0])  # Corrected: get_name
        axis = params["axes"][0]
        # keepdims is always False for jax.lax.argmin
        keepdims = 0  # Hardcoded: jax.lax.argmin always has keepdims=False

        node_1 = helper.make_node(
            "ArgMin",
            inputs=[input_name],
            outputs=[intermediate_name],
            name=s.get_unique_name("argmin"),
            axis=axis,
            keepdims=keepdims,
            select_last_index=int(
                params["index_dtype"] == jnp.int64
            ),  # Set select_last_index
        )
        s.add_node(node_1)

        node_2 = helper.make_node(
            "Cast",
            inputs=[intermediate_name],
            outputs=[output_name],
            to=TensorProto.INT32,  # Cast to the correct output type (INT32)
        )
        s.add_node(node_2)
