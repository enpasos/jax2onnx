from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from onnx import TensorProto, helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.argmin_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.argmin.html",
    onnx=[
        {
            "component": "ArgMin",
            "doc": "https://onnx.ai/onnx/operators/onnx__ArgMin.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="argmin",
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
class ArgMinPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.lax.argmin to ONNX.
    """

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        input_name = s.get_name(node_inputs[0])
        intermediate_name = s.get_unique_name("argmin_intermediate")
        output_name = s.get_name(node_outputs[0])
        axis = params["axes"][0]

        # ONNX ArgMin always returns int64
        argmin_node = helper.make_node(
            "ArgMin",
            inputs=[input_name],
            outputs=[intermediate_name],
            name=s.get_unique_name("argmin"),
            axis=axis,
            keepdims=0,
            select_last_index=0,  # You can parametrize this if needed
        )
        s.add_node(argmin_node)
        # ✅ Add correct shape and int64 dtype for ArgMin output
        s.add_shape_info(intermediate_name, node_outputs[0].aval.shape, dtype="int64")

        # Cast to the requested JAX dtype (e.g., int32)
        if params["index_dtype"] == jnp.int32:
            cast_node = helper.make_node(
                "Cast",
                inputs=[intermediate_name],
                outputs=[output_name],
                name=s.get_unique_name("cast_argmin"),
                to=TensorProto.INT32,
            )
            s.add_node(cast_node)
        else:
            identity_node = helper.make_node(
                "Identity",
                inputs=[intermediate_name],
                outputs=[output_name],
                name=s.get_unique_name("identity_argmin"),
            )
            s.add_node(identity_node)

        s.add_shape_info(intermediate_name, node_outputs[0].aval.shape, dtype="int64")
