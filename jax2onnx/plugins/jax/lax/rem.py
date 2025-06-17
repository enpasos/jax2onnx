from typing import TYPE_CHECKING
import logging

import jax
from onnx import helper
import jax.numpy as jnp

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

logger = logging.getLogger(__name__)


@register_primitive(
    jaxpr_primitive=jax.lax.rem_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.rem.html",
    onnx=[
        {
            "component": "Mod",
            "doc": "https://onnx.ai/onnx/operators/onnx__Mod.html",
        },
        {
            "component": "Div",
            "doc": "https://onnx.ai/onnx/operators/onnx__Div.html",
        },
    ],
    since="v0.6.5",
    context="primitives.lax",
    component="rem",
    testcases=[
        {
            "testcase": "rem_int",
            "callable": lambda x, y: jax.lax.rem(x, y),
            "input_values": [
                jnp.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=jnp.int32),
                jnp.array([4, 4, 3, 3, 2, 2, 1, 1, 5, 5], dtype=jnp.int32),
            ],
        },
        {
            "testcase": "rem_float",
            "callable": lambda x, y: jax.lax.rem(x, y),
            "input_values": [
                jnp.array(
                    [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                    dtype=jnp.float32,
                ),
                jnp.array(
                    [4.0, 4.0, 3.0, 3.0, 2.0, 2.0, 1.0, 1.0, 5.0, 5.0],
                    dtype=jnp.float32,
                ),
            ],
        },
        {
            "testcase": "rem_int_neg",
            "callable": lambda x, y: jax.lax.rem(x, y),
            "input_values": [
                jnp.array([-10, -9, -8, -7, 6, 5, 4, 3, -2, -1], dtype=jnp.int32),
                jnp.array([4, -4, 3, -3, 2, -2, 1, -1, 5, -5], dtype=jnp.int32),
            ],
        },
        {
            "testcase": "rem_float_neg",
            "callable": lambda x, y: jax.lax.rem(x, y),
            "input_values": [
                jnp.array(
                    [-10.0, -9.0, -8.0, -7.0, 6.0, 5.0, 4.0, 3.0, -2.0, -1.0],
                    dtype=jnp.float32,
                ),
                jnp.array(
                    [4.0, -4.0, 3.0, -3.0, 2.0, -2.0, 1.0, -1.0, 5.0, -5.0],
                    dtype=jnp.float32,
                ),
            ],
        },
    ],
)
class RemPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.rem (truncated remainder) to ONNX."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        def _register(name, shape, dtype):
            s.builder.register_value_info_metadata(name, shape, dtype)
            s.builder.add_value_info(name, shape, dtype)

        x_name = s.get_name(node_inputs[0])
        y_name = s.get_name(node_inputs[1])
        out_name = s.get_var_name(node_outputs[0])

        aval = node_inputs[0].aval
        shape = aval.shape
        dtype = aval.dtype
        is_float = jnp.issubdtype(dtype, jnp.floating)

        if is_float:
            # Float remainder: use ONNX Mod with fmod=1 (C fmod semantics = truncated remainder)
            tmp = s.get_unique_name("rem_fmod")
            s.add_node(
                helper.make_node(
                    "Mod",
                    inputs=[x_name, y_name],
                    outputs=[tmp],
                    name=s.get_unique_name("rem_mod"),
                    fmod=1,  # C fmod → truncated remainder
                )
            )
            _register(tmp, shape, dtype)
            s.add_node(
                helper.make_node(
                    "Identity",
                    inputs=[tmp],
                    outputs=[out_name],
                    name=s.get_unique_name("rem_out"),
                )
            )
            _register(out_name, shape, dtype)

        else:
            # Integer remainder: r = x - (trunc(x / y) * y)
            quot = s.get_unique_name("rem_div")
            s.add_node(
                helper.make_node(
                    "Div",
                    inputs=[x_name, y_name],
                    outputs=[quot],
                    name=s.get_unique_name("rem_div_node"),
                )
            )
            _register(quot, shape, dtype)

            prod = s.get_unique_name("rem_mul")
            s.add_node(
                helper.make_node(
                    "Mul",
                    inputs=[quot, y_name],
                    outputs=[prod],
                    name=s.get_unique_name("rem_mul_node"),
                )
            )
            _register(prod, shape, dtype)

            s.add_node(
                helper.make_node(
                    "Sub",
                    inputs=[x_name, prod],
                    outputs=[out_name],
                    name=s.get_unique_name("rem_sub_node"),
                )
            )
            _register(out_name, shape, dtype)
