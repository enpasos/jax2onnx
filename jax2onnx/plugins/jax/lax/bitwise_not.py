# jax2onnx/plugins/jax/lax/bitwise_not.py

from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    pass  # type hints only


@register_primitive(
    jaxpr_primitive=jax.lax.not_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.bitwise_not.html",
    onnx=[
        {
            "component": "BitwiseNot",
            "doc": "https://onnx.ai/onnx/operators/onnx__BitwiseNot.html",
        },
        {
            "component": "Not",
            "doc": "https://onnx.ai/onnx/operators/onnx__Not.html",
        },
    ],
    since="v0.7.5",
    context="primitives.lax",
    component="bitwise_not",
    testcases=[
        {
            "testcase": "bitwise_not_bool",
            "callable": lambda x: jax.lax.bitwise_not(x),
            "input_values": [np.array(True, dtype=np.bool_)],
            "expected_output_dtypes": [np.bool_],
        },
        {
            "testcase": "bitwise_not_i32",
            "callable": lambda x: jax.lax.bitwise_not(x),
            "input_values": [np.array(7, dtype=np.int32)],
            "expected_output_dtypes": [np.int32],
        },
    ],
)
class BitwiseNotPlugin(PrimitiveLeafPlugin):
    @staticmethod
    def abstract_eval(x):
        return x

    def lower(self, ctx, eqn):
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("not_in"))
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("not_out"))

        x_dtype = np.dtype(getattr(x_var.aval, "dtype", np.bool_))
        op_type = "Not" if x_dtype.kind == "b" else "BitwiseNot"

        node = ir.Node(
            op_type=op_type,
            domain="",
            inputs=[x_val],
            outputs=[out_val],
            name=ctx.fresh_name(op_type.lower()),
        )
        ctx.add_node(node)
