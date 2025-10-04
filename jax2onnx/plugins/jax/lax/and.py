# jax2onnx/plugins/jax/lax/and.py

from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    pass  # type hints only


@register_primitive(
    jaxpr_primitive=jax.lax.and_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.html#jax.lax.bitwise_and",
    onnx=[
        {
            "component": "And",
            "doc": "https://onnx.ai/onnx/operators/onnx__And.html",
        },
        {
            "component": "BitwiseAnd",
            "doc": "https://onnx.ai/onnx/operators/onnx__BitwiseAnd.html",
        },
    ],
    since="v0.6.5",
    context="primitives.lax",
    component="and",
    testcases=[
        {
            "testcase": "and_bool",
            "callable": lambda x, y: jax.lax.bitwise_and(x, y),
            "input_values": [
                np.array([True, True, False, False], dtype=np.bool_),
                np.array([True, False, True, False], dtype=np.bool_),
            ],
        },
        {
            "testcase": "and_int",
            "callable": lambda x, y: jax.lax.bitwise_and(x, y),
            "input_values": [
                np.array([1, 2, 3], dtype=np.int32),
                np.array([3, 1, 2], dtype=np.int32),
            ],
        },
    ],
)
class AndPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx, eqn):
        lhs_var, rhs_var = eqn.invars
        out_var = eqn.outvars[0]

        lhs_val = ctx.get_value_for_var(lhs_var, name_hint=ctx.fresh_name("and_lhs"))
        prefer_dtype = np.dtype(getattr(lhs_var.aval, "dtype", np.bool_))
        rhs_val = ctx.get_value_for_var(
            rhs_var,
            name_hint=ctx.fresh_name("and_rhs"),
            prefer_np_dtype=prefer_dtype,
        )
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("and_out"))

        is_bool = np.issubdtype(prefer_dtype, np.bool_)
        op_type = "And" if is_bool else "BitwiseAnd"

        node = ir.Node(
            op_type=op_type,
            domain="",
            inputs=[lhs_val, rhs_val],
            outputs=[out_val],
            name=ctx.fresh_name(op_type.lower()),
        )
        ctx.add_node(node)
