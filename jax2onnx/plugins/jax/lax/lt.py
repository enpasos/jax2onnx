from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    pass


@register_primitive(
    jaxpr_primitive=jax.lax.lt_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.lt.html",
    onnx=[
        {
            "component": "Less",
            "doc": "https://onnx.ai/onnx/operators/onnx__Less.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="lt",
    testcases=[
        {
            "testcase": "lt",
            "callable": lambda x1, x2: jax.lax.lt(x1, x2),
            "input_shapes": [(3,), (3,)],
        }
    ],
)
class LtPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx, eqn):
        lhs_var, rhs_var = eqn.invars
        out_var = eqn.outvars[0]

        lhs_val = ctx.get_value_for_var(lhs_var, name_hint=ctx.fresh_name("lt_lhs"))
        prefer_dtype = np.dtype(getattr(lhs_var.aval, "dtype", np.float32))
        rhs_val = ctx.get_value_for_var(
            rhs_var,
            name_hint=ctx.fresh_name("lt_rhs"),
            prefer_np_dtype=prefer_dtype,
        )
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("lt_out"))

        node = ir.Node(
            op_type="Less",
            domain="",
            inputs=[lhs_val, rhs_val],
            outputs=[out_val],
            name=ctx.fresh_name("Less"),
        )
        ctx.add_node(node)
