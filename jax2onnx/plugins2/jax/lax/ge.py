from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    pass


@register_primitive(
    jaxpr_primitive=jax.lax.ge_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.ge.html",
    onnx=[
        {
            "component": "GreaterOrEqual",
            "doc": "https://onnx.ai/onnx/operators/onnx__GreaterOrEqual.html",
        }
    ],
    since="v0.2.0",
    context="primitives2.lax",
    component="ge",
    testcases=[
        {
            "testcase": "ge",
            "callable": lambda x1, x2: jax.lax.ge(x1, x2),
            "input_shapes": [(3,), (3,)],
            "use_onnx_ir": True,
        }
    ],
)
class GePlugin(PrimitiveLeafPlugin):
    def lower(self, ctx, eqn):
        lhs_var, rhs_var = eqn.invars
        out_var = eqn.outvars[0]

        lhs_val = ctx.get_value_for_var(lhs_var, name_hint=ctx.fresh_name("ge_lhs"))
        prefer_dtype = np.dtype(getattr(lhs_var.aval, "dtype", np.float32))
        rhs_val = ctx.get_value_for_var(
            rhs_var,
            name_hint=ctx.fresh_name("ge_rhs"),
            prefer_np_dtype=prefer_dtype,
        )
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("ge_out"))

        node = ir.Node(
            op_type="GreaterOrEqual",
            domain="",
            inputs=[lhs_val, rhs_val],
            outputs=[out_val],
            name=ctx.fresh_name("GreaterOrEqual"),
        )
        ctx.add_node(node)
