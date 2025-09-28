from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    pass


@register_primitive(
    jaxpr_primitive=jax.lax.gt_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.gt.html",
    onnx=[
        {
            "component": "Greater",
            "doc": "https://onnx.ai/onnx/operators/onnx__Greater.html",
        }
    ],
    since="v0.2.0",
    context="primitives2.lax",
    component="gt",
    testcases=[
        {
            "testcase": "gt",
            "callable": lambda x1, x2: jax.lax.gt(x1, x2),
            "input_shapes": [(3,), (3,)],
            "use_onnx_ir": True,
        }
    ],
)
class GtPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx, eqn):
        lhs_var, rhs_var = eqn.invars
        out_var = eqn.outvars[0]

        lhs_val = ctx.get_value_for_var(lhs_var, name_hint=ctx.fresh_name("gt_lhs"))
        prefer_dtype = np.dtype(getattr(lhs_var.aval, "dtype", np.float32))
        rhs_val = ctx.get_value_for_var(
            rhs_var,
            name_hint=ctx.fresh_name("gt_rhs"),
            prefer_np_dtype=prefer_dtype,
        )
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("gt_out"))

        lhs_dtype_enum = getattr(getattr(lhs_val, "type", None), "dtype", None)
        rhs_dtype_enum = getattr(getattr(rhs_val, "type", None), "dtype", None)
        if (
            lhs_dtype_enum is not None
            and rhs_dtype_enum is not None
            and lhs_dtype_enum != rhs_dtype_enum
        ):
            rhs_val = ctx.cast_like(rhs_val, lhs_val, name_hint="gt_rhs")

        node = ir.Node(
            op_type="Greater",
            domain="",
            inputs=[lhs_val, rhs_val],
            outputs=[out_val],
            name=ctx.fresh_name("Greater"),
        )
        ctx.add_node(node)
