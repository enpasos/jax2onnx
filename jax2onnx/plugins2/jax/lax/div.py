from typing import TYPE_CHECKING, Optional

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.div_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.div.html",
    onnx=[{"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"}],
    since="v0.2.0",
    context="primitives2.lax",
    component="div",
    testcases=[
        {
            "testcase": "div",
            "callable": lambda x1, x2: x1 / x2,
            "input_shapes": [(3,), (3,)],
        },
        {
            "testcase": "div_const",
            "callable": lambda x: x / 2.0,
            "input_shapes": [(3,)],
        },
    ],
)
class DivPlugin(PrimitiveLeafPlugin):
    """IR-first lowering of ``lax.div`` to ONNX ``Div``."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        x_var, y_var = eqn.invars
        out_var = eqn.outvars[0]

        prefer_dt: Optional[np.dtype] = np.dtype(
            getattr(x_var.aval, "dtype", np.float32)
        )

        lhs_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("div_lhs"))
        rhs_val = ctx.get_value_for_var(
            y_var, name_hint=ctx.fresh_name("div_rhs"), prefer_np_dtype=prefer_dt
        )
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("div_out"))

        ctx.add_node(
            ir.Node(
                op_type="Div",
                domain="",
                inputs=[lhs_val, rhs_val],
                outputs=[out_val],
                name=ctx.fresh_name("Div"),
            )
        )
