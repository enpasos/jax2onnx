# jax2onnx/plugins/jax/lax/shift_right_logical.py

from __future__ import annotations

from typing import TYPE_CHECKING

import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive="shift_right_logical",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.shift_right_logical.html",
    onnx=[
        {
            "component": "BitShift",
            "doc": "https://onnx.ai/onnx/operators/onnx__BitShift.html",
        }
    ],
    since="v0.7.2",
    context="primitives.lax",
    component="shift_right_logical",
    testcases=[],
)
class ShiftRightLogicalPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.shift_right_logical`` to ONNX BitShift(direction="RIGHT")."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        lhs_var, rhs_var = eqn.invars
        out_var = eqn.outvars[0]

        lhs_val = ctx.get_value_for_var(lhs_var, name_hint=ctx.fresh_name("srl_input"))
        rhs_val = ctx.get_value_for_var(
            rhs_var, name_hint=ctx.fresh_name("srl_shift"), prefer_np_dtype=None
        )
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("srl_out"))

        ctx.add_node(
            ir.Node(
                op_type="BitShift",
                domain="",
                inputs=[lhs_val, rhs_val],
                outputs=[out_val],
                name=ctx.fresh_name("BitShift"),
                attributes=[IRAttr("direction", IRAttrType.STRING, "RIGHT")],
            )
        )
