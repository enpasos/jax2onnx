# file: jax2onnx/plugins2/jax/lax/or.py

from typing import TYPE_CHECKING

import numpy as np
import onnx_ir as ir
import jax

from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.or_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.bitwise_or.html",
    onnx=[
        {"component": "Or", "doc": "https://onnx.ai/onnx/operators/onnx__Or.html"},
        {
            "component": "BitwiseOr",
            "doc": "https://onnx.ai/onnx/operators/onnx__BitwiseOr.html",
        },
    ],
    since="v0.7.2",
    context="primitives2.lax",
    component="or",
    testcases=[],
)
class OrPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.bitwise_or`` and boolean ``or``."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        lhs_var, rhs_var = eqn.invars
        out_var = eqn.outvars[0]

        prefer_dtype = np.dtype(getattr(lhs_var.aval, "dtype", np.bool_))

        lhs_val = ctx.get_value_for_var(lhs_var, name_hint=ctx.fresh_name("or_lhs"))
        rhs_val = ctx.get_value_for_var(
            rhs_var,
            name_hint=ctx.fresh_name("or_rhs"),
            prefer_np_dtype=prefer_dtype,
        )
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("or_out"))

        op_type = "Or" if np.issubdtype(prefer_dtype, np.bool_) else "BitwiseOr"

        ctx.add_node(
            ir.Node(
                op_type=op_type,
                domain="",
                inputs=[lhs_val, rhs_val],
                outputs=[out_val],
                name=ctx.fresh_name(op_type.lower()),
            )
        )
