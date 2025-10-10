# jax2onnx/plugins/jax/lax/shift_right_logical.py

from __future__ import annotations

from typing import TYPE_CHECKING

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
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("srl_out"))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("srl_out")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("srl_out")

        result = ctx.builder.BitShift(
            lhs_val,
            rhs_val,
            direction="RIGHT",
            _outputs=[desired_name],
        )
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
