# jax2onnx/plugins/jax/lax/or.py

from typing import TYPE_CHECKING

import numpy as np
import jax

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


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
    context="primitives.lax",
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
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("or_out"))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("or_out")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("or_out")

        op_type = "Or" if np.issubdtype(prefer_dtype, np.bool_) else "BitwiseOr"
        builder_fn = getattr(ctx.builder, op_type)

        result = builder_fn(lhs_val, rhs_val, _outputs=[desired_name])
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
