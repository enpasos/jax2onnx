# jax2onnx/plugins/jax/core/call_hi_primitive.py

from __future__ import annotations

from jax2onnx._compat.jax import JaxprEqn, native_nonzero_expansion
from jax2onnx.converter.typing_support import ConverterContext
from jax2onnx.plugins.jax.core.jit import JitPlugin
from jax2onnx.plugins.jax.lax._control_flow_utils import lower_jaxpr_eqns
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)


@register_native_primitive(
    jaxpr_primitive="call_hi_primitive",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nonzero.html",
    onnx=[],
    since="0.17.1",
    context="primitives.core",
    component="native_nonzero",
    testcases=[],
)
class NativeNonzeroHiJaxPlugin(PrimitiveLeafPlugin):
    """Inline the selected native Nonzero composition from newer JAX releases."""

    def lower(self, ctx: ConverterContext, eqn: JaxprEqn) -> None:
        closed = JitPlugin._freshen_closed_jaxpr(native_nonzero_expansion(eqn))
        for var, value in zip(closed.jaxpr.constvars, closed.consts, strict=True):
            ctx.bind_const_for_var(var, value)
        for outer, inner in zip(eqn.invars, closed.jaxpr.invars, strict=True):
            ctx.bind_value_for_var(inner, ctx.require_value_for_var(outer))
        lower_jaxpr_eqns(ctx, closed.jaxpr, source="native_nonzero")
        for outer, inner in zip(eqn.outvars, closed.jaxpr.outvars, strict=True):
            ctx.bind_value_for_var(outer, ctx.require_value_for_var(inner))
