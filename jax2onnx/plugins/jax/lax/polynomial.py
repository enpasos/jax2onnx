# jax2onnx/plugins/jax/lax/polynomial.py

"""Inline JAX's native polynomial lowering without changing its recurrence."""

from __future__ import annotations

import jax
import numpy as np

from jax2onnx._compat.jax import (
    JaxprEqn,
    ShapedArray,
    native_polynomial_lowering,
    recorded_native_context,
)
from jax2onnx.converter.typing_support import ConverterContext
from jax2onnx.plugins.jax.lax._control_flow_utils import lower_jaxpr_eqns
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)


@register_native_primitive(
    jaxpr_primitive="polynomial",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.polynomial.html",
    onnx=[
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
        {"component": "Loop", "doc": "https://onnx.ai/onnx/operators/onnx__Loop.html"},
    ],
    since="0.17.1",
    context="primitives.lax",
    component="polynomial",
    testcases=[],
)
class PolynomialPlugin(PrimitiveLeafPlugin):
    """Reuse the installed native Horner/scan implementation and leaf plugins."""

    def lower(self, ctx: ConverterContext, eqn: JaxprEqn) -> None:
        native_lowering = native_polynomial_lowering(eqn.primitive)
        if native_lowering is None:
            raise NotImplementedError("The installed JAX has no polynomial primitive")

        unroll = eqn.params.get("unroll")
        if set(eqn.params) - {"unroll"}:
            raise NotImplementedError("Unsupported native polynomial parameters")

        inputs = []
        for var in eqn.invars:
            aval = var.aval
            if not isinstance(aval, ShapedArray):
                raise TypeError("Polynomial requires ordinary shaped array operands")
            inputs.append(
                jax.ShapeDtypeStruct(aval.shape, aval.dtype, weak_type=aval.weak_type)
            )

        # A fresh closure owns each trace's Var identities. Reusing a cached
        # closed JAXPR across polynomial equations would alias context bindings.
        def decompose(*operands: jax.Array) -> jax.Array:
            return native_lowering(*operands, unroll=unroll)

        # The native body preserves the unroll threshold (default 8), the scan's
        # initial +0*x step, coefficient broadcasting and per-operation dtype.
        with recorded_native_context(eqn.ctx):
            closed = jax.make_jaxpr(decompose)(*inputs)
        body = closed.jaxpr

        for var, value in zip(body.constvars, closed.consts, strict=True):
            ctx.bind_const_for_var(var, np.asarray(value))
        for outer, inner in zip(eqn.invars, body.invars, strict=True):
            ctx.bind_value_for_var(inner, ctx.require_value_for_var(outer))

        lower_jaxpr_eqns(ctx, body, source="polynomial")

        for outer, inner in zip(eqn.outvars, body.outvars, strict=True):
            ctx.bind_value_for_var(outer, ctx.require_value_for_var(inner))
