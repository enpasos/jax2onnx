# jax2onnx/plugins/jax/lax/polynomial.py

"""Inline JAX's reference decomposition of the polynomial primitive."""

from __future__ import annotations

from typing import cast

import jax

from jax2onnx._compat.jax import (
    JaxprEqn,
    polynomial_decomposition,
    trace_decomposition,
)
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.core.jit import JitPlugin
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)


def _linear_polynomial(
    x: jax.Array, constant: jax.Array, slope: jax.Array
) -> jax.Array:
    polynomial = getattr(jax.lax, "polynomial", None)
    if polynomial is not None:
        return cast(jax.Array, polynomial(x, [constant, slope]))
    return slope * x + constant


@register_native_primitive(
    jaxpr_primitive="polynomial",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.polynomial.html",
    onnx=[
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
        {"component": "Loop", "doc": "https://onnx.ai/onnx/operators/onnx__Loop.html"},
    ],
    since="0.18.0",
    context="primitives.lax",
    component="polynomial",
    testcases=[
        {
            "testcase": "polynomial_dynamic_coefficients",
            "callable": _linear_polynomial,
            "input_shapes": [(2, 3), (2, 1), (1, 3)],
            "post_check_onnx_graph": EG(["Mul:2x3 -> Add:2x3"], no_unused_inputs=True),
        }
    ],
)
class PolynomialPlugin(PrimitiveLeafPlugin):
    """Reuse upstream Horner/scan semantics and the existing primitive plugins."""

    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        # JAX's short recurrence starts at the highest coefficient; its scan
        # starts at zero. Reusing the decomposition preserves that distinction
        # for nonfinite inputs and signed zero, including explicit ``unroll``.
        closed = trace_decomposition(
            polynomial_decomposition,
            [var.aval for var in eqn.invars],
            eqn.params,
        )
        for original, expanded in (
            (eqn.invars, closed.jaxpr.invars),
            (eqn.outvars, closed.jaxpr.outvars),
        ):
            if len(original) != len(expanded) or any(
                outer.aval != inner.aval for outer, inner in zip(original, expanded)
            ):
                raise ValueError("polynomial decomposition changed its signature")
        JitPlugin().lower(ctx, eqn.replace(params={"jaxpr": closed}))
