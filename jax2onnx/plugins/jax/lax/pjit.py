from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Tuple

import numpy as np

from jax2onnx.plugins.plugin_system import (
    PLUGIN_REGISTRY,
    PrimitiveLeafPlugin,
    register_primitive,
)

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _extract_closed_jaxpr(params: dict[str, Any]) -> Tuple[Any, Iterable[Any]]:
    closed = params.get("call_jaxpr") or params.get("jaxpr")
    if closed is None:
        raise ValueError("pjit parameters missing inner jaxpr")
    if hasattr(closed, "jaxpr") and hasattr(closed, "consts"):
        return closed.jaxpr, getattr(closed, "consts")
    consts = params.get("consts", ())
    return closed, consts


@register_primitive(
    jaxpr_primitive="pjit",
    jax_doc="https://jax.readthedocs.io/en/latest/jax.experimental.pjit.html",
    onnx=[],
    since="v0.1.0",
    context="primitives.lax",
    component="pjit",
    testcases=[],
)
class PJITPlugin(PrimitiveLeafPlugin):
    """Inline the body of a ``pjit`` call directly into the current IR context."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        inner_jaxpr, consts = _extract_closed_jaxpr(eqn.params)

        # Bind constants into the current context so inner equations can use them
        for var, const_val in zip(inner_jaxpr.constvars, consts):
            np_const = np.asarray(const_val)
            ctx.bind_const_for_var(var, np_const)

        # Map outer inputs to inner invars
        for outer_var, inner_var in zip(eqn.invars, inner_jaxpr.invars):
            ctx.bind_value_for_var(inner_var, ctx.get_value_for_var(outer_var))

        # Lower the inner equations using existing plugins
        for inner_eqn in inner_jaxpr.eqns:
            prim = inner_eqn.primitive.name
            plugin = PLUGIN_REGISTRY.get(prim)
            if plugin is None:
                raise NotImplementedError(
                    f"[pjit] No plugins registered for primitive '{prim}' inside pjit body"
                )
            plugin.lower(ctx, inner_eqn)

        # Map inner outputs back to the outer graph
        for outer_var, inner_var in zip(eqn.outvars, inner_jaxpr.outvars):
            ctx.bind_value_for_var(outer_var, ctx.get_value_for_var(inner_var))
