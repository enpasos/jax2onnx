from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import numpy as np

import jax

from jax2onnx.plugins.plugin_system import (
    PLUGIN_REGISTRY2,
    PrimitiveLeafPlugin,
    register_primitive,
)


@register_primitive(
    jaxpr_primitive="jit",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.jit.html",
    onnx=[],
    since="v0.9.0",
    context="primitives.core",
    component="jit_inline",
    testcases=[
        {
            "testcase": "jit_identity",
            "callable": lambda x: jax.jit(lambda y: y + 1)(x),
            "input_shapes": [(3,)],
        }
    ],
)
class JitPlugin(PrimitiveLeafPlugin):
    """Inline the body of ``jit`` primitives into the active IR context."""

    _PRIM: ClassVar = None

    def lower(self, ctx, eqn):  # type: ignore[override]
        closed = eqn.params.get("call_jaxpr")
        if closed is None:
            thunk = eqn.params.get("call_jaxpr_thunk")
            if thunk is not None:
                closed = thunk()
        if closed is None:
            maybe_jaxpr = eqn.params.get("jaxpr")
            if maybe_jaxpr is not None:
                consts = eqn.params.get("consts", ())
                closed = SimpleNamespace(jaxpr=maybe_jaxpr, consts=consts)
        if closed is None:
            raise ValueError("jit lowering requires call_jaxpr parameter")

        inner_jaxpr = getattr(closed, "jaxpr", closed)
        consts = getattr(closed, "consts", eqn.params.get("consts", ()))

        for const_var, const_val in zip(inner_jaxpr.constvars, consts):
            ctx.bind_const_for_var(const_var, np.asarray(const_val))

        for outer_var, inner_var in zip(eqn.invars, inner_jaxpr.invars):
            ctx.bind_value_for_var(inner_var, ctx.get_value_for_var(outer_var))

        for inner_eqn in inner_jaxpr.eqns:
            prim_name = inner_eqn.primitive.name
            plugin = PLUGIN_REGISTRY2.get(prim_name)
            if plugin is None:
                raise NotImplementedError(
                    f"[jit] No plugins registered for primitive '{prim_name}' inside jit body"
                )
            plugin.lower(ctx, inner_eqn)

        for outer_var, inner_var in zip(eqn.outvars, inner_jaxpr.outvars):
            ctx.bind_value_for_var(outer_var, ctx.get_value_for_var(inner_var))
