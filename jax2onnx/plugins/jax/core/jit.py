# jax2onnx/plugins/jax/core/jit.py

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import numpy as np

import jax

from jax._src import core as jcore
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import (
    PLUGIN_REGISTRY,
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
            "post_check_onnx_graph": EG(
                [{"path": "Add:3", "inputs": {1: {"const": 1.0}}}],
                no_unused_inputs=True,
            ),
        }
    ],
)
class JitPlugin(PrimitiveLeafPlugin):
    """Inline the body of ``jit`` primitives into the active IR context."""

    _PRIM: ClassVar = None

    @staticmethod
    def _freshen_closed_jaxpr(closed):
        """Clone a ClosedJaxpr with fresh Vars so multiple jit bodies don't alias."""

        inner_jaxpr = getattr(closed, "jaxpr", closed)
        consts = getattr(closed, "consts", ())

        var_map: dict[jcore.Var, jcore.Var] = {}

        def _fresh_var(v):
            if not isinstance(v, jcore.Var):
                return v
            if v in var_map:
                return var_map[v]
            var_map[v] = jcore.Var(
                v.aval,
                getattr(v, "initial_qdd", None),
                getattr(v, "final_qdd", None),
            )
            return var_map[v]

        def _map_vars(seq):
            return [_fresh_var(v) for v in seq]

        constvars = _map_vars(inner_jaxpr.constvars)
        invars = _map_vars(inner_jaxpr.invars)
        outvars = _map_vars(inner_jaxpr.outvars)
        eqns = [
            eqn.replace(
                invars=_map_vars(eqn.invars),
                outvars=_map_vars(eqn.outvars),
            )
            for eqn in inner_jaxpr.eqns
        ]
        cloned = jcore.Jaxpr(
            constvars=constvars,
            invars=invars,
            outvars=outvars,
            eqns=eqns,
            effects=inner_jaxpr.effects,
            debug_info=inner_jaxpr.debug_info,
            is_high=getattr(inner_jaxpr, "is_high", False),
        )
        return jcore.ClosedJaxpr(cloned, consts)

    def lower(self, ctx, eqn):  # type: ignore[override]
        closed: SimpleNamespace | jcore.ClosedJaxpr | None = eqn.params.get(
            "call_jaxpr"
        )
        if closed is None:
            thunk = eqn.params.get("call_jaxpr_thunk")
            if thunk is not None:
                closed = thunk()
        if closed is None:
            maybe_jaxpr = eqn.params.get("jaxpr")
            # Newer JAX passes a ClosedJaxpr directly via `jaxpr`.
            if isinstance(maybe_jaxpr, jcore.ClosedJaxpr):
                closed = maybe_jaxpr
            elif maybe_jaxpr is not None:
                consts = eqn.params.get("consts", ())
                closed = SimpleNamespace(jaxpr=maybe_jaxpr, consts=consts)
        if closed is None:
            raise ValueError("jit lowering requires call_jaxpr parameter")

        fresh_closed = self._freshen_closed_jaxpr(
            closed
            if closed is not None
            else SimpleNamespace(
                jaxpr=getattr(closed, "jaxpr", closed),
                consts=eqn.params.get("consts", ()),
            )
        )
        inner_jaxpr = fresh_closed.jaxpr
        consts = fresh_closed.consts

        for const_var, const_val in zip(inner_jaxpr.constvars, consts):
            ctx.bind_const_for_var(const_var, np.asarray(const_val))

        for outer_var, inner_var in zip(eqn.invars, inner_jaxpr.invars):
            ctx.bind_value_for_var(inner_var, ctx.get_value_for_var(outer_var))

        for inner_eqn in inner_jaxpr.eqns:
            prim_name = inner_eqn.primitive.name
            plugin = PLUGIN_REGISTRY.get(prim_name)
            if plugin is None:
                raise NotImplementedError(
                    f"[jit] No plugins registered for primitive '{prim_name}' inside jit body"
                )
            plugin.lower(ctx, inner_eqn)

        for outer_var, inner_var in zip(eqn.outvars, inner_jaxpr.outvars):
            ctx.bind_value_for_var(outer_var, ctx.get_value_for_var(inner_var))
