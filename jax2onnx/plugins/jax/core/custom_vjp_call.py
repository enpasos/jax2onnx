# jax2onnx/plugins/jax/core/custom_vjp_call.py

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import (
    PLUGIN_REGISTRY,
    PrimitiveLeafPlugin,
    register_primitive,
)

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext

import jax

try:  # JAX 0.4+ lives in jax.extend.core
    from jax.extend.core import Primitive as JaxPrimitive  # type: ignore
except ImportError:  # pragma: no cover - fallback for older JAX
    from jax.core import Primitive as JaxPrimitive  # type: ignore


@jax.custom_vjp
def _square(x):
    return x * x


def _square_fwd(x):
    return _square(x), x


def _square_bwd(res, g):
    x = res
    return (2 * x * g,)


_square.defvjp(_square_fwd, _square_bwd)


@register_primitive(
    jaxpr_primitive="custom_vjp_call",
    jax_doc="Generic passthrough for custom VJP calls",
    onnx=[],
    since="0.7.1",
    context="primitives.core",
    component="custom_vjp_generic",
    testcases=[
        {
            "testcase": "custom_vjp_square",
            "callable": lambda x: _square(x),
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                ["Mul:3"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class CustomVjpCallPlugin(PrimitiveLeafPlugin):
    """Inline the body of a ``custom_vjp_call`` primitive into the current IR."""

    _PRIM: ClassVar[JaxPrimitive | None] = None

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        closed = eqn.params.get("call_jaxpr")
        if closed is None:
            raise ValueError("custom_vjp_call missing call_jaxpr parameter")
        inner_jaxpr = closed.jaxpr if hasattr(closed, "jaxpr") else closed
        consts = getattr(closed, "consts", eqn.params.get("consts", ()))

        for const_var, const_val in zip(inner_jaxpr.constvars, consts):
            ctx.bind_const_for_var(const_var, np.asarray(const_val))

        for outer_var, inner_var in zip(eqn.invars, inner_jaxpr.invars):
            ctx.bind_value_for_var(inner_var, ctx.get_value_for_var(outer_var))

        for inner_eqn in inner_jaxpr.eqns:
            prim_name = inner_eqn.primitive.name
            plugin = PLUGIN_REGISTRY.get(prim_name)
            if plugin is None:
                raise NotImplementedError(
                    f"No plugins registered for primitive '{prim_name}' inside custom_vjp body"
                )
            plugin.lower(ctx, inner_eqn)

        for outer_var, inner_var in zip(eqn.outvars, inner_jaxpr.outvars):
            ctx.bind_value_for_var(outer_var, ctx.get_value_for_var(inner_var))
