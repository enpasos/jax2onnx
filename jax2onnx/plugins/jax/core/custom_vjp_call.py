# jax2onnx/plugins/jax/core/custom_vjp_call.py

from __future__ import annotations

from typing import Any, ClassVar

from jax import core
import numpy as np

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._control_flow_utils import lower_jaxpr_eqns
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_primitive,
)

import jax

try:  # JAX 0.4+ lives in jax.extend.core
    from jax.extend.core import Primitive as JaxPrimitive
except ImportError:  # pragma: no cover - fallback for older JAX
    from jax.core import Primitive as JaxPrimitive


@jax.custom_vjp
def _square(x: Any) -> Any:
    return x * x


def _square_fwd(x: Any) -> tuple[Any, Any]:
    return _square(x), x


def _square_bwd(res: Any, g: Any) -> tuple[Any]:
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

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        closed = eqn.params.get("call_jaxpr")
        if closed is None:
            raise ValueError("custom_vjp_call missing call_jaxpr parameter")
        inner_jaxpr = closed.jaxpr if hasattr(closed, "jaxpr") else closed
        consts = getattr(closed, "consts", eqn.params.get("consts", ()))

        for const_var, const_val in zip(inner_jaxpr.constvars, consts):
            ctx.bind_const_for_var(const_var, np.asarray(const_val))

        for outer_var, inner_var in zip(eqn.invars, inner_jaxpr.invars):
            ctx.bind_value_for_var(inner_var, ctx.get_value_for_var(outer_var))

        lower_jaxpr_eqns(ctx, inner_jaxpr, source="custom_vjp")

        for outer_var, inner_var in zip(eqn.outvars, inner_jaxpr.outvars):
            ctx.bind_value_for_var(outer_var, ctx.get_value_for_var(inner_var))
