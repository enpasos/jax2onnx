# jax2onnx/plugins/jax/core/custom_jvp_call.py

from __future__ import annotations

from typing import Any, ClassVar

from jax import core
import numpy as np

from jax2onnx.converter.output_binding import (
    assert_eqn_outputs_bound,
    bind_returned_lowering_values,
)
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import (
    PLUGIN_REGISTRY,
    PrimitiveLeafPlugin,
    register_primitive,
)


import jax

try:  # JAX 0.4+ lives in jax.extend.core
    from jax.extend.core import Primitive as JaxPrimitive
except ImportError:  # pragma: no cover - fallback for older JAX
    from jax.core import Primitive as JaxPrimitive


@jax.custom_jvp
def _square(x: Any) -> Any:
    return x * x


@_square.defjvp
def _square_jvp(primals: tuple[Any, ...], tangents: tuple[Any, ...]) -> tuple[Any, Any]:
    (x,), (t,) = primals, tangents
    return _square(x), 2 * x * t


@register_primitive(
    jaxpr_primitive="custom_jvp_call",
    jax_doc="Generic passthrough for custom JVP calls",
    onnx=[],
    since="0.7.1",
    context="primitives.core",
    component="custom_jvp_generic",
    testcases=[
        {
            "testcase": "custom_jvp_square",
            "callable": lambda x: _square(x),
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                ["Mul:3"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class CustomJvpCallPlugin(PrimitiveLeafPlugin):
    """Inline the body of a ``custom_jvp_call`` primitive into the current IR."""

    _PRIM: ClassVar[JaxPrimitive | None] = None

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        closed = eqn.params.get("call_jaxpr")
        if closed is None:
            raise ValueError("custom_jvp_call missing call_jaxpr parameter")
        inner_jaxpr = closed.jaxpr if hasattr(closed, "jaxpr") else closed
        consts = getattr(closed, "consts", eqn.params.get("consts", ()))

        for const_var, const_val in zip(inner_jaxpr.constvars, consts):
            ctx.bind_const_for_var(const_var, np.asarray(const_val))

        for outer_var, inner_var in zip(eqn.invars, inner_jaxpr.invars):
            ctx.bind_value_for_var(inner_var, ctx.get_value_for_var(outer_var))

        for inner_eqn_index, inner_eqn in enumerate(inner_jaxpr.eqns):
            prim_name = inner_eqn.primitive.name
            plugin = PLUGIN_REGISTRY.get(prim_name)
            if plugin is None:
                raise NotImplementedError(
                    f"No plugins registered for primitive '{prim_name}' inside custom_jvp body"
                )
            lowering_result = plugin.lower(ctx, inner_eqn)
            bind_returned_lowering_values(
                ctx,
                inner_eqn,
                lowering_result,
                primitive_name=prim_name,
            )
            assert_eqn_outputs_bound(
                ctx,
                inner_eqn,
                primitive_name=prim_name,
                eqn_index=inner_eqn_index,
            )

        for outer_var, inner_var in zip(eqn.outvars, inner_jaxpr.outvars):
            ctx.bind_value_for_var(outer_var, ctx.get_value_for_var(inner_var))
