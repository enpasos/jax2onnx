# jax2onnx/plugins/jax/lax/remat2.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Iterable, Tuple

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._control_flow_utils import lower_jaxpr_eqns
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _unwrap_closed_jaxpr(jaxpr_like: Any) -> Tuple[Any, Iterable[Any]]:
    if hasattr(jaxpr_like, "jaxpr") and hasattr(jaxpr_like, "consts"):
        return jaxpr_like.jaxpr, getattr(jaxpr_like, "consts")
    return jaxpr_like, ()


def _init_remat2_primitive() -> Any | None:
    existing = getattr(lax, "remat2_p", None)
    if existing is not None:
        existing.multiple_results = True
        return existing
    base_core: Any | None = None
    try:  # pragma: no cover
        from jax.extend import core as jax_core_ext  # type: ignore

        base_core = jax_core_ext
    except ImportError:  # pragma: no cover
        try:
            from jax import core as jax_core

            base_core = jax_core
        except ImportError:
            base_core = None
    if base_core is None:
        return None
    remat2 = base_core.Primitive("remat2")
    remat2.multiple_results = True
    lax.remat2_p = remat2  # type: ignore[attr-defined]
    return remat2


_REMAT2_PRIM: Final[Any | None] = _init_remat2_primitive()


def _remat2_scalar_sin_chain(x: jax.Array) -> jax.Array:
    def body(v):
        first = jnp.sin(v)
        return jnp.sin(first)

    return jax.checkpoint(body)(x)


def _remat2_tuple_passthrough(
    a: jax.Array, b: jax.Array
) -> tuple[jax.Array, jax.Array]:
    def body(x, y):
        return x + jnp.cos(x), y

    return jax.checkpoint(body)(a, b)


@register_primitive(
    jaxpr_primitive="remat2",
    jax_doc="https://docs.jax.dev/en/latest/jep/11830-new-remat-checkpoint.html",
    onnx=[],
    since="v0.6.5",
    context="primitives.lax",
    component="remat2",
    testcases=[
        {
            "testcase": "remat2_scalar_sin_chain",
            "callable": _remat2_scalar_sin_chain,
            "input_values": [np.array(0.5, dtype=np.float32)],
            "expected_output_shapes": [()],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                ["Sin -> Sin"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "remat2_tuple_passthrough",
            "callable": _remat2_tuple_passthrough,
            "input_values": [
                np.array(1.0, dtype=np.float32),
                np.array(0.25, dtype=np.float32),
            ],
            "expected_output_shapes": [(), ()],
            "expected_output_dtypes": [np.float32, np.float32],
            "post_check_onnx_graph": EG(
                ["Add"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class Remat2Plugin(PrimitiveLeafPlugin):
    """Inline the inner jaxpr of ``lax.remat2`` into the surrounding context."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        jaxpr_like = eqn.params.get("jaxpr")
        if jaxpr_like is None:
            raise ValueError("remat2 lowering requires 'jaxpr' in params")
        inner_jaxpr, consts = _unwrap_closed_jaxpr(jaxpr_like)

        for const_var, const_value in zip(inner_jaxpr.constvars, consts):
            ctx.bind_const_for_var(const_var, np.asarray(const_value))

        for outer_var, inner_var in zip(eqn.invars, inner_jaxpr.invars):
            ctx.bind_value_for_var(inner_var, ctx.get_value_for_var(outer_var))

        lower_jaxpr_eqns(ctx, inner_jaxpr)

        for outer_var, inner_var in zip(eqn.outvars, inner_jaxpr.outvars):
            ctx.bind_value_for_var(outer_var, ctx.get_value_for_var(inner_var))
