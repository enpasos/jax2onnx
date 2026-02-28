# jax2onnx/plugins/jax/lax/custom_linear_solve.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

import jax
import numpy as np

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.core.jit import JitPlugin
from jax2onnx.plugins.plugin_system import (
    PLUGIN_REGISTRY,
    PrimitiveLeafPlugin,
    register_primitive,
)

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _flatten_length_info(obj: Any) -> tuple[int, int, int, int]:
    if obj is None:
        return (0, 0, 0, 0)
    names = ("matvec", "vecmat", "solve", "transpose_solve")
    if all(hasattr(obj, n) for n in names):
        return tuple(int(getattr(obj, n)) for n in names)
    if isinstance(obj, (tuple, list)) and len(obj) == 4:
        return tuple(int(v) for v in obj)
    raise TypeError(f"Unsupported const_lengths payload: {obj!r}")


def _as_closed_jaxpr(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "jaxpr") and hasattr(obj, "consts"):
        return obj
    # raw jaxpr without explicit consts
    return type("ClosedLike", (), {"jaxpr": obj, "consts": ()})()


def _slice_vars(seq: Iterable[Any], start: int, length: int) -> list[Any]:
    items = list(seq)
    if length <= 0:
        return []
    return items[start : start + length]


@register_primitive(
    jaxpr_primitive="custom_linear_solve",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html",
    onnx=[],
    since="0.12.1",
    context="primitives.lax",
    component="custom_linear_solve",
    testcases=[
        {
            "testcase": "custom_linear_solve_via_matvec",
            "callable": lambda a, b: jax.lax.custom_linear_solve(
                lambda x: a @ x,
                b,
                solve=lambda mv, rhs: mv(mv(rhs)),
            ),
            "input_values": [
                np.asarray([[2.0, 1.0], [0.0, 3.0]], dtype=np.float32),
                np.asarray([1.0, -1.0], dtype=np.float32),
            ],
            "post_check_onnx_graph": EG(
                ["Einsum -> Einsum"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class CustomLinearSolvePlugin(PrimitiveLeafPlugin):
    """Inline the `solve` branch of ``lax.custom_linear_solve`` for inference graphs."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        params = dict(getattr(eqn, "params", {}) or {})
        const_lengths = _flatten_length_info(params.get("const_lengths"))
        matvec_nconsts, vecmat_nconsts, solve_nconsts, transpose_nconsts = const_lengths

        jaxprs = params.get("jaxprs")
        if jaxprs is None or len(jaxprs) != 4:
            raise ValueError("custom_linear_solve missing expected jaxprs tuple")
        _matvec_jaxpr, vecmat_jaxpr, solve_jaxpr, _transpose_jaxpr = jaxprs

        # Forward conversion only needs `solve`. We reject variants that require vecmat consts.
        if vecmat_jaxpr is not None and vecmat_nconsts > 0:
            raise NotImplementedError(
                "custom_linear_solve with vecmat constants is not yet supported"
            )
        if transpose_nconsts > 0:
            raise NotImplementedError(
                "custom_linear_solve with transpose_solve constants is not yet supported"
            )
        if solve_jaxpr is None:
            raise NotImplementedError("custom_linear_solve requires a solve jaxpr")

        invars = list(eqn.invars)
        total_consts = (
            matvec_nconsts + vecmat_nconsts + solve_nconsts + transpose_nconsts
        )
        if total_consts > len(invars):
            raise ValueError(
                "custom_linear_solve const_lengths exceed number of inputs"
            )

        rhs_vars = invars[total_consts:]
        solve_const_start = matvec_nconsts + vecmat_nconsts
        solve_const_vars = _slice_vars(invars, solve_const_start, solve_nconsts)
        solve_inputs = solve_const_vars + rhs_vars

        solve_closed = _as_closed_jaxpr(solve_jaxpr)
        fresh_closed = JitPlugin._freshen_closed_jaxpr(solve_closed)
        inner_jaxpr = fresh_closed.jaxpr
        consts = fresh_closed.consts

        for cvar, cval in zip(inner_jaxpr.constvars, consts):
            ctx.bind_const_for_var(cvar, np.asarray(cval))

        if len(solve_inputs) != len(inner_jaxpr.invars):
            raise ValueError(
                "custom_linear_solve solve input arity mismatch: "
                f"got {len(solve_inputs)}, expected {len(inner_jaxpr.invars)}"
            )
        for outer_var, inner_var in zip(solve_inputs, inner_jaxpr.invars):
            ctx.bind_value_for_var(inner_var, ctx.get_value_for_var(outer_var))

        for inner_eqn in inner_jaxpr.eqns:
            prim_name = inner_eqn.primitive.name
            plugin = PLUGIN_REGISTRY.get(prim_name)
            if plugin is None:
                raise NotImplementedError(
                    f"[custom_linear_solve] No plugin registered for primitive '{prim_name}' in solve body"
                )
            plugin.lower(ctx, inner_eqn)

        if len(eqn.outvars) != len(inner_jaxpr.outvars):
            raise ValueError(
                "custom_linear_solve output arity mismatch with solve jaxpr outputs"
            )
        for outer_var, inner_var in zip(eqn.outvars, inner_jaxpr.outvars):
            ctx.bind_value_for_var(outer_var, ctx.get_value_for_var(inner_var))
