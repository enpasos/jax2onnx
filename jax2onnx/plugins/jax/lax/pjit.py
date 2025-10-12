# jax2onnx/plugins/jax/lax/pjit.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Tuple

from functools import partial
import numpy as np
import jax
from jax.experimental import mesh_utils, pjit
from jax.sharding import Mesh, PartitionSpec as P

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
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


def _single_device_mesh() -> Mesh:
    devices = mesh_utils.create_device_mesh((jax.device_count(),))
    return Mesh(devices, ("d",))


def _pjit_inline_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    mesh = _single_device_mesh()

    @partial(pjit.pjit, in_shardings=(P(), P()), out_shardings=P())
    def inner(x, y):
        return x * y

    with mesh:
        return inner(a, b)


def _pjit_inline_tuple(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mesh = _single_device_mesh()

    @partial(pjit.pjit, in_shardings=(P(), P()), out_shardings=(P(), P()))
    def inner(x, y):
        return x * y, x + y

    with mesh:
        return inner(a, b)


@register_primitive(
    jaxpr_primitive="pjit",
    jax_doc="https://jax.readthedocs.io/en/latest/jax.experimental.pjit.html",
    onnx=[],
    since="v0.1.0",
    context="primitives.lax",
    component="pjit",
    testcases=[
        {
            "testcase": "pjit_inline_mul",
            "callable": _pjit_inline_mul,
            "input_values": [
                np.array([2.0], dtype=np.float32),
                np.array([3.0], dtype=np.float32),
            ],
            "expected_output_shapes": [(1,)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                ["Mul:1"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "pjit_inline_tuple",
            "callable": _pjit_inline_tuple,
            "input_values": [
                np.array([4.0], dtype=np.float32),
                np.array([1.0], dtype=np.float32),
            ],
            "expected_output_shapes": [(1,), (1,)],
            "expected_output_dtypes": [np.float32, np.float32],
            "post_check_onnx_graph": EG(
                ["Add:1", "Mul:1"],
                no_unused_inputs=True,
            ),
        },
    ],
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
