# jax2onnx/plugins/jax/lax/shard_map.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Tuple, cast

import numpy as np
import jax
from jax.sharding import Mesh, PartitionSpec as P

from jax2onnx.converter.output_binding import (
    assert_eqn_outputs_bound,
    bind_returned_lowering_values,
)
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import (
    PLUGIN_REGISTRY,
    PrimitiveLeafPlugin,
    register_primitive,
)

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _extract_inner_jaxpr(params: dict[str, Any]) -> Tuple[Any, Iterable[Any]]:
    inner = params.get("call_jaxpr") or params.get("jaxpr")
    if inner is None:
        raise ValueError("shard_map parameters missing inner jaxpr")
    if hasattr(inner, "jaxpr") and hasattr(inner, "consts"):
        return inner.jaxpr, getattr(inner, "consts")
    return inner, params.get("consts", ())


def _single_device_mesh() -> Mesh:
    devices = np.array(jax.devices()).reshape((jax.device_count(),))
    return Mesh(devices, ("d",))


def _shard_map_inline_add(x: np.ndarray) -> np.ndarray:
    mesh = _single_device_mesh()

    @jax.shard_map(mesh=mesh, in_specs=(P(),), out_specs=P())
    def inner(val: np.ndarray) -> np.ndarray:
        return val + 1

    with mesh:
        return cast(np.ndarray, inner(x))


@register_primitive(
    jaxpr_primitive="shard_map",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.shard_map.html",
    onnx=[],
    since="0.10.2",
    context="primitives.lax",
    component="shard_map",
    testcases=[
        {
            "testcase": "shard_map_inline_add",
            "callable": _shard_map_inline_add,
            "input_values": [np.array([2.0], dtype=np.float32)],
            "expected_output_shapes": [(1,)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(["Add:1"], no_unused_inputs=True),
        }
    ],
)
class ShardMapPlugin(PrimitiveLeafPlugin):
    """Inline the body of a ``shard_map`` call directly into the current IR context."""

    def lower(self, ctx: "IRContext", eqn: Any) -> None:
        inner_jaxpr, consts = _extract_inner_jaxpr(eqn.params)

        for var, const_val in zip(inner_jaxpr.constvars, consts):
            ctx.bind_const_for_var(var, np.asarray(const_val))

        for outer_var, inner_var in zip(eqn.invars, inner_jaxpr.invars):
            ctx.bind_value_for_var(inner_var, ctx.get_value_for_var(outer_var))

        for inner_eqn_index, inner_eqn in enumerate(inner_jaxpr.eqns):
            prim = inner_eqn.primitive.name
            plugin = PLUGIN_REGISTRY.get(prim)
            if plugin is None:
                raise NotImplementedError(
                    f"[shard_map] No plugins registered for primitive '{prim}' inside shard_map body"
                )
            lowering_result = plugin.lower(ctx, inner_eqn)
            bind_returned_lowering_values(
                ctx,
                inner_eqn,
                lowering_result,
                primitive_name=prim,
            )
            assert_eqn_outputs_bound(
                ctx,
                inner_eqn,
                primitive_name=prim,
                eqn_index=inner_eqn_index,
            )

        for outer_var, inner_var in zip(eqn.outvars, inner_jaxpr.outvars):
            ctx.bind_value_for_var(outer_var, ctx.get_value_for_var(inner_var))
