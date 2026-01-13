# jax2onnx/plugins/jax/lax/ragged_dot_general.py

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import onnx_ir as ir

from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.jax.lax.dot_general import DotGeneralPlugin
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _adjust_axes(
    axes: tuple[int, ...], removed_axes: tuple[int, ...]
) -> tuple[int, ...]:
    adjusted = []
    for axis in axes:
        if axis in removed_axes:
            raise ValueError(
                "ragged_dot_general: rhs group dimension overlaps batch/contract axes"
            )
        shift = sum(1 for rem in removed_axes if rem < axis)
        adjusted.append(axis - shift)
    return tuple(adjusted)


def _gather_group_zero(
    ctx: "IRContext",
    value: ir.Value,
    axis: int,
    name_hint: str,
    shape: list[object],
) -> tuple[ir.Value, list[object]]:
    idx = _const_i64(ctx, np.asarray(0, dtype=np.int64), name_hint=f"{name_hint}_idx")
    out = ctx.builder.Gather(
        value,
        idx,
        axis=axis,
        _outputs=[ctx.fresh_name(name_hint)],
    )
    dtype = getattr(getattr(value, "type", None), "dtype", None)
    if dtype is not None:
        out.type = ir.TensorType(dtype)
        out.dtype = dtype
    if shape:
        new_shape = shape[:axis] + shape[axis + 1 :]
        _stamp_type_and_shape(out, tuple(new_shape))
        _ensure_value_metadata(ctx, out)
        return out, new_shape
    _ensure_value_metadata(ctx, out)
    return out, shape


@register_primitive(
    jaxpr_primitive="ragged_dot_general",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.ragged_dot_general.html",
    onnx=[{"component": "Einsum/MatMul", "doc": "https://onnx.ai/onnx/operators/"}],
    since="0.7.1",
    context="primitives.lax",
    component="ragged_dot_general",
)
class RaggedDotGeneralPlugin(PrimitiveLeafPlugin):
    """Lower ragged_dot_general by slicing to a single-group dot_general."""

    _PRIM: ClassVar[object | None] = None

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        if len(eqn.invars) < 3:
            raise ValueError("ragged_dot_general expects lhs, rhs, group_sizes inputs")
        lhs_var, rhs_var, group_sizes_var = eqn.invars[:3]
        out_var = eqn.outvars[0]
        if len(eqn.invars) > 3:
            raise NotImplementedError(
                "ragged_dot_general group_offset input is not supported"
            )

        params = getattr(eqn, "params", {})
        rdn = params.get("ragged_dot_dimension_numbers")
        if rdn is None:
            raise ValueError("ragged_dot_general missing ragged_dot_dimension_numbers")
        if params.get("group_offset", None) is not None:
            raise NotImplementedError(
                "ragged_dot_general with group_offset is not supported"
            )

        ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) = (
            rdn.dot_dimension_numbers
        )
        lhs_contract = tuple(lhs_contract)
        rhs_contract = tuple(rhs_contract)
        lhs_batch = tuple(lhs_batch)
        rhs_batch = tuple(rhs_batch)

        lhs_ragged = tuple(getattr(rdn, "lhs_ragged_dimensions", ()))
        rhs_group = tuple(getattr(rdn, "rhs_group_dimensions", ()))
        if len(lhs_ragged) != len(rhs_group):
            raise ValueError("ragged_dot_general ragged/group dimension mismatch")

        group_sizes_shape = tuple(getattr(group_sizes_var.aval, "shape", ()))
        if len(group_sizes_shape) != 1 or group_sizes_shape[0] != 1:
            raise NotImplementedError(
                "ragged_dot_general only supports a single group size"
            )

        lhs_val = ctx.get_value_for_var(lhs_var, name_hint=ctx.fresh_name("ragged_lhs"))
        rhs_val = ctx.get_value_for_var(rhs_var, name_hint=ctx.fresh_name("ragged_rhs"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("ragged_out")
        )

        lhs_shape = tuple(getattr(lhs_var.aval, "shape", ()))
        rhs_shape = list(getattr(rhs_var.aval, "shape", ()))
        out_shape = tuple(getattr(out_var.aval, "shape", ()))

        if rhs_group:
            removed = tuple(sorted(rhs_group))
            rhs_contract = _adjust_axes(rhs_contract, removed)
            rhs_batch = _adjust_axes(rhs_batch, removed)

            removed_count = 0
            for axis in sorted(rhs_group):
                adj_axis = axis - removed_count
                rhs_val, rhs_shape = _gather_group_zero(
                    ctx,
                    rhs_val,
                    adj_axis,
                    "ragged_rhs_group",
                    rhs_shape,
                )
                removed_count += 1
        rhs_shape = tuple(rhs_shape)

        dot_plugin = DotGeneralPlugin()
        if dot_plugin._maybe_lower_complex(
            ctx,
            lhs_var,
            rhs_var,
            out_var,
            lhs_val,
            rhs_val,
            out_spec,
            lhs_shape,
            rhs_shape,
            out_shape,
            lhs_contract,
            rhs_contract,
            lhs_batch,
            rhs_batch,
        ):
            return

        if dot_plugin._try_lower_matmul(
            ctx,
            lhs_var,
            rhs_var,
            out_var,
            lhs_val,
            rhs_val,
            out_spec,
            lhs_shape,
            rhs_shape,
            out_shape,
            lhs_contract,
            rhs_contract,
            lhs_batch,
            rhs_batch,
        ):
            return

        dot_plugin._lower_via_einsum(
            ctx,
            lhs_var,
            rhs_var,
            out_var,
            lhs_val,
            rhs_val,
            out_spec,
            lhs_shape,
            rhs_shape,
            out_shape,
            lhs_contract,
            rhs_contract,
            lhs_batch,
            rhs_batch,
        )
