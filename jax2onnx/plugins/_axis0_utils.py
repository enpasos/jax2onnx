# jax2onnx/plugins/_axis0_utils.py

from __future__ import annotations

from typing import Any
from types import SimpleNamespace

import os

import numpy as np

import onnx_ir as ir

from jax2onnx.plugins._ir_shapes import (
    _ensure_value_metadata,
    _stamp_type_and_shape,
    _to_ir_dim_for_shape,
)
from jax2onnx.plugins._loop_extent_meta import get_axis0_override, set_axis0_override
from jax2onnx.plugins.jax.lax._index_utils import _const_i64


def _axis0_debug_enabled() -> bool:
    flag = os.environ.get("J2O_DEBUG_AXIS0", "")
    if not flag:
        return False
    return flag.strip().lower() in ("1", "true", "yes", "on")


def _axis0_debug(message: str) -> None:
    if _axis0_debug_enabled():
        print(f"[axis0-debug] {message}", flush=True)


def ensure_axis0_extent(
    ctx: Any, value: Any, override: int | None, reference: Any | None = None
) -> Any:
    if override is None or override <= 1:
        _axis0_debug(
            f"ensure_axis0_extent skip override={override} value={getattr(value, 'name', None)}"
        )
        return value

    existing = get_axis0_override(value)
    if isinstance(existing, (int, np.integer)) and int(existing) == override:
        _axis0_debug(
            f"ensure_axis0_extent existing override matches override={override} value={getattr(value, 'name', None)}"
        )
        return value

    shape_obj = getattr(value, "shape", None)
    dims_tuple = getattr(shape_obj, "dims", None)
    dims = list(dims_tuple) if dims_tuple else None
    if (not dims or len(dims) == 0) and reference is not None:
        ref_shape = getattr(reference, "shape", None)
        ref_dims = getattr(ref_shape, "dims", None)
        if ref_dims and len(ref_dims) > 0:
            dims = list(ref_dims)
    if not dims or len(dims) == 0:
        _axis0_debug(
            f"ensure_axis0_extent no dims override={override} value={getattr(value, 'name', None)}"
        )
        return value
    if not dims_tuple or len(dims_tuple) == 0:
        try:
            stamp_dims = tuple(_to_ir_dim_for_shape(dim) for dim in dims)
            _stamp_type_and_shape(value, stamp_dims)
        except Exception:
            _axis0_debug(
                f"ensure_axis0_extent failed to stamp input shape value={getattr(value, 'name', None)}"
            )
    dim0 = dims[0]
    if isinstance(dim0, (int, np.integer)):
        dim0_int = int(dim0)
        if dim0_int == override:
            set_axis0_override(value, override)
            _axis0_debug(
                f"ensure_axis0_extent dim0 already {override} value={getattr(value, 'name', None)}"
            )
            return value
        if dim0_int != 1:
            _axis0_debug(
                f"ensure_axis0_extent dim0={dim0_int} incompatible with override={override} value={getattr(value, 'name', None)}"
            )
            return value
    else:
        _axis0_debug(
            f"ensure_axis0_extent non-static dim0 override={override} value={getattr(value, 'name', None)}"
        )

    rank = len(dims)
    override_vec = _const_i64(
        ctx,
        np.asarray([override], dtype=np.int64),
        ctx.fresh_name("axis0_override"),
    )

    _axis0_debug(
        "ensure_axis0_extent expanding "
        f"value={getattr(value, 'name', None)} override={override} dims={dims}"
    )
    if rank > 1:
        shape_tensor = ctx.builder.Shape(
            value,
            _outputs=[ctx.fresh_name("axis0_shape")],
        )
        shape_tensor.dtype = ir.DataType.INT64
        _stamp_type_and_shape(shape_tensor, (rank,))
        _ensure_value_metadata(ctx, shape_tensor)

        starts = _const_i64(
            ctx,
            np.asarray([1], dtype=np.int64),
            ctx.fresh_name("axis0_tail_starts"),
        )
        ends = _const_i64(
            ctx,
            np.asarray([np.iinfo(np.int64).max], dtype=np.int64),
            ctx.fresh_name("axis0_tail_ends"),
        )
        axes = _const_i64(
            ctx,
            np.asarray([0], dtype=np.int64),
            ctx.fresh_name("axis0_tail_axes"),
        )
        steps = _const_i64(
            ctx,
            np.asarray([1], dtype=np.int64),
            ctx.fresh_name("axis0_tail_steps"),
        )
        tail_shape = ctx.builder.Slice(
            shape_tensor,
            starts,
            ends,
            axes,
            steps,
            _outputs=[ctx.fresh_name("axis0_shape_tail")],
        )
        tail_shape.dtype = ir.DataType.INT64
        _stamp_type_and_shape(tail_shape, (max(rank - 1, 0),))
        _ensure_value_metadata(ctx, tail_shape)

        target_shape = ctx.builder.Concat(
            override_vec,
            tail_shape,
            axis=0,
            _outputs=[ctx.fresh_name("axis0_target_shape")],
        )
        target_shape.dtype = ir.DataType.INT64
        _stamp_type_and_shape(target_shape, (rank,))
        _ensure_value_metadata(ctx, target_shape)
    else:
        target_shape = override_vec
        _ensure_value_metadata(ctx, target_shape)

    expanded = ctx.builder.Expand(
        value,
        target_shape,
        _outputs=[ctx.fresh_name("axis0_expand")],
    )
    if getattr(value, "type", None) is not None:
        expanded.type = value.type
    try:
        original_dims = list(dims)
        if original_dims:
            original_dims[0] = override
            _stamp_type_and_shape(expanded, tuple(original_dims))
    except Exception:
        _axis0_debug(
            f"ensure_axis0_extent failed to stamp shape for value={getattr(expanded, 'name', None)}"
        )
    _ensure_value_metadata(ctx, expanded)
    set_axis0_override(expanded, override)
    _axis0_debug(
        f"ensure_axis0_extent produced expand value={getattr(expanded, 'name', None)} override={override}"
    )
    return expanded


def maybe_expand_binary_axis0(
    ctx: Any, lhs: Any, rhs: Any, out_val: Any, out_var: Any | None = None
):
    override_sources = [
        get_axis0_override(lhs),
        get_axis0_override(rhs),
        get_axis0_override(out_val),
        getattr(ctx, "_static_loop_extent_axis0", None),
    ]
    override_candidates = [
        int(val)
        for val in override_sources
        if isinstance(val, (int, np.integer)) and int(val) > 1
    ]
    override = max(override_candidates, default=None)

    _axis0_debug(
        "maybe_expand_binary_axis0 "
        f"lhs={getattr(lhs, 'name', None)} rhs={getattr(rhs, 'name', None)} "
        f"out={getattr(out_val, 'name', None)} override_candidates={override_candidates} "
        f"selected={override}"
    )

    if override is None:
        return lhs, rhs, None

    lhs = ensure_axis0_extent(ctx, lhs, override, reference=rhs or out_val)
    rhs = ensure_axis0_extent(ctx, rhs, override, reference=lhs or out_val)

    lhs_override = get_axis0_override(lhs)
    rhs_override = get_axis0_override(rhs)
    if lhs_override == override or rhs_override == override:
        return lhs, rhs, override

    fallback_lhs = ensure_axis0_extent(ctx, lhs, override, reference=out_val)
    fallback_rhs = ensure_axis0_extent(ctx, rhs, override, reference=out_val)
    lhs2_override = get_axis0_override(fallback_lhs)
    rhs2_override = get_axis0_override(fallback_rhs)
    if lhs2_override == override or rhs2_override == override:
        _axis0_debug(
            "maybe_expand_binary_axis0 override salvaged via out_spec "
            f"lhs_override={lhs2_override} rhs_override={rhs2_override} "
        )
        return fallback_lhs, fallback_rhs, override

    if out_var is not None:
        out_shape = tuple(getattr(getattr(out_var, "aval", None), "shape", ()) or ())
        if out_shape:
            fake_ref = SimpleNamespace(shape=SimpleNamespace(dims=out_shape))
            lhs_alt = ensure_axis0_extent(ctx, lhs, override, reference=fake_ref)
            rhs_alt = ensure_axis0_extent(ctx, rhs, override, reference=fake_ref)
            lhs3_override = get_axis0_override(lhs_alt)
            rhs3_override = get_axis0_override(rhs_alt)
            if lhs3_override == override or rhs3_override == override:
                _axis0_debug(
                    "maybe_expand_binary_axis0 override forced via target shape "
                    f"lhs_override={lhs3_override} rhs_override={rhs3_override} "
                )
                return lhs_alt, rhs_alt, override
    out_shape = ()
    if out_var is not None:
        out_shape = tuple(getattr(getattr(out_var, "aval", None), "shape", ()) or ())
    _axis0_debug(
        "maybe_expand_binary_axis0 override dropped "
        f"lhs_override={lhs_override} rhs_override={rhs_override} "
        f"selected={override} out_shape={out_shape}"
    )
    return lhs, rhs, None


def stamp_axis0_binary_result(
    result: Any, out_var: Any, out_spec: Any, override: int | None
) -> None:
    out_shape = tuple(getattr(getattr(out_var, "aval", None), "shape", ()) or ())
    if override is not None and out_shape:
        out_shape = (override,) + out_shape[1:]
    if out_shape:
        dims = tuple(_to_ir_dim_for_shape(dim) for dim in out_shape)
        _stamp_type_and_shape(result, dims)
    elif getattr(out_spec, "shape", None) is not None:
        result.shape = out_spec.shape
