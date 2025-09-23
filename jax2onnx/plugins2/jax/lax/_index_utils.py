"""Shared helpers for index-heavy lax primitives in plugins2."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import onnx_ir as ir

from jax2onnx.plugins2._ir_shapes import _ensure_value_info, _stamp_type_and_shape


def _append_initializer(ctx: Any, value: ir.Value) -> None:
    """Best-effort append of an initializer value to the active context."""
    if value is None:
        return
    try:
        inits = getattr(ctx, "_initializers", None)
        if inits is not None and hasattr(inits, "append"):
            inits.append(value)
            return
    except Exception:
        pass
    builder = getattr(ctx, "builder", None)
    if builder is not None:
        try:
            binits = getattr(builder, "initializers", None)
            if isinstance(binits, list):
                binits.append(value)
        except Exception:
            pass


def _const_i64(ctx: Any, values: Any, name_hint: str) -> ir.Value:
    """Create an INT64 initializer (scalar or vector) with a fresh name."""
    arr = np.asarray(values, dtype=np.int64)
    shape = () if arr.ndim == 0 else tuple(int(d) for d in arr.shape)
    val = ir.Value(
        name=ctx.fresh_name(name_hint),
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape(shape),
        const_value=ir.tensor(arr),
    )
    _append_initializer(ctx, val)
    return val


def _scalar_i64(ctx: Any, value: int, name_hint: str) -> ir.Value:
    return _const_i64(ctx, np.asarray(value, dtype=np.int64), name_hint)


def _cast_to_i64(ctx: Any, tensor_val: ir.Value, name_hint: str) -> ir.Value:
    """Cast the provided value to INT64 using CastLike (scalar safe)."""
    exemplar = _scalar_i64(ctx, 0, f"{name_hint}_exemplar")
    out = ctx.cast_like(tensor_val, exemplar, name_hint=name_hint)
    out.dtype = getattr(exemplar, "dtype", ir.DataType.INT64)
    shape_obj = getattr(tensor_val, "shape", None)
    dims: Sequence[Any] | None = None
    if shape_obj is None:
        dims = ()
    else:
        dims_attr = getattr(shape_obj, "dims", None)
        if dims_attr is not None:
            try:
                dims = tuple(dims_attr)
            except Exception:
                dims = tuple(dims_attr)
        else:
            try:
                dims = tuple(shape_obj)  # type: ignore[arg-type]
            except Exception:
                dims = ()
    _stamp_type_and_shape(out, dims or ())
    _ensure_value_info(ctx, out)
    return out


def _infer_rank(value: ir.Value, axis_hint: int) -> int:
    """Best-effort rank extraction with a fallback using the axis hint."""
    rank = None
    shape_obj = getattr(value, "shape", None)
    if shape_obj is not None:
        dims = getattr(shape_obj, "dims", None)
        if dims is not None:
            rank = len(dims)
        else:
            try:
                rank = len(tuple(shape_obj))
            except TypeError:
                rank = None
    if rank is None:
        type_obj = getattr(value, "type", None)
        if isinstance(type_obj, ir.TensorType):
            type_shape = getattr(type_obj, "shape", None)
            if type_shape is not None:
                dims = getattr(type_shape, "dims", None)
                if dims is not None:
                    rank = len(dims)
                else:
                    try:
                        rank = len(tuple(type_shape))
                    except TypeError:
                        rank = None
    if rank is None:
        aval = getattr(value, "aval", None)
        if aval is not None:
            rank = len(getattr(aval, "shape", ()) or ())
    if rank is None:
        rank = int(axis_hint) + 1
    return rank


__all__ = [
    "_append_initializer",
    "_const_i64",
    "_scalar_i64",
    "_cast_to_i64",
    "_infer_rank",
]
