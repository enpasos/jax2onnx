# jax2onnx/plugins/jax/lax/_index_utils.py

"""Shared helpers for index-heavy lax primitives in plugins."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import onnx_ir as ir

from jax2onnx.plugins._ir_shapes import _ensure_value_info, _stamp_type_and_shape


def _const_i64(
    ctx: Any,
    values: Any,
    name_hint: str | None = None,
    *,
    name: str | None = None,
) -> ir.Value:
    """Create an INT64 initializer (scalar or vector) with a fresh name."""

    if name is not None:
        name_hint = name
    if name_hint is None:
        raise TypeError("_const_i64 requires either a positional name_hint or 'name='")

    arr = np.asarray(values, dtype=np.int64)
    builder = getattr(ctx, "builder", None)
    base_name = ctx.fresh_name(name_hint) if hasattr(ctx, "fresh_name") else name_hint

    builder_mode = (
        bool(getattr(builder, "_function_mode", False))
        if builder is not None
        else False
    )
    inside_function = bool(
        getattr(ctx, "_inside_function_scope", False)
        or getattr(ctx, "_function_mode", False)
    )
    if builder is not None and not inside_function and not builder_mode:
        add_initializer = getattr(builder, "add_initializer_from_array", None)
        if callable(add_initializer):
            return add_initializer(base_name, arr)

    tensor_obj = ir.tensor(arr)
    shape = () if arr.ndim == 0 else tuple(int(d) for d in arr.shape)
    val = ir.Value(
        name=base_name,
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape(shape),
        const_value=tensor_obj,
    )

    handler = getattr(ctx, "_handle_initializer_append", None)
    if callable(handler):
        handler(val)
        return val

    init_list = getattr(ctx, "_initializers", None)
    if init_list is not None and hasattr(init_list, "append"):
        init_list.append(val)
        return val

    if builder is not None:
        builder_inits = getattr(builder, "initializers", None)
        if isinstance(builder_inits, list):
            builder_inits.append(val)
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
    "_const_i64",
    "_scalar_i64",
    "_cast_to_i64",
    "_infer_rank",
]
