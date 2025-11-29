# jax2onnx/plugins/jax/lax/_index_utils.py

"""Shared helpers for index-heavy lax primitives in plugins."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import onnx_ir as ir

from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape


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
    if builder is not None and (inside_function or builder_mode):
        tensor_obj = ir.tensor(arr)
        tensor_obj = ir.tensor(arr)
        val_name = ctx.fresh_name(name_hint)
        val = ir.Value(
            name=val_name,
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape(arr.shape),
            const_value=tensor_obj,
        )
        attributes = [ir.Attr("value", ir.AttributeType.TENSOR, tensor_obj)]
        node = ir.Node(
            op_type="Constant",
            domain="",
            inputs=[],
            outputs=[val],
            name=ctx.fresh_name("Constant"),
            attributes=attributes,
        )
        builder.nodes.append(node)
        _ensure_value_metadata(ctx, val)
        return val

    if builder is not None:
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
    """Cast the provided value to INT64 using Cast."""
    return _builder_op(
        ctx,
        "Cast",
        [tensor_val],
        name_hint=name_hint,
        dtype=ir.DataType.INT64,
        attributes={"to": int(ir.DataType.INT64.value)},
    )


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


def _builder_op(
    ctx: Any,
    op_type: str,
    inputs: Sequence[ir.Value | None],
    *,
    name_hint: str,
    dtype: ir.DataType | None = None,
    shape: Sequence[int | None] | None = None,
    attributes: Dict[str, Any] | None = None,
    output: ir.Value | None = None,
) -> ir.Value:
    """Invoke a builder op while preserving optional dtype/shape metadata."""

    builder = getattr(ctx, "builder", None)
    if builder is None:
        raise AttributeError("IR build context missing builder")

    attrs = dict(attributes or {})
    method = getattr(builder, op_type, None)

    if output is not None:
        if not getattr(output, "name", None):
            output.name = ctx.fresh_name(name_hint)
        result = builder.op(
            op_type,
            list(inputs),
            attrs,
            output=output,
            name=output.name,
        )
    else:
        out_name = ctx.fresh_name(name_hint)
        if callable(method):
            result = method(*inputs, _outputs=[out_name], **attrs)
        else:
            result = builder.op(op_type, list(inputs), attrs, name=out_name)

    if dtype is not None:
        result.type = ir.TensorType(dtype)
    if shape is not None:
        _stamp_type_and_shape(result, tuple(shape))
    _ensure_value_metadata(ctx, result)
    return result


def _shape_of(ctx: Any, value: ir.Value, name_hint: str) -> ir.Value:
    """Materialize ``Shape`` for ``value`` via the builder."""

    rank = _infer_rank(value, 0)
    return _builder_op(
        ctx,
        "Shape",
        [value],
        name_hint=name_hint,
        dtype=ir.DataType.INT64,
        shape=(rank,),
    )


def _gather_int_scalar(
    ctx: Any, shape_val: ir.Value, axis: int, name_hint: str
) -> ir.Value:
    """Gather a scalar INT64 entry from ``shape_val`` along ``axis``."""

    indices = _const_i64(ctx, np.asarray([axis], dtype=np.int64), f"{name_hint}_idx")
    gathered = _builder_op(
        ctx,
        "Gather",
        [shape_val, indices],
        name_hint=name_hint,
        dtype=ir.DataType.INT64,
        shape=(1,),
        attributes={"axis": 0},
    )

    axes = _const_i64(ctx, np.asarray([0], dtype=np.int64), f"{name_hint}_sq")
    scalar = _builder_op(
        ctx,
        "Squeeze",
        [gathered, axes],
        name_hint=f"{name_hint}_scalar",
        dtype=ir.DataType.INT64,
        shape=(),
    )
    return scalar


def _unsqueeze_scalar(
    ctx: Any, scalar: ir.Value, axis: int, name_hint: str
) -> ir.Value:
    """Unsqueeze ``scalar`` along ``axis`` (INT64 helper)."""

    axes = _const_i64(ctx, np.asarray([axis], dtype=np.int64), f"{name_hint}_axes")
    return _builder_op(
        ctx,
        "Unsqueeze",
        [scalar, axes],
        name_hint=name_hint,
        dtype=ir.DataType.INT64,
        shape=(1,),
    )


__all__ = [
    "_const_i64",
    "_scalar_i64",
    "_cast_to_i64",
    "_infer_rank",
    "_builder_op",
    "_shape_of",
    "_gather_int_scalar",
    "_unsqueeze_scalar",
]
