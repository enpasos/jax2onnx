# jax2onnx/plugins/jax/lax/_reduce_utils.py

"""Shared helpers for reduction primitives in plugins."""

from __future__ import annotations

from typing import Any, Final, Iterable, Optional, Sequence

import numpy as np
import onnx_ir as ir

from jax import core

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.jax.lax._index_utils import _const_i64, _scalar_i64


_REDUCESUM_INT64_WORK_DTYPES: Final[frozenset[np.dtype[Any]]] = frozenset(
    {
        np.dtype(np.uint8),
        np.dtype(np.uint16),
        np.dtype(np.uint32),
    }
)


def _normalize_axes(
    axes: Optional[Iterable[int]], rank: int
) -> Optional[tuple[int, ...]]:
    if axes is None:
        return None
    normalized: list[int] = []
    for ax in axes:
        ax_int = int(ax)
        if ax_int < 0:
            ax_int += rank
        if ax_int < 0 or ax_int >= rank:
            raise ValueError(f"reduction axis {ax} out of range for rank {rank}")
        normalized.append(ax_int)
    return tuple(normalized)


def _maybe_cast_input(
    ctx: Any,
    tensor: ir.Value,
    aval_shape: Sequence[Any],
    dtype: Optional[np.dtype],
) -> ir.Value:
    if dtype is None:
        return tensor

    dtype_enum = _dtype_to_ir(dtype, ctx.builder.enable_double_precision)
    cast_val = ctx.builder.Cast(
        tensor,
        _outputs=[ctx.fresh_name("reduce_cast")],
        to=int(dtype_enum.value),
    )
    cast_val.type = ir.TensorType(dtype_enum)
    cast_val.shape = tensor.shape
    _stamp_type_and_shape(cast_val, tuple(aval_shape))
    _ensure_value_metadata(ctx, cast_val)
    return cast_val


def lower_reduction(
    ctx: LoweringContextProtocol,
    eqn: core.JaxprEqn,
    *,
    op_type: str,
    allow_dtype_param: bool = True,
) -> None:
    operand_var = eqn.invars[0]
    out_var = eqn.outvars[0]

    params = getattr(eqn, "params", {})
    axes = params.get("axes")
    keepdims = bool(params.get("keepdims", False))

    requested_dtype = params.get("dtype") if allow_dtype_param else None
    if requested_dtype is not None:
        requested_dtype = np.dtype(requested_dtype)
    operand_dtype_param = getattr(getattr(operand_var, "aval", None), "dtype", None)
    operand_dtype = (
        np.dtype(operand_dtype_param) if operand_dtype_param is not None else None
    )

    work_dtype = requested_dtype
    needs_result_cast = False
    effective_dtype = requested_dtype or operand_dtype
    if op_type == "ReduceSum" and effective_dtype in _REDUCESUM_INT64_WORK_DTYPES:
        work_dtype = np.dtype(np.int64)
        needs_result_cast = True

    operand_val = ctx.get_value_for_var(
        operand_var, name_hint=ctx.fresh_name(f"{op_type.lower()}_in")
    )
    out_val = ctx.get_value_for_var(
        out_var, name_hint=ctx.fresh_name(f"{op_type.lower()}_out")
    )

    operand_shape = tuple(getattr(operand_var.aval, "shape", ()))
    axes_attr = _normalize_axes(axes, len(operand_shape))

    reduced_input = _maybe_cast_input(
        ctx,
        operand_val,
        operand_shape,
        work_dtype,
    )

    inputs = [reduced_input]
    if axes_attr is not None:
        axes_const = _const_i64(ctx, list(axes_attr), f"{op_type.lower()}_axes")
        inputs.append(axes_const)

    desired_name = getattr(out_val, "name", None) or ctx.fresh_name(op_type)
    producer = getattr(out_val, "producer", lambda: None)
    if callable(producer) and producer() is not None:
        desired_name = ctx.fresh_name(op_type)

    keepdims_attr = 1 if keepdims else 0
    reduce_outputs = [ctx.fresh_name(op_type)] if needs_result_cast else [desired_name]
    if op_type == "ReduceSum":
        result = ctx.builder.ReduceSum(
            *inputs,
            keepdims=keepdims_attr,
            _outputs=reduce_outputs,
        )
    elif op_type == "ReduceProd":
        result = ctx.builder.ReduceProd(
            *inputs,
            keepdims=keepdims_attr,
            _outputs=reduce_outputs,
        )
    elif op_type == "ReduceMax":
        result = ctx.builder.ReduceMax(
            *inputs,
            keepdims=keepdims_attr,
            _outputs=reduce_outputs,
        )
    elif op_type == "ReduceMin":
        result = ctx.builder.ReduceMin(
            *inputs,
            keepdims=keepdims_attr,
            _outputs=reduce_outputs,
        )
    elif op_type == "ReduceL1":
        result = ctx.builder.ReduceL1(
            *inputs,
            keepdims=keepdims_attr,
            _outputs=reduce_outputs,
        )
    elif op_type == "ReduceL2":
        result = ctx.builder.ReduceL2(
            *inputs,
            keepdims=keepdims_attr,
            _outputs=reduce_outputs,
        )
    elif op_type == "ReduceLogSum":
        result = ctx.builder.ReduceLogSum(
            *inputs,
            keepdims=keepdims_attr,
            _outputs=reduce_outputs,
        )
    elif op_type == "ReduceLogSumExp":
        result = ctx.builder.ReduceLogSumExp(
            *inputs,
            keepdims=keepdims_attr,
            _outputs=reduce_outputs,
        )
    elif op_type == "ReduceSumSquare":
        result = ctx.builder.ReduceSumSquare(
            *inputs,
            keepdims=keepdims_attr,
            _outputs=reduce_outputs,
        )
    else:
        raise ValueError(f"Unsupported reduction op: {op_type}")

    out_shape = tuple(getattr(out_var.aval, "shape", ()))
    aval_dtype = getattr(out_var.aval, "dtype", None)
    out_dtype_enum = None
    if aval_dtype is not None:
        out_dtype_enum = _dtype_to_ir(
            np.dtype(aval_dtype), ctx.builder.enable_double_precision
        )

    if needs_result_cast:
        result.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)

        target_dtype = out_dtype_enum or _dtype_to_ir(
            np.dtype(effective_dtype), ctx.builder.enable_double_precision
        )
        cast_result = ctx.builder.Cast(
            result,
            _outputs=[desired_name],
            to=int(target_dtype.value),
        )
        cast_result.type = ir.TensorType(target_dtype)
        _stamp_type_and_shape(cast_result, out_shape)
        _ensure_value_metadata(ctx, cast_result)
        ctx.bind_value_for_var(out_var, cast_result)
        return

    if out_dtype_enum is not None:
        result.type = ir.TensorType(out_dtype_enum)
    _stamp_type_and_shape(result, out_shape)

    _ensure_value_metadata(ctx, result)
    ctx.bind_value_for_var(out_var, result)


def lower_boolean_reduction(
    ctx: LoweringContextProtocol, eqn: core.JaxprEqn, *, mode: str
) -> None:
    operand_var = eqn.invars[0]
    out_var = eqn.outvars[0]

    params = getattr(eqn, "params", {})
    axes = params.get("axes")
    keepdims = bool(params.get("keepdims", False))

    operand_val = ctx.get_value_for_var(
        operand_var, name_hint=ctx.fresh_name(f"{mode}_in")
    )
    ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name(f"{mode}_out"))

    operand_shape = tuple(getattr(operand_var.aval, "shape", ()))
    axes_attr = _normalize_axes(axes, len(operand_shape))

    int_operand = _maybe_cast_input(ctx, operand_val, operand_shape, np.dtype(np.int64))

    out_shape = tuple(getattr(out_var.aval, "shape", ()))
    keepdims_attr = 1 if keepdims else 0

    inputs = [int_operand]
    if axes_attr is not None:
        axes_const = _const_i64(ctx, list(axes_attr), f"{mode}_axes")
        inputs.append(axes_const)

    if mode == "reduce_xor":
        reduce_out = ctx.builder.ReduceSum(
            *inputs,
            keepdims=keepdims_attr,
            _outputs=[ctx.fresh_name("ReduceSum")],
        )
    elif mode == "reduce_or":
        reduce_out = ctx.builder.ReduceMax(
            *inputs,
            keepdims=keepdims_attr,
            _outputs=[ctx.fresh_name("ReduceMax")],
        )
    else:
        reduce_out = ctx.builder.ReduceMin(
            *inputs,
            keepdims=keepdims_attr,
            _outputs=[ctx.fresh_name("ReduceMin")],
        )
    reduce_out.type = ir.TensorType(ir.DataType.INT64)
    _stamp_type_and_shape(reduce_out, out_shape)
    _ensure_value_metadata(ctx, reduce_out)

    if mode == "reduce_xor":
        two_const = _scalar_i64(ctx, 2, f"{mode}_two")
        mod_out = ctx.builder.Mod(
            reduce_out,
            two_const,
            fmod=0,
            _outputs=[ctx.fresh_name(f"{mode}_mod")],
        )
        mod_out.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(mod_out, out_shape)
        _ensure_value_metadata(ctx, mod_out)

        one_const = _scalar_i64(ctx, 1, f"{mode}_one")
        result = ctx.builder.Equal(
            mod_out,
            one_const,
            _outputs=[ctx.fresh_name(f"{mode}_eq")],
        )
        result.type = ir.TensorType(ir.DataType.BOOL)
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)
        return
    else:
        result = ctx.builder.Cast(
            reduce_out,
            _outputs=[ctx.fresh_name(f"{mode}_cast")],
            to=int(ir.DataType.BOOL.value),
        )
        result.type = ir.TensorType(ir.DataType.BOOL)
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)
