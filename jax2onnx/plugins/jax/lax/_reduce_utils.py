# jax2onnx/plugins/jax/lax/_reduce_utils.py

"""Shared helpers for reduction primitives in plugins."""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, cast

import numpy as np
import onnx_ir as ir

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.ir_utils import numpy_dtype_to_ir
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx._compat.jax import JaxprEqn
from jax2onnx.plugins.jax.lax._index_utils import (
    _builder_op,
    _const_i64,
    _scalar_i64,
)
from jax2onnx.plugins.jax.lax._opset_utils import builder_reduce_with_axes


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
    ctx: LoweringContextProtocol,
    tensor: ir.Value,
    aval_shape: Sequence[Any],
    dtype: Optional[np.dtype],
    *,
    target_dtype: ir.DataType | None = None,
) -> ir.Value:
    if dtype is None:
        return tensor

    dtype_enum = target_dtype or numpy_dtype_to_ir(dtype)
    cast_val = cast(
        ir.Value,
        ctx.builder.Cast(
            tensor,
            _outputs=[ctx.fresh_name("reduce_cast")],
            to=int(dtype_enum.value),
        ),
    )
    cast_val.type = ir.TensorType(dtype_enum)
    cast_val.shape = tensor.shape
    _stamp_type_and_shape(cast_val, tuple(aval_shape))
    _ensure_value_metadata(ctx, cast_val)
    return cast_val


def _integer_reduce_sum(
    ctx: LoweringContextProtocol,
    operand: ir.Value,
    shape: tuple[Any, ...],
    axes: tuple[int, ...] | None,
    keepdims: bool,
    output_dtype: ir.DataType,
) -> ir.Value:
    # CPU ReduceSum saturates signed overflow. CumSum wraps, so its final
    # prefix preserves addition modulo the source width. UINT64 uses the
    # same-width signed representation; smaller unsupported types use INT64.
    work_dtype = np.dtype(np.int32 if output_dtype == ir.DataType.INT32 else np.int64)
    work_enum = numpy_dtype_to_ir(work_dtype)
    value = operand
    if value.dtype != work_enum:
        value = _builder_op(
            ctx,
            "Cast",
            [value],
            name_hint="reduce_sum_work",
            dtype=work_enum,
            shape=shape,
            attributes={"to": int(work_enum.value)},
        )

    reduced_axes = tuple(range(len(shape))) if axes is None else axes
    if reduced_axes:
        zero = ctx.builder.add_initializer_from_array(
            name=ctx.fresh_name("reduce_sum_zero"),
            array=np.asarray(0, dtype=work_dtype),
        )
        last = _const_i64(ctx, [-1] if keepdims else -1, "reduce_sum_last")
        # Reducing the highest axes first keeps the remaining indices valid.
        for axis in sorted(reduced_axes, reverse=True):
            pads = [0] * (2 * len(shape))
            pads[axis] = 1
            extent = shape[axis]
            padded_extent = (
                int(extent) + 1 if isinstance(extent, (int, np.integer)) else None
            )
            padded_shape = shape[:axis] + (padded_extent,) + shape[axis + 1 :]
            # A leading zero makes the final prefix valid for empty axes,
            # including symbolic dimensions that become zero at runtime.
            value = _builder_op(
                ctx,
                "Pad",
                [value, _const_i64(ctx, pads, "reduce_sum_pads"), zero],
                name_hint="reduce_sum_pad",
                dtype=work_enum,
                shape=padded_shape,
                attributes={"mode": "constant"},
            )
            value = _builder_op(
                ctx,
                "CumSum",
                [value, _const_i64(ctx, axis, "reduce_sum_axis")],
                name_hint="reduce_sum_prefix",
                dtype=work_enum,
                shape=padded_shape,
            )
            shape = shape[:axis] + ((1,) if keepdims else ()) + shape[axis + 1 :]
            value = _builder_op(
                ctx,
                "Gather",
                [value, last],
                name_hint="reduce_sum_integer",
                dtype=work_enum,
                shape=shape,
                attributes={"axis": axis},
            )
    if work_enum != output_dtype:
        value = _builder_op(
            ctx,
            "Cast",
            [value],
            name_hint="reduce_sum_result",
            dtype=output_dtype,
            shape=shape,
            attributes={"to": int(output_dtype.value)},
        )
    return value


def lower_reduction(
    ctx: LoweringContextProtocol,
    eqn: JaxprEqn,
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

    operand_val = ctx.get_value_for_var(
        operand_var, name_hint=ctx.fresh_name(f"{op_type.lower()}_in")
    )
    out_val = ctx.get_value_for_var(
        out_var, name_hint=ctx.fresh_name(f"{op_type.lower()}_out")
    )
    out_dtype_enum = numpy_dtype_to_ir(out_var.aval.dtype)

    operand_shape = tuple(getattr(operand_var.aval, "shape", ()))
    axes_attr = _normalize_axes(axes, len(operand_shape))

    reduced_input = _maybe_cast_input(
        ctx,
        operand_val,
        operand_shape,
        requested_dtype,
        target_dtype=out_dtype_enum,
    )
    if axes_attr == ():
        desired_name = getattr(out_val, "name", None) or ctx.fresh_name(op_type)
        producer = getattr(out_val, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name(op_type)
        result = cast(
            ir.Value,
            ctx.builder.Identity(
                reduced_input,
                _outputs=[desired_name],
            ),
        )
        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        result.type = ir.TensorType(out_dtype_enum)
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)
        return

    desired_name = getattr(out_val, "name", None) or ctx.fresh_name(op_type)
    producer = getattr(out_val, "producer", lambda: None)
    if callable(producer) and producer() is not None:
        desired_name = ctx.fresh_name(op_type)

    keepdims_attr = 1 if keepdims else 0
    if op_type == "ReduceSum" and np.issubdtype(out_var.aval.dtype, np.integer):
        result = _integer_reduce_sum(
            ctx,
            reduced_input,
            operand_shape,
            axes_attr,
            keepdims,
            out_dtype_enum,
        )
    elif op_type in {
        "ReduceL1",
        "ReduceL2",
        "ReduceLogSum",
        "ReduceLogSumExp",
        "ReduceMax",
        "ReduceMin",
        "ReduceProd",
        "ReduceSum",
        "ReduceSumSquare",
    }:
        result = builder_reduce_with_axes(
            ctx,
            reduced_input,
            op_type=op_type,
            axes=axes_attr,
            keepdims=keepdims_attr,
            name_hint=op_type.lower(),
            output_name=desired_name,
        )
    else:
        raise ValueError(f"Unsupported reduction op: {op_type}")

    out_shape = tuple(getattr(out_var.aval, "shape", ()))
    result.type = ir.TensorType(out_dtype_enum)
    _stamp_type_and_shape(result, out_shape)

    _ensure_value_metadata(ctx, result)
    ctx.bind_value_for_var(out_var, result)


def lower_boolean_reduction(
    ctx: LoweringContextProtocol, eqn: JaxprEqn, *, mode: str
) -> None:
    operand_var = eqn.invars[0]
    out_var = eqn.outvars[0]

    if not np.issubdtype(np.dtype(out_var.aval.dtype), np.bool_):
        raise NotImplementedError(
            "integer bitwise reduce_and/reduce_or/reduce_xor are not supported; "
            "the boolean reduction lowering only supports boolean outputs"
        )

    params = getattr(eqn, "params", {})
    axes = params.get("axes")
    keepdims = bool(params.get("keepdims", False))

    operand_val = ctx.get_value_for_var(
        operand_var, name_hint=ctx.fresh_name(f"{mode}_in")
    )
    ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name(f"{mode}_out"))

    operand_shape = tuple(getattr(operand_var.aval, "shape", ()))
    axes_attr = _normalize_axes(axes, len(operand_shape))
    operand_dtype = np.dtype(operand_var.aval.dtype)
    boolean_operand = operand_val
    if not np.issubdtype(operand_dtype, np.bool_):
        # jnp.any/all accept numeric inputs, whose truth value must be computed
        # elementwise before reducing. Reducing the raw numbers can cancel for
        # OR (for example [1, -1]) or hide a zero for AND ([0, -1]).
        boolean_operand = _maybe_cast_input(
            ctx,
            operand_val,
            operand_shape,
            np.dtype(np.bool_),
        )
    if axes_attr == ():
        desired_name = ctx.fresh_name(mode)
        result = cast(
            ir.Value,
            ctx.builder.Identity(
                boolean_operand,
                _outputs=[desired_name],
            ),
        )
        result.type = ir.TensorType(ir.DataType.BOOL)
        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)
        return

    int_operand = _maybe_cast_input(
        ctx,
        boolean_operand,
        operand_shape,
        np.dtype(np.int64),
    )

    out_shape = tuple(getattr(out_var.aval, "shape", ()))
    keepdims_attr = 1 if keepdims else 0

    if mode == "reduce_xor":
        reduce_op_type = "ReduceSum"
    elif mode == "reduce_or":
        # ReduceMax returns the integer dtype's minimum value for an empty
        # reduced dimension. Casting that non-zero identity back to bool would
        # incorrectly make ``any(empty)`` true. A sum has the required zero
        # identity and remains equivalent for non-empty boolean inputs.
        reduce_op_type = "ReduceSum"
    else:
        reduce_op_type = "ReduceMin"
    reduce_out = builder_reduce_with_axes(
        ctx,
        int_operand,
        op_type=reduce_op_type,
        axes=axes_attr,
        keepdims=keepdims_attr,
        name_hint=reduce_op_type,
    )
    reduce_out.type = ir.TensorType(ir.DataType.INT64)
    _stamp_type_and_shape(reduce_out, out_shape)
    _ensure_value_metadata(ctx, reduce_out)

    if mode == "reduce_xor":
        two_const = _scalar_i64(ctx, 2, f"{mode}_two")
        mod_out = cast(
            ir.Value,
            ctx.builder.Mod(
                reduce_out,
                two_const,
                fmod=0,
                _outputs=[ctx.fresh_name(f"{mode}_mod")],
            ),
        )
        mod_out.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(mod_out, out_shape)
        _ensure_value_metadata(ctx, mod_out)

        one_const = _scalar_i64(ctx, 1, f"{mode}_one")
        result = cast(
            ir.Value,
            ctx.builder.Equal(
                mod_out,
                one_const,
                _outputs=[ctx.fresh_name(f"{mode}_eq")],
            ),
        )
        result.type = ir.TensorType(ir.DataType.BOOL)
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)
        return
    else:
        result = cast(
            ir.Value,
            ctx.builder.Cast(
                reduce_out,
                _outputs=[ctx.fresh_name(f"{mode}_cast")],
                to=int(ir.DataType.BOOL.value),
            ),
        )
        result.type = ir.TensorType(ir.DataType.BOOL)
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)
