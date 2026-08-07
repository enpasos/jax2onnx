# jax2onnx/plugins/jax/lax/_reduce_utils.py

"""Shared helpers for reduction primitives in plugins."""

from __future__ import annotations

from typing import Any, Final, Iterable, Optional, Sequence, cast

import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx._compat.jax import JaxprEqn
from jax2onnx.plugins.jax.lax._index_utils import _scalar_i64
from jax2onnx.plugins.jax.lax._opset_utils import builder_reduce_with_axes


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
    ctx: LoweringContextProtocol,
    tensor: ir.Value,
    aval_shape: Sequence[Any],
    dtype: Optional[np.dtype],
) -> ir.Value:
    if dtype is None:
        return tensor

    dtype_enum = _dtype_to_ir(dtype, ctx.builder.enable_double_precision)
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

    if axes_attr == () and needs_result_cast:
        # The INT64 work dtype only protects a real unsigned reduction. With
        # no axes, preserve the input dtype (or apply the requested result
        # dtype directly) instead of leaking the internal work dtype.
        work_dtype = requested_dtype
        needs_result_cast = False

    reduced_input = _maybe_cast_input(
        ctx,
        operand_val,
        operand_shape,
        work_dtype,
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
        aval_dtype = getattr(out_var.aval, "dtype", None)
        if aval_dtype is not None:
            identity_dtype_enum = _dtype_to_ir(
                np.dtype(aval_dtype), ctx.builder.enable_double_precision
            )
            result.type = ir.TensorType(identity_dtype_enum)
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)
        return

    desired_name = getattr(out_val, "name", None) or ctx.fresh_name(op_type)
    producer = getattr(out_val, "producer", lambda: None)
    if callable(producer) and producer() is not None:
        desired_name = ctx.fresh_name(op_type)

    keepdims_attr = 1 if keepdims else 0
    reduce_outputs = [ctx.fresh_name(op_type)] if needs_result_cast else [desired_name]
    if op_type in {
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
            output_name=reduce_outputs[0],
        )
    else:
        raise ValueError(f"Unsupported reduction op: {op_type}")

    out_shape = tuple(getattr(out_var.aval, "shape", ()))
    aval_dtype = getattr(out_var.aval, "dtype", None)
    out_dtype_enum: ir.DataType | None = None
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
        cast_result = cast(
            ir.Value,
            ctx.builder.Cast(
                result,
                _outputs=[desired_name],
                to=int(target_dtype.value),
            ),
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
    ctx: LoweringContextProtocol, eqn: JaxprEqn, *, mode: str
) -> None:
    operand_var = eqn.invars[0]
    out_var = eqn.outvars[0]

    out_dtype_param = getattr(getattr(out_var, "aval", None), "dtype", None)
    if out_dtype_param is not None and not np.issubdtype(
        np.dtype(out_dtype_param), np.bool_
    ):
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
    operand_dtype_param = getattr(getattr(operand_var, "aval", None), "dtype", None)
    operand_dtype = (
        np.dtype(operand_dtype_param) if operand_dtype_param is not None else None
    )
    boolean_operand = operand_val
    if operand_dtype is None or not np.issubdtype(operand_dtype, np.bool_):
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
