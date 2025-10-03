"""Shared helpers for reduction primitives in plugins."""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import numpy as np
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins.jax.lax._index_utils import _const_i64, _scalar_i64


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
    cast_val = ir.Value(
        name=ctx.fresh_name("reduce_cast"),
        type=ir.TensorType(dtype_enum),
        shape=tensor.shape,
    )
    ctx.add_node(
        ir.Node(
            op_type="Cast",
            domain="",
            inputs=[tensor],
            outputs=[cast_val],
            name=ctx.fresh_name("Cast"),
            attributes=[IRAttr("to", IRAttrType.INT, int(dtype_enum.value))],
        )
    )
    _stamp_type_and_shape(cast_val, tuple(aval_shape))
    _ensure_value_info(ctx, cast_val)
    return cast_val


def lower_reduction(
    ctx: Any,
    eqn,
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

    operand_shape = tuple(getattr(operand_var.aval, "shape", ()))
    axes_attr = _normalize_axes(axes, len(operand_shape))

    reduced_input = _maybe_cast_input(
        ctx,
        operand_val,
        operand_shape,
        requested_dtype,
    )

    attrs: list[IRAttr] = [IRAttr("keepdims", IRAttrType.INT, 1 if keepdims else 0)]

    inputs = [reduced_input]
    if axes_attr is not None:
        axes_const = _const_i64(ctx, list(axes_attr), f"{op_type.lower()}_axes")
        inputs.append(axes_const)

    node = ir.Node(
        op_type=op_type,
        domain="",
        inputs=inputs,
        outputs=[out_val],
        name=ctx.fresh_name(op_type),
        attributes=attrs,
    )
    ctx.add_node(node)

    out_shape = tuple(getattr(out_var.aval, "shape", ()))
    aval_dtype = getattr(out_var.aval, "dtype", None)
    if aval_dtype is not None:
        out_dtype_enum = _dtype_to_ir(
            np.dtype(aval_dtype), ctx.builder.enable_double_precision
        )
        out_val.type = ir.TensorType(out_dtype_enum)
    _stamp_type_and_shape(out_val, out_shape)

    _ensure_value_info(ctx, out_val)


def lower_boolean_reduction(ctx: Any, eqn, *, mode: str) -> None:
    operand_var = eqn.invars[0]
    out_var = eqn.outvars[0]

    params = getattr(eqn, "params", {})
    axes = params.get("axes")
    keepdims = bool(params.get("keepdims", False))

    operand_val = ctx.get_value_for_var(
        operand_var, name_hint=ctx.fresh_name(f"{mode}_in")
    )
    out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name(f"{mode}_out"))

    operand_shape = tuple(getattr(operand_var.aval, "shape", ()))
    axes_attr = _normalize_axes(axes, len(operand_shape))

    int_operand = _maybe_cast_input(ctx, operand_val, operand_shape, np.dtype(np.int64))

    out_shape = tuple(getattr(out_var.aval, "shape", ()))
    keepdims_attr = 1 if keepdims else 0

    reduce_out = ir.Value(
        name=ctx.fresh_name(f"{mode}_reduce"),
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape(tuple(out_shape)),
    )

    inputs = [int_operand]
    if axes_attr is not None:
        axes_const = _const_i64(ctx, list(axes_attr), f"{mode}_axes")
        inputs.append(axes_const)

    if mode == "reduce_xor":
        reduce_op = "ReduceSum"
    elif mode == "reduce_or":
        reduce_op = "ReduceMax"
    else:
        reduce_op = "ReduceMin"

    ctx.add_node(
        ir.Node(
            op_type=reduce_op,
            domain="",
            inputs=inputs,
            outputs=[reduce_out],
            name=ctx.fresh_name(reduce_op),
            attributes=[IRAttr("keepdims", IRAttrType.INT, keepdims_attr)],
        )
    )
    _stamp_type_and_shape(reduce_out, out_shape)
    _ensure_value_info(ctx, reduce_out)

    if mode == "reduce_xor":
        two_const = _scalar_i64(ctx, 2, f"{mode}_two")
        mod_out = ir.Value(
            name=ctx.fresh_name(f"{mode}_mod"),
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape(tuple(out_shape)),
        )
        ctx.add_node(
            ir.Node(
                op_type="Mod",
                domain="",
                inputs=[reduce_out, two_const],
                outputs=[mod_out],
                name=ctx.fresh_name("Mod"),
                attributes=[IRAttr("fmod", IRAttrType.INT, 0)],
            )
        )
        _stamp_type_and_shape(mod_out, out_shape)
        _ensure_value_info(ctx, mod_out)

        one_const = _scalar_i64(ctx, 1, f"{mode}_one")
        ctx.add_node(
            ir.Node(
                op_type="Equal",
                domain="",
                inputs=[mod_out, one_const],
                outputs=[out_val],
                name=ctx.fresh_name("Equal"),
            )
        )
    else:
        ctx.add_node(
            ir.Node(
                op_type="Cast",
                domain="",
                inputs=[reduce_out],
                outputs=[out_val],
                name=ctx.fresh_name("Cast"),
                attributes=[IRAttr("to", IRAttrType.INT, int(ir.DataType.BOOL.value))],
            )
        )

    out_val.type = ir.TensorType(ir.DataType.BOOL)
    _stamp_type_and_shape(out_val, out_shape)
    _ensure_value_info(ctx, out_val)
