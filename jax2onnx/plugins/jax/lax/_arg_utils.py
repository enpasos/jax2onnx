# jax2onnx/plugins/jax/lax/_arg_utils.py

"""Shared helpers for arg-reduction lax primitives."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import onnx_ir as ir

from jax import core

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import (
    _ensure_value_metadata,
    _stamp_type_and_shape,
    _to_ir_dim_for_shape,
)
from jax2onnx.plugins.jax.lax._control_flow_utils import builder_cast, builder_identity
from jax2onnx.plugins.jax.lax._index_utils import _builder_op


def lower_arg_reduction(
    ctx: LoweringContextProtocol,
    eqn: core.JaxprEqn,
    *,
    op_name: str,
    name_prefix: str,
) -> None:
    """Lower ``lax.argmax/argmin`` using ArgMax/ArgMin builder helpers."""

    operand_var = eqn.invars[0]
    out_var = eqn.outvars[0]

    params: dict[str, Any] = getattr(eqn, "params", {})
    axes_param = params.get("axes")
    if axes_param is None:
        axes_seq: tuple[int, ...] = (0,)
    elif isinstance(axes_param, Sequence):
        axes_seq = tuple(int(ax) for ax in axes_param)
    else:
        axes_seq = (int(axes_param),)
    axis = int(axes_seq[0])
    select_last = int(params.get("select_last_index", 0))
    index_dtype = np.dtype(params.get("index_dtype", np.int64))

    operand_val = ctx.get_value_for_var(
        operand_var,
        name_hint=ctx.fresh_name(f"{name_prefix}_in"),
    )
    operand_shape = tuple(getattr(operand_var.aval, "shape", ()))
    operand_dtype = np.dtype(getattr(operand_var.aval, "dtype", np.float32))
    rank = len(operand_shape)
    if rank > 0 and axis < 0:
        axis = axis % rank

    arg_input = operand_val
    if operand_dtype == np.bool_:
        arg_input = builder_cast(
            ctx,
            operand_val,
            _dtype_to_ir(np.dtype(np.int64), ctx.builder.enable_double_precision),
            name_hint=f"{name_prefix}_bool_cast",
        )
        _stamp_type_and_shape(arg_input, operand_shape)
        _ensure_value_metadata(ctx, arg_input)

    reduced_shape = tuple(
        _to_ir_dim_for_shape(dim) for dim in getattr(out_var.aval, "shape", ())
    )

    arg_tmp = _builder_op(
        ctx,
        op_name,
        [arg_input],
        name_hint=f"{name_prefix}_tmp",
        dtype=ir.DataType.INT64,
        shape=reduced_shape,
        attributes={
            "axis": axis,
            "keepdims": 0,
            "select_last_index": select_last,
        },
    )

    target_enum = _dtype_to_ir(index_dtype, ctx.builder.enable_double_precision)
    if target_enum == ir.DataType.INT64:
        out_val = builder_identity(
            ctx,
            arg_tmp,
            name_hint=f"{name_prefix}_out",
        )
    else:
        out_val = builder_cast(
            ctx,
            arg_tmp,
            target_enum,
            name_hint=f"{name_prefix}_out",
        )

    _stamp_type_and_shape(out_val, tuple(getattr(out_var.aval, "shape", ())))
    _ensure_value_metadata(ctx, out_val)
    out_val.type = ir.TensorType(target_enum)
    ctx.bind_value_for_var(out_var, out_val)


__all__ = ["lower_arg_reduction"]
