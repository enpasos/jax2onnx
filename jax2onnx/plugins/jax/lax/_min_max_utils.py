# jax2onnx/plugins/jax/lax/_min_max_utils.py

"""Finite SOURCE zero-tie signs for equal floating Min/Max at opset 23.

GatherElements preserves selected negative zeros on the measured CPU targets.
This bounded representation does not settle other opsets, BF16, subnormal
differences or whole-family numerical admission. LEGACY keeps its existing path.
"""

from __future__ import annotations

from typing import Literal

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx._compat.jax import JaxprEqn, definitely_equal_shape
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_representation import representation_for_aval
from jax2onnx.converter.typing_support import NativeLoweringContext


def lower_source_min_max(
    ctx: NativeLoweringContext, eqn: JaxprEqn, *, operation: Literal["Min", "Max"]
) -> bool:
    """Emit the reviewed dense selection only for the finite eligible route."""
    if ctx.precision_policy is not PrecisionPolicy.SOURCE or ctx.opset != 23:
        return False
    out_var = eqn.outvars[0]
    dtype = np.dtype(out_var.aval.dtype)
    if dtype not in (np.dtype(np.float16), np.dtype(np.float32), np.dtype(np.float64)):
        return False

    lhs_var, rhs_var = eqn.invars
    if any(np.dtype(var.aval.dtype) != dtype for var in (lhs_var, rhs_var)):
        raise ValueError(
            f"SOURCE {operation} requires equal floating operand/result dtypes"
        )
    lhs = ctx.require_value_for_var(lhs_var)
    rhs = ctx.require_value_for_var(rhs_var)
    for var, value in ((lhs_var, lhs), (rhs_var, rhs)):
        if not representation_for_aval(var.aval).matches_ir_value(value):
            raise ValueError(
                f"SOURCE {operation} canonical operand representation differs from its source"
            )
    expected_shape = jax.lax.broadcast_shapes(lhs_var.aval.shape, rhs_var.aval.shape)
    if not definitely_equal_shape(expected_shape, out_var.aval.shape):
        raise ValueError(
            f"SOURCE {operation} result does not match operand broadcasting"
        )
    lhs_shape, rhs_shape = lhs.shape, rhs.shape
    if lhs_shape is None or rhs_shape is None:
        raise ValueError(f"SOURCE {operation} requires known canonical shapes")
    output = ctx.allocate_value_for_var(
        out_var, name_hint=ctx.fresh_name(operation.lower())
    )
    if (
        not representation_for_aval(out_var.aval).matches_ir_value(output)
        or output.shape is None
    ):
        raise ValueError(
            f"SOURCE {operation} result representation differs from its source"
        )
    shape = output.shape
    data_type = ir.DataType.from_numpy(dtype)
    prefix = operation.lower()

    def emit(
        op: str,
        inputs: list[ir.Value],
        result_shape: ir.Shape,
        result_dtype: ir.DataType = data_type,
        *,
        axis: int | None = None,
        to: int | None = None,
    ) -> ir.Value:
        attributes: dict[str, int] = {}
        if axis is not None:
            attributes["axis"] = axis
        if to is not None:
            attributes["to"] = to
        return ctx.builder.op(
            op,
            inputs,
            attributes,
            output=ir.Value(
                name=ctx.fresh_name(f"{prefix}_{op.lower()}"),
                type=ir.TensorType(result_dtype),
                shape=result_shape,
            ),
        )

    zero = ctx.builder.add_initializer_from_array(
        name=ctx.fresh_name(f"{prefix}_positive_zero"), array=np.asarray(0, dtype=dtype)
    )
    axes = ctx.builder.add_initializer_from_array(
        name=ctx.fresh_name(f"{prefix}_selection_axis"),
        array=np.asarray([0], dtype=np.int64),
    )
    base = emit(operation, [lhs, rhs], shape)
    left_zero = emit("Equal", [lhs, zero], lhs_shape, ir.DataType.BOOL)
    right_zero = emit("Equal", [rhs, zero], rhs_shape, ir.DataType.BOOL)
    ties = emit("And", [left_zero, right_zero], shape, ir.DataType.BOOL)
    if operation == "Min":
        negative_left = emit("Neg", [lhs], lhs_shape)
        negative_right = emit("Neg", [rhs], rhs_shape)
        correction = emit(
            "Neg", [emit("Add", [negative_left, negative_right], shape)], shape
        )
    else:
        correction = emit("Add", [lhs, rhs], shape)
    # Reuse actual IR dimensions, including symbols; only the artificial choice
    # axis is added. No dimension is concretized or inferred from complex storage.
    selected_shape = ir.Shape((1, *shape.dims))
    choices_shape = ir.Shape((2, *shape.dims))
    base_choices = emit("Unsqueeze", [base, axes], selected_shape)
    correction_choices = emit("Unsqueeze", [correction, axes], selected_shape)
    choices = emit("Concat", [base_choices, correction_choices], choices_shape, axis=0)
    indices = emit("Cast", [ties], shape, ir.DataType.INT64, to=int(ir.DataType.INT64))
    indices = emit("Unsqueeze", [indices, axes], selected_shape, ir.DataType.INT64)
    selected = emit("GatherElements", [choices, indices], selected_shape, axis=0)
    result = ctx.builder.op("Squeeze", [selected, axes], output=output)
    ctx.bind_value_for_var(out_var, result)
    return True
