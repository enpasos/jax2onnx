# jax2onnx/plugins/_precision_utils.py

"""Explicit operand adaptations for the legacy double-precision policy."""

from __future__ import annotations

import onnx_ir as ir

from jax2onnx.converter.typing_support import NativeLoweringContext
from jax2onnx.plugins._loop_extent_meta import propagate_axis0_override


def promote_float32_operand(
    ctx: NativeLoweringContext,
    value: ir.Value,
    *,
    target_dtype: ir.DataType | None,
    name_hint: str,
) -> ir.Value:
    """Adapt a FLOAT data operand when its consumer explicitly requires DOUBLE.

    The caller supplies the consumer's effective dtype, including scope policy.
    Keep the original binding and any preceding FLOAT rounding available to
    other consumers, graph outputs, and representation-sensitive operations.
    """
    if value.dtype != ir.DataType.FLOAT or target_dtype != ir.DataType.DOUBLE:
        return value

    output = ir.Value(
        name=ctx.fresh_name(name_hint),
        type=ir.TensorType(ir.DataType.DOUBLE),
        shape=value.shape,
    )
    result = ctx.builder.op(
        "Cast", [value], {"to": int(ir.DataType.DOUBLE.value)}, output=output
    )
    propagate_axis0_override(value, result)
    return result
