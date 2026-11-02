# jax2onnx/plugins/_precision_utils.py

"""Operation-owned accuracy requests and rounding boundaries."""

from __future__ import annotations

from typing import Literal

import jax
import onnx_ir as ir

from jax2onnx._compat.jax import JaxprEqn
from jax2onnx.converter.typing_support import NativeLoweringContext
from jax2onnx.plugins._loop_extent_meta import propagate_axis0_override


def require_default_accuracy(eqn: JaxprEqn) -> None:
    """Reject an explicit accuracy request that this lowering cannot preserve."""

    accuracy = eqn.params.get("accuracy")
    if accuracy is None or accuracy is jax.lax.AccuracyMode.DEFAULT:
        return
    raise NotImplementedError(
        f"{eqn.primitive.name} accuracy={accuracy!r}: the exporter does not currently "
        "preserve non-default accuracy requests."
    )


def emit_float16_arithmetic_with_rounding(
    ctx: NativeLoweringContext,
    op_type: Literal["Add", "Sub", "Mul"],
    lhs: ir.Value,
    rhs: ir.Value,
    *,
    output: ir.Value,
) -> ir.Value:
    """Keep FP16 arithmetic rounding when ORT CPU adapts half kernels to FLOAT.

    Explicit FLOAT arithmetic and a source-typed result Cast retain the half
    boundary that ORT's inserted precision-free casts otherwise remove.
    Every adaptation is local to this consumer; source bindings stay intact.
    Products of finite FP16 operands are exactly representable in FLOAT, so the
    final Cast supplies their source FP16 result boundary.
    """
    if any(value.dtype != ir.DataType.FLOAT16 for value in (lhs, rhs, output)):
        raise TypeError(f"{op_type} FP16 rounding requires FLOAT16 operands and result")

    operands = []
    for side, value in (("lhs", lhs), ("rhs", rhs)):
        operand = ir.Value(
            name=ctx.fresh_name(f"{op_type.lower()}_{side}_float"),
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=value.shape,
        )
        ctx.builder.op(
            "Cast", [value], {"to": int(ir.DataType.FLOAT.value)}, output=operand
        )
        propagate_axis0_override(value, operand)
        operands.append(operand)
    computed = ir.Value(
        name=ctx.fresh_name(f"{op_type.lower()}_float_result"),
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=output.shape,
    )
    ctx.builder.op(op_type, operands, output=computed)
    return ctx.builder.op(
        "Cast", [computed], {"to": int(ir.DataType.FLOAT16.value)}, output=output
    )
