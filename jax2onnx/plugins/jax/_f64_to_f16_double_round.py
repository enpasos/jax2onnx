# jax2onnx/plugins/jax/_f64_to_f16_double_round.py

"""Private, unused emission of FP64-to-FP32-to-FP16 rounding.

The FLOAT Cast supplies the first rounding. Exact widening to DOUBLE then feeds
the existing mathematical half-RNE construction, whose arithmetic requires
DOUBLE values. This helper neither selects a source backend contract nor
registers a lowering. Numerical qualification belongs to the pinned CPU owner;
NaN payload/sign preservation and SOURCE acceptance remain outside its scope.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import onnx_ir as ir

from jax2onnx.plugins.jax._f64_to_f16_rne import emit_f64_to_f16_rne

if TYPE_CHECKING:
    from jax2onnx.converter.typing_support import NativeLoweringContext


def emit_f64_to_f16_double_round(
    ctx: NativeLoweringContext, value: ir.Value
) -> ir.Value:
    """Round to FLOAT, widen exactly, then round mathematically to FLOAT16.

    The half-RNE arithmetic requires opset 12. Its input is the exactly widened
    FLOAT value, preserving the first rounding rather than returning to the
    original input. The numerical owner targets opsets 21 and 23 on pinned CPU.
    """
    if ctx.builder.opset < 12:
        raise ValueError("Two-stage FP64-to-FP16 emission requires opset 12 or newer")
    if not isinstance(value.type, ir.TensorType) or value.dtype != ir.DataType.DOUBLE:
        raise TypeError("Two-stage FP64-to-FP16 emission requires a DOUBLE tensor")

    def output(name: str, dtype: ir.DataType) -> ir.Value:
        return ir.Value(
            name=ctx.fresh_name("f64_to_f16_double_round_" + name),
            type=ir.TensorType(dtype),
            shape=value.shape,
        )

    intermediate = ctx.builder.op(
        "Cast",
        [value],
        {"to": int(ir.DataType.FLOAT.value)},
        output=output("f32", ir.DataType.FLOAT),
    )
    widened = ctx.builder.op(
        "Cast",
        [intermediate],
        {"to": int(ir.DataType.DOUBLE.value)},
        output=output("wide", ir.DataType.DOUBLE),
    )
    return emit_f64_to_f16_rne(ctx, widened)
