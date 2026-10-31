# jax2onnx/plugins/jax/_f64_to_f16_rne.py

"""Private, unused emission of direct FP64-to-FP16 round-to-nearest-even.

Select the reciprocal FP16 spacing with exact powers of two, round in DOUBLE,
then cast the already rounded value. This construction does not select a source
backend contract and is not registered or used by conversion. NaN payloads are
outside its contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnx_ir as ir

if TYPE_CHECKING:
    from jax2onnx.converter.typing_support import NativeLoweringContext


def emit_f64_to_f16_rne(ctx: NativeLoweringContext, value: ir.Value) -> ir.Value:
    """Emit a typed, shape-preserving RNE construction without tracing source.

    FP16 subnormals and its lowest normal binade use spacing 2**-24. Each
    subsequent binade doubles that spacing. The largest binade's spacing also
    rounds the overflow midpoint correctly before the final FLOAT16 Cast.
    All numerical constants and arithmetic preceding that Cast are DOUBLE.
    """
    if ctx.builder.opset < 12:
        raise ValueError("FP64-to-FP16 RNE emission requires opset 12 or newer")
    if not isinstance(value.type, ir.TensorType) or value.dtype != ir.DataType.DOUBLE:
        raise TypeError("FP64-to-FP16 RNE emission requires a DOUBLE tensor")

    def output(name: str, dtype: ir.DataType) -> ir.Value:
        return ir.Value(
            name=ctx.fresh_name("f64_to_f16_rne_" + name),
            type=ir.TensorType(dtype),
            shape=value.shape,
        )

    def scalar(number: float) -> ir.Value:
        # The context owns graph initializers and Function Constant emission.
        constant = ctx.bind_const_for_var(
            object(), np.asarray(number, dtype=np.float64)
        )
        if constant.dtype != ir.DataType.DOUBLE:
            raise TypeError("FP64-to-FP16 RNE constants must retain DOUBLE storage")
        return constant

    def op(
        name: str, *inputs: ir.Value, dtype: ir.DataType = ir.DataType.DOUBLE
    ) -> ir.Value:
        return ctx.builder.op(name, list(inputs), output=output(name.lower(), dtype))

    absolute = op("Abs", value)
    unit = scalar(2.0**24)
    for exponent in range(-13, 16):
        above = op(
            "GreaterOrEqual",
            absolute,
            scalar(2.0**exponent),
            dtype=ir.DataType.BOOL,
        )
        unit = op("Where", above, scalar(2.0 ** (10 - exponent)), unit)

    scaled = op("Mul", value, unit)
    rounded = op("Round", scaled)
    rescaled = op("Div", rounded, unit)
    return ctx.builder.op(
        "Cast",
        [rescaled],
        {"to": int(ir.DataType.FLOAT16.value)},
        output=output("out", ir.DataType.FLOAT16),
    )
