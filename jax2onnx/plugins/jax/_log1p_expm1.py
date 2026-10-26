# jax2onnx/plugins/jax/_log1p_expm1.py

"""Accurate ONNX lowerings for ``log1p``, ``expm1`` and ``sinh``.

ONNX has no Log1p/Expm1 operator. ``Log(1 + x)`` and ``Exp(x) - 1`` round the
intermediate to one for small ``|x|`` and lose all relative accuracy there.
The compensated forms below (Goldberg for log1p, Kahan for expm1) divide by a
difference taken from that same rounded intermediate, so the rounding error
cancels. They need no dtype-specific constants and stay within a few ulps from
float16 through float64; bfloat16 is computed in float32 (see
``ElementwiseEmitter``). ``sinh`` builds on expm1 the way libm does, avoiding
the same cancellation in ``(Exp(x) - Exp(-x)) / 2``.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir

from jax2onnx.converter.typing_support import NativeLoweringContext


def emit_log1p(
    ctx: NativeLoweringContext,
    x: ir.Value,
    *,
    name_hint: str,
    output: ir.Value | None = None,
) -> ir.Value:
    """Emit ``log1p(x) = x * (log(u) / (u - 1))`` with ``u = fl(1 + x)``."""
    emit = ElementwiseEmitter(ctx, x, output, name_hint)
    x = emit.input(x)
    one = emit.scalar(1.0)
    u = emit.op("Add", x, one, name="u")
    ratio = emit.op(
        "Div", emit.op("Log", u, name="log_u"), emit.op("Sub", u, one, name="d")
    )
    # u == 1 makes the ratio 0/0 although log1p(x) rounds to x; x == +inf
    # makes it inf/inf.
    use_x = emit.op(
        "Or",
        emit.op("Equal", u, one, boolean=True, name="exact"),
        emit.op("Equal", x, emit.scalar(np.inf), boolean=True, name="pos_inf"),
        boolean=True,
    )
    safe_ratio = emit.op("Where", use_x, one, ratio, name="safe_ratio")
    # Ending in Mul rather than Where keeps the sign of -0.0: ONNX Runtime's
    # CPU Where returns +0.0 when it selects -0.0.
    return emit.final("Mul", x, safe_ratio)


def emit_expm1(
    ctx: NativeLoweringContext,
    x: ir.Value,
    *,
    name_hint: str,
    output: ir.Value | None = None,
) -> ir.Value:
    """Emit ``expm1(x) = (u - 1) * (x / log(u))`` with ``u = fl(exp(x))``."""
    emit = ElementwiseEmitter(ctx, x, output, name_hint)
    x = emit.input(x)
    one = emit.scalar(1.0)
    u = emit.op("Exp", x, name="u")
    d = emit.op("Sub", u, one, name="d")
    scaled = emit.op(
        "Mul", d, emit.op("Div", x, emit.op("Log", u, name="log_u"), name="ratio")
    )
    # Where exp(x) underflows to 0 or overflows to inf, u - 1 is already the
    # rounded result; the formula would give -0.0 or inf * 0.
    saturated = emit.op(
        "Or",
        emit.op("Equal", d, emit.scalar(-1.0), boolean=True, name="underflow"),
        emit.op("Equal", u, emit.scalar(np.inf), boolean=True, name="overflow"),
        boolean=True,
    )
    unsaturated = emit.op("Where", saturated, d, scaled, name="unsaturated")
    # u == 1 makes log(u) zero although expm1(x) rounds to x.
    exact = emit.op("Equal", u, one, boolean=True, name="exact")
    return emit.final("Where", exact, x, unsaturated)


def emit_sinh(
    ctx: NativeLoweringContext,
    x: ir.Value,
    *,
    name_hint: str,
    output: ir.Value | None = None,
) -> ir.Value:
    """Emit ``sinh(x) = sign(x) * t * (1 + 1 / (t + 1)) / 2``, ``t = expm1(|x|)``."""
    emit = ElementwiseEmitter(ctx, x, output, name_hint)
    x = emit.input(x)
    one = emit.scalar(1.0)
    half = emit.scalar(0.5)
    abs_x = emit.op("Abs", x, name="abs")
    t = emit_expm1(ctx, abs_x, name_hint=f"{name_hint}_expm1")
    # t + t / (t + 1) is exp(|x|) - exp(-|x|) as a sum of two non-negative
    # terms, so nothing cancels; factoring out t keeps t == inf from giving
    # inf / inf.
    exp_neg_abs = emit.op(
        "Div", one, emit.op("Add", t, one, name="exp_abs"), name="exp_neg_abs"
    )
    twice = emit.op(
        "Mul", t, emit.op("Add", one, exp_neg_abs, name="factor"), name="twice"
    )
    regular = emit.op("Mul", half, twice, name="regular")
    # exp(|x|) overflows up to ln(2) below sinh's own overflow point. There
    # exp(-|x|) is negligible and sinh(|x|) = (w / 2) * w with w = exp(|x| / 2).
    w = emit.op("Exp", emit.op("Mul", abs_x, half, name="half_abs"), name="w")
    large = emit.op("Mul", emit.op("Mul", half, w, name="half_w"), w, name="large")
    overflow = emit.op("Equal", t, emit.scalar(np.inf), boolean=True, name="overflow")
    sinh_abs = emit.op("Where", overflow, large, regular, name="sinh_abs")
    # x < 0 misses -0.0, and 1 / x < 0 misses -inf.
    zero = emit.scalar(0.0)
    negative = emit.op(
        "Or",
        emit.op("Less", x, zero, boolean=True, name="below_zero"),
        emit.op(
            "Less",
            emit.op("Div", one, x, name="recip"),
            zero,
            boolean=True,
            name="negative_zero",
        ),
        boolean=True,
        name="negative",
    )
    sign = emit.op("Where", negative, emit.scalar(-1.0), one, name="sign")
    return emit.final("Mul", sign, sinh_abs)


class ElementwiseEmitter:
    """Typed elementwise emission sharing one shape and compute dtype.

    bfloat16 is computed in float32 and rounded back at the output: ONNX
    Runtime has no bfloat16 ``Log`` kernel, and float32 rounds to the nearest
    bfloat16 result.
    """

    def __init__(
        self,
        ctx: NativeLoweringContext,
        like: ir.Value,
        output: ir.Value | None,
        name_hint: str,
    ) -> None:
        dtype = output.dtype if output is not None else like.dtype
        if dtype is None:
            raise TypeError(f"{name_hint} lowering requires a typed value")
        self._ctx = ctx
        self._result_dtype = dtype
        self._dtype = ir.DataType.FLOAT if dtype == ir.DataType.BFLOAT16 else dtype
        self._shape = like.shape
        self._output = output
        self._name_hint = name_hint

    def input(self, value: ir.Value) -> ir.Value:
        if value.dtype == self._dtype:
            return value
        return self._cast(value, self._value("in", self._dtype), self._dtype)

    def scalar(self, value: float) -> ir.Value:
        array = np.asarray(value, dtype=self._dtype.numpy())
        scalar = self._ctx.bind_const_for_var(object(), array)
        if scalar.dtype == self._dtype:
            return scalar
        # With enable_double_precision, bind_const_for_var promotes the scalar
        # to DOUBLE while float16 and bfloat16 inputs keep their compute dtype.
        # Retyping the constant in place does not survive the later promotion
        # of FLOAT constants, so cast it explicitly.
        output = ir.Value(
            name=self._ctx.fresh_name(f"{self._name_hint}_scalar"),
            type=ir.TensorType(self._dtype),
            shape=scalar.shape,
        )
        return self._cast(scalar, output, self._dtype)

    def op(
        self, op_type: str, *inputs: ir.Value, boolean: bool = False, name: str = ""
    ) -> ir.Value:
        dtype = ir.DataType.BOOL if boolean else self._dtype
        output = self._value(name or op_type.lower(), dtype)
        return self._ctx.builder.op(op_type, list(inputs), output=output)

    def final(self, op_type: str, *inputs: ir.Value) -> ir.Value:
        output = self._output
        if output is None:
            output = self._value("out", self._result_dtype)
        if self._dtype == self._result_dtype:
            return self._ctx.builder.op(op_type, list(inputs), output=output)
        unrounded = self.op(op_type, *inputs, name="unrounded")
        return self._cast(unrounded, output, self._result_dtype)

    def _value(self, name: str, dtype: ir.DataType) -> ir.Value:
        return ir.Value(
            name=self._ctx.fresh_name(f"{self._name_hint}_{name}"),
            type=ir.TensorType(dtype),
            shape=self._shape,
        )

    def _cast(self, value: ir.Value, output: ir.Value, dtype: ir.DataType) -> ir.Value:
        to = int(dtype.value)
        return self._ctx.builder.op("Cast", [value], {"to": to}, output=output)
