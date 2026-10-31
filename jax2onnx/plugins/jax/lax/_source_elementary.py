# jax2onnx/plugins/jax/lax/_source_elementary.py

"""Typed SOURCE algorithms for the owning elementary primitive lowerers.

No tracing patches or host-value computation occurs here. Every coefficient
and intermediate uses the original operator's precision and actual IR shape.
"""

from __future__ import annotations

from math import factorial
from typing import Any, Literal

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_representation import representation_for_aval
from jax2onnx.converter.typing_support import NativeLoweringContext


def lower_source_elementary(
    ctx: NativeLoweringContext,
    eqn: Any,
    *,
    operation: Literal[
        "asin",
        "sinh",
        "cosh",
        "log1p",
        "expm1",
        "round",
        "integer_pow",
        "sign",
        "neg",
        "div",
        "rem",
    ],
) -> bool:
    if ctx.precision_policy is not PrecisionPolicy.SOURCE:
        return False
    if operation in {"div", "neg"} and not np.issubdtype(
        np.dtype(eqn.invars[0].aval.dtype), np.integer
    ):
        return False
    if operation == "asin" and np.dtype(eqn.invars[0].aval.dtype) not in {
        np.dtype(np.float32),
        np.dtype(np.float64),
    }:
        return False
    values = [ctx.require_value_for_var(var) for var in eqn.invars]
    output = ctx.allocate_value_for_var(
        eqn.outvars[0], name_hint=ctx.fresh_name(f"source_{operation}")
    )
    for var, value in zip((*eqn.invars, *eqn.outvars), (*values, output), strict=True):
        if not representation_for_aval(var.aval).matches_ir_value(value):
            raise ValueError(
                f"SOURCE {operation} canonical representation differs from its native typed value"
            )
    if output.shape is None or output.dtype is None:
        raise ValueError(f"SOURCE {operation} needs an actual typed output shape")
    dtype = np.dtype(eqn.outvars[0].aval.dtype)
    shape = output.shape
    tensor_dtype = output.dtype

    def emit(
        op: str, args: list[ir.Value], *, boolean: bool = False, **attrs: Any
    ) -> ir.Value:
        return ctx.builder.op(
            op,
            args,
            attrs,
            output=ir.Value(
                name=ctx.fresh_name(f"{operation}_{op.lower()}"),
                type=ir.TensorType(ir.DataType.BOOL if boolean else tensor_dtype),
                shape=shape,
            ),
        )

    def constant(value: Any) -> ir.Value:
        return ctx.bind_const_for_var(object(), np.asarray(value, dtype=dtype))

    def select(
        condition: ir.Value, when_true: ir.Value, when_false: ir.Value
    ) -> ir.Value:
        # This is the same dense selection used by scoped SOURCE Min/Max:
        # GatherElements preserves selected zero bits on the measured targets.
        # All calls here have the native output shape (binary integer operands
        # are explicitly expanded before entering a selection).
        axis = ctx.bind_const_for_var(object(), np.asarray([0], dtype=np.int64))
        selected_shape = ir.Shape((1, *shape.dims))
        choices_shape = ir.Shape((2, *shape.dims))

        def shaped(
            op: str,
            args: list[ir.Value],
            result_shape: ir.Shape,
            result_dtype: ir.DataType = tensor_dtype,
            **attrs: Any,
        ) -> ir.Value:
            return ctx.builder.op(
                op,
                args,
                attrs,
                output=ir.Value(
                    name=ctx.fresh_name(f"{operation}_{op.lower()}"),
                    type=ir.TensorType(result_dtype),
                    shape=result_shape,
                ),
            )

        # Broadcast a scalar branch through the actual output's Shape rather
        # than concretizing B or introducing a floating arithmetic consumer.
        def expanded(value: ir.Value) -> ir.Value:
            if value.shape == shape:
                return value
            dimensions = ctx.builder.op(
                "Shape",
                [output_template],
                output=ir.Value(
                    name=ctx.fresh_name(f"{operation}_shape"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((len(shape),)),
                ),
            )
            return shaped("Expand", [value, dimensions], shape)

        choices = shaped(
            "Concat",
            [
                shaped("Unsqueeze", [expanded(when_false), axis], selected_shape),
                shaped("Unsqueeze", [expanded(when_true), axis], selected_shape),
            ],
            choices_shape,
            axis=0,
        )
        index = shaped(
            "Cast", [condition], shape, ir.DataType.INT64, to=int(ir.DataType.INT64)
        )
        index = shaped("Unsqueeze", [index, axis], selected_shape, ir.DataType.INT64)
        picked = shaped("GatherElements", [choices, index], selected_shape, axis=0)
        return shaped("Squeeze", [picked, axis], shape)

    x = values[0]
    output_template = x
    if operation in {"div", "rem"}:
        # Native lax binary broadcasting supplies a result shape independent
        # of either input. Shape(Add) would alter a consumed integer value;
        # the existing output shape's dimensions are not available as a value.
        # Expand through a Bool comparison with the same native broadcasting.
        predicate = emit("Equal", values, boolean=True)
        output_template = predicate
        dimensions = ctx.builder.op(
            "Shape",
            [predicate],
            output=ir.Value(
                name=ctx.fresh_name(f"{operation}_shape"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape((len(shape),)),
            ),
        )
        values = [
            ctx.builder.op(
                "Expand",
                [value, dimensions],
                output=ir.Value(
                    name=ctx.fresh_name(f"{operation}_broadcast"),
                    type=ir.TensorType(tensor_dtype),
                    shape=shape,
                ),
            )
            for value in values
        ]
        x = values[0]
    if operation in {"sinh", "cosh"}:
        result = emit(operation.title(), [x])
    elif operation == "asin":
        # Rational coefficients and half-angle identity adapted from OpenLibm
        # v0.8.1 src/e_asin.c and src/e_asinf.c. This same-width adaptation
        # does not inherit their full correctly-rounded implementation.
        # Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
        # Developed at SunSoft, a Sun Microsystems, Inc. business.
        # Permission to use, copy, modify, and distribute this
        # software is freely granted, provided that this notice
        # is preserved.
        #
        # e_asinf.c conversion to float by Ian Lance Taylor, Cygnus Support.
        # Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
        # Developed at SunPro, a Sun Microsystems, Inc. business.
        # Permission to use, copy, modify, and distribute this
        # software is freely granted, provided that this notice
        # is preserved.
        coefficient_words: tuple[tuple[int, ...], tuple[int, ...], int, int] = {
            "float32": (
                (1042983541, 3173979066, 3155022699),
                (1065353216, 3207914926),
                1070141402,
                866263400,
            ),
            "float64": (
                (
                    4595172819793696085,
                    13822908529170411389,
                    4596417465768494165,
                    13809305468778614587,
                    4560439845004096136,
                    4540259411154564873,
                ),
                (
                    4607182418800017408,
                    13835966419869248843,
                    4611733184086379208,
                    13827746767276147033,
                    4590215604441354882,
                ),
                4609753056924675352,
                4364452196894661639,
            ),
        }[dtype.name]
        unsigned_dtype = np.uint32 if dtype == np.dtype(np.float32) else np.uint64

        def coefficient(word: int) -> ir.Value:
            # Decode only frozen coefficient words, never model values.
            return constant(np.asarray(word, dtype=unsigned_dtype).view(dtype))

        def rational(t: ir.Value) -> ir.Value:
            p_words, q_words, _, _ = coefficient_words
            numerator = coefficient(p_words[-1])
            for word in reversed(p_words[:-1]):
                numerator = emit(
                    "Add", [coefficient(word), emit("Mul", [t, numerator])]
                )
            denominator = coefficient(q_words[-1])
            for word in reversed(q_words[:-1]):
                denominator = emit(
                    "Add", [coefficient(word), emit("Mul", [t, denominator])]
                )
            return emit("Div", [emit("Mul", [t, numerator]), denominator])

        zero, half, one, two = (constant(value) for value in (0, 0.5, 1, 2))
        a = emit("Abs", [x])
        small_t = emit("Mul", [a, a])
        small = emit("Add", [a, emit("Mul", [a, rational(small_t)])])
        large_t = emit("Mul", [emit("Sub", [one, a]), half])
        s = emit("Sqrt", [large_t])
        large = emit(
            "Sub",
            [
                coefficient(coefficient_words[2]),
                emit(
                    "Sub",
                    [
                        emit(
                            "Mul",
                            [
                                two,
                                emit("Add", [s, emit("Mul", [s, rational(large_t)])]),
                            ],
                        ),
                        coefficient(coefficient_words[3]),
                    ],
                ),
            ],
        )
        magnitude = select(emit("LessOrEqual", [a, half], boolean=True), small, large)
        signed = select(
            emit("Less", [x, zero], boolean=True), emit("Neg", [magnitude]), magnitude
        )
        preserve = emit(
            "Or",
            [emit("IsNaN", [x], boolean=True), emit("Equal", [x, zero], boolean=True)],
            boolean=True,
        )
        result = select(preserve, x, signed)
    elif operation == "log1p":
        # Goldberg's compensation recovers the bits lost in 1+x. Neither the
        # input nor Log is narrowed. At u==1 use the exact input (including
        # -0); infinite u uses Log directly, avoiding inf/inf contamination.
        one = constant(1)
        u = emit("Add", [one, x])
        logarithm = emit("Log", [u])
        denominator = emit("Sub", [u, one])
        corrected = emit("Mul", [logarithm, emit("Div", [x, denominator])])
        corrected = select(emit("IsInf", [u], boolean=True), logarithm, corrected)
        result = select(emit("Equal", [u, one], boolean=True), x, corrected)
    elif operation == "expm1":
        # Typed Horner evaluation of expm1(x)/x on [-1/2,1/2]. The omitted
        # Taylor remainder is below 1e-15 (F64) / 1e-10 (F32) relative to x;
        # all arithmetic still rounds in the native operand precision.
        degree = 18 if dtype == np.dtype(np.float64) else 12
        polynomial = constant(1 / factorial(degree))
        for index in range(degree - 1, 0, -1):
            polynomial = emit(
                "Add", [constant(1 / factorial(index)), emit("Mul", [x, polynomial])]
            )
        small = emit("Mul", [x, polynomial])
        ordinary = emit("Sub", [emit("Exp", [x]), constant(1)])
        result = select(
            emit("LessOrEqual", [emit("Abs", [x]), constant(0.5)], boolean=True),
            small,
            ordinary,
        )
    elif operation == "integer_pow":
        exponent = eqn.params["y"]
        if type(exponent) is not int:
            raise ValueError(
                "SOURCE integer_pow requires the native static integer exponent"
            )
        if exponent == 0:
            result = select(
                emit("Equal", [x, x], boolean=True), constant(1), constant(1)
            )
        elif exponent == 1:
            result = x
        elif exponent == 3:
            # JAX has a separate cubic MLIR rule with this association.
            result = emit("Mul", [emit("Mul", [x, x]), x])
        else:
            remaining = abs(exponent)
            accumulator = None
            power = x
            while remaining:
                if remaining & 1:
                    accumulator = (
                        power
                        if accumulator is None
                        else emit("Mul", [accumulator, power])
                    )
                remaining >>= 1
                if remaining:
                    power = emit("Mul", [power, power])
            assert accumulator is not None
            result = (
                emit("Div", [constant(1), accumulator]) if exponent < 0 else accumulator
            )
    elif operation == "round":
        method = eqn.params["rounding_method"]
        if method is jax.lax.RoundingMethod.TO_NEAREST_EVEN:
            result = emit("Round", [x])
        elif method is jax.lax.RoundingMethod.AWAY_FROM_ZERO:
            magnitude = emit("Abs", [x])
            integral = emit("Floor", [magnitude])
            fraction = emit("Sub", [magnitude, integral])
            rounded = select(
                emit("GreaterOrEqual", [fraction, constant(0.5)], boolean=True),
                emit("Add", [integral, constant(1)]),
                integral,
            )
            signed = select(
                emit("Less", [x, constant(0)], boolean=True),
                emit("Neg", [rounded]),
                rounded,
            )
            result = select(emit("Equal", [x, constant(0)], boolean=True), x, signed)
        else:
            raise ValueError(
                "SOURCE round requires the native supported rounding_method"
            )
    elif operation == "sign":
        result = emit("Sign", [x])
        if np.issubdtype(dtype, np.floating) or dtype.name == "bfloat16":
            preserve = emit(
                "Or",
                [
                    emit("IsNaN", [x], boolean=True),
                    emit("Equal", [x, constant(0)], boolean=True),
                ],
                boolean=True,
            )
            result = select(preserve, x, result)
    elif operation == "neg":
        result = (
            emit("Sub", [constant(0), x])
            if np.issubdtype(dtype, np.unsignedinteger)
            else emit("Neg", [x])
        )
    elif operation in {"div", "rem"}:
        y = values[1]
        if np.issubdtype(dtype, np.floating) or dtype.name == "bfloat16":
            result = emit("Mod", [x, y], fmod=1)
        else:
            zero, one = constant(0), constant(1)
            by_zero = emit("Equal", [y, zero], boolean=True)
            exceptional = by_zero
            overflow = None
            if np.issubdtype(dtype, np.signedinteger):
                overflow = emit(
                    "And",
                    [
                        emit("Equal", [x, constant(np.iinfo(dtype).min)], boolean=True),
                        emit("Equal", [y, constant(-1)], boolean=True),
                    ],
                    boolean=True,
                )
                exceptional = emit("Or", [by_zero, overflow], boolean=True)
            safe_y = select(exceptional, one, y)
            quotient = emit("Div", [x, safe_y])
            if operation == "div":
                result = select(
                    by_zero,
                    constant(
                        -1
                        if np.issubdtype(dtype, np.signedinteger)
                        else np.iinfo(dtype).max
                    ),
                    quotient,
                )
            else:
                remainder = emit("Sub", [x, emit("Mul", [quotient, safe_y])])
                result = select(by_zero, x, remainder)
                if overflow is not None:
                    result = select(overflow, zero, result)
    else:
        raise AssertionError(operation)
    ctx.bind_value_for_var(
        eqn.outvars[0], ctx.builder.op("Identity", [result], output=output)
    )
    return True


def lower_source_boolean_min_max(
    ctx: NativeLoweringContext, eqn: Any, *, operation: Literal["Min", "Max"]
) -> bool:
    if ctx.precision_policy is not PrecisionPolicy.SOURCE or np.dtype(
        eqn.outvars[0].aval.dtype
    ) != np.dtype(np.bool_):
        return False
    inputs = [ctx.require_value_for_var(var) for var in eqn.invars]
    output = ctx.allocate_value_for_var(
        eqn.outvars[0], name_hint=ctx.fresh_name("logical_extremum")
    )
    for var, value in zip((*eqn.invars, *eqn.outvars), (*inputs, output), strict=True):
        if not representation_for_aval(var.aval).matches_ir_value(value):
            raise ValueError("SOURCE Boolean extremum has physical type drift")
    result = ctx.builder.op(
        "And" if operation == "Min" else "Or", inputs, output=output
    )
    ctx.bind_value_for_var(eqn.outvars[0], result)
    return True
