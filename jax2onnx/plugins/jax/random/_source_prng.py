# jax2onnx/plugins/jax/random/_source_prng.py

"""Selected native Threefry words and the proven mantissa uniform expression.

Words use a nonnegative INT64 carrier between modulo-2**32 operations. Integer
shifts use unsigned carriers; the largest intermediate word sum is below 2**34.
The flat counter Range is bounded by ONNX's signed-int64 tensor element count.
No floating conversion is used until the exact 23/52-bit mantissa is formed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx._compat.jax import (
    ClosedJaxpr,
    DropVar,
    JaxprEqn,
    Literal,
    ShapedArray,
    Var,
    has_selected_prng_context,
    is_selected_prng_key_dtype,
    is_selected_threefry_impl,
    is_unpartitioned_prng_placement,
    native_mul_parameter_names,
)
from jax2onnx.converter.source_representation import (
    normalized_shape_dimensions,
    source_shape_contract,
)
from jax2onnx.converter.typing_support import NativeLoweringContext


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"selected native PRNG: {message}")


class NativePRNGExclusion(ValueError):
    """A real native request beyond the selected algorithm or placement packet."""


def _supported(condition: bool, message: str) -> None:
    if not condition:
        raise NativePRNGExclusion(f"selected native PRNG: {message}")


def require_native_prng_opset(opset: int, eqn: Any) -> None:
    """Keep this emitter's audited version boundary at every direct entry."""
    if opset not in (23, 26):
        from jax2onnx.converter.source_profile import SourcePrecisionError

        raise SourcePrecisionError(
            "selected native PRNG emission is outside the audited opset packet",
            phase="profile-validation",
            category="profile-exclusion",
            location="native PRNG entry",
            primitive=eqn.primitive.name,
            requested_dtypes=tuple(
                str(var.aval.dtype) for var in (*eqn.invars, *eqn.outvars)
            ),
            opset=opset,
            requirement="the selected native PRNG lowering audited at opset 23 or 26",
            action="retain the original opset and report this unaudited operation/version intersection; export never upgrades the opset",
        )


def _shape_signature(shape: Any) -> Any:
    return source_shape_contract(tuple(shape))


def _native_aval(var: Any) -> ShapedArray:
    aval = var.aval
    if not isinstance(aval, ShapedArray):
        raise ValueError(
            "selected native PRNG: requires an ordinary native tensor aval"
        )
    return aval


def _aval(var: Any, dtype: Any, shape: tuple[object, ...], weak: bool) -> None:
    aval = _native_aval(var)
    _supported(
        is_unpartitioned_prng_placement(aval.sharding, len(aval.shape)),
        "requires actual absent or empty unpartitioned native aval placement",
    )
    if dtype == "key":
        valid_dtype = is_selected_prng_key_dtype(aval.dtype)
    else:
        try:
            valid_dtype = np.dtype(aval.dtype) == np.dtype(dtype)
        except TypeError:
            valid_dtype = False
    _require(valid_dtype, "dtype or native key implementation differs")
    _require(
        type(aval.weak_type) is bool and aval.weak_type is weak, "weakness differs"
    )
    _require(
        _shape_signature(aval.shape) == _shape_signature(shape),
        "shape or native scope differs",
    )


def _literal(var: Any, value: int | float, dtype: Any) -> None:
    _require(isinstance(var, Literal), "constant operand is not the actual literal")
    _aval(var, dtype, (), False)
    array = np.asarray(var.val)
    expected = np.asarray(value, dtype=dtype)
    _require(
        array.dtype == expected.dtype
        and array.shape == ()
        and array.tobytes() == expected.tobytes(),
        "literal payload differs",
    )


def _uniform_bound_literal(
    var: Any, value: float, dtype: Any, *, selected_only: bool = False
) -> None:
    """Check the uncast payload; other valid mixed ranges remain exclusions."""
    require_bound = _supported if selected_only else _require
    require_bound(isinstance(var, Literal), "uniform bound is not the actual literal")
    _aval(var, dtype, (), True)
    _require(
        isinstance(var.val, (float, np.floating, np.ndarray)),
        "uniform bound payload is not a native numeric scalar",
    )
    payload = np.asarray(var.val)
    _require(
        payload.shape == () and payload.dtype.kind == "f",
        "uniform bound literal payload metadata differs",
    )
    if isinstance(var.val, (np.floating, np.ndarray)):
        _require(
            payload.dtype == np.dtype(dtype),
            "uniform bound literal payload dtype differs from its aval",
        )
    require_bound(
        payload.tobytes() == np.asarray(value, dtype=payload.dtype).tobytes(),
        "uniform bound literal payload differs",
    )


def _operands(eqn: Any, *variables: Any) -> None:
    _require(
        len(eqn.invars) == len(variables)
        and all(
            actual is expected
            for actual, expected in zip(eqn.invars, variables, strict=True)
        ),
        "operand variable bindings differ",
    )


def _parameters(eqn: Any, expected: dict[str, Any]) -> None:
    _require(set(eqn.params) == set(expected), "primitive parameter fields differ")
    for name, wanted in expected.items():
        actual = eqn.params[name]
        if name == "shape":
            valid = type(actual) is tuple and _shape_signature(
                actual
            ) == _shape_signature(wanted)
        elif wanted is None or type(wanted) is bool:
            valid = actual is wanted
        else:
            valid = type(actual) is type(wanted) and actual == wanted
        _require(bool(valid), f"primitive parameter {name} differs")


def _native_bindings(body: Any) -> None:
    """Keep ordinary scope definition/use proofs in the finite carrier route."""
    formals = (*body.invars, *body.constvars)
    _require(
        all(isinstance(var, Var) and not isinstance(var, DropVar) for var in formals),
        "carrier formals are not actual native variable definitions",
    )
    bound = {id(var) for var in formals}
    _require(len(bound) == len(formals), "carrier has duplicate formal definitions")
    for equation in body.eqns:
        _require(
            isinstance(equation, JaxprEqn)
            and not equation.effects
            and len(equation.outvars) == 1,
            "carrier equation kind, effects or result count differs",
        )
        _require(
            all(
                isinstance(var, Literal) or id(var) in bound for var in equation.invars
            ),
            "carrier has an unresolved binding",
        )
        output = equation.outvars[0]
        _require(
            isinstance(output, Var)
            and not isinstance(output, DropVar)
            and id(output) not in bound,
            "carrier redefines or drops a result binding",
        )
        bound.add(id(output))
    _require(
        all(isinstance(var, Literal) or id(var) in bound for var in body.outvars),
        "carrier output has no local definition",
    )


def validate_selected_random_equation(eqn: Any) -> None:
    """Prove native scalar logical keys and exact ordinary word operands."""
    name = eqn.primitive.name
    counts = {
        "random_wrap": (1, 1),
        "random_unwrap": (1, 1),
        "random_seed": (1, 1),
        "random_fold_in": (2, 1),
        "random_bits": (1, 1),
    }
    _require(
        name in counts and (len(eqn.invars), len(eqn.outvars)) == counts[name],
        "random primitive arity differs",
    )
    _require(not eqn.effects, "random equation has effects")
    if name == "random_wrap":
        _require(
            set(eqn.params) == {"impl"}
            and is_selected_threefry_impl(eqn.params["impl"]),
            "wrap requires the installed selected implementation",
        )
        _aval(eqn.invars[0], np.uint32, (2,), False)
        _aval(eqn.outvars[0], "key", (), False)
    elif name == "random_unwrap":
        _parameters(eqn, {})
        _aval(eqn.invars[0], "key", (), False)
        _aval(eqn.outvars[0], np.uint32, (2,), False)
    elif name == "random_seed":
        _require(
            set(eqn.params) == {"impl"}
            and is_selected_threefry_impl(eqn.params["impl"]),
            "seed requires the installed selected implementation",
        )
        source = eqn.invars[0].aval
        _require(
            np.dtype(source.dtype).name in {"int32", "int64", "uint32", "uint64"},
            "seed requires a 32/64-bit integer scalar",
        )
        _aval(eqn.invars[0], source.dtype, (), source.weak_type)
        _aval(eqn.outvars[0], "key", (), False)
    elif name == "random_fold_in":
        _parameters(eqn, {})
        _aval(eqn.invars[0], "key", (), False)
        _aval(eqn.invars[1], np.uint32, (), False)
        _aval(eqn.outvars[0], "key", (), False)
    else:
        _require(
            set(eqn.params) == {"bit_width", "shape"}, "random bits parameters differ"
        )
        width, shape = eqn.params["bit_width"], eqn.params["shape"]
        _require(
            type(width) is int and width in (32, 64) and type(shape) is tuple,
            "requires 32/64-bit native bits and a tuple shape",
        )
        _supported(
            jax.config.jax_threefry_partitionable is True
            and has_selected_prng_context(eqn),
            "requires recorded native partitionable=True context and unchanged selected ambient mode",
        )
        _, symbol = _shape_signature(shape)
        _supported(
            symbol is None or len(shape) in (2, 3),
            "symbolic random bits retain the audited rank-two/rank-three packet",
        )
        if all(isinstance(dim, (int, np.integer)) for dim in shape):
            _require(
                math.prod(shape) <= np.iinfo(np.int64).max,
                "counter grid exceeds the signed-int64 physical tensor limit",
            )
        _aval(eqn.invars[0], "key", (), False)
        _aval(eqn.outvars[0], np.uint32 if width == 32 else np.uint64, shape, False)


@dataclass(frozen=True)
class _Uniform:
    shape: tuple[object, ...]
    dtype: np.dtype[Any]
    bits: Any


def prove_native_uniform(eqn: Any) -> _Uniform | None:
    """Inspect the actual native graph; a descriptor or function name is no proof."""
    if not isinstance(eqn, JaxprEqn) or eqn.primitive.name != "jit":
        return None
    closed = eqn.params.get("jaxpr")
    if not isinstance(closed, ClosedJaxpr):
        return None
    body = closed.jaxpr
    expected_names = (
        "convert_element_type",
        "convert_element_type",
        "broadcast_in_dim",
        "broadcast_in_dim",
        "random_bits",
        "shift_right_logical",
        "or",
        "bitcast_convert_type",
        "sub",
        "sub",
        "mul",
        "add",
        "max",
    )
    if tuple(item.primitive.name for item in body.eqns) != expected_names:
        return None
    _require(
        not eqn.effects and not body.effects and not getattr(body, "is_high", False),
        "uniform must be a pure low-level graph",
    )
    _require(
        not closed.consts
        and not body.constvars
        and len(body.invars) == len(eqn.invars) == 3
        and len(body.outvars) == len(eqn.outvars) == 1,
        "uniform boundary or constants differ",
    )
    _native_bindings(body)
    output_aval = _native_aval(body.outvars[0])
    dtype = np.dtype(output_aval.dtype)
    _supported(
        dtype in (np.dtype(np.float32), np.dtype(np.float64)),
        "uniform requires the observed FP32/FP64 probability dtype",
    )
    shape = tuple(output_aval.shape)
    _supported(
        len(shape) in (2, 3)
        or len(shape) == 1
        and type(shape[0]) is int
        and shape[0] >= 0,
        "uniform requires an audited rank-two/rank-three shape or a static rank-one shape",
    )
    scalar_shape = (1,) * len(shape)
    key, low, high = body.invars
    _aval(key, "key", (), False)
    for formal, actual in zip(body.invars, eqn.invars, strict=True):
        formal_aval = _native_aval(formal)
        _require(type(formal_aval.weak_type) is bool, "invalid formal weakness")
        _aval(
            actual,
            "key" if formal is key else formal_aval.dtype,
            (),
            formal_aval.weak_type,
        )
    low_aval, high_aval = _native_aval(low), _native_aval(high)
    bound_dtypes = (np.dtype(low_aval.dtype), np.dtype(high_aval.dtype))
    if bound_dtypes != (dtype, dtype):
        # Native x64-on FP32 Bernoulli passes weak FP64 +0/+1 literals before
        # the two explicit conversions below. No other mixed range is audited.
        _supported(
            dtype == np.dtype(np.float32)
            and bound_dtypes == (np.dtype(np.float64),) * 2
            and low_aval.weak_type is high_aval.weak_type is True,
            "uniform bound dtypes differ from the observed native carriers",
        )
        for actual, value in zip(eqn.invars[1:], (0.0, 1.0), strict=True):
            _uniform_bound_literal(actual, value, np.float64, selected_only=True)
    _aval(low, bound_dtypes[0], (), True)
    _aval(high, bound_dtypes[1], (), True)
    _aval(body.outvars[0], dtype, shape, False)
    _aval(eqn.outvars[0], dtype, shape, False)
    equations = body.eqns
    outputs = [item.outvars[0] for item in equations]
    for index, formal in ((0, low), (1, high)):
        _operands(equations[index], formal)
        _parameters(
            equations[index], {"new_dtype": dtype, "weak_type": False, "sharding": None}
        )
        _aval(outputs[index], dtype, (), False)
    for index, source in ((2, outputs[0]), (3, outputs[1])):
        _operands(equations[index], source)
        _parameters(
            equations[index],
            {"shape": scalar_shape, "broadcast_dimensions": (), "sharding": None},
        )
        _aval(outputs[index], dtype, scalar_shape, False)
    bits = equations[4]
    _operands(bits, key)
    validate_selected_random_equation(bits)
    width = 32 if dtype == np.dtype(np.float32) else 64
    _require(
        bits.params["bit_width"] == width, "random width differs from probability dtype"
    )
    _require(
        _shape_signature(bits.params["shape"]) == _shape_signature(shape),
        "random grid differs from uniform output",
    )
    integer_dtype = np.dtype(np.uint32 if width == 32 else np.uint64)
    for index, constant in (
        (5, 9 if width == 32 else 12),
        (6, 0x3F800000 if width == 32 else 0x3FF0000000000000),
    ):
        item = equations[index]
        _require(
            len(item.invars) == 2 and item.invars[0] is outputs[index - 1],
            "uniform integer dataflow differs",
        )
        _literal(item.invars[1], constant, integer_dtype)
        _parameters(item, {})
        _aval(outputs[index], integer_dtype, shape, False)
    _operands(equations[7], outputs[6])
    _parameters(equations[7], {"new_dtype": dtype})
    _aval(outputs[7], dtype, shape, False)
    _require(
        len(equations[8].invars) == 2 and equations[8].invars[0] is outputs[7],
        "uniform subtraction binding differs",
    )
    _literal(equations[8].invars[1], 1.0, dtype)
    _parameters(equations[8], {})
    _aval(outputs[8], dtype, shape, False)
    for index, operands, expected_shape, parameters in (
        (9, (outputs[3], outputs[2]), scalar_shape, {}),
        (
            10,
            (outputs[8], outputs[9]),
            shape,
            {name: None for name in native_mul_parameter_names()},
        ),
        (11, (outputs[10], outputs[2]), shape, {}),
        (12, (outputs[2], outputs[11]), shape, {}),
    ):
        _operands(equations[index], *operands)
        _parameters(equations[index], parameters)
        _aval(outputs[index], dtype, expected_shape, False)
    _require(body.outvars[0] is outputs[12], "uniform returns a different binding")
    return _Uniform(shape, dtype, bits)


def native_bernoulli_uniform(eqn: Any) -> Any | None:
    """Prove the two-equation native Bernoulli carrier around the same uniform."""
    if not isinstance(eqn, JaxprEqn) or eqn.primitive.name != "jit":
        return None
    closed = eqn.params.get("jaxpr")
    if not isinstance(closed, ClosedJaxpr):
        return None
    body = closed.jaxpr
    if len(body.eqns) != 2 or tuple(item.primitive.name for item in body.eqns) != (
        "jit",
        "lt",
    ):
        return None
    uniform, comparison = body.eqns
    proof = prove_native_uniform(uniform)
    if proof is None:
        return None
    _require(
        not closed.consts
        and not body.constvars
        and not body.effects
        and not getattr(body, "is_high", False),
        "Bernoulli carrier is not pure",
    )
    _require(not eqn.effects, "Bernoulli equation has effects")
    _native_bindings(body)
    _require(
        len(body.invars) == len(eqn.invars) == 2
        and len(body.outvars) == len(eqn.outvars) == 1,
        "Bernoulli boundary differs",
    )
    key, probability = body.invars
    _aval(key, "key", (), False)
    _aval(probability, proof.dtype, (), False)
    _aval(eqn.invars[0], "key", (), False)
    _aval(eqn.invars[1], proof.dtype, (), False)
    _require(
        len(uniform.invars) == 3 and uniform.invars[0] is key,
        "Bernoulli key binding differs",
    )
    for variable, value in zip(uniform.invars[1:], (0.0, 1.0), strict=True):
        _uniform_bound_literal(variable, value, _native_aval(variable).dtype)
    _require(
        isinstance(comparison, JaxprEqn)
        and not comparison.effects
        and len(comparison.outvars) == 1,
        "Bernoulli comparison differs",
    )
    _operands(comparison, uniform.outvars[0], probability)
    _parameters(comparison, {})
    _aval(comparison.outvars[0], np.bool_, proof.shape, False)
    _require(
        body.outvars[0] is comparison.outvars[0], "Bernoulli output binding differs"
    )
    _aval(eqn.outvars[0], np.bool_, proof.shape, False)
    return uniform


class _Words:
    def __init__(self, ctx: NativeLoweringContext) -> None:
        self.ctx = ctx

    def constant(self, value: Any, dtype: Any = np.int64) -> ir.Value:
        return self.ctx.builder.add_initializer_from_array(
            name=self.ctx.fresh_name("prng_constant"),
            array=np.asarray(value, dtype=dtype),
        )

    def op(
        self,
        name: str,
        *values: ir.Value,
        dtype: ir.DataType = ir.DataType.INT64,
        shape: ir.Shape | None = None,
        **attributes: str | int | float,
    ) -> ir.Value:
        result = self.ctx.builder.op(name, values, attributes)
        result.type = ir.TensorType(dtype)
        result.shape = shape if shape is not None else values[0].shape
        return result

    def cast(self, value: ir.Value, dtype: ir.DataType) -> ir.Value:
        return self.op("Cast", value, dtype=dtype, to=int(dtype.value))

    def add(self, left: ir.Value, right: ir.Value) -> ir.Value:
        return self.op(
            "BitwiseAnd", self.op("Add", left, right), self.constant(0xFFFFFFFF)
        )

    def shift(self, value: ir.Value, count: int, direction: str) -> ir.Value:
        unsigned = self.cast(value, ir.DataType.UINT64)
        shifted = self.op(
            "BitShift",
            unsigned,
            self.constant(count, np.uint64),
            dtype=ir.DataType.UINT64,
            direction=direction,
        )
        return self.cast(shifted, ir.DataType.INT64)

    def rotate(self, value: ir.Value, count: int) -> ir.Value:
        return self.op(
            "BitwiseAnd",
            self.op(
                "BitwiseOr",
                self.shift(value, count, "LEFT"),
                self.shift(value, 32 - count, "RIGHT"),
            ),
            self.constant(0xFFFFFFFF),
        )

    def key(self, value: ir.Value) -> tuple[ir.Value, ir.Value]:
        _require(
            value.dtype == ir.DataType.UINT32 and value.shape == ir.Shape((2,)),
            "physical key binding must contain two uint32 words",
        )
        pair = [
            self.cast(
                self.op(
                    "Gather",
                    value,
                    self.constant(index),
                    dtype=ir.DataType.UINT32,
                    shape=ir.Shape(()),
                    axis=0,
                ),
                ir.DataType.INT64,
            )
            for index in (0, 1)
        ]
        return pair[0], pair[1]

    def threefry(
        self, key: ir.Value, high: ir.Value, low: ir.Value
    ) -> tuple[ir.Value, ir.Value]:
        key0, key1 = self.key(key)
        parity = self.op(
            "BitwiseXor", self.op("BitwiseXor", key0, key1), self.constant(0x1BD11BDA)
        )
        keys = (key0, key1, parity)
        left, right = self.add(high, key0), self.add(low, key1)
        for group in range(5):
            for rotation in (13, 15, 26, 6) if group % 2 == 0 else (17, 29, 16, 24):
                left = self.add(left, right)
                right = self.op("BitwiseXor", self.rotate(right, rotation), left)
            injection = group + 1
            left = self.add(left, keys[injection % 3])
            right = self.add(
                self.add(right, keys[(injection + 1) % 3]), self.constant(injection)
            )
        return left, right

    def grid(self, shape: tuple[object, ...]) -> tuple[ir.Value, ir.Value]:
        pieces: list[ir.Value] = []
        for dimension in shape:
            if isinstance(dimension, (int, np.integer)):
                pieces.append(self.constant([int(dimension)]))
            else:
                origin = self.ctx.get_symbolic_dim_origin(dimension)
                _require(
                    origin is not None,
                    "counter grid has no actual symbolic input binding",
                )
                assert origin is not None
                pieces.append(
                    self.op(
                        "Shape",
                        origin.value,
                        shape=ir.Shape((1,)),
                        start=origin.axis,
                        end=origin.axis + 1,
                    )
                )
        dimensions = (
            self.op("Concat", *pieces, shape=ir.Shape((len(shape),)), axis=0)
            if pieces
            else self.constant([])
        )
        size = self.op("ReduceProd", dimensions, shape=ir.Shape(()), keepdims=0)
        counter = self.op(
            "Range", self.constant(0), size, self.constant(1), shape=ir.Shape((None,))
        )
        counter = self.op(
            "Reshape",
            counter,
            dimensions,
            shape=ir.Shape(normalized_shape_dimensions(shape)),
            allowzero=1,
        )
        return self.shift(counter, 32, "RIGHT"), self.op(
            "BitwiseAnd", counter, self.constant(0xFFFFFFFF)
        )


def emit_fold_in(ctx: NativeLoweringContext, eqn: Any) -> None:
    require_native_prng_opset(ctx.opset, eqn)
    validate_selected_random_equation(eqn)
    words = _Words(ctx)
    data = words.cast(ctx.require_value_for_var(eqn.invars[1]), ir.DataType.INT64)
    high, low = words.threefry(
        ctx.require_value_for_var(eqn.invars[0]), words.constant(0), data
    )
    axes = words.constant([0])
    pair = [
        words.op(
            "Unsqueeze",
            words.cast(value, ir.DataType.UINT32),
            axes,
            dtype=ir.DataType.UINT32,
            shape=ir.Shape((1,)),
        )
        for value in (high, low)
    ]
    result = words.op(
        "Concat", *pair, dtype=ir.DataType.UINT32, shape=ir.Shape((2,)), axis=0
    )
    ctx.bind_value_for_var(eqn.outvars[0], result)


def emit_seed(ctx: NativeLoweringContext, eqn: Any) -> None:
    require_native_prng_opset(ctx.opset, eqn)
    validate_selected_random_equation(eqn)
    words = _Words(ctx)
    source = ctx.require_value_for_var(eqn.invars[0])
    low = words.cast(source, ir.DataType.UINT32)
    if np.dtype(eqn.invars[0].aval.dtype).itemsize == 8:
        unsigned = words.cast(source, ir.DataType.UINT64)
        high = words.cast(
            words.op(
                "BitShift",
                unsigned,
                words.constant(32, np.uint64),
                dtype=ir.DataType.UINT64,
                direction="RIGHT",
            ),
            ir.DataType.UINT32,
        )
    else:
        high = words.constant(0, np.uint32)
    axes = words.constant([0])
    pair = [
        words.op(
            "Unsqueeze", value, axes, dtype=ir.DataType.UINT32, shape=ir.Shape((1,))
        )
        for value in (high, low)
    ]
    ctx.bind_value_for_var(
        eqn.outvars[0],
        words.op(
            "Concat", *pair, dtype=ir.DataType.UINT32, shape=ir.Shape((2,)), axis=0
        ),
    )


def _random_words(
    words: _Words, key: ir.Value, shape: tuple[object, ...]
) -> tuple[ir.Value, ir.Value]:
    high, low = words.grid(shape)
    return words.threefry(key, high, low)


def emit_bits(ctx: NativeLoweringContext, eqn: Any) -> None:
    require_native_prng_opset(ctx.opset, eqn)
    validate_selected_random_equation(eqn)
    words = _Words(ctx)
    high, low = _random_words(
        words, ctx.require_value_for_var(eqn.invars[0]), eqn.params["shape"]
    )
    if eqn.params["bit_width"] == 32:
        result = words.cast(words.op("BitwiseXor", high, low), ir.DataType.UINT32)
    else:
        result = words.op(
            "BitwiseOr",
            words.op(
                "BitShift",
                words.cast(high, ir.DataType.UINT64),
                words.constant(32, np.uint64),
                dtype=ir.DataType.UINT64,
                direction="LEFT",
            ),
            words.cast(low, ir.DataType.UINT64),
            dtype=ir.DataType.UINT64,
        )
    ctx.bind_value_for_var(eqn.outvars[0], result)


def emit_native_uniform(ctx: NativeLoweringContext, eqn: Any) -> bool:
    proof = prove_native_uniform(eqn)
    if proof is None:
        return False
    require_native_prng_opset(ctx.opset, eqn)
    words = _Words(ctx)
    high, low = _random_words(
        words, ctx.require_value_for_var(eqn.invars[0]), proof.shape
    )
    if proof.dtype == np.dtype(np.float32):
        mantissa = words.shift(words.op("BitwiseXor", high, low), 9, "RIGHT")
        target, fraction = ir.DataType.FLOAT, 2.0**-23
    else:
        mantissa = words.op(
            "Add",
            words.op("Mul", high, words.constant(1 << 20)),
            words.shift(low, 12, "RIGHT"),
        )
        target, fraction = ir.DataType.DOUBLE, 2.0**-52
    uniform = words.op(
        "Mul",
        words.cast(mantissa, target),
        words.constant(fraction, proof.dtype),
        dtype=target,
    )
    singleton_shape = ir.Shape((1,) * len(proof.shape))
    dimensions = words.constant([1] * len(proof.shape))
    bounds = [
        words.op(
            "Reshape",
            words.cast(ctx.require_value_for_var(var), target),
            dimensions,
            dtype=target,
            shape=singleton_shape,
        )
        for var in eqn.invars[1:]
    ]
    low_bound, high_bound = bounds
    width = words.op("Sub", high_bound, low_bound, dtype=target)
    scaled = words.op("Mul", uniform, width, dtype=target)
    translated = words.op("Add", scaled, low_bound, dtype=target)
    result = words.op("Max", low_bound, translated, dtype=target, shape=uniform.shape)
    ctx.bind_value_for_var(eqn.outvars[0], result)
    return True


def emit_native_bernoulli(ctx: NativeLoweringContext, eqn: Any) -> bool:
    """Emit the proved scalar-formal carrier with its actual enclosing B binding.

    The shared proof covers both equations and all native caller/body ports.
    Its key/probability formals alone do not declare B, so this finite carrier
    is emitted as a unit rather than retraced as an independent root scope.
    """
    uniform = native_bernoulli_uniform(eqn)
    if uniform is None:
        return False
    require_native_prng_opset(ctx.opset, eqn)
    body = eqn.params["jaxpr"].jaxpr
    for inner, outer in zip(body.invars, eqn.invars, strict=True):
        ctx.bind_value_for_var(inner, ctx.require_value_for_var(outer))
    if not emit_native_uniform(ctx, uniform):
        raise ValueError("native Bernoulli uniform proof changed during emission")
    sample = ctx.require_value_for_var(uniform.outvars[0])
    probability = ctx.require_value_for_var(body.invars[1])
    result = _Words(ctx).op(
        "Less", sample, probability, dtype=ir.DataType.BOOL, shape=sample.shape
    )
    ctx.bind_value_for_var(body.outvars[0], result)
    ctx.bind_value_for_var(eqn.outvars[0], result)
    return True
