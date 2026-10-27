# tests/extra_tests/converter/test_f64_to_f16_rne_emission.py

"""Target-only checks for an unused RNE emitter, independent of SOURCE policy.

The reference reads integer IEEE fields and rounds an exact rational by integer
quotient/remainder. It never casts through a host or JAX half-precision value.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest

from jax2onnx.converter.function_scope import FunctionScope
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.plugins.jax._f64_to_f16_rne import emit_f64_to_f16_rne


_FRACTION64 = (1 << 52) - 1
_SIGN64 = 1 << 63
_MASK64 = (1 << 64) - 1
_OPTIMIZATIONS = ("disabled", "all")
_ORIGINAL_SIX_WORDS = (
    0x3FF0020000000000,
    0x3FF001FFFFFFFFFF,
    0x3FF0020000000001,
    0xBFF0020000000000,
    0x40EFFC0000000000,
    0x40EFFE0000000000,
)


def _nearest_integer(significand: int, binary_shift: int) -> int:
    if binary_shift >= 0:
        return significand << binary_shift
    divisor = 1 << -binary_shift
    quotient, remainder = divmod(significand, divisor)
    twice = remainder * 2
    return quotient + int(twice > divisor or (twice == divisor and quotient & 1))


def _rne_half_bits(raw: int) -> int:
    """Round one exact binary64 storage word; NaNs have no payload oracle."""
    sign = (raw >> 63) << 15
    exponent = (raw >> 52) & 0x7FF
    fraction = raw & _FRACTION64
    if exponent == 0x7FF:
        if fraction:
            raise ValueError("NaN payloads are observations, not finite references")
        return sign | 0x7C00
    significand = fraction if exponent == 0 else (1 << 52) | fraction
    binary_exponent = -1074 if exponent == 0 else exponent - 1023 - 52
    if significand == 0:
        return sign
    leading_exponent = significand.bit_length() - 1 + binary_exponent
    if leading_exponent < -14:
        # All half subnormals share one exact 2**-24 spacing. A carry produces
        # the minimum normal encoding without a second rounding operation.
        return sign | _nearest_integer(significand, binary_exponent + 24)
    if leading_exponent > 15:
        return sign | 0x7C00
    rounded = _nearest_integer(significand, binary_exponent - leading_exponent + 10)
    if rounded == 2048:
        leading_exponent += 1
        rounded = 1024
    if leading_exponent > 15:
        return sign | 0x7C00
    return sign | ((leading_exponent + 15) << 10) | (rounded - 1024)


def _exact_binary64(integer: int, binary_exponent: int) -> int:
    """Encode an exactly representable positive scaled integer without floats."""
    if integer == 0:
        return 0
    top = integer.bit_length() - 1
    assert 0 <= top <= 52
    exponent = top + binary_exponent
    assert -1022 <= exponent <= 1023
    fraction = (integer << (52 - top)) & _FRACTION64
    return ((exponent + 1023) << 52) | fraction


def _half_units(encoded: int) -> int:
    """Exact nonnegative half values in units of 2**-24, plus 65536."""
    if encoded == 0x7C00:
        return 1 << 40  # The finite extrapolated endpoint at the overflow tie.
    exponent, fraction = encoded >> 10, encoded & 0x3FF
    return fraction if exponent == 0 else (1024 + fraction) << (exponent - 1)


@pytest.fixture(scope="module")
def exhaustive_words() -> tuple[int, ...]:
    finite = [_exact_binary64(_half_units(word), -24) for word in range(0x7C00)]
    around_midpoints = []
    for lower in range(0x7C00):
        midpoint = _exact_binary64(_half_units(lower) + _half_units(lower + 1), -25)
        around_midpoints.extend((midpoint - 1, midpoint, midpoint + 1))
    positive = finite + around_midpoints
    return tuple(positive + [word | _SIGN64 for word in positive])


def _edge_words() -> list[int]:
    positive = [
        0,
        1,
        _FRACTION64,
        1 << 52,
        0x7FEFFFFFFFFFFFFF,
        0x7FF0000000000000,
    ]
    for integer, exponent in (
        (1, -25),
        (1, -24),
        (1023, -24),
        (2047, -25),
        (1, -14),
        (1, 0),
        (2047, 5),
        (4095, 4),
        (1, 16),
    ):
        middle = _exact_binary64(integer, exponent)
        positive.extend((middle - 1, middle, middle + 1))
    for exponent in range(-13, 16):
        threshold = _exact_binary64(1, exponent)
        positive.extend((threshold - 1, threshold, threshold + 1))
    return positive + [word | _SIGN64 for word in positive] + list(_ORIGINAL_SIX_WORDS)


def _deterministic_words() -> list[int]:
    # Fixed integer mixing, without global NumPy/JAX state or import-time seeds.
    result = []
    state = 0xC3A5C85C97CB3127
    for _ in range(8192):
        state = (state + 0x9E3779B97F4A7C15) & _MASK64
        mixed = state
        mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & _MASK64
        mixed ^= mixed >> 31
        if (mixed >> 52) & 0x7FF != 0x7FF:
            result.append(mixed)
    return result + _edge_words()


def _input(shape: tuple[int | str, ...], name: str = "x") -> ir.Value:
    return ir.Value(
        name=name, type=ir.TensorType(ir.DataType.DOUBLE), shape=ir.Shape(shape)
    )


def _graph(
    shape: tuple[int | str, ...],
    *,
    opset: int = 23,
    constant: np.ndarray | None = None,
    local_function: bool = False,
) -> ir.Model:
    ctx = IRContext(opset=opset, enable_double_precision=False)
    value = _input(shape)
    if constant is None:
        ctx.builder.inputs.append(value)
    else:
        value = ctx.bind_const_for_var(object(), constant)
        assert value.dtype == ir.DataType.DOUBLE
        assert value.const_value is not None
        storage = value.const_value.numpy()
        assert storage.dtype == constant.dtype == np.dtype("<f8")
        assert storage.shape == constant.shape
        assert storage.tobytes() == constant.tobytes()
    function = None
    if local_function:
        scope = FunctionScope(ctx, name="UnusedDirectHalfRNE", domain="test.unused.rne")
        formal = scope.begin([value])[0]
        rounded = emit_f64_to_f16_rne(scope.ctx, formal)
        function = scope.end([rounded])
        result = ctx.builder.op(
            function.name,
            [value],
            domain=function.domain,
            version=opset,
            output=ir.Value(
                name="result",
                type=ir.TensorType(ir.DataType.FLOAT16),
                shape=value.shape,
            ),
        )
    else:
        result = emit_f64_to_f16_rne(ctx, value)
    assert result.dtype == ir.DataType.FLOAT16
    assert result.shape == ir.Shape(shape)
    ctx.builder.outputs.append(result)
    model = ctx.builder.to_ir_model(name="isolated_unused_half_rne")
    if function is not None:
        model.functions[function.identifier()] = function
        model.graph.opset_imports[function.domain] = opset
        assert model.graph.opset_imports[function.domain] == opset
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    return model


def _session(model: ir.Model, optimization: str) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = {
        "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }[optimization]
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(
        ir.to_proto(model).SerializeToString(deterministic=True),
        options,
        providers=["CPUExecutionProvider"],
    )


def _assert_bits(
    session: ort.InferenceSession,
    words: list[int] | tuple[int, ...],
    shape: tuple[int, ...],
    *,
    constant: bool = False,
) -> None:
    storage = np.asarray(words, dtype="<u8").reshape(shape)
    values = storage.view("<f8")
    expected = np.fromiter(
        (_rne_half_bits(int(word)) for word in storage.flat),
        dtype="<u2",
        count=storage.size,
    ).reshape(shape)
    outputs = session.run(None, {} if constant else {"x": values})
    assert len(outputs) == 1 and type(outputs[0]) is np.ndarray
    actual = outputs[0]
    assert actual.dtype == np.dtype("<f2") and actual.shape == shape
    actual_bits = actual.view("<u2")
    different = np.flatnonzero(actual_bits.reshape(-1) != expected.reshape(-1))
    details = [
        {
            "index": int(index),
            "input": f"{int(storage.flat[index]):016x}",
            "expected": f"{int(expected.flat[index]):04x}",
            "actual": f"{int(actual_bits.flat[index]):04x}",
        }
        for index in different[:12]
    ]
    assert different.size == 0, {
        "different_count": int(different.size),
        "first": details,
    }


@pytest.mark.parametrize("sign", [0, _SIGN64], ids=["positive", "negative"])
@pytest.mark.parametrize(
    "raw,expected",
    [
        (0, 0x0000),
        (1, 0x0000),
        (0x3E60000000000000, 0x0000),
        (0x3E60000000000001, 0x0001),
        (0x3E70000000000000, 0x0001),
        (0x3F10000000000000, 0x0400),
        (0x3FF0000000000000, 0x3C00),
        (0x3FF0020000000000, 0x3C00),
        (0x3FF0020000000001, 0x3C01),
        (0x3FF0060000000000, 0x3C02),
        (0x40EFFDFFFFFFFFFF, 0x7BFF),
        (0x40EFFE0000000000, 0x7C00),
        (0x7FEFFFFFFFFFFFFF, 0x7C00),
        (0x7FF0000000000000, 0x7C00),
    ],
)
def test_integer_oracle_known_storage_boundaries(
    raw: int, expected: int, sign: int
) -> None:
    assert _rne_half_bits(raw | sign) == expected | ((sign >> 63) << 15)


def test_exhaustive_corpus_keeps_all_finite_values_and_midpoint_neighbors(
    exhaustive_words: tuple[int, ...],
) -> None:
    assert len(exhaustive_words) == len(set(exhaustive_words)) == 253952
    assert sum(word < _SIGN64 for word in exhaustive_words) == 126976
    for encoded in range(0x7C00):
        positive = _exact_binary64(_half_units(encoded), -24)
        assert _rne_half_bits(positive) == encoded
        assert _rne_half_bits(positive | _SIGN64) == encoded | 0x8000
        midpoint = _exact_binary64(_half_units(encoded) + _half_units(encoded + 1), -25)
        for raw, expected in (
            (midpoint - 1, encoded),
            (midpoint, encoded + (encoded & 1)),
            (midpoint + 1, encoded + 1),
        ):
            assert _rne_half_bits(raw) == expected
            assert _rne_half_bits(raw | _SIGN64) == expected | 0x8000


@pytest.mark.parametrize("optimization", _OPTIMIZATIONS)
def test_all_finite_half_values_and_adjacent_midpoint_neighbors(
    optimization: str,
    exhaustive_words: tuple[int, ...],
) -> None:
    model = _graph((len(exhaustive_words),))
    _assert_bits(
        _session(model, optimization), exhaustive_words, (len(exhaustive_words),)
    )


@pytest.mark.parametrize("optimization", _OPTIMIZATIONS)
def test_deterministic_raw64_extremals_and_subnormal_inputs(optimization: str) -> None:
    words = _deterministic_words()
    _assert_bits(_session(_graph((len(words),)), optimization), words, (len(words),))


@pytest.mark.parametrize("optimization", _OPTIMIZATIONS)
@pytest.mark.parametrize("shape", [(), (17,), (19,), (65,), (3, 7)])
def test_scalar_odd_tail_and_rank_two_shapes(
    optimization: str,
    shape: tuple[int, ...],
) -> None:
    session = _session(_graph(shape), optimization)
    edges = _edge_words()
    size = int(np.prod(shape)) if shape else 1
    # Reuse the scalar graph across all edges, and rotate odd tails so critical
    # midpoint/zero/overflow values occupy both vector lanes and the tail.
    for start in range(0, len(edges), size):
        words = [edges[(start + index) % len(edges)] for index in range(size)]
        _assert_bits(session, words, shape)


@pytest.mark.parametrize("optimization", _OPTIMIZATIONS)
@pytest.mark.parametrize("shape", [(), (17,), (19,), (65,), (3, 7)])
def test_original_six_words_in_scalar_and_every_final_tail_slot(
    optimization: str,
    shape: tuple[int, ...],
) -> None:
    assert _ORIGINAL_SIX_WORDS[2] == 0x3FF0020000000001
    assert [_rne_half_bits(word) for word in _ORIGINAL_SIX_WORDS] == [
        0x3C00,
        0x3C00,
        0x3C01,
        0xBC00,
        0x7BFF,
        0x7C00,
    ]
    session = _session(_graph(shape), optimization)
    size = int(np.prod(shape)) if shape else 1
    for last in _ORIGINAL_SIX_WORDS:
        words = [_ORIGINAL_SIX_WORDS[index % 6] for index in range(size)]
        words[-1] = last
        assert words[-1] == last
        _assert_bits(session, words, shape)


@pytest.mark.parametrize("optimization", _OPTIMIZATIONS)
def test_one_dynamic_graph_reused_for_distinct_runtime_lengths(
    optimization: str,
) -> None:
    session = _session(_graph(("N",)), optimization)
    words = _deterministic_words()
    for length in (1, 19, 257, len(words)):
        _assert_bits(session, words[:length], (length,))


@pytest.mark.parametrize("optimization", _OPTIMIZATIONS)
def test_constant_only_graph_preserves_actual_input_and_rounded_bits(
    optimization: str,
) -> None:
    words = _edge_words()
    constant = np.asarray(words, dtype="<u8").view("<f8")
    model = _graph((len(words),), constant=constant)
    assert not model.graph.inputs
    _assert_bits(_session(model, optimization), words, (len(words),), constant=True)


@pytest.mark.parametrize("optimization", _OPTIMIZATIONS)
def test_local_function_executes_with_its_own_double_constants(
    optimization: str,
) -> None:
    words = _edge_words()
    model = _graph((len(words),), local_function=True)
    assert not model.graph.initializers
    assert len(model.functions) == 1
    function = next(iter(model.functions.values()))
    assert not function.graph.initializers
    constants = [node for node in function.graph if node.op_type == "Constant"]
    assert len(constants) == 59
    owned = {id(value) for value in function.inputs}
    for node in function.graph:
        assert all(value is None or id(value) in owned for value in node.inputs)
        owned.update(id(value) for value in node.outputs)
    assert all(id(value) in owned for value in function.outputs)
    _assert_bits(_session(model, optimization), words, (len(words),))


@pytest.mark.parametrize("opset", [12, 23])
def test_actual_ir_and_constants_keep_double_until_the_final_half_cast(
    opset: int,
) -> None:
    model = _graph(("N", 7), opset=opset)
    nodes = list(model.graph)
    assert [node.op_type for node in nodes].count("Round") == 1
    assert [node.op_type for node in nodes].count("GreaterOrEqual") == 29
    assert [node.op_type for node in nodes].count("Where") == 29
    assert nodes[-1].op_type == "Cast"
    assert all(node.op_type != "Cast" for node in nodes[:-1])
    for node in nodes:
        expected = (
            ir.DataType.BOOL
            if node.op_type == "GreaterOrEqual"
            else ir.DataType.FLOAT16 if node.op_type == "Cast" else ir.DataType.DOUBLE
        )
        assert all(
            value.dtype == expected and value.shape == ir.Shape(("N", 7))
            for value in node.outputs
        )
        assert all(
            value is None or value.dtype in (ir.DataType.DOUBLE, ir.DataType.BOOL)
            for value in node.inputs
        )
    constants = list(model.graph.initializers.values())
    assert len(constants) == 59
    actual_constants = []
    for value in constants:
        assert value.dtype == ir.DataType.DOUBLE and value.shape == ir.Shape(())
        assert value.const_value is not None
        storage = value.const_value.numpy()
        assert storage.dtype == np.dtype("<f8") and storage.shape == ()
        actual_constants.append(storage.tobytes())
    expected_constants = [np.asarray(2.0**24, dtype="<f8").tobytes()]
    for exponent in range(-13, 16):
        expected_constants.extend(
            (
                np.asarray(2.0**exponent, dtype="<f8").tobytes(),
                np.asarray(2.0 ** (10 - exponent), dtype="<f8").tobytes(),
            )
        )
    assert actual_constants == expected_constants


@pytest.mark.parametrize(
    "dtype", [None, ir.DataType.FLOAT, ir.DataType.INT64, ir.DataType.BOOL]
)
def test_non_double_inputs_fail_before_any_graph_mutation(
    dtype: ir.DataType | None,
) -> None:
    ctx = IRContext(opset=23, enable_double_precision=False)
    value = ir.Value(
        name="x",
        type=ir.TensorType(dtype) if dtype is not None else None,
        shape=ir.Shape((3,)),
    )
    ctx.builder.inputs.append(value)
    before = (
        tuple(ctx.builder.graph),
        tuple(ctx.builder.initializers),
        tuple(ctx.builder.outputs),
        tuple(ctx.builder.inputs),
    )
    with pytest.raises(TypeError):
        emit_f64_to_f16_rne(ctx, value)
    assert before == (
        tuple(ctx.builder.graph),
        tuple(ctx.builder.initializers),
        tuple(ctx.builder.outputs),
        tuple(ctx.builder.inputs),
    )


def test_schema_floor_rejects_opset_eleven_before_graph_mutation() -> None:
    ctx = IRContext(opset=11, enable_double_precision=False)
    value = _input((3,))
    ctx.builder.inputs.append(value)
    before = (
        tuple(ctx.builder.graph),
        tuple(ctx.builder.initializers),
        tuple(ctx.builder.outputs),
        tuple(ctx.builder.inputs),
    )
    with pytest.raises(ValueError):
        emit_f64_to_f16_rne(ctx, value)
    assert before == (
        tuple(ctx.builder.graph),
        tuple(ctx.builder.initializers),
        tuple(ctx.builder.outputs),
        tuple(ctx.builder.inputs),
    )


@pytest.mark.parametrize("optimization", _OPTIMIZATIONS)
def test_nan_class_with_payload_observations_separate_from_source_acceptance(
    optimization: str,
    record_property: Callable[[str, object], None],
) -> None:
    words = [
        ((0x7FF << 52) | payload | sign)
        for sign in (0, _SIGN64)
        for payload in (1, 0x12345, 1 << 51, (1 << 51) | 0x1555555555555, _FRACTION64)
    ]
    values = np.asarray(words, dtype="<u8").view("<f8")
    session = _session(_graph((len(words),)), optimization)
    result = session.run(None, {"x": values})[0]
    assert (
        type(result) is np.ndarray
        and result.dtype == np.dtype("<f2")
        and result.shape == (len(words),)
    )
    output_words = result.view("<u2").reshape(-1)
    assert all(
        (int(word) & 0x7C00) == 0x7C00 and int(word) & 0x3FF for word in output_words
    )
    record_property(
        "unused-rne-nan-observations",
        json.dumps(
            {
                "optimization": optimization,
                "scope": "NaN class; payload and sign observed only",
                "SOURCE_acceptance_claimed": False,
                "original_criteria_changed": False,
                "words": [
                    {"input": f"{raw:016x}", "output": f"{int(actual):04x}"}
                    for raw, actual in zip(words, output_words)
                ],
            },
            sort_keys=True,
        ),
    )
