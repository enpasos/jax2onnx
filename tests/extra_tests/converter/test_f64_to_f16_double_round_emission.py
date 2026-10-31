# tests/extra_tests/converter/test_f64_to_f16_double_round_emission.py

"""Target-only owner for unused composed DOUBLE-to-FLOAT-to-half RNE emission.

The oracle composes two integer IEEE encoders; neither host floating casts nor
source execution supplies expected values. The full opset-23 CPU matrix and
focused opset-21 controls require ordinary ORT disabled/ALL optimization and
inspect the complete prefix and mathematical body in each saved runtime graph.
The genuine FLOAT rounding is widened exactly for the DOUBLE half-RNE body.
No session enables unsafe cast-chain
elimination or sets the floating-point environment. NaNs have a class-only
observation scope; payload/sign preservation and SOURCE acceptance are unproved.
"""

from __future__ import annotations

import ctypes
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest

from jax2onnx.converter.ir_context import IRContext
from jax2onnx.converter.source_structure import StructureMismatch, StructureUnavailable
from jax2onnx.plugins.jax._f64_to_f16_double_round import (
    emit_f64_to_f16_double_round,
)
from jax2onnx.plugins.jax._source_f64_f16_structure import (
    F64F16DoubleRoundSpec,
    RneGraphMatch,
    check_f64_f16_double_round_graph,
)


_FRACTION64 = (1 << 52) - 1
_SIGN64 = 1 << 63
_MASK64 = (1 << 64) - 1
_OPTIMIZATIONS = ("disabled", "all")
_STATIC_SHAPES = ((), (0,), (1,), (2,), (4,), (8,), (6,), (17,), (19,), (65,), (3, 7))
_ORIGINAL_SIX_WORDS = (
    0x3FF0020000000000,
    0x3FF001FFFFFFFFFF,
    0x3FF0020000000001,
    0xBFF0020000000000,
    0x40EFFC0000000000,
    0x40EFFE0000000000,
)


def _round_integer(significand: int, shift: int) -> int:
    """Round an exact nonnegative integer times 2**shift, with even ties."""
    if shift >= 0:
        return significand << shift
    divisor = 1 << -shift
    quotient, remainder = divmod(significand, divisor)
    return quotient + int(
        2 * remainder > divisor or (2 * remainder == divisor and quotient & 1)
    )


def _round_ieee(
    raw: int,
    *,
    source_exponent_bits: int,
    source_fraction_bits: int,
    target_exponent_bits: int,
    target_fraction_bits: int,
) -> int:
    """Decode exact integer fields and encode one correctly rounded IEEE word."""
    source_width = 1 + source_exponent_bits + source_fraction_bits
    assert 0 <= raw < 1 << source_width
    source_max_exponent = (1 << source_exponent_bits) - 1
    target_max_exponent = (1 << target_exponent_bits) - 1
    source_bias = source_max_exponent >> 1
    target_bias = target_max_exponent >> 1
    sign = (raw >> (source_width - 1)) << (target_exponent_bits + target_fraction_bits)
    exponent = (raw >> source_fraction_bits) & source_max_exponent
    fraction = raw & ((1 << source_fraction_bits) - 1)
    infinity = target_max_exponent << target_fraction_bits
    if exponent == source_max_exponent:
        if fraction:
            raise ValueError("NaN payloads have no integer finite-value reference")
        return sign | infinity
    significand = fraction if exponent == 0 else (1 << source_fraction_bits) | fraction
    if not significand:
        return sign
    unit_exponent = (
        1 - source_bias if exponent == 0 else exponent - source_bias
    ) - source_fraction_bits
    leading_exponent = significand.bit_length() - 1 + unit_exponent
    minimum_normal_exponent = 1 - target_bias
    quantum_exponent = (
        max(leading_exponent, minimum_normal_exponent) - target_fraction_bits
    )
    rounded = _round_integer(significand, unit_exponent - quantum_exponent)
    if leading_exponent < minimum_normal_exponent:
        # The integer subnormal encoding also covers carry into the first normal.
        return sign | rounded
    if rounded == 1 << (target_fraction_bits + 1):
        leading_exponent += 1
        rounded >>= 1
    if leading_exponent > target_bias:
        return sign | infinity
    return (
        sign
        | ((leading_exponent + target_bias) << target_fraction_bits)
        | (rounded - (1 << target_fraction_bits))
    )


def _binary32_bits(raw64: int) -> int:
    return _round_ieee(
        raw64,
        source_exponent_bits=11,
        source_fraction_bits=52,
        target_exponent_bits=8,
        target_fraction_bits=23,
    )


def _double_round_bits(raw64: int) -> int:
    return _round_ieee(
        _binary32_bits(raw64),
        source_exponent_bits=8,
        source_fraction_bits=23,
        target_exponent_bits=5,
        target_fraction_bits=10,
    )


def _exact_binary64(integer: int, binary_exponent: int) -> int:
    """Encode a positive exact scaled integer, without host floating arithmetic."""
    if integer == 0:
        return 0
    top = integer.bit_length() - 1
    assert 0 <= top <= 52
    exponent = top + binary_exponent
    assert -1022 <= exponent <= 1023
    return ((exponent + 1023) << 52) | ((integer << (52 - top)) & _FRACTION64)


def _half_units(encoded: int) -> int:
    """Nonnegative half values in 2**-24 units, extrapolating 65536 at infinity."""
    if encoded == 0x7C00:
        return 1 << 40
    exponent, fraction = encoded >> 10, encoded & 0x3FF
    return fraction if exponent == 0 else (1024 + fraction) << (exponent - 1)


def _half_midpoint(lower: int) -> int:
    return _exact_binary64(_half_units(lower) + _half_units(lower + 1), -25)


def _binary32_midpoint(left: int, right: int) -> int:
    """Encode the exact binary64 midpoint of two adjacent positive normals."""
    assert right == left + 1
    left_exponent = ((left >> 23) & 0xFF) - 127 - 23
    right_exponent = ((right >> 23) & 0xFF) - 127 - 23
    assert 0 < (left >> 23) < 0xFF and 0 < (right >> 23) < 0xFF
    left_significand = (1 << 23) | (left & 0x7FFFFF)
    right_significand = (1 << 23) | (right & 0x7FFFFF)
    unit = min(left_exponent, right_exponent)
    integer = (left_significand << (left_exponent - unit)) + (
        right_significand << (right_exponent - unit)
    )
    return _exact_binary64(integer, unit - 1)


@pytest.fixture(scope="module")
def half_boundary_words() -> tuple[int, ...]:
    finite = [_exact_binary64(_half_units(word), -24) for word in range(0x7C00)]
    positive = finite + [
        _half_midpoint(lower) + offset
        for lower in range(0x7C00)
        for offset in (-1, 0, 1)
    ]
    return tuple(positive + [word | _SIGN64 for word in positive])


@pytest.fixture(scope="module")
def first_stage_boundary_words() -> tuple[int, ...]:
    # Each half midpoint is exactly representable in binary32. Its binary32
    # rounding basin has two boundaries: just outside it, the final half can
    # change. Merely checking midpoint +/- one binary64 ULP misses these edges.
    positive = []
    for lower in range(0x7C00):
        middle32 = _binary32_bits(_half_midpoint(lower))
        for edge in (
            _binary32_midpoint(middle32 - 1, middle32),
            _binary32_midpoint(middle32, middle32 + 1),
        ):
            positive.extend((edge - 1, edge, edge + 1))
    return tuple(positive + [word | _SIGN64 for word in positive])


def _edge_words() -> list[int]:
    positive = [0, 1, _FRACTION64, 1 << 52, 0x7FEFFFFFFFFFFFFF, 0x7FF0000000000000]
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
        (1, -150),
        (1, -149),
        (1, -126),
        ((1 << 25) - 1, 103),
    ):
        middle = _exact_binary64(integer, exponent)
        positive.extend((middle - 1, middle, middle + 1))
    for exponent in range(-13, 16):
        middle = _exact_binary64(1, exponent)
        positive.extend((middle - 1, middle, middle + 1))
    return positive + [word | _SIGN64 for word in positive] + list(_ORIGINAL_SIX_WORDS)


def _mixed_words() -> list[int]:
    # Fixed integer mixing needs neither global RNG state nor an import-time seed.
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


def _assert_ir_composition(
    graph: ir.Graph,
    value: ir.Value,
    result: ir.Value,
    shape: tuple[int | str, ...] | None,
    *,
    saved: bool = False,
) -> RneGraphMatch:
    """Check independent math and exact owners, before or after protective clone."""
    assert len(graph.inputs) == len(graph.outputs) == 1
    assert graph.inputs[0] is value and graph.outputs[0] is result
    try:
        match = check_f64_f16_double_round_graph(
            F64F16DoubleRoundSpec(shape),
            graph,
            value,
            result,
            allow_cast_schema_defaults=saved,
        )
    except (StructureMismatch, StructureUnavailable) as exc:
        raise AssertionError(str(exc)) from exc
    assert match.nodes == tuple(graph)
    nodes = [node for node in graph if node.op_type != "Constant"]
    assert len(nodes) == 65 and len(tuple(graph)) <= 124
    assert len(match.literal_obligations) == 59
    assert 1 <= len(match.constants) <= 59
    if not saved:
        assert len(match.constants) == 59
    first, widened = nodes[:2]
    assert first.inputs[0] is value
    assert widened.inputs[0] is first.outputs[0]
    assert first.outputs[0].dtype == ir.DataType.FLOAT
    assert widened.outputs[0].dtype == ir.DataType.DOUBLE
    assert value.producer() is None and first.outputs[0].producer() is first
    assert widened.outputs[0].producer() is widened
    assert nodes[-1].outputs[0] is result and result.producer() is nodes[-1]
    return match


def _graph(shape: tuple[int | str, ...], *, opset: int = 23) -> ir.Model:
    ctx = IRContext(opset=opset, enable_double_precision=False)
    value = ir.Value(
        name="x", type=ir.TensorType(ir.DataType.DOUBLE), shape=ir.Shape(shape)
    )
    ctx.builder.inputs.append(value)
    result = emit_f64_to_f16_double_round(ctx, value)
    ctx.builder.outputs.append(result)
    _assert_ir_composition(ctx.builder.graph, value, result, shape)
    model = ctx.builder.to_ir_model(name="isolated_unused_double_round")
    assert not model.functions
    assert len(model.graph.inputs) == len(model.graph.outputs) == 1
    _assert_ir_composition(
        model.graph, model.graph.inputs[0], model.graph.outputs[0], shape
    )
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    return model


def _proto_shape(value: onnx.ValueInfoProto) -> tuple[int | str, ...]:
    tensor = value.type.tensor_type
    assert value.type.HasField("tensor_type") and tensor.HasField("shape")
    result: list[int | str] = []
    for dim in tensor.shape.dim:
        assert dim.WhichOneof("value") in ("dim_value", "dim_param")
        result.append(dim.dim_value if dim.HasField("dim_value") else dim.dim_param)
    return tuple(result)


def _assert_saved_composition(
    model: onnx.ModelProto, shape: tuple[int | str, ...]
) -> None:
    """Read the complete actual saved graph without rewriting or shape inference."""
    graph = model.graph
    assert len(graph.input) == len(graph.output) == 1
    assert not model.functions
    assert not graph.sparse_initializer
    assert graph.input[0].name == "x"
    assert graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.DOUBLE
    assert graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    assert _proto_shape(graph.input[0]) == _proto_shape(graph.output[0]) == shape
    values = list(graph.input) + list(graph.value_info) + list(graph.output)
    assert len({value.name for value in values}) == len(values)
    definitions = [graph.input[0].name] + [
        name for node in graph.node for name in node.output
    ]
    initializer_names = {value.name for value in graph.initializer}
    assert len(set(definitions)) == len(definitions)
    assert all(definitions) and not set(definitions).intersection(initializer_names)
    metadata_names = {value.name for value in values}
    assert set(definitions) <= metadata_names
    referenced_names = initializer_names | {
        value.name for value in list(graph.input) + list(graph.output)
    }
    referenced_names.update(
        name for node in graph.node for name in (*node.input, *node.output)
    )
    # ORT may keep descriptive ValueInfo for a deduplicated initializer. These
    # rows define no value and supply no type, shape or literal-storage witness.
    surplus = metadata_names - set(definitions) - initializer_names
    by_name = {value.name: value for value in values}
    for name in surplus:
        assert name not in referenced_names
        assert by_name[name].type.tensor_type.elem_type == onnx.TensorProto.DOUBLE
        assert _proto_shape(by_name[name]) == ()
    for node in graph.node:
        assert all(not attribute.ref_attr_name for attribute in node.attribute)
    actual = ir.from_proto(model)
    _assert_ir_composition(
        actual.graph,
        actual.graph.inputs[0],
        actual.graph.outputs[0],
        shape,
        saved=True,
    )


def _assert_default_rounding_environment() -> None:
    # The pinned Linux CPU qualification uses FE_TONEAREST (0). Observe it;
    # do not call fesetround or import source compiler ScopedSetRound helpers.
    function = ctypes.CDLL(None).fegetround
    function.argtypes = []
    function.restype = ctypes.c_int
    assert function() == 0, "Pinned CPU qualification requires ordinary FE_TONEAREST"


def _session(
    model: ir.Model,
    optimization: str,
    directory: Path,
    shape: tuple[int | str, ...],
) -> ort.InferenceSession:
    _assert_default_rounding_environment()
    path = directory / "actual-optimized.onnx"
    assert not path.exists()
    options = ort.SessionOptions()
    options.graph_optimization_level = {
        "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }[optimization]
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.optimized_model_filepath = str(path)
    session = ort.InferenceSession(
        ir.to_proto(model).SerializeToString(deterministic=True),
        options,
        providers=["CPUExecutionProvider"],
    )
    _assert_default_rounding_environment()
    assert path.is_file() and 0 < path.stat().st_size < 2 * 1024 * 1024
    saved = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(saved, full_check=True)
    if all(dim != 0 for dim in shape):
        _assert_saved_composition(saved, shape)
    return session


def _assert_bits(
    session: ort.InferenceSession,
    words: list[int] | tuple[int, ...],
    shape: tuple[int, ...],
) -> None:
    storage = np.asarray(words, dtype="<u8").reshape(shape)
    values = storage.view("<f8")
    expected = np.fromiter(
        (_double_round_bits(int(word)) for word in storage.flat),
        dtype="<u2",
        count=storage.size,
    ).reshape(shape)
    _assert_default_rounding_environment()
    outputs = session.run(None, {"x": values})
    _assert_default_rounding_environment()
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
        (0x3E60000000000001, 0x0000),
        (0x3E70000000000000, 0x0001),
        (0x3F10000000000000, 0x0400),
        (0x3FF0000000000000, 0x3C00),
        (0x3FF0020000000000, 0x3C00),
        (0x3FF0020000000001, 0x3C00),
        (0x3FF0060000000000, 0x3C02),
        (0x40EFFDFFFFFFFFFF, 0x7C00),
        (0x40EFFE0000000000, 0x7C00),
        (0x7FEFFFFFFFFFFFFF, 0x7C00),
        (0x7FF0000000000000, 0x7C00),
    ],
)
def test_composed_integer_oracle_known_storage_boundaries(
    raw: int, expected: int, sign: int
) -> None:
    assert _double_round_bits(raw | sign) == expected | ((sign >> 63) << 15)


@pytest.mark.parametrize("sign", [0, _SIGN64], ids=["positive", "negative"])
@pytest.mark.parametrize(
    "raw,expected",
    [
        (0, 0x00000000),
        (1, 0x00000000),
        (0x3690000000000000, 0x00000000),
        (0x3690000000000001, 0x00000001),
        (0x36A0000000000000, 0x00000001),
        (0x3810000000000000, 0x00800000),
        (0x3FF0000010000000, 0x3F800000),
        (0x3FF0000010000001, 0x3F800001),
        (0x3FF0000030000000, 0x3F800002),
        (0x47EFFFFFF0000000, 0x7F800000),
        (0x47EFFFFFEFFFFFFF, 0x7F7FFFFF),
        (0x7FEFFFFFFFFFFFFF, 0x7F800000),
        (0x7FF0000000000000, 0x7F800000),
    ],
)
def test_first_stage_integer_oracle_known_storage_boundaries(
    raw: int, expected: int, sign: int
) -> None:
    assert _binary32_bits(raw | sign) == expected | ((sign >> 63) << 31)


def test_complete_corpora_keep_half_grid_and_both_first_stage_basin_edges(
    half_boundary_words: tuple[int, ...],
    first_stage_boundary_words: tuple[int, ...],
) -> None:
    assert len(half_boundary_words) == len(set(half_boundary_words)) == 253952
    assert (
        len(first_stage_boundary_words)
        == len(set(first_stage_boundary_words))
        == 380928
    )
    assert not set(half_boundary_words).intersection(first_stage_boundary_words)
    for lower in range(0x7C00):
        exact = _exact_binary64(_half_units(lower), -24)
        middle = _half_midpoint(lower)
        middle32 = _binary32_bits(middle)
        even_half = lower + (lower & 1)
        for sign in (0, _SIGN64):
            half_sign = (sign >> 63) << 15
            assert _double_round_bits(exact | sign) == lower | half_sign
            for offset in (-1, 0, 1):
                actual = _double_round_bits((middle + offset) | sign)
                assert actual == even_half | half_sign
            before = _binary32_midpoint(middle32 - 1, middle32)
            after = _binary32_midpoint(middle32, middle32 + 1)
            for raw, expected32, expected16 in (
                (before - 1, middle32 - 1, lower),
                (before, middle32, even_half),
                (before + 1, middle32, even_half),
                (after - 1, middle32, even_half),
                (after, middle32, even_half),
                (after + 1, middle32 + 1, lower + 1),
            ):
                assert _binary32_bits(raw | sign) == expected32 | ((sign >> 63) << 31)
                assert _double_round_bits(raw | sign) == expected16 | half_sign


@pytest.mark.parametrize("optimization", _OPTIMIZATIONS)
def test_all_finite_half_values_and_midpoint_binary64_neighbors(
    optimization: str,
    half_boundary_words: tuple[int, ...],
    tmp_path: Path,
) -> None:
    shape = (len(half_boundary_words),)
    _assert_bits(
        _session(_graph(shape), optimization, tmp_path, shape),
        half_boundary_words,
        shape,
    )


@pytest.mark.parametrize("optimization", _OPTIMIZATIONS)
def test_first_stage_rounding_basin_neighbors_at_every_half_midpoint(
    optimization: str,
    first_stage_boundary_words: tuple[int, ...],
    tmp_path: Path,
) -> None:
    shape = (len(first_stage_boundary_words),)
    _assert_bits(
        _session(_graph(shape), optimization, tmp_path, shape),
        first_stage_boundary_words,
        shape,
    )


@pytest.mark.parametrize("optimization", _OPTIMIZATIONS)
def test_raw64_mixture_extremals_and_both_stage_subnormal_boundaries(
    optimization: str, tmp_path: Path
) -> None:
    words = _mixed_words()
    shape = (len(words),)
    _assert_bits(_session(_graph(shape), optimization, tmp_path, shape), words, shape)


@pytest.mark.parametrize("optimization", _OPTIMIZATIONS)
@pytest.mark.parametrize("shape", _STATIC_SHAPES)
def test_scalars_empty_lane_widths_odd_tails_rank_two_and_unchanged_six(
    optimization: str, shape: tuple[int, ...], tmp_path: Path
) -> None:
    session = _session(_graph(shape), optimization, tmp_path, shape)
    size = int(np.prod(shape)) if shape else 1
    if size == 0:
        _assert_bits(session, [], shape)
        return
    edges = _edge_words()
    for start in range(0, len(edges), size):
        words = [edges[(start + index) % len(edges)] for index in range(size)]
        _assert_bits(session, words, shape)
    assert [_double_round_bits(word) for word in _ORIGINAL_SIX_WORDS] == [
        0x3C00,
        0x3C00,
        0x3C00,
        0xBC00,
        0x7BFF,
        0x7C00,
    ]
    if shape == (6,):
        _assert_bits(session, _ORIGINAL_SIX_WORDS, shape)
    # All six are already individual scalar inputs above. Here each also appears
    # in the final scalar/tail slot, including the full eight-lane vector case.
    for last in _ORIGINAL_SIX_WORDS:
        words = [_ORIGINAL_SIX_WORDS[index % 6] for index in range(size)]
        words[-1] = last
        _assert_bits(session, words, shape)


@pytest.mark.parametrize("optimization", _OPTIMIZATIONS)
def test_one_dynamic_graph_reused_for_distinct_lengths(
    optimization: str, tmp_path: Path
) -> None:
    session = _session(_graph(("N",)), optimization, tmp_path, ("N",))
    words = _mixed_words()
    for length in (0, 1, 2, 4, 8, 6, 17, 19, 65, 257, len(words)):
        _assert_bits(session, words[:length], (length,))


@pytest.mark.parametrize("optimization", _OPTIMIZATIONS)
def test_supported_opset_twenty_one_executes_the_same_connected_chain(
    optimization: str, tmp_path: Path
) -> None:
    words = _edge_words()
    shape = (len(words),)
    _assert_bits(
        _session(_graph(shape, opset=21), optimization, tmp_path, shape), words, shape
    )


@pytest.mark.parametrize("opset", [12, 21, 23])
@pytest.mark.parametrize("shape", [(), (0,), ("N", 7)])
def test_ir_and_serialized_chain_keep_real_float_intermediate_and_shape(
    opset: int, shape: tuple[int | str, ...]
) -> None:
    _assert_saved_composition(ir.to_proto(_graph(shape, opset=opset)), shape)


def test_unknown_rank_metadata_is_preserved_by_both_casts() -> None:
    ctx = IRContext(opset=23, enable_double_precision=False)
    value = ir.Value(name="x", type=ir.TensorType(ir.DataType.DOUBLE), shape=None)
    ctx.builder.inputs.append(value)
    result = emit_f64_to_f16_double_round(ctx, value)
    ctx.builder.outputs.append(result)
    _assert_ir_composition(ctx.builder.graph, value, result, None)
    model = ctx.builder.to_ir_model(name="unknown_rank_double_round")
    _assert_ir_composition(
        model.graph, model.graph.inputs[0], model.graph.outputs[0], None
    )
    assert all(value.shape is None for node in model.graph for value in node.outputs)


@pytest.mark.parametrize(
    "dtype",
    [None, ir.DataType.FLOAT16, ir.DataType.FLOAT, ir.DataType.INT64, ir.DataType.BOOL],
)
def test_invalid_input_fails_before_any_graph_mutation(
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
        emit_f64_to_f16_double_round(ctx, value)
    assert before == (
        tuple(ctx.builder.graph),
        tuple(ctx.builder.initializers),
        tuple(ctx.builder.outputs),
        tuple(ctx.builder.inputs),
    )


@pytest.mark.parametrize("opset", [5, 6, 7, 8, 9, 10, 11])
def test_integer_cast_attribute_schema_floor_fails_before_graph_mutation(
    opset: int,
) -> None:
    ctx = IRContext(opset=opset, enable_double_precision=False)
    value = ir.Value(
        name="x", type=ir.TensorType(ir.DataType.DOUBLE), shape=ir.Shape((3,))
    )
    ctx.builder.inputs.append(value)
    before = (
        tuple(ctx.builder.graph),
        tuple(ctx.builder.initializers),
        tuple(ctx.builder.outputs),
        tuple(ctx.builder.inputs),
    )
    with pytest.raises(ValueError):
        emit_f64_to_f16_double_round(ctx, value)
    assert before == (
        tuple(ctx.builder.graph),
        tuple(ctx.builder.initializers),
        tuple(ctx.builder.outputs),
        tuple(ctx.builder.inputs),
    )


@pytest.mark.parametrize(
    "fault",
    [
        "collapse",
        "intermediate-type",
        "intermediate-to",
        "final-to",
        "final-type",
        "input-type",
        "missing-metadata",
        "bypass",
        "reverse",
        "domain",
        "overload",
        "attribute",
        "shape",
        "input-shape",
        "output-shape",
        "owner",
        "initializer",
        "extra-node",
    ],
)
def test_saved_graph_checker_rejects_chain_contract_mutations(fault: str) -> None:
    shape = ("N", 7)
    model = ir.to_proto(_graph(shape))
    first, second = model.graph.node[:2]
    final = model.graph.node[-1]
    metadata = {value.name: value for value in model.graph.value_info}
    if fault == "collapse":
        second.input[0] = first.input[0]
        del model.graph.node[0]
    elif fault == "intermediate-type":
        metadata[first.output[0]].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    elif fault == "intermediate-to":
        first.attribute[0].i = onnx.TensorProto.DOUBLE
    elif fault == "final-to":
        final.attribute[0].i = onnx.TensorProto.FLOAT
    elif fault == "final-type":
        model.graph.output[0].type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    elif fault == "input-type":
        model.graph.input[0].type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    elif fault == "missing-metadata":
        del model.graph.value_info[:]
    elif fault == "bypass":
        second.input[0] = first.input[0]
    elif fault == "reverse":
        copies = [onnx.NodeProto(), onnx.NodeProto()]
        copies[0].CopyFrom(second)
        copies[1].CopyFrom(first)
        model.graph.node[0].CopyFrom(copies[0])
        model.graph.node[1].CopyFrom(copies[1])
    elif fault == "domain":
        first.domain = "unreviewed"
    elif fault == "overload":
        first.overload = "unreviewed"
    elif fault == "attribute":
        first.attribute.append(onnx.helper.make_attribute("saturate", 0))
    elif fault == "shape":
        metadata[first.output[0]].type.tensor_type.shape.dim[1].dim_value = 8
    elif fault == "input-shape":
        model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 8
    elif fault == "output-shape":
        model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 8
    elif fault == "owner":
        model.graph.input[0].name = "foreign_input"
        first.input[0] = "foreign_input"
    elif fault == "initializer":
        model.graph.initializer.append(
            onnx.helper.make_tensor("unused", onnx.TensorProto.FLOAT, [], [1.0])
        )
    elif fault == "extra-node":
        model.graph.node.append(
            onnx.helper.make_node("Identity", [first.input[0]], ["unused"])
        )
    with pytest.raises(AssertionError):
        _assert_saved_composition(model, shape)


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "type",
        "shape",
        "duplicate",
        "referenced",
        "missing-compute",
        "physical-type",
        "physical-shape",
        "storage",
    ],
)
def test_saved_metadata_only_rows_cannot_supply_definitions_or_literal_storage(
    fault: str,
) -> None:
    shape = ("N", 7)
    model = ir.to_proto(_graph(shape))
    unused = onnx.helper.make_tensor_value_info(
        "deduplicated_literal_description", onnx.TensorProto.DOUBLE, []
    )
    model.graph.value_info.append(unused)
    if fault == "none":
        _assert_saved_composition(model, shape)
        return
    if fault == "type":
        model.graph.value_info[-1].type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    elif fault == "shape":
        model.graph.value_info[-1].type.tensor_type.shape.dim.add().dim_value = 1
    elif fault == "duplicate":
        model.graph.value_info.append(unused)
    elif fault == "referenced":
        comparison = next(
            node for node in model.graph.node if node.op_type == "GreaterOrEqual"
        )
        comparison.input[1] = unused.name
    elif fault == "missing-compute":
        first_output = model.graph.node[0].output[0]
        position = next(
            index
            for index, value in enumerate(model.graph.value_info)
            if value.name == first_output
        )
        del model.graph.value_info[position]
    elif fault in ("physical-type", "physical-shape"):
        first_output = model.graph.node[0].output[0]
        physical = next(
            value for value in model.graph.value_info if value.name == first_output
        )
        if fault == "physical-type":
            physical.type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
        else:
            physical.type.tensor_type.shape.dim[1].dim_value = 8
    else:
        tensor = model.graph.initializer[0]
        assert len(tensor.raw_data) == 8
        tensor.raw_data = bytes([tensor.raw_data[0] ^ 1]) + tensor.raw_data[1:]
    with pytest.raises(AssertionError):
        _assert_saved_composition(model, shape)


@pytest.mark.parametrize("optimization", _OPTIMIZATIONS)
def test_nan_class_with_explicit_payload_unverified_scope(
    optimization: str,
    tmp_path: Path,
    record_property: Callable[[str, object], None],
) -> None:
    words = [
        (0x7FF << 52) | payload | sign
        for sign in (0, _SIGN64)
        for payload in (1, 0x12345, 1 << 51, (1 << 51) | 0x1555555555555, _FRACTION64)
    ]
    for raw in words:
        with pytest.raises(ValueError, match="NaN payloads"):
            _double_round_bits(raw)
    shape = (len(words),)
    session = _session(_graph(shape), optimization, tmp_path, shape)
    _assert_default_rounding_environment()
    outputs = session.run(None, {"x": np.asarray(words, dtype="<u8").view("<f8")})
    _assert_default_rounding_environment()
    assert len(outputs) == 1 and type(outputs[0]) is np.ndarray
    actual = outputs[0]
    assert actual.dtype == np.dtype("<f2") and actual.shape == shape
    output_words = actual.view("<u2")
    assert all(
        (int(word) & 0x7C00) == 0x7C00 and int(word) & 0x3FF for word in output_words
    )
    record_property(
        "unused-double-round-nan-observations",
        json.dumps(
            {
                "optimization": optimization,
                "scope": "NaN class only; payload and sign observed but unverified",
                "SOURCE_acceptance_claimed": False,
                "original_criteria_changed": False,
                "words": [
                    {"input": f"{raw:016x}", "output": f"{int(output):04x}"}
                    for raw, output in zip(words, output_words)
                ],
            },
            sort_keys=True,
        ),
    )
