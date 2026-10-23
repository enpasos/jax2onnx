# internal_docs/precision/poc/test_source_export.py

"""Executable checks for the bounded source-dtype integration experiment."""

from collections.abc import Callable, Iterator
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from internal_docs.precision.poc.source_export import (
    PrototypeLimit,
    RepresentationLimit,
    export_source_dtypes,
)
from jax2onnx import onnx_function, to_onnx
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.converter.precision_policy import PrecisionPolicy


@pytest.fixture(autouse=True)
def _restore_x64() -> Iterator[None]:
    previous = bool(jax.config.x64_enabled)
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _spec(value: np.ndarray) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(value.shape, value.dtype)


def _run_cpu(model: onnx.ModelProto, *inputs: np.ndarray) -> list[np.ndarray]:
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    feeds = {
        spec.name: value
        for spec, value in zip(session.get_inputs(), inputs, strict=True)
    }
    return session.run(None, feeds)


def _assert_output(actual: np.ndarray, expected: object) -> None:
    reference = np.asarray(expected)
    assert actual.dtype == reference.dtype
    np.testing.assert_array_equal(actual, reference)


def test_float32_rounding_and_arithmetic_differ_from_legacy_double_export() -> None:
    def fn(x: jax.Array) -> jax.Array:
        rounded = x.astype(jnp.float32)
        return (rounded + np.float32(1)) - rounded

    x = np.asarray(1e8, dtype=np.float64)
    source_model = export_source_dtypes(fn, [_spec(x)])
    legacy_model = to_onnx(fn, [_spec(x)], enable_double_precision=True)
    (source,) = _run_cpu(source_model, x)
    (legacy,) = _run_cpu(legacy_model, x)

    _assert_output(source, fn(jnp.asarray(x)))
    _assert_output(source, np.asarray(0, dtype=np.float32))
    _assert_output(legacy, np.asarray(1, dtype=np.float64))
    assert jax.config.x64_enabled


def test_shared_value_retains_float32_alongside_float64_branches() -> None:
    def fn(x: jax.Array) -> tuple[jax.Array, ...]:
        rounded = x.astype(jnp.float32)
        return x + x, rounded, -rounded, rounded + rounded

    x = np.asarray([0.1, 2**24 + 1, -0.0], dtype=np.float64)
    outputs = _run_cpu(export_source_dtypes(fn, [_spec(x)]), x)
    for actual, expected in zip(outputs, fn(jnp.asarray(x)), strict=True):
        _assert_output(actual, expected)
    assert [value.dtype for value in outputs] == [
        np.float64,
        np.float32,
        np.float32,
        np.float32,
    ]


def test_float16_work_selected_float32_work_and_deliberate_cast_back() -> None:
    one16 = np.ones((2,), dtype=np.float16)
    one32 = np.ones((2,), dtype=np.float32)

    def fn(x: jax.Array) -> tuple[jax.Array, ...]:
        lower = jax.lax.add(x, one16)
        wider = jax.lax.add(x.astype(jnp.float32), one32)
        return lower, wider, wider.astype(jnp.float16)

    x = np.asarray([2048, 1], dtype=np.float16)
    outputs = _run_cpu(export_source_dtypes(fn, [_spec(x)]), x)
    expected = (
        np.asarray([2048, 2], dtype=np.float16),
        np.asarray([2049, 2], dtype=np.float32),
        np.asarray([2048, 2], dtype=np.float16),
    )
    for actual, reference, jax_result in zip(
        outputs, expected, fn(jnp.asarray(x)), strict=True
    ):
        _assert_output(actual, reference)
        _assert_output(actual, jax_result)


def test_explicit_float64_reduction_from_float32_input() -> None:
    def fn(x: jax.Array) -> jax.Array:
        return jax.lax.reduce_sum(x.astype(jnp.float64), axes=(0,))

    x = np.asarray([2**24, 1, -(2**24)], dtype=np.float32)
    (actual,) = _run_cpu(export_source_dtypes(fn, [_spec(x)]), x)
    _assert_output(actual, fn(jnp.asarray(x)))
    _assert_output(actual, np.asarray(1, dtype=np.float64))


@pytest.mark.parametrize(
    ("constant_dtype", "input_dtype"),
    [(np.float32, np.float32), (np.float64, np.float64), (np.float32, np.float64)],
    ids=["fp32", "fp64", "fp32_constant_with_fp64_input"],
)
def test_captured_constant_payload_and_output_keep_their_dtype(
    constant_dtype, input_dtype
) -> None:
    weights = np.asarray([1 + 2**-30, -2.5], dtype=constant_dtype)

    def fn(x: jax.Array) -> tuple[np.ndarray, jax.Array]:
        return weights, x + weights

    x = np.asarray([3.0, 4.0], dtype=input_dtype)
    model = export_source_dtypes(fn, [_spec(x)])
    direct, added = _run_cpu(model, x)
    _assert_output(direct, weights)
    _assert_output(added, fn(jnp.asarray(x))[1])

    payloads = [onnx.numpy_helper.to_array(t) for t in model.graph.initializer]
    payloads.extend(
        onnx.numpy_helper.to_array(attribute.t)
        for node in model.graph.node
        if node.op_type == "Constant"
        for attribute in node.attribute
        if attribute.name == "value"
    )
    assert any(
        value.dtype == weights.dtype and np.array_equal(value, weights)
        for value in payloads
    )


def test_bfloat16_boundaries_are_structurally_valid_without_runtime_claim() -> None:
    def fn(x: jax.Array) -> tuple[jax.Array, ...]:
        wider = x.astype(jnp.float32)
        return x, -x, -wider, (-wider).astype(jnp.bfloat16)

    model = export_source_dtypes(fn, [jax.ShapeDtypeStruct((2,), jnp.bfloat16)])
    assert model.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.BFLOAT16
    assert [value.type.tensor_type.elem_type for value in model.graph.output] == [
        onnx.TensorProto.BFLOAT16,
        onnx.TensorProto.BFLOAT16,
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.BFLOAT16,
    ]
    onnx.checker.check_model(model, full_check=True)


def test_bitcast_requires_explicit_opset_and_keeps_shared_float32_bits() -> None:
    def fn(x: jax.Array) -> tuple[jax.Array, ...]:
        rounded = x.astype(jnp.float32)
        bits = jax.lax.bitcast_convert_type(rounded, new_dtype=jnp.uint32)
        return rounded, -rounded, bits

    x = np.asarray([0.1, 2**24 + 1, -0.0, -2.5], dtype=np.float64)
    with pytest.raises(RepresentationLimit, match="selected opset is 23"):
        export_source_dtypes(fn, [_spec(x)])
    assert jax.config.x64_enabled

    direct, negated, bits = _run_cpu(export_source_dtypes(fn, [_spec(x)], opset=26), x)
    rounded = x.astype(np.float32)
    _assert_output(direct, rounded)
    _assert_output(negated, -rounded)
    _assert_output(bits, rounded.view(np.uint32))
    assert np.signbit(direct[2])


def test_float64_input_is_rejected_before_calling_model_when_x64_is_disabled() -> None:
    called = []

    def fn(x: jax.Array) -> jax.Array:
        called.append(True)
        return -x

    jax.config.update("jax_enable_x64", False)
    with pytest.raises(ValueError, match="float64, but JAX x64 is disabled"):
        export_source_dtypes(fn, [jax.ShapeDtypeStruct((2,), jnp.float64)])
    assert not called
    assert not jax.config.x64_enabled


@pytest.mark.parametrize("ambient_x64", [False, True])
def test_success_leaves_ambient_x64_and_explicit_float32_unchanged(ambient_x64) -> None:
    jax.config.update("jax_enable_x64", ambient_x64)
    x = np.asarray([0.1, -2.5], dtype=np.float32)
    model = export_source_dtypes(lambda value: -value, [_spec(x)])
    assert bool(jax.config.x64_enabled) is ambient_x64
    (actual,) = _run_cpu(model, x)
    _assert_output(actual, -x)


@pytest.mark.parametrize("ambient_x64", [False, True])
@pytest.mark.parametrize("failure", ["tracing", "unsupported_primitive"])
def test_failure_leaves_ambient_x64_unchanged(ambient_x64, failure) -> None:
    def fn(x: jax.Array) -> jax.Array:
        if failure == "tracing":
            raise RuntimeError("intentional tracing failure")
        return jax.lax.sin(x)

    jax.config.update("jax_enable_x64", ambient_x64)
    error = RuntimeError if failure == "tracing" else PrototypeLimit
    match = "intentional tracing failure" if failure == "tracing" else "outside the PoC"
    with pytest.raises(error, match=match):
        export_source_dtypes(fn, [jax.ShapeDtypeStruct((2,), jnp.float32)])
    assert bool(jax.config.x64_enabled) is ambient_x64


@onnx_function
def _function_body(x: jax.Array) -> jax.Array:
    return -x


@pytest.mark.parametrize(
    "fn",
    [
        jax.jit(lambda x: -x),
        lambda x: jax.lax.scan(lambda carry, _: (-carry, carry), x, None, length=2)[0],
    ],
    ids=["jit", "scan"],
)
def test_nested_scopes_are_explicitly_outside_the_experiment(
    fn: Callable[[jax.Array], jax.Array],
) -> None:
    with pytest.raises(PrototypeLimit, match="outside the PoC subset"):
        export_source_dtypes(fn, [jax.ShapeDtypeStruct((), jnp.float32)])


def test_bitcast_with_different_word_width_needs_another_representation() -> None:
    with pytest.raises(RepresentationLimit, match="changes word width"):
        export_source_dtypes(
            lambda x: jax.lax.bitcast_convert_type(x, new_dtype=jnp.uint32),
            [jax.ShapeDtypeStruct((2,), jnp.float64)],
            opset=26,
        )


def test_uint32_bit_representation_does_not_enable_integer_arithmetic() -> None:
    def fn(x: jax.Array) -> jax.Array:
        bits = jax.lax.bitcast_convert_type(x, new_dtype=jnp.uint32)
        return bits + bits

    with pytest.raises(PrototypeLimit, match="only floating arithmetic is covered"):
        export_source_dtypes(fn, [jax.ShapeDtypeStruct((2,), jnp.float32)], opset=26)


@pytest.mark.parametrize(
    ("inputs", "opset", "message"),
    [
        ([(2,)], 23, "requires typed"),
        ([jax.ShapeDtypeStruct(("batch",), jnp.float32)], 23, "only static shapes"),
        ([jax.ShapeDtypeStruct((2,), jnp.float32)], 17, "only opsets 23 and 26"),
        ([jax.ShapeDtypeStruct((2,), jnp.uint32)], 23, "only floating graph inputs"),
    ],
    ids=["shape_only", "dynamic_shape", "unsupported_opset", "integer_input"],
)
def test_input_scope_limits_are_reported_before_tracing(inputs, opset, message) -> None:
    def should_not_run(x):
        raise AssertionError("input validation must precede tracing")

    with pytest.raises(PrototypeLimit, match=message):
        export_source_dtypes(should_not_run, inputs, opset=opset)


def test_reallocating_output_preserves_the_existing_producer_and_binding_value() -> (
    None
):
    ctx = IRContext(
        opset=23,
        enable_double_precision=False,
        precision_policy=PrecisionPolicy.SOURCE,
    )
    source = ir.Value(name="input", type=ir.TensorType(ir.DataType.FLOAT), shape=(2,))
    ctx.add_graph_input_value(source)
    previous = ctx.builder.op(
        "Neg",
        [source],
        output=ir.Value(name="previous", type=source.type, shape=source.shape),
    )

    class Var:
        aval = SimpleNamespace(shape=(2,), dtype=np.dtype(np.float64))

    var = Var()
    ctx.bind_value_for_var(var, previous)
    replacement = ctx.allocate_value_for_var(var)
    assert replacement is not previous
    assert replacement.producer() is None
    assert replacement.dtype == ir.DataType.DOUBLE
    assert ctx.require_value_for_var(var) is replacement
    assert previous.producer().op_type == "Neg"
    assert previous.dtype == source.dtype == ir.DataType.FLOAT
