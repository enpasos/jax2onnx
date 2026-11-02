# tests/extra_tests/converter/test_conversion_weak_type.py

"""Input weakness must survive tracing and ONNX Function specialization."""

from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function, to_onnx
from jax2onnx._compat.jax import ShapedArray
from jax2onnx.converter.conversion_api import _as_sds_list
from jax2onnx.user_interface import _normalize_input_specs


@pytest.fixture(autouse=True)
def _use_default_jax_precision() -> Iterator[None]:
    previous = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _add_explicit_half(x):
    return x + jnp.asarray(1.0, dtype=jnp.float16)


@onnx_function
def _weak_type_child(x):
    return _add_explicit_half(x)


def _call_weak_and_strong(weak, strong):
    return _weak_type_child(weak), _weak_type_child(strong)


@onnx_function
def _weak_type_negation(x):
    return -x


def _consume_weak_function_result(x):
    return _add_explicit_half(-_weak_type_negation(x))


@onnx_function
def _weak_type_capture(x, *, scalar):
    return x + scalar


def _call_weak_and_strong_captures(x, weak, strong):
    return (
        _weak_type_capture(x, scalar=weak),
        _weak_type_capture(x, scalar=strong),
    )


def _run_model(model, *inputs):
    onnx.checker.check_model(model, full_check=True)
    results = []
    for optimization in (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    ):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = optimization
        session = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        feeds = {
            value.name: np.asarray(data)
            for value, data in zip(session.get_inputs(), inputs)
        }
        results.append(session.run(None, feeds))
    assert len(results[0]) == len(results[1])
    for unoptimized, optimized in zip(*results):
        assert optimized.dtype == unoptimized.dtype
        assert optimized.shape == unoptimized.shape
        np.testing.assert_array_equal(optimized, unoptimized)
    return results[0]


def _native_reference(fn, *inputs):
    reference = fn(*inputs)

    def fresh_call(*values):
        return fn(*values)

    compiled = jax.jit(fresh_call)
    for result in (compiled(*inputs), compiled(*inputs)):
        assert jax.tree.structure(result) == jax.tree.structure(reference)
        for actual, expected in zip(
            jax.tree.leaves(result), jax.tree.leaves(reference)
        ):
            actual, expected = np.asarray(actual), np.asarray(expected)
            assert actual.dtype == expected.dtype
            assert actual.shape == expected.shape
            np.testing.assert_array_equal(actual, expected)
    return reference


@pytest.mark.parametrize("weak_type", [False, True], ids=["strong", "weak"])
@pytest.mark.parametrize("input_kind", ["array", "spec", "aval"])
def test_public_inputs_preserve_weakness_and_numeric_promotion(input_kind, weak_type):
    weak = jnp.asarray(2048.0)
    value = weak if weak_type else jnp.asarray(2048.0, dtype=weak.dtype)
    if input_kind == "array":
        spec = value
    elif input_kind == "spec":
        spec = jax.ShapeDtypeStruct(value.shape, value.dtype, weak_type=weak_type)
    else:
        spec = ShapedArray(value.shape, value.dtype, weak_type=weak_type)

    model = to_onnx(_add_explicit_half, [spec])
    expected = np.asarray(_native_reference(_add_explicit_half, value))
    (actual,) = _run_model(model, value)

    assert expected.dtype == (np.float16 if weak_type else np.float32)
    assert expected.item() == (2048.0 if weak_type else 2049.0)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    if weak_type:
        casts = [node for node in model.graph.node if node.op_type == "Cast"]
        assert any(
            attribute.name == "to" and attribute.i == onnx.TensorProto.FLOAT16
            for node in casts
            for attribute in node.attribute
        )


def test_same_function_specializes_for_weak_and_strong_operands():
    weak = jnp.asarray(2048.0)
    strong = jnp.asarray(2048.0, dtype=weak.dtype)
    model = to_onnx(_call_weak_and_strong, [weak, strong])
    expected = _native_reference(_call_weak_and_strong, weak, strong)
    actual = _run_model(model, weak, strong)

    assert len(model.functions) == 2
    calls = [node for node in model.graph.node if node.op_type == "_weak_type_child"]
    assert len(calls) == 2
    assert len({node.domain for node in calls}) == 2
    for result, reference in zip(actual, expected):
        assert result.dtype == reference.dtype
        np.testing.assert_array_equal(result, np.asarray(reference))
    assert actual[0].item() == 2048.0
    assert actual[1].item() == 2049.0


def test_function_output_retains_weakness_for_its_consumer():
    weak = jnp.asarray(2048.0)
    model = to_onnx(_consume_weak_function_result, [weak])
    expected = _native_reference(_consume_weak_function_result, weak)
    (actual,) = _run_model(model, weak)

    assert model.functions
    assert actual.dtype == np.float16
    np.testing.assert_array_equal(actual, np.asarray(expected))
    assert actual.item() == 2048.0


def test_dynamic_keyword_captures_keep_weakness_in_function_specs_and_keys():
    half = np.asarray([1.0], dtype=np.float16)
    weak = jnp.asarray(2048.0)
    strong = jnp.asarray(2048.0, dtype=weak.dtype)
    model = to_onnx(_call_weak_and_strong_captures, [half, weak, strong])
    expected = _native_reference(
        _call_weak_and_strong_captures, jnp.asarray(half), weak, strong
    )
    actual = _run_model(model, half, weak, strong)

    assert len(model.functions) == 2
    for result, reference in zip(actual, expected):
        assert result.dtype == reference.dtype
        np.testing.assert_array_equal(result, np.asarray(reference))
    assert actual[0].item() == 2048.0
    assert actual[1].item() == 2049.0


def test_symbolic_input_normalization_preserves_declared_weakness():
    normalized = _normalize_input_specs(
        [jax.ShapeDtypeStruct(("B", 2), jnp.float32, weak_type=True)]
    )
    (spec,) = _as_sds_list(normalized, enable_double_precision=False)

    assert spec.weak_type
    assert spec.dtype == np.dtype(np.float32)
    assert str(spec.shape[0]) == "B"
    assert spec.shape[1] == 2


def test_shape_only_and_numpy_inputs_remain_strong():
    normalized = _normalize_input_specs([(2,), np.asarray(2048.0, dtype=np.float32)])
    specs = _as_sds_list(normalized, enable_double_precision=False)

    assert all(not spec.weak_type for spec in specs)
    model = to_onnx(_add_explicit_half, [np.asarray(2048.0, dtype=np.float32)])
    (actual,) = _run_model(model, np.asarray(2048.0, dtype=np.float32))
    assert actual.dtype == np.float32
    assert actual.item() == 2049.0
