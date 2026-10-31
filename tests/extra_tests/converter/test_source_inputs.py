# tests/extra_tests/converter/test_source_inputs.py

"""Weak input properties survive the private source export boundary."""

from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

import tests.extra_tests.converter.precision_ux_support as migration_ux
from tests.extra_tests.converter.precision_ux_support import (
    _run_cpu,
    compact_report,
    export_candidate,
)
from tests.extra_tests.converter.precision_test_support import export_source_dtypes
from jax2onnx import onnx_function, to_onnx


@pytest.fixture(autouse=True)
def _restore_jax_config() -> Iterator[None]:
    previous_x64 = bool(jax.config.x64_enabled)
    previous_platforms = jax.config.jax_platforms
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
        jax.config.update("jax_platforms", previous_platforms)


def _add_explicit_half(x: jax.Array) -> jax.Array:
    return x + jnp.asarray(1, dtype=jnp.float16)


@onnx_function
def _weak_sensitive_function(x: jax.Array) -> jax.Array:
    return _add_explicit_half(x)


def _call_input_pair(weak: jax.Array, strong: jax.Array) -> tuple[jax.Array, jax.Array]:
    return _weak_sensitive_function(weak), _weak_sensitive_function(strong)


def _run_pair(model: onnx.ModelProto, *inputs: np.ndarray) -> list[np.ndarray]:
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    return session.run(
        None,
        {
            value.name: np.asarray(
                supplied,
                dtype=onnx.helper.tensor_dtype_to_np_dtype(
                    spec.type.tensor_type.elem_type
                ),
            )
            for value, spec, supplied in zip(
                session.get_inputs(), model.graph.input, inputs, strict=True
            )
        },
    )


def _input_pair() -> tuple[jax.Array, jax.Array]:
    weak = jnp.asarray(2048.0)
    strong = jnp.asarray(2048.0, dtype=weak.dtype)
    assert weak.shape == strong.shape == ()
    assert weak.dtype == strong.dtype
    assert weak.weak_type and not strong.weak_type
    return weak, strong


def _spec(value: jax.Array) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(
        value.shape, value.dtype, weak_type=bool(value.weak_type)
    )


def _assert_source_output(
    model: onnx.ModelProto, original: jax.Array, weak: bool
) -> None:
    # Evaluate the original JAX value. Reconstructing it through NumPy would
    # silently make the weak reference strong and conceal the regression.
    reference = _add_explicit_half(original)
    expected_dtype = np.dtype(np.float16 if weak else original.dtype)
    assert reference.dtype == expected_dtype
    expected = np.asarray(2048 if weak else 2049, dtype=expected_dtype)
    np.testing.assert_array_equal(np.asarray(reference), expected, strict=True)
    onnx.checker.check_model(model, full_check=True)
    actual = _run_cpu(model, np.asarray(original))
    np.testing.assert_array_equal(actual, np.asarray(reference), strict=True)
    assert model.graph.output[0].type.tensor_type.elem_type == (
        onnx.helper.np_dtype_to_tensor_dtype(expected_dtype)
    )


@pytest.mark.parametrize("ambient_x64", [False, True])
@pytest.mark.parametrize("kind", ["array", "spec"])
@pytest.mark.parametrize("weak", [False, True], ids=["strong", "weak"])
def test_source_facade_preserves_weakness_and_numeric_promotion(
    ambient_x64: bool, kind: str, weak: bool
) -> None:
    jax.config.update("jax_enable_x64", ambient_x64)
    weak_value, strong_value = _input_pair()
    original = weak_value if weak else strong_value
    supplied = original if kind == "array" else _spec(original)
    result = export_candidate(_add_explicit_half, [supplied], precision="source")
    report = result.report
    assert report.input_specs[0].weak_type is weak
    assert report.source_input_weak_types == (weak,)
    assert report.traced_input_weak_types == (weak,)
    assert (
        report.source_input_dtypes
        == report.traced_inputs
        == (np.dtype(original.dtype).name,)
    )
    rendered = compact_report(report)
    assert f":weak_type={weak}" in rendered
    assert f"source-normalized weak_type=({weak},)" in rendered
    assert f"trace input weak_type=({weak},)" in rendered
    assert report.ambient_x64 is ambient_x64
    assert report.tracing_x64 is ambient_x64
    assert report.after_x64 is ambient_x64
    assert bool(jax.config.x64_enabled) is ambient_x64
    _assert_source_output(result.model, original, weak)


@pytest.mark.parametrize("ambient_x64", [False, True])
@pytest.mark.parametrize("weak", [False, True], ids=["strong", "weak"])
def test_direct_source_helper_preserves_supplied_spec_weakness(
    ambient_x64: bool, weak: bool
) -> None:
    jax.config.update("jax_enable_x64", ambient_x64)
    weak_value, strong_value = _input_pair()
    original = weak_value if weak else strong_value
    traced_weak_types = []

    def observed(x: jax.Array) -> jax.Array:
        traced_weak_types.append(bool(x.weak_type))
        return _add_explicit_half(x)

    model = export_source_dtypes(observed, [_spec(original)])
    assert traced_weak_types == [weak]
    assert bool(jax.config.x64_enabled) is ambient_x64
    _assert_source_output(model, original, weak)


@pytest.mark.parametrize("ambient_x64", [False, True])
@pytest.mark.parametrize("kind", ["numpy", "shape-only"])
def test_untyped_shapes_and_numpy_inputs_remain_strong(
    ambient_x64: bool, kind: str
) -> None:
    jax.config.update("jax_enable_x64", ambient_x64)
    x = np.asarray(2048, dtype=np.float32)
    supplied = x if kind == "numpy" else ()
    result = export_candidate(_add_explicit_half, [supplied], precision="source")
    assert result.report.input_specs[0].weak_type is (
        False if kind == "numpy" else None
    )
    assert result.report.source_input_weak_types == (False,)
    assert result.report.traced_input_weak_types == (False,)
    np.testing.assert_array_equal(
        _run_cpu(result.model, x), np.asarray(2049, dtype=np.float32), strict=True
    )
    assert bool(jax.config.x64_enabled) is ambient_x64


@pytest.mark.parametrize("ambient_x64", [False, True])
@pytest.mark.parametrize("kind", ["array", "spec"])
@pytest.mark.parametrize("weak", [False, True], ids=["strong", "weak"])
@pytest.mark.parametrize(
    "legacy", [None, False, True], ids=["omitted", "false", "true"]
)
def test_legacy_forwards_original_inputs_and_matches_the_actual_public_call(
    monkeypatch: pytest.MonkeyPatch,
    ambient_x64: bool,
    kind: str,
    weak: bool,
    legacy: bool | None,
) -> None:
    jax.config.update("jax_enable_x64", ambient_x64)
    weak_value, strong_value = _input_pair()
    original = weak_value if weak else strong_value
    inputs = [original if kind == "array" else _spec(original)]
    forwarded_inputs = []

    def public_call(fn: Any, supplied: Any, **options: Any) -> onnx.ModelProto:
        forwarded_inputs.append(supplied)
        return to_onnx(fn, supplied, **options)

    monkeypatch.setattr(migration_ux, "to_onnx", public_call)
    options = {} if legacy is None else {"enable_double_precision": legacy}
    candidate = export_candidate(_add_explicit_half, inputs, **options)
    assert len(forwarded_inputs) == 1 and forwarded_inputs[0] is inputs
    public_weak_types = []

    def public_observed(x: jax.Array) -> jax.Array:
        public_weak_types.append(bool(x.weak_type))
        return _add_explicit_half(x)

    public_model = to_onnx(public_observed, inputs, **options)
    assert candidate.report.input_specs[0].weak_type is weak
    assert candidate.report.source_input_weak_types == ()
    assert candidate.report.traced_input_weak_types == tuple(public_weak_types)
    assert public_weak_types == [False]
    assert [
        value.type.tensor_type.elem_type
        for value in (*candidate.model.graph.input, *candidate.model.graph.output)
    ] == [
        value.type.tensor_type.elem_type
        for value in (*public_model.graph.input, *public_model.graph.output)
    ]
    assert [node.op_type for node in candidate.model.graph.node] == [
        node.op_type for node in public_model.graph.node
    ]
    feed_dtype = onnx.helper.tensor_dtype_to_np_dtype(
        public_model.graph.input[0].type.tensor_type.elem_type
    )
    feed = np.asarray(original, dtype=feed_dtype)
    np.testing.assert_array_equal(
        _run_cpu(candidate.model, feed), _run_cpu(public_model, feed), strict=True
    )
    assert candidate.report.tracing_x64 is bool(legacy)
    assert candidate.report.after_x64 is ambient_x64
    assert bool(jax.config.x64_enabled) is ambient_x64


def _assert_distinct_source_functions(
    model: onnx.ModelProto, inputs: tuple[jax.Array, jax.Array]
) -> None:
    assert [value.type.tensor_type.elem_type for value in model.graph.input] == [
        onnx.helper.np_dtype_to_tensor_dtype(np.dtype(value.dtype)) for value in inputs
    ]
    assert len(model.functions) == 2
    definitions = {(function.domain, function.name) for function in model.functions}
    calls = [
        (node.domain, node.op_type)
        for node in model.graph.node
        if (node.domain, node.op_type) in definitions
    ]
    assert len(calls) == 2 and len(set(calls)) == 2
    outputs = _run_pair(model, *(np.asarray(value) for value in inputs))
    for actual, original in zip(outputs, inputs, strict=True):
        np.testing.assert_array_equal(
            actual, np.asarray(_add_explicit_half(original)), strict=True
        )
    assert outputs[0].dtype == np.float16
    assert outputs[1].dtype == inputs[1].dtype
    np.testing.assert_array_equal(outputs[0], np.asarray(2048, dtype=np.float16))
    np.testing.assert_array_equal(outputs[1], np.asarray(2049, dtype=inputs[1].dtype))


@pytest.mark.parametrize("ambient_x64", [False, True])
def test_same_function_specializes_for_weak_and_strong_source_operands(
    ambient_x64: bool,
) -> None:
    jax.config.update("jax_enable_x64", ambient_x64)
    inputs = _input_pair()
    result = export_candidate(_call_input_pair, inputs, precision="source")
    assert result.report.traced_input_weak_types == (True, False)
    assert result.report.traced_outputs == (
        "float16",
        np.dtype(inputs[1].dtype).name,
    )
    _assert_distinct_source_functions(result.model, inputs)
    assert bool(jax.config.x64_enabled) is ambient_x64


@pytest.mark.parametrize("ambient_x64", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_source_function_specializations_survive_intervening_legacy_export(
    ambient_x64: bool, legacy: bool
) -> None:
    jax.config.update("jax_enable_x64", ambient_x64)
    inputs = _input_pair()
    before = export_candidate(_call_input_pair, inputs, precision="source")
    _assert_distinct_source_functions(before.model, inputs)
    legacy_result = export_candidate(
        _call_input_pair, inputs, enable_double_precision=legacy
    )
    legacy_public = to_onnx(_call_input_pair, inputs, enable_double_precision=legacy)
    assert len(legacy_result.model.functions) == len(legacy_public.functions) == 1
    assert legacy_result.report.traced_input_weak_types == (False, False)
    for actual, expected in zip(
        _run_pair(legacy_result.model, *(np.asarray(value) for value in inputs)),
        _run_pair(legacy_public, *(np.asarray(value) for value in inputs)),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected, strict=True)
    after = export_candidate(_call_input_pair, inputs, precision="source")
    _assert_distinct_source_functions(after.model, inputs)
    assert after.report.traced_inputs == before.report.traced_inputs
    assert after.report.traced_input_weak_types == before.report.traced_input_weak_types
    assert after.report.traced_outputs == before.report.traced_outputs
    assert after.report.model_outputs == before.report.model_outputs
    assert bool(jax.config.x64_enabled) is ambient_x64
