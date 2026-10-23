# internal_docs/precision/poc/test_numerical_contract.py

"""Checks for the bounded numerical experiment; runtime defects may disappear."""

from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import pytest
from onnx import TensorProto

from internal_docs.precision.poc.numerical_contract import (
    explicit_rounding_square_sum,
    graph_summary,
    normalization_control,
    normalization_input,
    normalization_reference,
    preferred_dot,
    rounding_status,
    run_experiments,
    square_sum,
    unfused_square_sum_control,
)


@pytest.fixture(autouse=True)
def _restore_jax_config() -> Iterator[None]:
    previous_x64 = bool(jax.config.x64_enabled)
    previous_platforms = jax.config.jax_platforms
    try:
        jax.config.update("jax_enable_x64", True)
        yield
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
        jax.config.update("jax_platforms", previous_platforms)


@pytest.fixture(scope="module")
def observations():
    return run_experiments()


def test_dot_preferred_element_type_is_a_source_result_requirement() -> None:
    x = np.asarray([[[2048, 1]]], dtype=np.float16)
    w = np.ones((1, 2, 1), dtype=np.float16)
    traced = jax.make_jaxpr(preferred_dot)(x, w)
    (equation,) = traced.jaxpr.eqns
    assert equation.primitive.name == "dot_general"
    assert equation.params["preferred_element_type"] == np.dtype(np.float32)
    assert traced.jaxpr.outvars[0].aval.dtype == np.float32
    np.testing.assert_array_equal(preferred_dot(x, w), [[[2049]]])


def test_public_dot_gap_is_separate_from_validity_and_kernel_execution(
    observations,
) -> None:
    baseline, candidate = observations[:2]
    for case in (baseline, candidate):
        onnx.checker.check_model(case.model, full_check=True)
    assert (
        baseline.model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT16
    )
    assert (
        candidate.model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
    )
    assert "MatMul[FLOAT16]" in graph_summary(baseline.model)
    assert "Cast[FLOAT]" in graph_summary(candidate.model)
    assert "MatMul[FLOAT]" in graph_summary(candidate.model)
    assert baseline.semantic_status.startswith("known result-type/representation gap")
    for result in (candidate.reference, candidate.disabled, candidate.enabled):
        assert result[0].dtype == np.float32
        np.testing.assert_array_equal(result[0], baseline.eager[0])


def test_normalization_source_export_is_distinct_from_control(observations) -> None:
    case = observations[6]
    assert "bounded source export" in case.origin
    assert "ReduceSum[DOUBLE]" in graph_summary(case.model)
    expected = normalization_reference(normalization_input())
    for values in (case.reference, case.disabled, case.enabled):
        for value, reference in zip(values[:3], expected[:3], strict=True):
            np.testing.assert_array_equal(value, reference)
        np.testing.assert_allclose(values[3], expected[3], rtol=3e-7, atol=1e-7)


def test_source_dot_preserves_preferred_result_without_changing_inputs(
    observations,
) -> None:
    case = observations[7]
    assert "unchanged FP16-input JAX callable" in case.origin
    assert all(
        value.type.tensor_type.elem_type == TensorProto.FLOAT16
        for value in case.model.graph.input
    )
    for values in (case.reference, case.disabled, case.enabled):
        assert values[0].dtype == np.float32
        np.testing.assert_array_equal(values[0], [[[2049]]])


def test_normalization_control_keeps_fp32_squares_before_double_sum() -> None:
    model = normalization_control()
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    types = {
        value.name: value.type.tensor_type.elem_type
        for value in (*inferred.graph.value_info, *inferred.graph.output)
    }
    assert types["centered"] == types["squares"] == TensorProto.FLOAT
    assert types["squares64"] == types["variance_sum64"] == TensorProto.DOUBLE
    assert types["variance64"] == TensorProto.DOUBLE
    assert types["variance"] == types["normalized"] == TensorProto.FLOAT
    producer = {output: node for node in model.graph.node for output in node.output}
    assert producer["squares"].op_type == "Mul"
    assert producer["squares64"].input == ["squares"]
    assert producer["variance_sum64"].input[0] == "squares64"
    assert producer["variance"].input == ["variance64"]


def test_selected_double_variance_has_independent_numerical_evidence(
    observations,
) -> None:
    case = observations[2]
    expected = normalization_reference(normalization_input())
    assert expected[2].dtype == np.float32
    np.testing.assert_array_equal(expected[2], [[np.float32(8388609)], [0]])
    # The small squares matter after the large terms. A naive FP32 sum loses
    # them on this independent NumPy control; widening only the final output
    # cannot repair the already-lost contribution.
    assert np.sum(expected[1][0], dtype=np.float32) / np.float32(16) == np.float32(
        8388608
    )
    for values in (
        case.eager,
        case.compiled,
        case.reference,
        case.disabled,
        case.enabled,
    ):
        for actual, reference in zip(values[:3], expected[:3], strict=True):
            assert actual.dtype == reference.dtype
            np.testing.assert_array_equal(actual, reference)
        np.testing.assert_allclose(values[3], expected[3], rtol=3e-7, atol=1e-7)
        np.testing.assert_array_equal(values[3][1], np.zeros(16, dtype=np.float32))
    assert "NOT export" in case.origin


def test_fp16_rounding_runtime_observations_allow_a_future_runtime_fix(
    observations,
) -> None:
    baseline, control = observations[3:5]
    for case in (baseline, control):
        assert "Mul[FLOAT16]" in graph_summary(case.model)
        for values in (case.eager, case.compiled, case.reference):
            assert rounding_status(values) == "accepted: sample matches JAX"
        for values, status in (
            (case.disabled, case.disabled_status),
            (case.enabled, case.enabled_status),
        ):
            assert status == rounding_status(values)
    assert "ReduceSum[FLOAT16]" in graph_summary(baseline.model)
    assert "ReduceSumSquare" not in graph_summary(baseline.model)
    assert "ReduceSum[FLOAT16]" in graph_summary(control.model)
    assert [node.op_type for node in unfused_square_sum_control().graph.node] == [
        "Mul",
        "ReduceSum",
    ]


def test_rounding_diagnostic_rejects_unfamiliar_results() -> None:
    zero = np.zeros(8, dtype=np.float16)
    assert rounding_status((zero, np.asarray(0, dtype=np.float16))).startswith(
        "accepted"
    )
    known = np.asarray(2**-23, dtype=np.float16)
    assert rounding_status((zero, known)).startswith("diagnostic")
    with pytest.raises(AssertionError, match="Unexpected FP16 intermediate result"):
        rounding_status((zero, np.asarray(1, dtype=np.float16)))
    with pytest.raises(AssertionError):
        rounding_status((np.ones(8, dtype=np.float16), known))


def test_explicit_rounding_candidate_retains_boundary_and_original_result_dtypes(
    observations,
) -> None:
    case = observations[5]
    summary = graph_summary(case.model)
    assert "Mul[FLOAT] -> Cast[FLOAT16] -> Cast[FLOAT] -> ReduceSum[FLOAT]" in summary
    assert summary.endswith("Cast[FLOAT16]")
    for values in (
        case.eager,
        case.compiled,
        case.reference,
        case.disabled,
        case.enabled,
    ):
        assert rounding_status(values) == "accepted: sample matches JAX"
    small = jnp.full(8, 2**-13, dtype=jnp.float16)
    assert [v.dtype for v in explicit_rounding_square_sum(small)] == [
        v.dtype for v in square_sum(small)
    ]


def test_experiment_restores_disabled_x64_even_on_success() -> None:
    jax.config.update("jax_enable_x64", False)
    run_experiments()
    assert not jax.config.x64_enabled


def test_experiment_restores_x64_on_unexpected_failure(monkeypatch) -> None:
    import internal_docs.precision.poc.numerical_contract as experiment

    def fail(*args, **kwargs):
        raise RuntimeError("unexpected kernel failure must propagate")

    jax.config.update("jax_enable_x64", False)
    monkeypatch.setattr(experiment, "run_cpu", fail)
    with pytest.raises(RuntimeError, match="unexpected kernel failure must propagate"):
        run_experiments()
    assert not jax.config.x64_enabled
