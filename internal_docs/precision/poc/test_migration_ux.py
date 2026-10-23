# internal_docs/precision/poc/test_migration_ux.py

"""Observable calling behavior of the internal migration UX candidate."""

from collections.abc import Iterator
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import pytest

from internal_docs.precision.poc.migration_ux import (
    FailureCategory,
    MigrationError,
    _run_cpu,
    compact_report,
    export_candidate,
)
from jax2onnx import to_onnx


@pytest.fixture(autouse=True)
def _restore_jax_config() -> Iterator[None]:
    previous = bool(jax.config.x64_enabled)
    platforms = jax.config.jax_platforms
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)
        jax.config.update("jax_platforms", platforms)


def _exact_region(x: jax.Array) -> jax.Array:
    y = x.astype(jnp.float32)
    return (y + 1) - y


def _types(model: onnx.ModelProto) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (
        tuple(value.type.tensor_type.elem_type for value in model.graph.input),
        tuple(value.type.tensor_type.elem_type for value in model.graph.output),
    )


@pytest.mark.parametrize("ambient", [False, True])
@pytest.mark.parametrize("kind", ["shape", "spec", "numpy", "jax"])
@pytest.mark.parametrize(
    "legacy", [None, False, True], ids=["omitted", "false", "true"]
)
def test_legacy_routes_retain_actual_public_call_behavior(
    ambient, kind, legacy
) -> None:
    jax.config.update("jax_enable_x64", ambient)
    inputs = {
        "shape": [(2,)],
        "spec": [jax.ShapeDtypeStruct((2,), np.float32)],
        "numpy": [np.asarray([1, 2], dtype=np.float32)],
        "jax": [jnp.asarray([1, 2], dtype=jnp.float32)],
    }[kind]
    options = {} if legacy is None else {"enable_double_precision": legacy}
    actual = export_candidate(_exact_region, inputs, **options)
    expected = to_onnx(_exact_region, inputs, **options)
    assert _types(actual.model) == _types(expected)
    assert [node.op_type for node in actual.model.graph.node] == [
        node.op_type for node in expected.graph.node
    ]
    assert (
        actual.report.route
        == {
            None: "legacy-omitted",
            False: "legacy-false",
            True: "legacy-true",
        }[legacy]
    )
    assert actual.report.tracing_x64 is bool(legacy)
    assert actual.report.ambient_x64 is ambient
    assert actual.report.after_x64 is ambient
    assert bool(jax.config.x64_enabled) is ambient


@pytest.mark.parametrize("ambient", [False, True])
@pytest.mark.parametrize("kind", ["shape", "spec", "numpy", "jax"])
def test_source_normalizes_supported_inputs_and_preserves_ambient_config(
    ambient, kind
) -> None:
    jax.config.update("jax_enable_x64", ambient)
    inputs = {
        "shape": [(2,)],
        "spec": [jax.ShapeDtypeStruct((2,), np.float16)],
        "numpy": [np.asarray([1, 2], dtype=np.float16)],
        "jax": [jnp.asarray([1, 2], dtype=jnp.float16)],
    }[kind]
    result = export_candidate(lambda x: -x, inputs, precision="source")
    dtype = "float32" if kind == "shape" else "float16"
    assert result.report.source_input_dtypes == (dtype,)
    assert result.report.traced_inputs == result.report.traced_outputs == (dtype,)
    assert (
        result.report.model_inputs
        == result.report.model_outputs
        == ("FLOAT" if kind == "shape" else "FLOAT16",)
    )
    assert result.report.tracing_x64 is ambient
    assert result.report.after_x64 is ambient
    assert bool(jax.config.x64_enabled) is ambient


@pytest.mark.parametrize("ambient", [False, True])
@pytest.mark.parametrize(
    "legacy", [None, False, True], ids=["omitted", "false", "true"]
)
def test_legacy_fp64_input_keeps_public_truncation_or_promotion(
    ambient, legacy
) -> None:
    jax.config.update("jax_enable_x64", ambient)
    inputs = [jax.ShapeDtypeStruct((1,), np.float64)]
    options = {} if legacy is None else {"enable_double_precision": legacy}
    actual = export_candidate(_exact_region, inputs, **options)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        expected = to_onnx(_exact_region, inputs, **options)
    assert _types(actual.model) == _types(expected)
    assert actual.report.traced_inputs == ("float64" if legacy else "float32",)
    assert actual.report.traced_outputs == ("float32",)
    assert actual.report.model_outputs == ("DOUBLE" if legacy else "FLOAT",)
    # JAX 0.10.2 truncates this specification without JAX 0.11.2's warning.
    assert any(
        "truncated" in warning.message for warning in actual.report.warnings
    ) == any("truncated" in str(warning.message) for warning in recorded)
    assert bool(jax.config.x64_enabled) is ambient


@pytest.mark.parametrize("ambient", [False, True])
def test_source_uses_ambient_jax_default_types_without_changing_fp32_input(
    ambient,
) -> None:
    jax.config.update("jax_enable_x64", ambient)

    def model(x):
        return -x, jnp.asarray(1.0)

    source = export_candidate(model, [(1,)], precision="source")
    omitted = export_candidate(model, [(1,)])
    assert source.report.traced_inputs == omitted.report.traced_inputs == ("float32",)
    assert source.report.traced_outputs == (
        "float32",
        "float64" if ambient else "float32",
    )
    assert source.report.model_outputs == ("FLOAT", "DOUBLE" if ambient else "FLOAT")
    assert omitted.report.traced_outputs == ("float32", "float32")
    assert bool(jax.config.x64_enabled) is ambient


@pytest.mark.parametrize("ambient", [False, True])
@pytest.mark.parametrize(
    "options", [{}, {"enable_double_precision": True}, {"precision": "source"}]
)
def test_observation_preserves_python_scalar_outputs(ambient, options) -> None:
    jax.config.update("jax_enable_x64", ambient)
    result = export_candidate(lambda x: (-x, 1.0), [(1,)], **options)
    expected_scalar = (
        "float64"
        if options.get("enable_double_precision")
        or (ambient and options.get("precision") == "source")
        else "float32"
    )
    assert result.report.traced_outputs[1] == expected_scalar
    assert result.report.model_outputs[1] == (
        "DOUBLE" if expected_scalar == "float64" else "FLOAT"
    )
    assert bool(jax.config.x64_enabled) is ambient


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("ambient", [False, True])
def test_explicit_legacy_flag_conflicts_with_source_before_trace(
    legacy, ambient
) -> None:
    jax.config.update("jax_enable_x64", ambient)
    called = []

    def should_not_run(x):
        called.append(x)
        return -x

    with pytest.raises(MigrationError, match="including False") as caught:
        export_candidate(
            should_not_run,
            [(2,)],
            precision="source",
            enable_double_precision=legacy,
        )
    assert caught.value.category is FailureCategory.OPTIONS
    assert not called
    assert bool(jax.config.x64_enabled) is ambient


@pytest.mark.parametrize(
    "options", [{"precision": "automatic"}, {"enable_double_precision": "false"}]
)
def test_invalid_options_are_rejected_before_trace(options) -> None:
    def should_not_run(x):
        raise AssertionError("option validation must precede tracing")

    with pytest.raises(MigrationError) as caught:
        export_candidate(should_not_run, [(2,)], **options)
    assert caught.value.category is FailureCategory.OPTIONS


@pytest.mark.parametrize("kind", ["spec", "numpy", "jax"])
def test_explicit_fp64_input_rejects_disabled_x64_before_trace(kind) -> None:
    jax.config.update("jax_enable_x64", True)
    inputs = {
        "spec": [jax.ShapeDtypeStruct((1,), np.float64)],
        "numpy": [np.asarray([1e8], dtype=np.float64)],
        "jax": [jnp.asarray([1e8], dtype=jnp.float64)],
    }[kind]
    jax.config.update("jax_enable_x64", False)
    called = []

    def should_not_run(x):
        called.append(x)
        return -x

    with pytest.raises(MigrationError, match="enable JAX x64 before") as caught:
        export_candidate(should_not_run, inputs, precision="source")
    assert caught.value.category is FailureCategory.JAX_CONFIG
    assert not called
    assert not jax.config.x64_enabled


def test_trace_warning_is_visible_but_truncated_array_does_not_imply_fp64_intent() -> (
    None
):
    jax.config.update("jax_enable_x64", False)
    with pytest.warns(UserWarning, match="not available"):
        already_truncated = jnp.asarray([1.0], dtype=jnp.float64)
    assert already_truncated.dtype == jnp.float32
    plain = export_candidate(lambda x: -x, [already_truncated], precision="source")
    assert plain.report.input_specs[0].dtype == "float32"
    assert not plain.report.warnings
    converted = export_candidate(
        lambda x: x.astype(jnp.float64), [already_truncated], precision="source"
    )
    assert converted.report.traced_outputs == ("float32",)
    assert converted.report.model_outputs == ("FLOAT",)
    assert any(
        warning.category == "UserWarning"
        and "float64" in warning.message
        and "not available" in warning.message
        for warning in converted.report.warnings
    )
    assert "UserWarning" in compact_report(converted.report)
    assert not jax.config.x64_enabled


@pytest.mark.parametrize("ambient", [False, True])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"enable_double_precision": False},
        {"enable_double_precision": True},
        {"precision": "source"},
    ],
)
def test_failed_trace_restores_config_and_reports_the_actual_tracing_mode(
    ambient, options
) -> None:
    jax.config.update("jax_enable_x64", ambient)

    def fail(x):
        raise RuntimeError("intentional model tracing failure")

    with pytest.raises(MigrationError, match="intentional model tracing") as caught:
        export_candidate(fail, [(2,)], **options)
    error = caught.value
    assert error.category is FailureCategory.TRACE
    assert isinstance(error.__cause__, RuntimeError)
    assert error.report is not None
    assert error.report.after_x64 is ambient
    assert error.report.tracing_x64 is (
        ambient
        if options.get("precision") == "source"
        else bool(options.get("enable_double_precision"))
    )
    assert bool(jax.config.x64_enabled) is ambient


@pytest.mark.parametrize("ambient", [False, True])
def test_source_scope_failure_is_distinct_from_onnx_representation_failure(
    ambient,
) -> None:
    jax.config.update("jax_enable_x64", ambient)
    spec = jax.ShapeDtypeStruct((2,), np.float32)
    with pytest.raises(MigrationError, match="outside the PoC subset") as scope:
        export_candidate(jnp.sin, [spec], precision="source")
    assert scope.value.category is FailureCategory.PROTOTYPE
    with pytest.raises(MigrationError, match="selected opset is 23") as representation:
        export_candidate(
            lambda x: jax.lax.bitcast_convert_type(x, jnp.uint32),
            [spec],
            precision="source",
        )
    assert representation.value.category is FailureCategory.REPRESENTATION
    assert bool(jax.config.x64_enabled) is ambient


def test_exact_fp32_region_has_observable_legacy_source_difference() -> None:
    jax.config.update("jax_enable_x64", True)
    x = np.asarray([1e8], dtype=np.float64)
    legacy = export_candidate(_exact_region, [x], enable_double_precision=True)
    source = export_candidate(_exact_region, [x], precision="source")
    assert legacy.report.traced_inputs == source.report.traced_inputs == ("float64",)
    assert legacy.report.traced_outputs == source.report.traced_outputs == ("float32",)
    assert legacy.report.model_outputs == ("DOUBLE",)
    assert source.report.model_outputs == ("FLOAT",)
    np.testing.assert_array_equal(
        _run_cpu(legacy.model, x), np.asarray([1], np.float64), strict=True
    )
    np.testing.assert_array_equal(
        _run_cpu(source.model, x), np.asarray([0], np.float32), strict=True
    )


def test_same_dtype_dependent_model_is_instantiated_by_typed_inputs() -> None:
    jax.config.update("jax_enable_x64", True)

    def minimum_region(x):
        y = x.astype(jnp.promote_types(x.dtype, jnp.float32))
        return (y + 1) - y

    for dtype, value, expected in (
        (np.float16, 2048, np.float32),
        (np.float64, 1e8, np.float64),
    ):
        x = np.asarray([value], dtype=dtype)
        result = export_candidate(minimum_region, [x], precision="source")
        assert result.report.traced_outputs == (np.dtype(expected).name,)
        np.testing.assert_array_equal(
            _run_cpu(result.model, x), np.asarray([1], expected), strict=True
        )


@pytest.mark.parametrize("inputs", [[("batch",)], [(-1,)], [object()]])
def test_input_errors_are_actionable_and_precede_tracing(inputs) -> None:
    def should_not_run(x):
        raise AssertionError("input validation must precede tracing")

    with pytest.raises(MigrationError) as caught:
        export_candidate(should_not_run, inputs, precision="source")
    assert caught.value.category is FailureCategory.INPUT
