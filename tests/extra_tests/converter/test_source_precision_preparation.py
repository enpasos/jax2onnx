# tests/extra_tests/converter/test_source_precision_preparation.py

"""The migration oracle observes original typed JAX, not legacy precision variants."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import pytest

from jax2onnx.plugins.plugin_system import construct_and_call, with_requested_dtype
from jax2onnx.precision import CheckStatus, SampleCriteria, get_precision_report
from jax2onnx.converter.precision_policy import PrecisionPolicy
from tests.extra_tests.converter import source_precision_support as source
from tests.extra_tests.converter.source_precision_support import (
    SourcePreparationError,
    SourceScenario,
    is_source_float,
    prepare_source_case,
    raw_source_case,
    run_source_case,
)


@pytest.fixture(autouse=True)
def _restore_jax_settings() -> Iterator[None]:
    previous = (
        bool(jax.config.x64_enabled),
        jax.config.jax_default_matmul_precision,
        jax.config.jax_platforms,
    )
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous[0])
        jax.config.update("jax_default_matmul_precision", previous[1])
        jax.config.update("jax_platforms", previous[2])


def _metadata(fn: Any, **options: Any) -> dict[str, Any]:
    return {
        "context": "migration-test",
        "component": "original",
        "testcase": "oracle",
        "callable": fn,
        **options,
    }


def test_original_fp32_cancellation_on_fp64_input_is_zero_not_legacy_one() -> None:
    def function(x: jax.Array) -> jax.Array:
        rounded = x.astype(jnp.float32)
        return (rounded + np.float32(1)) - rounded

    value = np.asarray([2**24], np.float64)
    metadata = _metadata(
        function, input_values=[value], expected_output_dtypes=[np.float64]
    )
    result = run_source_case(
        metadata, SourceScenario("explicit-fp32", True, np.float64)
    )
    assert result.audit.status is CheckStatus.VALIDATED
    assert result.numerical_status is CheckStatus.VALIDATED
    prepared = result.prepared
    assert prepared.traced_inputs[0].dtype == "float64"
    assert prepared.traced_outputs[0].dtype == "float32"
    np.testing.assert_array_equal(
        prepared.reference_outputs[0], np.asarray([0], np.float32), strict=True
    )
    assert not prepared.reference_outputs[0].flags.writeable
    assert (
        result.model.graph.output[0].type.tensor_type.elem_type
        == onnx.TensorProto.FLOAT
    )
    result.validation.require_sample_acceptance()
    # Raw legacy expectations are retained as metadata, never rewritten into
    # the independent original-source reference.
    assert metadata["expected_output_dtypes"] == [np.float64]


def test_weak_and_strong_scalar_sources_retain_distinct_promotion() -> None:
    def function(weak: jax.Array, strong: jax.Array) -> tuple[jax.Array, ...]:
        return weak + np.float16(1), strong + np.float16(1)

    scenario = SourceScenario(
        "weak-and-strong",
        False,
        np.float32,
        input_specs=(
            jax.ShapeDtypeStruct((), np.float32, weak_type=True),
            jax.ShapeDtypeStruct((), np.float32),
        ),
        input_values=(2048.0, np.asarray(2048.0, np.float32)),
    )
    result = run_source_case(_metadata(function), scenario)
    assert result.audit.status is CheckStatus.VALIDATED
    prepared = result.prepared
    assert [item.weak_type for item in prepared.traced_inputs] == [True, False]
    assert [item.dtype for item in prepared.traced_outputs] == ["float16", "float32"]
    np.testing.assert_array_equal(
        prepared.reference_outputs[0], np.asarray(2048, np.float16), strict=True
    )
    np.testing.assert_array_equal(
        prepared.reference_outputs[1], np.asarray(2049, np.float32), strict=True
    )
    result.validation.require_sample_acceptance()


def test_bfloat16_input_domain_is_floating_even_when_numpy_issubdtype_is_false() -> (
    None
):
    assert not np.issubdtype(np.dtype(jnp.bfloat16), np.floating)
    assert is_source_float(jnp.bfloat16)
    result = run_source_case(
        _metadata(lambda x: -x, input_shapes=[(2,)], input_dtypes=[jnp.bfloat16]),
        SourceScenario("bf16", False, jnp.bfloat16),
        optimization=None,
    )
    assert result.audit.status is CheckStatus.VALIDATED
    assert result.prepared.arguments[0].dtype == jnp.bfloat16
    assert result.prepared.reference_outputs[0].dtype.name == "bfloat16"
    assert result.numerical_status is CheckStatus.UNVERIFIED


def test_mixed_tuple_output_types_follow_original_trace_without_blanket_promotion() -> (
    None
):
    def function(x: jax.Array) -> tuple[jax.Array, ...]:
        return x.astype(jnp.float32), -x, x > np.float64(0)

    metadata = _metadata(
        function,
        input_values=[np.asarray([-2, 3], np.float64)],
        expected_output_dtypes=[np.float64] * 3,
    )
    result = run_source_case(metadata, SourceScenario("mixed-tuple", True, np.float64))
    assert result.audit.status is CheckStatus.VALIDATED
    assert [item.dtype for item in result.prepared.traced_outputs] == [
        "float32",
        "float64",
        "bool",
    ]
    assert [item.dtype.name for item in result.prepared.reference_outputs] == [
        "float32",
        "float64",
        "bool",
    ]
    result.validation.require_sample_acceptance()


def test_factory_instantiates_once_with_declared_dtype_and_restores_config_on_failure() -> (
    None
):
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_matmul_precision", "high")
    calls = []

    def constructor(*, dtype: Any):
        calls.append(
            (
                np.dtype(dtype).name,
                bool(jax.config.x64_enabled),
                jax.config.jax_default_matmul_precision,
            )
        )
        return lambda x: x + np.asarray(1, dtype=dtype)

    metadata = _metadata(
        construct_and_call(constructor, dtype=with_requested_dtype()),
        input_values=[np.asarray([1], np.float32)],
    )
    result = run_source_case(metadata, SourceScenario("factory", True, np.float32))
    result.validation.require_sample_acceptance()
    assert calls == [("float32", True, None)]
    assert (
        not jax.config.x64_enabled and jax.config.jax_default_matmul_precision == "high"
    )

    def broken_constructor(*, dtype: Any):
        assert jax.config.x64_enabled
        assert jax.config.jax_default_matmul_precision is None
        raise RuntimeError("constructor failure")

    broken = _metadata(
        construct_and_call(broken_constructor, dtype=with_requested_dtype()),
        input_values=[],
    )
    with pytest.raises(RuntimeError, match="constructor failure"):
        prepare_source_case(broken, SourceScenario("failure", True, np.float32))
    assert (
        not jax.config.x64_enabled and jax.config.jax_default_matmul_precision == "high"
    )


def test_schema_valid_wrong_output_dtype_cannot_be_hidden_by_reference_casts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export = source._export_prepared

    def corrupted(prepared):
        model = export(prepared)
        original = model.graph.output[0].name
        model.graph.node.append(
            onnx.helper.make_node(
                "Cast", [original], ["wrong_dtype"], to=onnx.TensorProto.DOUBLE
            )
        )
        model.graph.output[0].name = "wrong_dtype"
        model.graph.output[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
        onnx.checker.check_model(model, full_check=True)
        return model

    monkeypatch.setattr(source, "_export_prepared", corrupted)
    result = run_source_case(
        _metadata(lambda x: -x, input_values=[np.asarray([1], np.float32)]),
        SourceScenario("wrong-dtype", False, np.float32),
    )
    assert result.audit.status is CheckStatus.FAILED
    assert result.category == "implementation-gap"
    assert "dtype float64 differs from original source float32" in result.audit.detail
    assert result.numerical_status is CheckStatus.UNVERIFIED
    assert result.validation is None


def test_skipped_numerics_and_unaudited_functionality_remain_unverified() -> None:
    identity = _metadata(
        lambda x: -x,
        input_values=[np.asarray([1], np.float32)],
        skip_numeric_validation=True,
    )
    skipped = run_source_case(
        identity, SourceScenario("structural-only", False, np.float32)
    )
    assert skipped.audit.status is CheckStatus.VALIDATED
    assert skipped.stage == "artifact-checked"
    assert skipped.numerical_status is CheckStatus.UNVERIFIED
    assert skipped.validation.target is None
    sine = run_source_case(
        _metadata(lambda x: jax.lax.sin(x), input_values=[np.asarray([1], np.float32)]),
        SourceScenario("pending-sin", False, np.float32),
    )
    assert sine.model is not None, sine.audit.detail
    assert sine.category == "runtime-evidence" and sine.stage == "runtime-checked"
    assert (
        sine.schema
        is sine.audit.status
        is sine.numerical_status
        is CheckStatus.VALIDATED
    )
    unsupported = run_source_case(
        _metadata(lambda x: jax.lax.erf(x), input_values=[np.asarray([1], np.float32)]),
        SourceScenario("pending-erf", False, np.float32),
    )
    assert unsupported.category == "migration-gap"
    assert (
        unsupported.audit.status
        is unsupported.numerical_status
        is CheckStatus.UNVERIFIED
    )
    # The helper delegates admission to the actual SOURCE facade. This exact
    # original uint32 identity is now ordinary typed data with actual evidence.
    integers = run_source_case(
        _metadata(lambda x: x, input_values=[np.asarray([1], np.uint32)]),
        SourceScenario("pending-integer", False, np.float32),
    )
    assert (
        integers.stage == "runtime-checked" and integers.category == "runtime-evidence"
    )
    assert integers.schema is CheckStatus.VALIDATED
    assert integers.audit.status is integers.numerical_status is CheckStatus.VALIDATED
    assert integers.prepared.traced_inputs[0].dtype == "uint32"
    assert integers.prepared.reference_outputs[0].dtype == np.dtype(np.uint32)
    assert (
        integers.prepared.reference_outputs[0].tobytes()
        == np.asarray([1], np.uint32).tobytes()
    )
    integers.validation.require_sample_acceptance()


def test_explicit_raw_criteria_are_frozen_without_legacy_f64_defaults() -> None:
    metadata = _metadata(
        lambda x: x,
        input_values=[np.asarray([1], np.float64)],
        rtol=2e-5,
        atol=1e-6,
        rtol_f64=99,
        atol_f64=99,
    )
    prepared = prepare_source_case(
        metadata, SourceScenario("criteria", True, np.float64)
    )
    assert prepared.criteria == SampleCriteria("tolerant", rtol=2e-5, atol=1e-6)
    metadata["rtol"] = 9
    assert prepared.criteria.rtol == 2e-5
    exact = prepare_source_case(
        _metadata(
            lambda x: x,
            input_values=[np.asarray([1], np.float64)],
            rtol_f64=99,
            atol_f64=99,
        ),
        SourceScenario("exact", True, np.float64),
    )
    assert exact.criteria.mode == "exact"


def test_symbolic_preparation_keeps_declared_shapes_and_explicit_concrete_sample() -> (
    None
):
    metadata = _metadata(lambda x: -x, input_shapes=[("B", 3)])
    scenario = SourceScenario("symbolic", False, np.float32, concrete_shapes=((2, 3),))
    result = run_source_case(metadata, scenario)
    assert result.prepared.reference_outputs[0].shape == (2, 3)
    assert result.prepared.traced_inputs[0].shape == ("B", 3)
    assert result.prepared.traced_outputs[0].shape == ("B", 3)
    assert result.prepared.arguments[0].shape == (2, 3)
    assert (
        result.prepared.traced_inputs[0].dtype
        == result.prepared.traced_outputs[0].dtype
        == "float32"
    )
    assert (
        result.prepared.reference_outputs[0].tobytes()
        == (-np.asarray(result.prepared.arguments[0])).tobytes()
    )
    assert result.stage == "runtime-checked" and result.category == "runtime-evidence"
    assert (
        result.schema
        is result.audit.status
        is result.numerical_status
        is CheckStatus.VALIDATED
    )
    assert result.model is not None and result.validation is not None
    assert len(result.model.graph.input) == len(result.model.graph.output) == 1
    for port in (*result.model.graph.input, *result.model.graph.output):
        tensor = port.type.tensor_type
        assert tensor.elem_type == onnx.TensorProto.FLOAT
        assert len(tensor.shape.dim) == 2
        assert tensor.shape.dim[0].dim_param == "B"
        assert tensor.shape.dim[1].dim_value == 3
    assert len(result.validation.samples) == 1
    sample = result.validation.samples[0]
    assert sample.criteria == result.prepared.criteria == SampleCriteria("exact")
    assert sample.status is CheckStatus.VALIDATED
    for evidence in (*sample.inputs, *sample.references, *sample.outputs):
        assert evidence.shape == (2, 3) and evidence.dtype == "float32"
    assert sample.references[0].sha256 == sample.outputs[0].sha256
    result.validation.require_sample_acceptance(result.model)


def test_existing_raw_registry_case_runs_without_legacy_oracle_seams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests import t_generator
    from jax2onnx import user_interface

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("SOURCE oracle must not call legacy preparation/allclose coercions")

    monkeypatch.setattr(t_generator, "generate_test_params", forbidden)
    monkeypatch.setattr(t_generator, "make_test_function", forbidden)
    monkeypatch.setattr(user_interface, "allclose", forbidden)
    monkeypatch.setattr(user_interface, "_to_numpy_input", forbidden)
    metadata = raw_source_case("plugin::add", 0)
    result = run_source_case(
        metadata, SourceScenario("registered-add", False, np.float32)
    )
    assert result.audit.status is CheckStatus.VALIDATED
    assert result.prepared.identity == ("primitives.lax", "add", "add")
    result.validation.require_sample_acceptance()


def test_typed_input_mismatch_is_not_silently_cast_for_runtime() -> None:
    scenario = SourceScenario(
        "no-feed-cast",
        False,
        np.float32,
        input_specs=(jax.ShapeDtypeStruct((1,), np.float32),),
        input_values=(np.asarray([1], np.float16),),
    )
    with pytest.raises(SourcePreparationError, match="no implicit feed cast"):
        prepare_source_case(_metadata(lambda x: x), scenario)


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.uint64, np.complex128])
@pytest.mark.parametrize("container", ["array", "scalar"])
@pytest.mark.parametrize("declaration", ["inferred", "shape-default", "explicit"])
def test_typed_inputs_reject_canonicalization_before_conversion_or_model_construction(
    dtype, container: str, declaration: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "high")
    value = np.asarray([1 + 2**-30] if dtype is np.float64 else [1], dtype=dtype)
    if container == "scalar":
        value = value[0]
    calls = []

    def constructor():
        calls.append("constructor")
        return lambda x: calls.append("callable") or x

    metadata = _metadata(construct_and_call(constructor), input_values=[value])
    if declaration == "shape-default":
        metadata["input_shapes"] = [value.shape]
    specs = (
        (jax.ShapeDtypeStruct(value.shape, dtype),)
        if declaration == "explicit"
        else None
    )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Typed invalid inputs must be rejected before jnp.asarray")

    monkeypatch.setattr(source.jnp, "asarray", forbidden)
    with pytest.raises(SourcePreparationError, match="no implicit feed cast"):
        prepare_source_case(
            metadata,
            SourceScenario(
                "preserve-typed-input", False, np.float32, input_specs=specs
            ),
        )
    assert not calls
    assert jax.config.x64_enabled
    assert jax.config.jax_default_matmul_precision == "high"


def test_existing_typed_jax_array_cannot_be_canonicalized_by_preparation() -> None:
    jax.config.update("jax_enable_x64", True)
    value = jnp.asarray([1 + 2**-30], dtype=jnp.float64)
    calls = []

    def function(x):
        calls.append(x)
        return x

    with pytest.raises(
        SourcePreparationError, match="float64 would canonicalize to float32"
    ):
        prepare_source_case(
            _metadata(function, input_values=[value]),
            SourceScenario("typed-jax-input", False, np.float32),
        )
    assert not calls
    assert jax.config.x64_enabled


def test_inferred_fp64_payload_cannot_receive_fp32_sample_acceptance() -> None:
    value = np.asarray([1 + 2**-30], np.float64)
    calls = []

    def function(x):
        calls.append(np.dtype(x.dtype))
        return x

    metadata = _metadata(function, input_values=[value])
    rejected = run_source_case(
        metadata, SourceScenario("inferred-fp64-off", False, np.float32)
    )
    assert rejected.stage == "source-preparation"
    assert rejected.audit.category == "original-source"
    assert rejected.audit.status is CheckStatus.FAILED
    assert rejected.prepared is None and rejected.validation is None
    assert rejected.numerical_status is CheckStatus.UNVERIFIED
    assert not calls
    accepted = run_source_case(
        metadata, SourceScenario("inferred-fp64-on", True, np.float32)
    )
    assert accepted.prepared.traced_inputs[0].dtype == "float64"
    assert accepted.prepared.reference_outputs[0].dtype == np.dtype(np.float64)
    assert accepted.prepared.reference_outputs[0].tobytes() == value.tobytes()
    accepted.validation.require_sample_acceptance()
    assert set(calls) == {np.dtype(np.float64)}


def test_real_bitcast_opset_limit_is_distinct_from_unaudited_support() -> None:
    metadata = _metadata(
        lambda x: jax.lax.bitcast_convert_type(x, jnp.uint32),
        input_values=[np.asarray([1], np.float32)],
        opset_version=23,
    )
    result = run_source_case(
        metadata, SourceScenario("bitcast-representation", False, np.float32)
    )
    assert result.category == "representation-limit"
    assert result.audit.status is CheckStatus.FAILED
    assert "26" in result.audit.detail
    assert result.numerical_status is CheckStatus.UNVERIFIED


def test_oracle_export_uses_shared_private_facade_and_real_trace_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual = source.user_interface._to_onnx
    calls = []

    def observed(*args: Any, **kwargs: Any):
        calls.append(kwargs["precision_policy"])
        return actual(*args, **kwargs)

    monkeypatch.setattr(source.user_interface, "_to_onnx", observed)
    metadata = _metadata(
        lambda x: -x,
        input_values=[np.asarray([1], np.float32)],
        input_names=["source_input"],
        output_names=["source_output"],
    )
    result = run_source_case(
        metadata, SourceScenario("shared-facade", False, np.float32)
    )
    assert calls == [PrecisionPolicy.SOURCE]
    assert result.audit.status is CheckStatus.VALIDATED
    assert result.model.graph.input[0].name == "source_input"
    assert result.model.graph.output[0].name == "source_output"
    report = get_precision_report(result.model)
    assert report.inputs[0].traced.dtype == result.prepared.traced_inputs[0].dtype
    assert report.exported_outputs[0].name == "source_output"
    result.validation.require_sample_acceptance()
