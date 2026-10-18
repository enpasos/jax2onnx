# tests/extra_tests/converter/test_precision_evidence.py

"""Public provenance is distinct from bounded, artifact-bound CPU evidence."""

from collections.abc import Iterator
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from jax2onnx.precision import (
    CheckStatus,
    PRECISION_METADATA_KEY,
    PrecisionCheck,
    PrecisionInputSummary,
    PrecisionReport,
    PrecisionReportError,
    PrecisionWarning,
    SampleAcceptanceError,
    SampleCriteria,
    SourceSample,
    TensorProperties,
    attach_precision_report,
    get_precision_report,
    precision_report_to_json,
    validate_source_samples,
)
from tests.extra_tests.converter.precision_test_support import export_source_dtypes


@pytest.fixture(autouse=True)
def _restore_jax_settings() -> Iterator[None]:
    settings = {
        "jax_enable_x64": jax.config.x64_enabled,
        "jax_platforms": jax.config.jax_platforms,
        "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
    }
    try:
        yield
    finally:
        for name, value in settings.items():
            jax.config.update(name, value)


def _identity() -> onnx.ModelProto:
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["x"], ["y"])],
        "sample",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])],
    )
    return onnx.helper.make_model(
        graph, ir_version=10, opset_imports=[onnx.helper.make_opsetid("", 23)]
    )


def _report() -> PrecisionReport:
    properties = TensorProperties("x", (1,), "float32", False)
    return PrecisionReport(
        exporter_version="test-exporter",
        jax_version=jax.__version__,
        opset=23,
        caller_x64_enabled=False,
        caller_default_matmul_precision=None,
        inputs=(
            PrecisionInputSummary(
                0,
                TensorProperties(None, (1,), None),
                properties,
                properties,
                properties,
            ),
        ),
        traced_outputs=(TensorProperties(None, (1,), "float32", False),),
        exported_outputs=(TensorProperties("y", (1,), "float32"),),
        warnings=(PrecisionWarning("UserWarning", "recorded warning"),),
        adaptations=("Audited test adaptation",),
        checks=(
            PrecisionCheck(
                "representation", CheckStatus.VALIDATED, "Bounded schema checked"
            ),
            PrecisionCheck(
                "lowering", CheckStatus.VALIDATED, "Bounded lowering checked"
            ),
            PrecisionCheck(
                "runtime-execution", CheckStatus.UNVERIFIED, "No target executed"
            ),
            PrecisionCheck(
                "numerical-acceptance", CheckStatus.UNVERIFIED, "No samples supplied"
            ),
        ),
    )


@pytest.mark.parametrize("representation", ["proto", "ir", "file"])
def test_export_report_round_trips_without_implying_runtime_acceptance(
    representation: str, tmp_path: Path
) -> None:
    proto = _identity()
    model = ir.from_proto(proto) if representation == "ir" else proto
    report = _report()
    attach_precision_report(model, report)
    if representation == "file":
        path = tmp_path / "export.onnx"
        onnx.save_model(model, path)
        model = path
    recovered = get_precision_report(model)
    assert recovered == report
    assert recovered is not report
    assert recovered.inputs[0].supplied.dtype is None
    assert recovered.schema_version == 1 and recovered.profile == "source-static-v1"
    assert recovered.checks[-1].status is CheckStatus.UNVERIFIED
    with pytest.raises(FrozenInstanceError):
        recovered.profile = "mutated"  # type: ignore[misc]


def test_report_defaults_are_unverified_and_legacy_metadata_is_absent() -> None:
    assert get_precision_report(_identity()) is None
    properties = TensorProperties(None, (1,), "float32")
    report = PrecisionReport(
        "test", "test", 23, False, None, (), (properties,), (properties,)
    )
    assert all(check.status is CheckStatus.UNVERIFIED for check in report.checks)


@pytest.mark.parametrize(
    "change",
    ["json", "version", "profile", "dtype", "status", "collection", "duplicate"],
)
def test_present_invalid_metadata_is_not_silently_treated_as_legacy(
    change: str,
) -> None:
    model = _identity()
    data = json.loads(precision_report_to_json(_report()))
    if change == "version":
        data["schema_version"] = 2
    elif change == "profile":
        data["profile"] = "future-profile"
    elif change == "dtype":
        data["inputs"][0]["supplied"]["dtype"] = 17
    elif change == "status":
        data["checks"][0]["status"] = "accepted-by-assumption"
    elif change == "collection":
        data["adaptations"] = "invalid collection"
    onnx.helper.set_model_props(
        model, {PRECISION_METADATA_KEY: "{" if change == "json" else json.dumps(data)}
    )
    if change == "duplicate":
        item = model.metadata_props.add()
        item.key, item.value = PRECISION_METADATA_KEY, json.dumps(data)
    with pytest.raises(PrecisionReportError):
        get_precision_report(model)


@pytest.mark.parametrize("optimization", ["disabled", "all"])
def test_actual_cpu_session_records_identity_criteria_and_settings(
    optimization: str,
) -> None:
    model = _identity()
    value = np.asarray([1], np.float32)
    report = validate_source_samples(
        model,
        samples=[SourceSample({"x": value}, [value.copy()])],
        optimization=optimization,
    )
    assert report.status is CheckStatus.VALIDATED
    report.require_sample_acceptance()
    assert report.target is not None
    assert report.target.providers == ("CPUExecutionProvider",)
    assert report.target.runtime_version == ort.__version__
    assert report.target.optimization == (
        "ORT_ENABLE_ALL" if optimization == "all" else "ORT_DISABLE_ALL"
    )
    assert report.target.intra_op_threads == report.target.inter_op_threads == 1
    assert report.target.execution_mode == "ORT_SEQUENTIAL"
    assert report.artifact_sha256 and len(report.artifact_sha256) == 64
    sample = report.samples[0]
    assert sample.inputs[0].name == "x"
    assert (
        sample.inputs[0].sha256
        == sample.references[0].sha256
        == sample.outputs[0].sha256
    )
    assert sample.criteria.mode == "exact"
    assert len(sample.sample_sha256) == 64
    assert (
        tuple(provider for provider, _ in report.target.provider_options)
        == report.target.providers
    )
    # Evidence contains hashes and properties, never references to caller arrays.
    digest = sample.inputs[0].sha256
    value[0] = 77
    assert sample.inputs[0].sha256 == digest


@pytest.mark.parametrize("difference", ["dtype", "shape", "value", "count"])
def test_reference_mismatches_are_known_failures(difference: str) -> None:
    value = np.asarray([1], np.float32)
    expected = {
        "dtype": [value.astype(np.float64)],
        "shape": [value.reshape(())],
        "value": [np.asarray([2], np.float32)],
        "count": [value, value],
    }[difference]
    report = validate_source_samples(
        _identity(), samples=[SourceSample([value], expected)], optimization="disabled"
    )
    assert report.status is CheckStatus.FAILED
    assert report.samples[0].checks[0].category == "runtime-execution"
    assert report.samples[0].checks[0].status is CheckStatus.VALIDATED
    assert report.samples[0].checks[-1].category == "numerical-acceptance"
    with pytest.raises(SampleAcceptanceError, match="failed"):
        report.require_sample_acceptance()


def test_tolerance_is_explicit_and_distinct_from_exact_values() -> None:
    value = np.asarray([1], np.float32)
    reference = np.asarray([1.00001], np.float32)
    exact = validate_source_samples(
        _identity(),
        samples=[SourceSample([value], [reference])],
        optimization="disabled",
    )
    tolerant = validate_source_samples(
        _identity(),
        samples=[
            SourceSample(
                [value], [reference], SampleCriteria("tolerant", rtol=2e-5, atol=0)
            )
        ],
        optimization="disabled",
    )
    assert exact.status is CheckStatus.FAILED
    tolerant.require_sample_acceptance()
    assert exact.samples[0].sample_sha256 != tolerant.samples[0].sample_sha256
    with pytest.raises(ValueError, match="explicit"):
        SampleCriteria("tolerant")
    with pytest.raises(ValueError, match="only"):
        SampleCriteria("exact", rtol=1e-3)
    for tolerance in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError):
            SampleCriteria("tolerant", rtol=tolerance, atol=0)


def test_bitwise_criterion_deliberately_distinguishes_signed_zero() -> None:
    value = np.asarray([-0.0], np.float32)
    reference = np.asarray([0.0], np.float32)
    exact = validate_source_samples(
        _identity(),
        samples=[SourceSample([value], [reference])],
        optimization="disabled",
    )
    bitwise = validate_source_samples(
        _identity(),
        samples=[SourceSample([value], [reference], SampleCriteria("bitwise"))],
        optimization="disabled",
    )
    exact.require_sample_acceptance()
    assert bitwise.status is CheckStatus.FAILED
    assert (
        bitwise.samples[0].inputs[0].sha256 != bitwise.samples[0].references[0].sha256
    )
    with pytest.raises(SampleAcceptanceError, match="bitwise"):
        bitwise.require_sample_acceptance()


@pytest.mark.parametrize(
    "evidence", ["no-target", "no-samples", "no-reference", "empty-reference"]
)
def test_missing_evidence_never_executes_or_accepts(
    evidence: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_session(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Missing evidence must not select/execute a runtime")

    monkeypatch.setattr(ort, "InferenceSession", unexpected_session)
    value = np.asarray([1], np.float32)
    expected = (
        None
        if evidence == "no-reference"
        else ([] if evidence == "empty-reference" else [value])
    )
    report = validate_source_samples(
        _identity(),
        samples=[] if evidence == "no-samples" else [SourceSample([value], expected)],
        optimization=None if evidence == "no-target" else "disabled",
    )
    assert report.status is CheckStatus.UNVERIFIED
    with pytest.raises(SampleAcceptanceError, match="unverified"):
        report.require_sample_acceptance()


def test_failed_sample_cannot_be_overwritten_by_later_success() -> None:
    value = np.asarray([1], np.float32)
    report = validate_source_samples(
        _identity(),
        samples=[
            SourceSample([value], [value + np.float32(1)]),
            SourceSample([value], [value]),
        ],
        optimization="disabled",
    )
    assert [sample.status for sample in report.samples] == [
        CheckStatus.FAILED,
        CheckStatus.VALIDATED,
    ]
    assert report.status is CheckStatus.FAILED
    with pytest.raises(SampleAcceptanceError):
        report.require_sample_acceptance()


def test_schema_failure_is_distinct_from_runtime_and_numerical_failures() -> None:
    model = _identity()
    model.graph.node[0].op_type = "MissingOnnxOperation"
    report = validate_source_samples(model, samples=[], optimization=None)
    assert report.status is CheckStatus.FAILED
    assert report.checks[-1].category == "onnx-schema"
    assert report.target is None


def test_runtime_load_failure_is_known_and_not_a_numerical_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_kernel(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("No CPU kernel for this operation/dtype")

    monkeypatch.setattr(ort, "InferenceSession", missing_kernel)
    value = np.asarray([1], np.float32)
    report = validate_source_samples(
        _identity(), samples=[SourceSample([value], [value])], optimization="disabled"
    )
    assert report.status is CheckStatus.FAILED
    assert report.checks[-1].category == "runtime-load"
    assert "No CPU kernel" in report.checks[-1].detail
    assert not any(
        check.category == "numerical-acceptance" and check.status is CheckStatus.FAILED
        for check in report.checks
    )


def test_runtime_run_failure_is_distinct_and_dominates_unverified_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ort.InferenceSession

    class RunFailure:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.actual = original(*args, **kwargs)

        def disable_fallback(self) -> None:
            self.actual.disable_fallback()

        def get_providers(self) -> list[str]:
            return self.actual.get_providers()

        def get_provider_options(self) -> dict[str, dict[str, str]]:
            return self.actual.get_provider_options()

        def get_inputs(self) -> Any:
            return self.actual.get_inputs()

        def run(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("CPU execution failed")

    monkeypatch.setattr(ort, "InferenceSession", RunFailure)
    value = np.asarray([1], np.float32)
    report = validate_source_samples(
        _identity(), samples=[SourceSample([value], [value])], optimization="disabled"
    )
    assert report.status is CheckStatus.FAILED
    assert report.samples[0].checks[-2].category == "runtime-execution"
    assert report.samples[0].checks[-2].status is CheckStatus.FAILED
    assert report.samples[0].checks[-1].status is CheckStatus.UNVERIFIED


def test_actual_provider_list_must_be_cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    original = ort.InferenceSession

    class ProviderMismatch:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.actual = original(*args, **kwargs)

        def disable_fallback(self) -> None:
            self.actual.disable_fallback()

        def get_providers(self) -> list[str]:
            return ["UnexpectedExecutionProvider", "CPUExecutionProvider"]

        def get_provider_options(self) -> dict[str, dict[str, str]]:
            return self.actual.get_provider_options()

    monkeypatch.setattr(ort, "InferenceSession", ProviderMismatch)
    value = np.asarray([1], np.float32)
    report = validate_source_samples(
        _identity(), samples=[SourceSample([value], [value])], optimization="all"
    )
    assert report.status is CheckStatus.FAILED
    assert report.target.providers == (
        "UnexpectedExecutionProvider",
        "CPUExecutionProvider",
    )
    assert report.checks[-1].category == "runtime-load"


@pytest.mark.parametrize("representation", ["proto", "ir", "file"])
def test_guard_rechecks_artifact_identity_after_mutation(
    representation: str, tmp_path: Path
) -> None:
    proto = _identity()
    model = ir.from_proto(proto) if representation == "ir" else proto
    if representation == "file":
        model = tmp_path / "model.onnx"
        onnx.save_model(proto, model)
    value = np.asarray([1], np.float32)
    report = validate_source_samples(
        model, samples=[SourceSample([value], [value])], optimization="disabled"
    )
    report.require_sample_acceptance()
    if representation == "file":
        proto.graph.doc_string = "changed"
        onnx.save_model(proto, model)
    elif representation == "ir":
        model.graph.doc_string = "changed"
    else:
        model.graph.doc_string = "changed"
    with pytest.raises(SampleAcceptanceError, match="changed"):
        report.require_sample_acceptance()


def test_saved_external_payload_is_inlined_hashed_and_rechecked(tmp_path: Path) -> None:
    model = _identity()
    model.graph.node[0].op_type = "Add"
    model.graph.node[0].input.append("bias")
    model.graph.initializer.append(
        onnx.numpy_helper.from_array(np.asarray([2], np.float32), "bias")
    )
    path = tmp_path / "external.onnx"
    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="payload.bin",
        size_threshold=0,
    )
    value = np.asarray([1], np.float32)
    report = validate_source_samples(
        path,
        samples=[SourceSample([value], [value + np.float32(2)])],
        optimization="disabled",
    )
    report.require_sample_acceptance()
    assert report.artifact_sha256
    (tmp_path / "payload.bin").write_bytes(np.asarray([3], np.float32).tobytes())
    with pytest.raises(SampleAcceptanceError, match="changed"):
        report.require_sample_acceptance()
    unresolved = onnx.load_model(path, load_external_data=False)
    rejected = validate_source_samples(unresolved, samples=[], optimization=None)
    assert rejected.status is CheckStatus.FAILED
    assert rejected.checks[-1].category == "artifact-load"


def test_metadata_reader_does_not_require_external_payload(tmp_path: Path) -> None:
    model = _identity()
    model.graph.initializer.append(
        onnx.numpy_helper.from_array(np.asarray([2], np.float32), "unused_bias")
    )
    attach_precision_report(model, _report())
    path = tmp_path / "external.onnx"
    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="payload.bin",
        size_threshold=0,
    )
    (tmp_path / "payload.bin").unlink()
    assert get_precision_report(path) == _report()
    validation = validate_source_samples(path, samples=[], optimization=None)
    assert validation.status is CheckStatus.FAILED
    assert validation.checks[-1].category == "artifact-load"


def test_artifact_loader_rejects_symlink_paths(tmp_path: Path) -> None:
    path = tmp_path / "actual.onnx"
    onnx.save_model(_identity(), path)
    link = tmp_path / "symlink.onnx"
    link.symlink_to(path)
    report = validate_source_samples(link, samples=[], optimization=None)
    assert report.status is CheckStatus.FAILED
    assert "Symlink" in report.checks[-1].detail


@pytest.mark.parametrize("witness", ["fp16-square-sum", "weak-bias", "fp32-reciprocal"])
@pytest.mark.parametrize("optimization", ["disabled", "all"])
def test_known_source_witnesses_use_independent_references_without_freezing_runtime_defects(
    witness: str, optimization: str
) -> None:
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_matmul_precision", None)
    if witness == "fp16-square-sum":

        def function(x: jax.Array) -> tuple[jax.Array, ...]:
            square = x * x
            return square, jax.lax.reduce_sum(square, axes=(0,))

        values = [np.full(8, 2**-13, np.float16)]
        expected = [np.zeros(8, np.float16), np.asarray(0, np.float16)]
    elif witness == "weak-bias":
        weak_bias = jnp.asarray(4096.0)
        assert weak_bias.weak_type

        def function(x: jax.Array) -> tuple[jax.Array, ...]:
            return ((x + weak_bias).astype(jnp.float32) * np.float32(2**-20),)

        values = [np.asarray([1], np.float16)]
        rounded = np.add(values[0], np.float16(4096), dtype=np.float16)
        expected = [
            np.multiply(
                rounded.astype(np.float32), np.float32(2**-20), dtype=np.float32
            )
        ]
    else:

        def function(x: jax.Array, y: jax.Array) -> tuple[jax.Array, ...]:
            return (x * (np.float32(1) / jax.lax.sqrt(y)),)

        values = [np.asarray([3], np.float32), np.asarray([2], np.float32)]
        root = np.sqrt(values[1], dtype=np.float32)
        reciprocal = np.divide(np.float32(1), root, dtype=np.float32)
        expected = [np.multiply(values[0], reciprocal, dtype=np.float32)]
    model = export_source_dtypes(
        function, [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in values]
    )
    report = validate_source_samples(
        model, samples=[SourceSample(values, expected)], optimization=optimization
    )
    assert report.status in (CheckStatus.VALIDATED, CheckStatus.FAILED)
    assert report.samples[0].checks[0].status is CheckStatus.VALIDATED
    if report.status is CheckStatus.FAILED:
        assert report.samples[0].checks[-1].category == "numerical-acceptance"
        with pytest.raises(SampleAcceptanceError, match="failed"):
            report.require_sample_acceptance()
    else:
        report.require_sample_acceptance()
