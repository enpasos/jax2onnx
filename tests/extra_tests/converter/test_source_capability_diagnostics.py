# tests/extra_tests/converter/test_source_capability_diagnostics.py

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from typing import Any

import numpy as np
import onnx
import pytest

from jax2onnx import _source_capability as capability
from jax2onnx.precision import (
    CheckStatus,
    SampleAcceptanceError,
    SampleCriteria,
    SourceSample,
    validate_source_samples,
)

from tests.extra_tests.converter.source_capability_counterparts import (
    SourceSchemaObservation,
)


_SCHEMA_RAW = "[ShapeInferenceError] (op_type:Range, node name: range0): start typestr: T, has unsupported type: tensor(bool)"
_RUNTIME_RAW = (
    "[ONNXRuntimeError] : 1 : FAIL : actual model_load_utils.cc:46 "
    "ONNX Runtime only *guarantees* support for models stamped with official "
    "released onnx opset versions. Opset 26 is under development and support "
    "for this is limited. The operator schemas and or other functionality may "
    "change before next ONNX release and in this case ONNX Runtime will not "
    "guarantee backward compatibility. Current official support for domain "
    "ai.onnx is till opset 25."
)


def _range(dtype: int = onnx.TensorProto.BOOL, opset: int = 27) -> onnx.ModelProto:
    return onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node(
                    "Range", ["start", "limit", "delta"], ["out"], name="range0"
                )
            ],
            "typed_range",
            [
                onnx.helper.make_tensor_value_info(name, dtype, [])
                for name in ("start", "limit", "delta")
            ],
            [onnx.helper.make_tensor_value_info("out", dtype, [1])],
        ),
        opset_imports=[onnx.helper.make_opsetid("", opset)],
    )


def _identity(opset: int = 26) -> onnx.ModelProto:
    return onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("Identity", ["x"], ["y"], name="identity0")],
            "identity",
            [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2])],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2])],
        ),
        opset_imports=[onnx.helper.make_opsetid("", opset)],
    )


def _schema_failure(
    monkeypatch: pytest.MonkeyPatch,
    model: onnx.ModelProto | None = None,
    raw: str = _SCHEMA_RAW,
) -> tuple[Exception, onnx.ModelProto]:
    model = _range() if model is None else model
    error = onnx.shape_inference.InferenceError(raw)
    calls = []

    def checker(blob: bytes, *, full_check: bool) -> None:
        assert type(blob) is bytes and blob == model.SerializeToString()
        assert full_check is True
        calls.append(blob)
        raise error

    monkeypatch.setattr(onnx.checker, "check_model", checker)
    with pytest.raises(onnx.shape_inference.InferenceError) as caught:
        capability.check_source_schema(model)
    assert caught.value is error and str(error) == raw
    assert calls == [model.SerializeToString()]
    return error, model


def test_actual_checker_rejection_identifies_actual_schema_and_original_cause() -> None:
    model = _range()
    original = model.SerializeToString()
    schema = onnx.defs.get_schema("Range", 27, "")
    constraint = next(
        item for item in schema.type_constraints if item.type_param_str == "T"
    )
    assert "tensor(bool)" not in constraint.allowed_type_strs
    with pytest.raises(onnx.shape_inference.InferenceError) as caught:
        capability.check_source_schema(model)
    diagnostic = capability.diagnostic_from_detail(
        capability.exception_detail(caught.value),
        artifact_sha256=hashlib.sha256(original).hexdigest(),
    )
    assert diagnostic is not None
    assert diagnostic.context["schema_since_version"] == schema.since_version
    assert diagnostic.context["requested_opset"] == 27
    assert diagnostic.context["allowed_types"] == list(constraint.allowed_type_strs)
    assert diagnostic.cause_message == str(caught.value)
    assert (
        diagnostic.cause_type
        == f"{type(caught.value).__module__}.{type(caught.value).__qualname__}"
    )
    assert model.SerializeToString() == original


def test_checker_valid_existing_operator_at_higher_declared_opset_is_not_pre_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _identity(onnx.defs.onnx_opset_version() + 1)
    checker = onnx.checker.check_model
    calls = []

    def actual(blob: bytes, *, full_check: bool) -> None:
        calls.append(blob)
        checker(blob, full_check=full_check)

    monkeypatch.setattr(onnx.checker, "check_model", actual)
    capability.check_source_schema(model)
    assert calls == [model.SerializeToString()]


@pytest.mark.parametrize(
    "defect",
    [
        "unrelated",
        "wrong-operator",
        "wrong-node",
        "wrong-port",
        "wrong-parameter",
        "wrong-type",
        "supported-type",
        "wrong-domain",
        "duplicate-import",
        "unknown-domain",
        "ambiguous-node",
        "conflicting-metadata",
    ],
)
def test_schema_unrecognized_failures_keep_exact_exception_and_text(
    monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    model, raw = _range(), _SCHEMA_RAW
    if defect == "unrelated":
        raw = "[ShapeInferenceError] actual malformed shape, not a port capability"
    elif defect == "wrong-operator":
        raw = raw.replace("op_type:Range", "op_type:Identity")
    elif defect == "wrong-node":
        raw = raw.replace("range0", "other_range")
    elif defect == "wrong-port":
        raw = raw.replace(": start typestr", ": absent typestr")
    elif defect == "wrong-parameter":
        raw = raw.replace("typestr: T,", "typestr: U,")
    elif defect == "wrong-type":
        raw = raw.replace("tensor(bool)", "tensor(float16)")
    elif defect == "supported-type":
        model = _range(onnx.TensorProto.FLOAT)
        raw = raw.replace("tensor(bool)", "tensor(float)")
    elif defect == "wrong-domain":
        model.graph.node[0].domain = "example.unsupported"
    elif defect == "duplicate-import":
        model.opset_import.append(onnx.helper.make_opsetid("ai.onnx", 27))
    elif defect == "unknown-domain":
        model.graph.node[0].domain = "example.unsupported"
        model.opset_import.append(onnx.helper.make_opsetid("example.unsupported", 27))
    elif defect == "ambiguous-node":
        extra = deepcopy(model.graph.node[0])
        extra.output[0] = "other"
        model.graph.node.append(extra)
    elif defect == "conflicting-metadata":
        model.graph.value_info.append(
            onnx.helper.make_tensor_value_info("start", onnx.TensorProto.FLOAT, [])
        )
    error, _ = _schema_failure(monkeypatch, model, raw)
    assert capability.exception_detail(error) == raw
    assert not getattr(error, "__notes__", ())


def test_schema_note_survives_same_attempt_text_adapter_and_precision_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error, model = _schema_failure(monkeypatch)
    detail = capability.exception_detail(error)
    assert detail.startswith(_SCHEMA_RAW + "\n")
    assert capability.diagnostic_from_detail(detail) is not None
    # The existing precision schema gate observes the same original exception;
    # no runtime import or sample execution is introduced by the adapter.
    result = validate_source_samples(model, samples=[], optimization=None)
    assert result.status is CheckStatus.FAILED and not result.samples
    schema = next(item for item in result.checks if item.category == "onnx-schema")
    assert schema.status is CheckStatus.FAILED
    assert capability.diagnostic_from_detail(schema.detail) is not None
    with pytest.raises(SampleAcceptanceError):
        result.require_sample_acceptance()


def _runtime_failure_detail(
    model: onnx.ModelProto | None = None, *, raw: str = _RUNTIME_RAW
) -> tuple[str, bytes]:
    import onnxruntime as ort
    from onnxruntime.capi.onnxruntime_pybind11_state import Fail

    blob = (_identity() if model is None else model).SerializeToString()
    error = Fail(raw)
    return (
        capability.runtime_load_detail(
            error, blob, ort, providers_requested=("CPUExecutionProvider",)
        ),
        blob,
    )


def test_runtime_actual_causal_class_and_artifact_binding_are_preserved() -> None:
    import onnxruntime as ort
    from onnxruntime.capi.onnxruntime_pybind11_state import Fail

    detail, blob = _runtime_failure_detail()
    diagnostic = capability.diagnostic_from_detail(
        detail,
        artifact_sha256=hashlib.sha256(blob).hexdigest(),
        providers_requested=("CPUExecutionProvider",),
    )
    assert diagnostic is not None and diagnostic.cause_message == _RUNTIME_RAW
    assert diagnostic.cause_type == f"{Fail.__module__}.{Fail.__qualname__}"
    assert diagnostic.opset_imports == (("", 26),)
    assert diagnostic.context == {
        "domain": "",
        "requested_opset": 26,
        "supported_opset": 25,
        "runtime_version": ort.__version__,
        "runtime_build": ort.get_build_info(),
        "providers_requested": ["CPUExecutionProvider"],
    }
    assert capability.diagnostic_from_detail(detail, artifact_sha256="0" * 64) is None
    assert (
        capability.diagnostic_from_detail(
            detail, providers_requested=("CUDAExecutionProvider",)
        )
        is None
    )


@pytest.mark.parametrize(
    "defect",
    [
        "unrelated-class",
        "unrelated-message",
        "wrong-domain",
        "wrong-target",
        "supported-target",
        "wrong-provider",
        "runtime-absent",
    ],
)
def test_unrelated_runtime_failures_are_not_classified(defect: str) -> None:
    import onnxruntime as ort
    from onnxruntime.capi.onnxruntime_pybind11_state import Fail

    model, raw, runtime = _identity(), _RUNTIME_RAW, ort
    providers = ("CPUExecutionProvider",)
    error: Exception = Fail(raw)
    if defect == "unrelated-class":
        error = ValueError(raw)
    elif defect == "unrelated-message":
        error = Fail("actual malformed input/output or unknown operator")
    elif defect == "wrong-domain":
        error = Fail(raw.replace("domain ai.onnx", "domain example.other"))
    elif defect == "wrong-target":
        model = _identity(23)
    elif defect == "supported-target":
        error = Fail(raw.replace("till opset 25", "till opset 26"))
    elif defect == "wrong-provider":
        providers = ("CUDAExecutionProvider",)
    elif defect == "runtime-absent":
        runtime = None
    assert capability.runtime_load_detail(
        error, model.SerializeToString(), runtime, providers_requested=providers
    ) == str(error)


@pytest.mark.parametrize("boundary", ["onnx-schema", "runtime-load"])
@pytest.mark.parametrize(
    "defect",
    [
        "protocol",
        "cause-type",
        "cause-text",
        "digest",
        "domain",
        "duplicate-import",
        "opset-string",
        "opset-bool",
        "extra-field",
        "provider-or-type",
    ],
)
def test_serialized_diagnostic_rejects_hostile_metadata(
    monkeypatch: pytest.MonkeyPatch, boundary: str, defect: str
) -> None:
    if boundary == "onnx-schema":
        error, _ = _schema_failure(monkeypatch)
        detail = capability.exception_detail(error)
    else:
        detail, _ = _runtime_failure_detail()
    diagnostic = capability.diagnostic_from_detail(detail)
    assert diagnostic is not None
    data = asdict(diagnostic)
    if defect == "protocol":
        data["protocol"] = 2
    elif defect == "cause-type":
        data["cause_type"] = "builtins.ValueError"
    elif defect == "cause-text":
        data["cause_message"] = "different attempt"
    elif defect == "digest":
        data["artifact_sha256"] = "not-a-digest"
    elif defect == "domain":
        data["context"]["domain"] = "unrelated.domain"
    elif defect == "duplicate-import":
        data["opset_imports"] = [*data["opset_imports"], data["opset_imports"][0]]
    elif defect == "opset-string":
        data["context"]["requested_opset"] = str(data["context"]["requested_opset"])
    elif defect == "opset-bool":
        data["context"]["requested_opset"] = True
    elif defect == "extra-field":
        data["pretend_accepted"] = True
    elif defect == "provider-or-type":
        key = "actual_type" if boundary == "onnx-schema" else "providers_requested"
        data["context"][key] = (
            "tensor(float)" if boundary == "onnx-schema" else ["CUDAExecutionProvider"]
        )
    hostile = diagnostic.cause_message + "\n" + capability._PREFIX + json.dumps(data)
    assert capability.diagnostic_from_detail(hostile) is None


def test_runtime_load_report_keeps_failed_status_no_outputs_and_rejects_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import onnxruntime as ort
    from onnxruntime.capi.onnxruntime_pybind11_state import Fail

    model, attempts = _identity(), []
    error = Fail(_RUNTIME_RAW)

    def session(blob: bytes, options: Any, *, providers: list[str]) -> None:
        attempts.append((blob, providers, options.graph_optimization_level))
        raise error

    monkeypatch.setattr(ort, "InferenceSession", session)
    values = np.asarray([1, -0.0], np.float32)
    sample = SourceSample((values,), (values,), SampleCriteria("bitwise"))
    for optimization in ("disabled", "all"):
        result = validate_source_samples(
            model, samples=[sample], optimization=optimization
        )
        assert result.status is CheckStatus.FAILED and len(result.samples) == 1
        assert result.samples[0].outputs == ()
        checks = {item.category: item for item in result.samples[0].checks}
        assert checks["runtime-load"].status is CheckStatus.FAILED
        assert checks["numerical-acceptance"].status is CheckStatus.UNVERIFIED
        diagnostic = capability.diagnostic_from_detail(
            checks["runtime-load"].detail,
            artifact_sha256=result.artifact_sha256,
            providers_requested=("CPUExecutionProvider",),
        )
        assert diagnostic is not None and diagnostic.cause_message == str(error)
        with pytest.raises(SampleAcceptanceError):
            result.require_sample_acceptance()
        serialized = json.dumps(
            {
                "artifact_sha256": result.artifact_sha256,
                "checks": [asdict(item) for item in result.checks],
                "samples": [asdict(item) for item in result.samples],
            }
        )
        restored = json.loads(serialized)
        assert (
            capability.diagnostic_from_detail(
                next(
                    item["detail"]
                    for item in restored["checks"]
                    if item["category"] == "runtime-load"
                ),
                artifact_sha256=result.artifact_sha256,
            )
            == diagnostic
        )
    assert len(attempts) == 2 and all(
        item[1] == ["CPUExecutionProvider"] for item in attempts
    )


@pytest.mark.parametrize("kind", ["missing-reference", "malformed-input"])
def test_invalid_only_samples_do_not_import_runtime_or_construct_session(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    # Intercept the import statement itself: metadata/parser/checker work must
    # not turn an unexecuted sample into an optional-runtime dependency.
    import builtins

    original_import = builtins.__import__
    attempts = []

    def import_guard(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "onnxruntime" or name.startswith("onnxruntime."):
            attempts.append(name)
            raise AssertionError("Invalid-only sample imported the runtime")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_guard)
    values = np.asarray([1, 2], np.float32)
    sample = (
        SourceSample((values,), None, SampleCriteria("bitwise"))
        if kind == "missing-reference"
        else SourceSample((object(),), (values,), SampleCriteria("bitwise"))
    )
    result = validate_source_samples(
        _identity(), samples=[sample], optimization="disabled"
    )
    assert result.status is not CheckStatus.VALIDATED
    assert not result.samples[0].outputs
    assert not attempts


def test_unknown_exception_notes_leave_existing_text_unchanged() -> None:
    error = ValueError("unchanged original")
    error.add_note("unrecognized note")
    error.add_note(capability._PREFIX + "{broken-json")
    assert capability.exception_detail(error) == str(error)


_MISSING_OPERATOR = "Jax2OnnxUnregisteredSchemaControl"


def _missing_schema_model(operator: str = _MISSING_OPERATOR) -> onnx.ModelProto:
    model = _identity()
    model.graph.node[0].op_type = operator
    model.graph.node[0].name = "missing0"
    return model


def _missing_schema_text(operator: str = _MISSING_OPERATOR) -> str:
    return f"No Op registered for {operator} with domain_version of 26\n\n==> Context: Bad node spec for node. Name: missing0 OpType: {operator}"


def test_missing_schema_observer_preserves_actual_checker_exception_and_candidate() -> (
    None
):
    model = _missing_schema_model()
    observation = SourceSchemaObservation(capability.check_source_schema)
    before = model.SerializeToString()
    with pytest.raises(onnx.checker.ValidationError) as caught:
        observation(model)
    attempt = observation.assert_unchanged()
    assert attempt.original is model and attempt.cause is caught.value
    diagnostic = observation.missing_schema_failure(
        caught.value, operator=_MISSING_OPERATOR, opset=26
    )
    assert diagnostic is not None
    assert diagnostic["schema"] == "FAILED"
    assert diagnostic["successful_export"] is False
    assert (
        diagnostic["numerical_acceptance"]
        == diagnostic["output_acceptance"]
        == "UNVERIFIED"
    )
    assert diagnostic["artifact_sha256"] == hashlib.sha256(before).hexdigest()
    assert diagnostic["opset_imports"] == (("", 26),)
    assert diagnostic["cause_text"] == str(caught.value) == _missing_schema_text()
    assert (
        diagnostic["cause_type"]
        == f"{type(caught.value).__module__}.{type(caught.value).__qualname__}"
    )
    assert (
        diagnostic["registry_cause_type"]
        == f"{onnx.defs.SchemaError.__module__}.{onnx.defs.SchemaError.__qualname__}"
    )
    assert diagnostic["checker_calls"] == 1
    assert attempt.blob == attempt.blob_after == before
    assert not attempt.returned


def test_missing_schema_observer_keeps_success_result_and_original_arguments_once() -> (
    None
):
    model, sentinel, calls = _identity(), object(), []

    def original(candidate: Any, marker: Any, *, option: Any) -> Any:
        calls.append((candidate, marker, option))
        capability.check_source_schema(candidate)
        return sentinel

    observation = SourceSchemaObservation(original)
    assert observation(model, sentinel, option=sentinel) is sentinel
    attempt = observation.assert_unchanged()
    assert calls == [(model, sentinel, sentinel)]
    assert attempt.result is sentinel and attempt.returned and attempt.cause is None
    assert attempt.args == (sentinel,) and attempt.kwargs == {"option": sentinel}


@pytest.mark.parametrize(
    "defect",
    [
        "wrong-operator",
        "wrong-node",
        "wrong-name",
        "wrong-domain",
        "wrong-overload",
        "wrong-import",
        "duplicate-import",
        "alias-import",
        "wrong-target",
        "target-bool",
        "duplicate-node",
        "available-schema",
        "unrelated-message",
        "unrelated-class",
        "changed-bytes",
        "mutated-during-check",
        "changed-cause",
        "changed-notes",
        "nested-graph",
        "local-function",
    ],
)
def test_missing_schema_counterpart_rejects_unrelated_or_changed_failures(
    defect: str,
) -> None:
    model, raw = _missing_schema_model(), _missing_schema_text()
    operator, opset = _MISSING_OPERATOR, 26
    if defect == "wrong-operator":
        operator = "AnotherOperator"
    elif defect == "wrong-node":
        model.graph.node[0].op_type = "Identity"
    elif defect == "wrong-name":
        model.graph.node[0].name = "another_name"
    elif defect == "wrong-domain":
        model.graph.node[0].domain = "example.custom"
    elif defect == "wrong-overload":
        model.graph.node[0].overload = "custom_overload"
    elif defect == "wrong-import":
        model.opset_import[0].version = 23
    elif defect == "duplicate-import":
        model.opset_import.append(onnx.helper.make_opsetid("", 26))
    elif defect == "alias-import":
        model.opset_import.append(onnx.helper.make_opsetid("ai.onnx", 26))
    elif defect == "wrong-target":
        opset = 27
    elif defect == "target-bool":
        opset = True
    elif defect == "duplicate-node":
        model.graph.node.append(deepcopy(model.graph.node[0]))
    elif defect == "available-schema":
        model = _missing_schema_model("Identity")
        operator, raw = "Identity", _missing_schema_text("Identity")
    elif defect == "unrelated-message":
        raw = "actual malformed graph/output/checker failure"
    elif defect == "nested-graph":
        model.graph.node[0].attribute.append(
            onnx.helper.make_attribute("body", _identity().graph)
        )
    elif defect == "local-function":
        model.functions.append(
            onnx.helper.make_function(
                "example", "Local", ["x"], ["y"], [], [onnx.helper.make_opsetid("", 26)]
            )
        )
    error = (
        ValueError(raw)
        if defect == "unrelated-class"
        else onnx.checker.ValidationError(raw)
    )
    calls = []

    def original(candidate: Any) -> None:
        calls.append(candidate)
        if defect == "mutated-during-check":
            candidate.graph.name = "changed during original checker"
        raise error

    observation = SourceSchemaObservation(original)
    with pytest.raises(type(error)) as caught:
        observation(model)
    assert caught.value is error and calls == [model]
    supplied = error
    if defect == "changed-bytes":
        model.graph.name = "changed after original checker"
    elif defect == "changed-cause":
        supplied = onnx.checker.ValidationError(str(error))
    elif defect == "changed-notes":
        error.add_note("changed after original checker")
    assert (
        observation.missing_schema_failure(supplied, operator=operator, opset=opset)
        is None
    )
    if defect in {"changed-bytes", "mutated-during-check"}:
        with pytest.raises(AssertionError):
            observation.assert_unchanged()


@pytest.mark.parametrize(
    "change", ["none", "metadata", "node", "import", "report", "duplicate-report"]
)
def test_source_schema_success_allows_only_exact_report_attachment(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    from jax2onnx import user_interface as ui
    from jax2onnx.converter.precision_policy import PrecisionPolicy
    from jax2onnx.precision import PRECISION_METADATA_KEY, get_precision_report

    def model(value: Any) -> Any:
        return -value

    observation = SourceSchemaObservation(ui.check_source_schema)
    with monkeypatch.context() as active:
        active.setattr(ui, "check_source_schema", observation)
        artifact = ui._to_onnx(model, [(2,)], precision_policy=PrecisionPolicy.SOURCE)
    report = get_precision_report(artifact)
    assert report is not None
    attempt = observation.assert_checker_boundary()
    assert attempt.returned and attempt.result is None and attempt.cause is None
    assert artifact is attempt.original
    if change == "metadata":
        entry = artifact.metadata_props.add()
        entry.key, entry.value = "unrelated", "changed"
    elif change == "node":
        artifact.graph.node[0].op_type = "Identity"
    elif change == "import":
        artifact.opset_import[0].version += 1
    elif change == "report":
        next(
            item
            for item in artifact.metadata_props
            if item.key == PRECISION_METADATA_KEY
        ).value += " "
    elif change == "duplicate-report":
        original = next(
            item
            for item in artifact.metadata_props
            if item.key == PRECISION_METADATA_KEY
        )
        artifact.metadata_props.add().CopyFrom(original)
    before = artifact.SerializeToString()
    if change == "none":
        observation.assert_report_attachment(artifact, report)
    else:
        with pytest.raises(AssertionError):
            observation.assert_report_attachment(artifact, report)
    assert artifact.SerializeToString() == before
