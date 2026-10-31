# tests/extra_tests/converter/source_capability_counterparts.py

"""Exact failed-capability counterparts for unchanged original test attempts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import hashlib
import re
from typing import Any

import onnx
import pytest

from jax2onnx._source_capability import diagnostic_from_detail
from jax2onnx.precision import (
    CheckStatus,
    PRECISION_METADATA_KEY,
    get_precision_report,
    precision_report_to_json,
    SampleAcceptanceError,
    SampleCriteria,
    SourceSample,
    _artifact_bytes,
    validate_source_samples,
)
from tests.extra_tests.converter.source_precision_support import SourceCaseResult


def assert_schema_capability_failure(
    result: SourceCaseResult,
    record_property: Callable[[str, object], None],
) -> None:
    """Require the diagnostic retained from this exact original export attempt."""
    assert result.model is None and result.validation is None
    assert result.stage == "source-export"
    assert result.audit.status is CheckStatus.FAILED
    assert result.schema is CheckStatus.FAILED
    assert result.numerical_status is CheckStatus.UNVERIFIED
    assert result.raw_callback_status == "not-executed"
    assert result.prepared is not None
    diagnostic = diagnostic_from_detail(result.audit.detail)
    assert diagnostic is not None, result.audit.detail
    assert diagnostic.boundary == "onnx-schema"
    assert diagnostic.cause_type in {
        f"{kind.__module__}.{kind.__qualname__}"
        for kind in (onnx.checker.ValidationError, onnx.shape_inference.InferenceError)
    }
    context = diagnostic.context
    requested = int(result.prepared.metadata.get("opset_version", 23))
    assert context["requested_opset"] == requested
    assert context["onnx_version"] == onnx.__version__
    schema = onnx.defs.get_schema(context["operator"], requested, context["domain"])
    assert schema.since_version == context["schema_since_version"]
    port = schema.inputs[context["input_index"]]
    assert (port.name, port.type_str) == (context["port"], context["type_parameter"])
    constraints = [
        item for item in schema.type_constraints if item.type_param_str == port.type_str
    ]
    assert len(constraints) == 1
    assert list(constraints[0].allowed_type_strs) == context["allowed_types"]
    assert context["actual_type"] not in context["allowed_types"]
    record_property("source_failed_schema_capability", asdict(diagnostic))


def accept_samples_or_assert_runtime_capability_failure(
    proto: Any,
    feeds: Any,
    references: Any,
    criteria: SampleCriteria,
    record_property: Callable[[str, object], None],
) -> None:
    """Keep both CPU modes and native roles; accept only the observed failure."""
    _, digest = _artifact_bytes(proto)
    for optimization in ("disabled", "all"):
        for reference_index in (0, 1):
            result = validate_source_samples(
                proto,
                samples=[
                    SourceSample(values, expected[reference_index], criteria)
                    for values, expected in zip(feeds, references, strict=True)
                ],
                optimization=optimization,
            )
            if result.status is CheckStatus.VALIDATED:
                result.require_sample_acceptance()
                assert result.status is CheckStatus.VALIDATED
                continue
            assert result.status is CheckStatus.FAILED
            assert result.artifact_sha256 == digest
            assert result.target is not None and result.target.providers == ()
            assert all(
                item.status is CheckStatus.VALIDATED
                for item in result.checks
                if item.category != "runtime-load"
            )
            failures = [
                item for item in result.checks if item.category == "runtime-load"
            ]
            assert len(failures) == 1 and failures[0].status is CheckStatus.FAILED
            diagnostic = diagnostic_from_detail(
                failures[0].detail,
                artifact_sha256=digest,
                providers_requested=("CPUExecutionProvider",),
            )
            assert diagnostic is not None, failures[0].detail
            assert diagnostic.boundary == "runtime-load"
            # The runtime was already loaded by this exact validation attempt.
            import onnxruntime as ort
            from onnxruntime.capi.onnxruntime_pybind11_state import Fail

            assert diagnostic.cause_type == f"{Fail.__module__}.{Fail.__qualname__}"
            assert diagnostic.context["runtime_version"] == ort.__version__
            assert result.target.runtime_version == ort.__version__
            expected_level = (
                ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                if optimization == "disabled"
                else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            assert result.target.optimization == expected_level.name
            assert diagnostic.context["runtime_build"] == ort.get_build_info()
            imports = tuple(
                sorted(
                    ("" if item.domain == "ai.onnx" else item.domain, item.version)
                    for item in proto.opset_import
                )
            )
            assert diagnostic.opset_imports == imports
            assert all(
                item.status is CheckStatus.VALIDATED
                for item in result.checks
                if item.category in {"artifact-load", "onnx-schema"}
            )
            assert len(result.samples) == 1
            sample = result.samples[0]
            assert sample.criteria == criteria and not sample.outputs
            assert sample.inputs and sample.references
            assert [
                (item.category, item.status)
                for item in sample.checks
                if item.category in {"runtime-load", "numerical-acceptance"}
            ] == [
                ("runtime-load", CheckStatus.FAILED),
                ("numerical-acceptance", CheckStatus.UNVERIFIED),
            ]
            assert (
                next(
                    item.detail
                    for item in sample.checks
                    if item.category == "runtime-load"
                )
                == failures[0].detail
            )
            with pytest.raises(SampleAcceptanceError):
                result.require_sample_acceptance()
            record_property(
                f"source_failed_runtime_capability_{optimization}_{reference_index}",
                asdict(diagnostic),
            )


@dataclass
class _CheckedSourceCandidate:
    original: Any
    blob: bytes
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    blob_after: bytes | None = None
    cause: Exception | None = None
    cause_text: str | None = None
    cause_notes: tuple[Any, ...] = ()
    result: Any = None
    returned: bool = False


class SourceSchemaObservation:
    """Observe one original checker without replacing its result or exception."""

    def __init__(self, checker: Callable[..., Any]) -> None:
        self.checker = checker
        self.attempts: list[_CheckedSourceCandidate] = []

    @staticmethod
    def _blob(model: Any) -> bytes:
        return model if isinstance(model, bytes) else model.SerializeToString()

    def __call__(self, model: Any, *args: Any, **kwargs: Any) -> Any:
        attempt = _CheckedSourceCandidate(model, self._blob(model), args, dict(kwargs))
        self.attempts.append(attempt)
        try:
            result = self.checker(model, *args, **kwargs)
        except Exception as cause:
            attempt.cause = cause
            attempt.cause_text = str(cause)
            attempt.cause_notes = tuple(getattr(cause, "__notes__", ()))
            raise
        else:
            attempt.result, attempt.returned = result, True
            return result
        finally:
            try:
                attempt.blob_after = self._blob(model)
            except Exception:
                # Observation must not replace the original checker outcome.
                attempt.blob_after = None

    def assert_checker_boundary(self) -> _CheckedSourceCandidate:
        assert len(self.attempts) == 1, "one actual SOURCE checker attempt required"
        (attempt,) = self.attempts
        assert attempt.blob == attempt.blob_after
        return attempt

    def assert_unchanged(self) -> _CheckedSourceCandidate:
        attempt = self.assert_checker_boundary()
        assert attempt.blob == self._blob(attempt.original)
        return attempt

    def assert_report_attachment(self, artifact: Any, report: Any) -> None:
        """Require the unchanged checked artifact and its final canonical report."""
        attempt = self.assert_unchanged()
        assert attempt.returned and attempt.result is None and attempt.cause is None
        assert artifact is attempt.original and report is not None
        assert self._blob(artifact) == attempt.blob
        frozen = onnx.load_model_from_string(attempt.blob)
        entries = [
            item
            for item in artifact.metadata_props
            if item.key == PRECISION_METADATA_KEY
        ]
        assert len(entries) == 1
        (entry,) = entries
        assert get_precision_report(frozen) == report
        assert get_precision_report(artifact) == report
        assert entry.value == precision_report_to_json(report)

    def missing_schema_failure(
        self, cause: Exception, *, operator: str, opset: int
    ) -> dict[str, Any] | None:
        """Recognize only an exact observed root-node missing-schema rejection."""
        if len(self.attempts) != 1 or type(cause) is not onnx.checker.ValidationError:
            return None
        (attempt,) = self.attempts
        if (
            attempt.cause is not cause
            or attempt.returned
            or str(cause) != attempt.cause_text
            or tuple(getattr(cause, "__notes__", ())) != attempt.cause_notes
        ):
            return None
        if type(opset) is not int or opset < 1 or not operator:
            return None
        try:
            if attempt.blob != attempt.blob_after or attempt.blob != self._blob(
                attempt.original
            ):
                return None
            match = re.fullmatch(
                r"No Op registered for (?P<operator>\w+) with domain_version of (?P<opset>\d+)\n\n==> Context: Bad node spec for node\. Name: (?P<node>\S+) OpType: (?P<type>\w+)",
                str(cause),
            )
            if (
                match is None
                or match.group("operator") != operator
                or match.group("type") != operator
                or int(match.group("opset")) != opset
            ):
                return None
            candidate = onnx.load_model_from_string(attempt.blob)
            if candidate.functions or any(
                attribute.type
                in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                for node in candidate.graph.node
                for attribute in node.attribute
            ):
                return None
            imports = tuple(
                (item.domain, item.version) for item in candidate.opset_import
            )
            if (
                len({domain for domain, _ in imports}) != len(imports)
                or ("", opset) not in imports
                or any(domain == "ai.onnx" for domain, _ in imports)
            ):
                return None
            nodes = [
                node
                for node in candidate.graph.node
                if node.name == match.group("node")
            ]
            if len(nodes) != 1:
                return None
            (node,) = nodes
            if node.op_type != operator or node.domain or node.overload:
                return None
            # Registry lookup follows the actual failed checker; it is not a
            # predictive version guard and never substitutes an export attempt.
            try:
                onnx.defs.get_schema(operator, opset, "")
            except onnx.defs.SchemaError as registry_cause:
                if type(registry_cause) is not onnx.defs.SchemaError:
                    return None
                registry_type = f"{type(registry_cause).__module__}.{type(registry_cause).__qualname__}"
                registry_text = str(registry_cause)
            else:
                return None
            return dict(
                protocol="source-missing-installed-schema-v1",
                schema="FAILED",
                numerical_acceptance="UNVERIFIED",
                output_acceptance="UNVERIFIED",
                successful_export=False,
                artifact_sha256=hashlib.sha256(attempt.blob).hexdigest(),
                opset_imports=imports,
                requested_opset=opset,
                domain="",
                operator=operator,
                node=node.name,
                overload=node.overload,
                checker_calls=len(self.attempts),
                cause_type=f"{type(cause).__module__}.{type(cause).__qualname__}",
                cause_text=str(cause),
                cause_notes=attempt.cause_notes,
                onnx_version=onnx.__version__,
                registry_cause_type=registry_type,
                registry_cause_text=registry_text,
            )
        except Exception:
            # An unrecognized diagnostic must preserve the original failure.
            return None
