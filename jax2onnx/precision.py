# jax2onnx/precision.py

"""Export provenance and bounded, artifact-bound CPU sample evidence.

Export metadata describes representation and lowering checks. It is not runtime
or numerical evidence. CPU validation accepts only the supplied samples, their
independent references, and the recorded execution settings; it makes no claim
about other inputs, providers, architectures, or deployment environments.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import onnx
import onnx_ir as ir


PRECISION_METADATA_KEY = "jax2onnx.precision"
PRECISION_SCHEMA_VERSION = 1
SOURCE_PRECISION_PROFILE = "source-static-v1"
SOURCE_DIAGNOSTIC_PROFILE = "source-diagnostic-v1"


class CheckStatus(str, Enum):
    VALIDATED = "validated"
    FAILED = "failed"
    UNVERIFIED = "unverified"


class PrecisionReportError(ValueError):
    """Precision metadata exists but is malformed or uses an unknown schema."""


@dataclass(frozen=True)
class TensorProperties:
    name: str | None
    shape: tuple[int | str | None, ...]
    dtype: str | None
    weak_type: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))


@dataclass(frozen=True)
class PrecisionInputSummary:
    index: int
    supplied: TensorProperties
    normalized: TensorProperties
    traced: TensorProperties
    exported: TensorProperties


@dataclass(frozen=True)
class PrecisionWarning:
    category: str
    message: str


@dataclass(frozen=True)
class PrecisionCheck:
    category: str
    status: CheckStatus
    detail: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", CheckStatus(self.status))


def _unverified_export_checks() -> tuple[PrecisionCheck, ...]:
    return tuple(
        PrecisionCheck(category, CheckStatus.UNVERIFIED, "No evidence recorded.")
        for category in (
            "representation",
            "lowering",
            "runtime-execution",
            "numerical-acceptance",
        )
    )


@dataclass(frozen=True)
class PrecisionReport:
    exporter_version: str
    jax_version: str
    opset: int
    caller_x64_enabled: bool
    caller_default_matmul_precision: str | None
    inputs: tuple[PrecisionInputSummary, ...]
    traced_outputs: tuple[TensorProperties, ...]
    exported_outputs: tuple[TensorProperties, ...]
    warnings: tuple[PrecisionWarning, ...] = ()
    adaptations: tuple[str, ...] = ()
    checks: tuple[PrecisionCheck, ...] = field(
        default_factory=_unverified_export_checks
    )
    schema_version: int = PRECISION_SCHEMA_VERSION
    profile: str = SOURCE_PRECISION_PROFILE
    caller_numpy_dtype_promotion: str | None = None
    caller_numpy_rank_promotion: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != PRECISION_SCHEMA_VERSION:
            raise PrecisionReportError(
                f"Unsupported precision schema version: {self.schema_version!r}"
            )
        if self.profile not in {SOURCE_PRECISION_PROFILE, SOURCE_DIAGNOSTIC_PROFILE}:
            raise PrecisionReportError(
                f"Unsupported precision profile: {self.profile!r}"
            )
        if self.caller_numpy_dtype_promotion is not None and (
            not isinstance(self.caller_numpy_dtype_promotion, str)
            or self.caller_numpy_dtype_promotion not in ("standard", "strict")
        ):
            raise PrecisionReportError("Invalid caller NumPy dtype promotion setting")
        if self.caller_numpy_rank_promotion is not None and (
            not isinstance(self.caller_numpy_rank_promotion, str)
            or self.caller_numpy_rank_promotion not in ("allow", "warn", "raise")
        ):
            raise PrecisionReportError("Invalid caller NumPy rank promotion setting")
        for name in (
            "inputs",
            "traced_outputs",
            "exported_outputs",
            "warnings",
            "adaptations",
            "checks",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.profile == SOURCE_DIAGNOSTIC_PROFILE and any(
            check.category == "lowering" and check.status is CheckStatus.VALIDATED
            for check in self.checks
        ):
            raise PrecisionReportError(
                "Diagnostic precision reports cannot claim validated lowering"
            )


def precision_report_to_json(report: PrecisionReport) -> str:
    """Serialize the versioned export record without adding sample evidence."""
    return json.dumps(asdict(report), sort_keys=True, separators=(",", ":"))


def attach_precision_report(
    model: ir.Model | onnx.ModelProto, report: PrecisionReport
) -> None:
    """Attach provenance through the model's supported metadata interface."""
    payload = precision_report_to_json(report)
    if isinstance(model, ir.Model):
        model.metadata_props[PRECISION_METADATA_KEY] = payload
    elif isinstance(model, onnx.ModelProto):
        properties = {item.key: item.value for item in model.metadata_props}
        properties[PRECISION_METADATA_KEY] = payload
        onnx.helper.set_model_props(model, properties)
    else:
        raise TypeError("Expected an onnx_ir.Model or onnx.ModelProto")


ModelArtifact: TypeAlias = ir.Model | onnx.ModelProto | str | os.PathLike[str]


def _json_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PrecisionReportError("Expected a JSON object in precision metadata")
    return cast(dict[str, Any], value)


def _tensor_from_json(value: Any) -> TensorProperties:
    data = _json_object(value)
    shape = data["shape"]
    if not isinstance(shape, list) or any(
        dim is not None and (isinstance(dim, bool) or not isinstance(dim, (int, str)))
        for dim in shape
    ):
        raise PrecisionReportError("Invalid tensor shape in precision metadata")
    for key in ("name", "dtype"):
        if data[key] is not None and not isinstance(data[key], str):
            raise PrecisionReportError(f"Invalid tensor {key} in precision metadata")
    weak_type = data.get("weak_type")
    if weak_type is not None and not isinstance(weak_type, bool):
        raise PrecisionReportError("Invalid weak_type in precision metadata")
    return TensorProperties(data["name"], tuple(shape), data["dtype"], weak_type)


def _report_from_json(payload: str) -> PrecisionReport:
    try:
        data = _json_object(json.loads(payload))
        if type(data["schema_version"]) is not int:
            raise PrecisionReportError("Invalid precision schema version")
        for key in ("exporter_version", "jax_version", "profile"):
            if not isinstance(data[key], str):
                raise PrecisionReportError(f"Invalid {key} in precision metadata")
        if (
            type(data["opset"]) is not int
            or type(data["caller_x64_enabled"]) is not bool
        ):
            raise PrecisionReportError("Invalid precision opset or caller x64 setting")
        precision = data["caller_default_matmul_precision"]
        if precision is not None and not isinstance(precision, str):
            raise PrecisionReportError("Invalid caller matmul precision setting")
        for key in (
            "inputs",
            "traced_outputs",
            "exported_outputs",
            "warnings",
            "adaptations",
            "checks",
        ):
            if not isinstance(data[key], list):
                raise PrecisionReportError(
                    f"Invalid {key} collection in precision metadata"
                )
        inputs = []
        for raw in data["inputs"]:
            item = _json_object(raw)
            if type(item["index"]) is not int:
                raise PrecisionReportError("Invalid precision input index")
            inputs.append(
                PrecisionInputSummary(
                    item["index"],
                    *(
                        _tensor_from_json(item[key])
                        for key in ("supplied", "normalized", "traced", "exported")
                    ),
                )
            )
        warnings = tuple(
            PrecisionWarning(**_json_object(item)) for item in data["warnings"]
        )
        checks = tuple(PrecisionCheck(**_json_object(item)) for item in data["checks"])
        if any(not isinstance(value, str) for value in data["adaptations"]):
            raise PrecisionReportError("Invalid lowering adaptation")
        if any(
            not isinstance(w.category, str) or not isinstance(w.message, str)
            for w in warnings
        ):
            raise PrecisionReportError("Invalid precision warning")
        if any(
            not isinstance(c.category, str) or not isinstance(c.detail, str)
            for c in checks
        ):
            raise PrecisionReportError("Invalid precision check")
        return PrecisionReport(
            exporter_version=data["exporter_version"],
            jax_version=data["jax_version"],
            opset=data["opset"],
            caller_x64_enabled=data["caller_x64_enabled"],
            caller_default_matmul_precision=precision,
            inputs=tuple(inputs),
            traced_outputs=tuple(
                _tensor_from_json(item) for item in data["traced_outputs"]
            ),
            exported_outputs=tuple(
                _tensor_from_json(item) for item in data["exported_outputs"]
            ),
            warnings=warnings,
            adaptations=tuple(data["adaptations"]),
            checks=checks,
            schema_version=data["schema_version"],
            profile=data["profile"],
            caller_numpy_dtype_promotion=data.get("caller_numpy_dtype_promotion"),
            caller_numpy_rank_promotion=data.get("caller_numpy_rank_promotion"),
        )
    except PrecisionReportError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise PrecisionReportError(f"Malformed precision metadata: {exc}") from exc


def get_precision_report(model_or_path: ModelArtifact) -> PrecisionReport | None:
    """Read provenance from IR, protobuf, or a saved model; legacy models return None.

    Malformed or unsupported metadata raises :class:`PrecisionReportError`.
    Reading metadata does not load external tensor data or execute a model.
    """
    if isinstance(model_or_path, ir.Model):
        payload = model_or_path.metadata_props.get(PRECISION_METADATA_KEY)
    else:
        proto = (
            model_or_path
            if isinstance(model_or_path, onnx.ModelProto)
            else onnx.load_model(model_or_path, load_external_data=False)
        )
        matching = [
            item.value
            for item in proto.metadata_props
            if item.key == PRECISION_METADATA_KEY
        ]
        if len(matching) > 1:
            raise PrecisionReportError("Duplicate precision metadata entries")
        payload = matching[0] if matching else None
    return None if payload is None else _report_from_json(payload)


@dataclass(frozen=True)
class SampleCriteria:
    """Exact values, explicit tolerances, or identical dtype/shape/bit patterns.

    Exact values compare signed zero equally and, by default, matching NaNs.
    Bitwise comparison distinguishes signed zero and NaN payloads.
    Integer and Boolean outputs always require exact values.
    """

    mode: Literal["exact", "tolerant", "bitwise"] = "exact"
    rtol: float | None = None
    atol: float | None = None
    equal_nan: bool = True

    def __post_init__(self) -> None:
        if self.mode not in ("exact", "tolerant", "bitwise"):
            raise ValueError(f"Unknown sample comparison mode: {self.mode!r}")
        if self.mode == "tolerant":
            for name, value in (("rtol", self.rtol), ("atol", self.atol)):
                if value is None or not math.isfinite(value) or value < 0:
                    raise ValueError(
                        f"Tolerant comparison requires explicit finite {name} >= 0"
                    )
        elif self.rtol is not None or self.atol is not None:
            raise ValueError("rtol/atol apply only to tolerant comparison")


@dataclass(frozen=True)
class SourceSample:
    inputs: Sequence[npt.ArrayLike] | Mapping[str, npt.ArrayLike]
    expected_outputs: Sequence[npt.ArrayLike] | None = None
    criteria: SampleCriteria = field(default_factory=SampleCriteria)


@dataclass(frozen=True)
class ArrayEvidence:
    name: str
    dtype: str
    shape: tuple[int, ...]
    sha256: str


@dataclass(frozen=True)
class CpuSampleTarget:
    runtime_version: str
    optimization: str
    architecture: str
    providers: tuple[str, ...]
    provider_options: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = ()
    intra_op_threads: int = 1
    inter_op_threads: int = 1
    execution_mode: str = "ORT_SEQUENTIAL"
    session_config: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class SourceSampleEvidence:
    index: int
    inputs: tuple[ArrayEvidence, ...]
    references: tuple[ArrayEvidence, ...]
    outputs: tuple[ArrayEvidence, ...]
    criteria: SampleCriteria
    checks: tuple[PrecisionCheck, ...]
    sample_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        identity = {
            "inputs": [asdict(item) for item in self.inputs],
            "references": [asdict(item) for item in self.references],
            "criteria": asdict(self.criteria),
        }
        payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        object.__setattr__(self, "sample_sha256", hashlib.sha256(payload).hexdigest())

    @property
    def status(self) -> CheckStatus:
        return _aggregate(self.checks)


class SampleAcceptanceError(ValueError):
    """The supplied artifact has no accepted, still-matching sample evidence."""


@dataclass(frozen=True)
class SourceValidationReport:
    artifact_sha256: str | None
    target: CpuSampleTarget | None
    checks: tuple[PrecisionCheck, ...]
    samples: tuple[SourceSampleEvidence, ...]
    _artifact: ModelArtifact = field(repr=False, compare=False)
    scope: str = (
        "Only the supplied samples, references, criteria, artifact, and CPU settings."
    )

    @property
    def status(self) -> CheckStatus:
        checks = self.checks + tuple(
            check for sample in self.samples for check in sample.checks
        )
        if not self.samples and not any(
            check.status is CheckStatus.FAILED for check in checks
        ):
            return CheckStatus.UNVERIFIED
        return _aggregate(checks)

    def require_sample_acceptance(
        self, model_or_path: ModelArtifact | None = None
    ) -> None:
        """Require accepted samples and the identical current execution artifact.

        With no argument, recheck the original object/path, so later mutation of
        an in-memory model or saved file cannot reuse this evidence unnoticed.
        """
        if self.status is not CheckStatus.VALIDATED:
            details = "; ".join(
                check.detail
                for check in self.checks
                + tuple(c for sample in self.samples for c in sample.checks)
                if check.status is not CheckStatus.VALIDATED
            )
            raise SampleAcceptanceError(f"{self.status.value}: {details}")
        try:
            digest = _artifact_bytes(
                self._artifact if model_or_path is None else model_or_path
            )[1]
        except Exception as exc:
            raise SampleAcceptanceError(
                f"Could not verify current artifact identity: {exc}"
            ) from exc
        if digest != self.artifact_sha256:
            raise SampleAcceptanceError(
                "Artifact changed after sample validation; validate the current artifact again."
            )


def require_sample_acceptance(
    report: SourceValidationReport, model_or_path: ModelArtifact | None = None
) -> None:
    """Require sample acceptance; see :meth:`SourceValidationReport.require_sample_acceptance`."""
    report.require_sample_acceptance(model_or_path)


def _aggregate(checks: Sequence[PrecisionCheck]) -> CheckStatus:
    if any(check.status is CheckStatus.FAILED for check in checks):
        return CheckStatus.FAILED
    if not checks or any(check.status is CheckStatus.UNVERIFIED for check in checks):
        return CheckStatus.UNVERIFIED
    return CheckStatus.VALIDATED


def _all_tensors(message: Any) -> list[onnx.TensorProto]:
    """Walk protobuf fields, including tensors in functions and nested graphs."""
    found = []
    if isinstance(message, onnx.TensorProto):
        found.append(message)
    for descriptor, value in message.ListFields():
        if descriptor.message_type is None:
            continue
        if descriptor.is_repeated:
            for child in value:
                found.extend(_all_tensors(child))
        else:
            found.extend(_all_tensors(value))
    return found


def _safe_file(path: Path) -> Path:
    absolute = path.absolute()
    if any(part.is_symlink() for part in (absolute, *absolute.parents)):
        raise ValueError(
            f"Symlink paths are not supported for sample validation: {path}"
        )
    if not absolute.is_file():
        raise ValueError(f"Not a regular model/payload file: {path}")
    return absolute


def _artifact_bytes(model_or_path: ModelArtifact) -> tuple[bytes, str]:
    if isinstance(model_or_path, ir.Model):
        proto = cast(onnx.ModelProto, ir.to_proto(model_or_path))
        base = None
    elif isinstance(model_or_path, onnx.ModelProto):
        proto = onnx.ModelProto()
        proto.CopyFrom(model_or_path)
        base = None
    else:
        path = _safe_file(Path(model_or_path))
        proto = onnx.load_model(path, load_external_data=False)
        base = path.parent
    for tensor in _all_tensors(proto):
        if not onnx.external_data_helper.uses_external_data(tensor):
            continue
        if base is None:
            raise ValueError("External tensor data needs an explicit saved model path")
        external = {entry.key: entry.value for entry in tensor.external_data}
        location = Path(external.get("location", ""))
        if not str(location) or location.is_absolute() or ".." in location.parts:
            raise ValueError(
                "External tensor data must use a relative local payload path"
            )
        payload = _safe_file(base / location)
        if not payload.is_relative_to(base):
            raise ValueError(
                "External tensor data must stay within the model directory"
            )
        # Load this validated file directly. No external-data resolver/network
        # fallback is involved, and execution receives the inlined bytes below.
        offset = int(external.get("offset", "0"))
        length = int(external["length"]) if "length" in external else None
        if offset < 0 or (length is not None and length < 0):
            raise ValueError("External tensor offset/length must be nonnegative")
        with payload.open("rb") as stream:
            stream.seek(offset)
            raw = stream.read() if length is None else stream.read(length)
        if length is not None and len(raw) != length:
            raise ValueError("External tensor data is shorter than its declared length")
        tensor.raw_data = raw
        tensor.ClearField("external_data")
        tensor.data_location = onnx.TensorProto.DEFAULT
    blob = proto.SerializeToString(deterministic=True)
    return blob, hashlib.sha256(blob).hexdigest()


def _array(value: npt.ArrayLike) -> np.ndarray[Any, Any]:
    array = np.asarray(value)
    if array.dtype.hasobject or (
        array.dtype.kind in ("U", "S", "V") and array.dtype.name != "bfloat16"
    ):
        raise ValueError(
            f"Sample evidence needs numeric arrays, received {array.dtype}"
        )
    return cast(np.ndarray[Any, Any], array.copy(order="C"))


def _array_evidence(name: str, array: np.ndarray[Any, Any]) -> ArrayEvidence:
    header = json.dumps(
        {
            "dtype": array.dtype.name,
            "dtype_str": array.dtype.str,
            "shape": list(array.shape),
        },
        sort_keys=True,
    ).encode()
    digest = hashlib.sha256(header + b"\0" + array.tobytes(order="C")).hexdigest()
    return ArrayEvidence(name, array.dtype.name, tuple(array.shape), digest)


def _compare(
    actual: np.ndarray[Any, Any],
    reference: np.ndarray[Any, Any],
    criteria: SampleCriteria,
) -> str | None:
    if actual.dtype != reference.dtype:
        return f"dtype differs: actual {actual.dtype}, reference {reference.dtype}"
    if actual.shape != reference.shape:
        return f"shape differs: actual {actual.shape}, reference {reference.shape}"
    if criteria.mode == "bitwise":
        matches = actual.tobytes(order="C") == reference.tobytes(order="C")
    elif actual.dtype.kind in "biu":
        return (
            None
            if np.array_equal(actual, reference)
            else "values fail exact integer/Boolean comparison"
        )
    elif criteria.mode == "exact":
        matches = bool(np.array_equal(actual, reference, equal_nan=criteria.equal_nan))
    else:
        matches = bool(
            np.allclose(
                actual,
                reference,
                rtol=criteria.rtol,
                atol=criteria.atol,
                equal_nan=criteria.equal_nan,
            )
        )
    return None if matches else f"values fail {criteria.mode} comparison"


def _source_batch_boundaries(
    model: onnx.ModelProto,
) -> tuple[tuple[TensorProperties, ...], tuple[TensorProperties, ...]] | None:
    """Check the finite audited symbolic signature against its source report."""
    report = get_precision_report(model)
    if report is None or report.profile != SOURCE_PRECISION_PROFILE:
        return None
    properties = (
        tuple(
            item
            for entry in report.inputs
            for item in (entry.supplied, entry.normalized, entry.traced, entry.exported)
        )
        + report.traced_outputs
        + report.exported_outputs
    )
    artifact_symbolic = any(
        dim.HasField("dim_param")
        for value in (*model.graph.input, *model.graph.output)
        for dim in value.type.tensor_type.shape.dim
    )
    if not artifact_symbolic and not any(
        isinstance(dim, str) for item in properties for dim in item.shape
    ):
        return None
    symbol: str | None = None
    for item in properties:
        labels = [dim for dim in item.shape if isinstance(dim, str)]
        if labels:
            if (
                len(item.shape) not in (2, 3)
                or len(labels) != 1
                or item.shape[0] != labels[0]
                or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", labels[0]) is None
                or item.dtype
                not in {
                    None,
                    "bool",
                    "int8",
                    "int16",
                    "int32",
                    "int64",
                    "uint8",
                    "uint16",
                    "uint32",
                    "uint64",
                    "float16",
                    "bfloat16",
                    "float32",
                    "float64",
                }
                or any(
                    not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0
                    for dim in item.shape[1:]
                )
                or (symbol is not None and symbol != labels[0])
            ):
                raise PrecisionReportError("Unsupported SOURCE symbolic batch contract")
            symbol = labels[0]
    inputs = tuple(entry.exported for entry in report.inputs)
    outputs = report.exported_outputs
    for entries, declarations in (
        (model.graph.input, inputs),
        (model.graph.output, outputs),
    ):
        if len(entries) != len(declarations):
            raise PrecisionReportError(
                "SOURCE report boundary count differs from artifact"
            )
        for value, declaration in zip(entries, declarations, strict=True):
            tensor = value.type.tensor_type
            shape: list[int | str] = []
            if not tensor.HasField("shape"):
                raise PrecisionReportError("SOURCE artifact boundary has no shape")
            for dim in tensor.shape.dim:
                if dim.HasField("dim_value"):
                    shape.append(dim.dim_value)
                elif dim.HasField("dim_param") and dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    raise PrecisionReportError(
                        "SOURCE artifact boundary has an unknown dimension"
                    )
            dtype = np.dtype(
                onnx.helper.tensor_dtype_to_np_dtype(tensor.elem_type)
            ).name
            if (value.name, tuple(shape), dtype) != (
                declaration.name,
                declaration.shape,
                declaration.dtype,
            ):
                raise PrecisionReportError(
                    "SOURCE report boundary name, order, dtype or shape differs from artifact"
                )
    for index, entry in enumerate(report.inputs):
        if entry.index != index or any(
            item.shape != entry.exported.shape
            or (item.dtype is not None and item.dtype != entry.exported.dtype)
            for item in (entry.supplied, entry.normalized, entry.traced)
        ):
            raise PrecisionReportError(
                "SOURCE input report does not preserve its declared batch contract"
            )
    if len(report.traced_outputs) != len(outputs) or any(
        (traced.shape, traced.dtype) != (exported.shape, exported.dtype)
        for traced, exported in zip(report.traced_outputs, outputs, strict=True)
    ):
        raise PrecisionReportError(
            "SOURCE output report does not preserve its declared batch contract"
        )
    if not any(symbol in item.shape for item in inputs):
        raise PrecisionReportError("SOURCE symbolic output has no input batch binding")
    return inputs, outputs


def _check_bound_sample_tensor(
    array: np.ndarray[Any, Any],
    declaration: TensorProperties,
    bindings: dict[str, int],
    *,
    bind: bool,
) -> None:
    if array.dtype.name != declaration.dtype or array.ndim != len(declaration.shape):
        raise ValueError(
            f"{declaration.name}: requires dtype {declaration.dtype} and shape "
            f"{declaration.shape}; received {array.dtype.name} {array.shape}"
        )
    for axis, (actual, expected) in enumerate(
        zip(array.shape, declaration.shape, strict=True)
    ):
        if isinstance(expected, str):
            if actual <= 0:
                raise ValueError(
                    f"{declaration.name}: batch {expected} must be positive"
                )
            if expected not in bindings and bind:
                bindings[expected] = actual
            if bindings.get(expected) != actual:
                raise ValueError(
                    f"{declaration.name}: axis {axis} must match shared batch {expected}={bindings.get(expected)}; received {actual}"
                )
        elif isinstance(expected, bool) or expected != actual:
            raise ValueError(
                f"{declaration.name}: axis {axis} requires {expected}; received {actual}"
            )


def validate_source_samples(
    model_or_path: ModelArtifact,
    *,
    samples: Sequence[SourceSample],
    optimization: Literal["disabled", "all"] | None,
) -> SourceValidationReport:
    """Execute explicitly selected ORT CPU settings against independent references.

    This also works for legacy artifacts with no precision metadata. Metadata
    presence is never runtime evidence. ``optimization=None`` or empty/missing
    references execute nothing and remain unverified. Schema, load, run, and
    comparison failures remain distinct, and any known failure dominates.
    """
    if optimization not in (None, "disabled", "all"):
        raise ValueError("optimization must be explicitly 'disabled', 'all', or None")
    checks: list[PrecisionCheck] = []
    evidence: list[SourceSampleEvidence] = []
    digest = None
    target = None

    def result() -> SourceValidationReport:
        return SourceValidationReport(
            digest, target, tuple(checks), tuple(evidence), model_or_path
        )

    try:
        blob, digest = _artifact_bytes(model_or_path)
    except Exception as exc:
        checks.append(PrecisionCheck("artifact-load", CheckStatus.FAILED, str(exc)))
        return result()
    checks.append(
        PrecisionCheck(
            "artifact-load",
            CheckStatus.VALIDATED,
            "Complete inlined execution artifact loaded and hashed.",
        )
    )
    try:
        model = onnx.load_model_from_string(blob)
        onnx.checker.check_model(model, full_check=True)
    except Exception as exc:
        checks.append(PrecisionCheck("onnx-schema", CheckStatus.FAILED, str(exc)))
        return result()
    checks.append(
        PrecisionCheck(
            "onnx-schema", CheckStatus.VALIDATED, "ONNX schema and shape checks passed."
        )
    )
    try:
        batch_boundaries = _source_batch_boundaries(model)
    except Exception as exc:
        checks.extend(
            (
                PrecisionCheck("source-shape-contract", CheckStatus.FAILED, str(exc)),
                PrecisionCheck(
                    "numerical-acceptance",
                    CheckStatus.UNVERIFIED,
                    "SOURCE boundary metadata does not match the artifact.",
                ),
            )
        )
        return result()
    if optimization is None or not samples:
        reason = (
            "No CPU optimization profile selected."
            if optimization is None
            else "No independent samples supplied."
        )
        checks.extend(
            PrecisionCheck(category, CheckStatus.UNVERIFIED, reason)
            for category in ("runtime-execution", "numerical-acceptance")
        )
        return result()

    session = None
    for index, sample in enumerate(samples):
        sample_checks: list[PrecisionCheck] = []
        input_records: tuple[ArrayEvidence, ...] = ()
        references: tuple[ArrayEvidence, ...] = ()
        output_records: tuple[ArrayEvidence, ...] = ()
        try:
            named = isinstance(sample.inputs, Mapping)
            values = (
                {str(name): _array(value) for name, value in sample.inputs.items()}
                if isinstance(sample.inputs, Mapping)
                else {str(i): _array(value) for i, value in enumerate(sample.inputs)}
            )
            input_records = tuple(
                _array_evidence(name, value) for name, value in values.items()
            )
            expected = (
                tuple(_array(value) for value in sample.expected_outputs)
                if sample.expected_outputs is not None
                else ()
            )
            references = tuple(
                _array_evidence(str(i), value) for i, value in enumerate(expected)
            )
        except Exception as exc:
            sample_checks.append(
                PrecisionCheck("sample-input", CheckStatus.FAILED, str(exc))
            )
            evidence.append(
                SourceSampleEvidence(
                    index,
                    input_records,
                    references,
                    (),
                    sample.criteria,
                    tuple(sample_checks),
                )
            )
            continue
        bindings: dict[str, int] = {}
        if batch_boundaries is not None:
            input_contract, output_contract = batch_boundaries
            try:
                names = tuple(cast(str, item.name) for item in input_contract)
                if named:
                    if set(values) != set(names):
                        raise ValueError(
                            f"Sample input names {tuple(values)} do not match artifact inputs {names}"
                        )
                    feeds = {name: values[name] for name in names}
                else:
                    feeds = dict(zip(names, values.values(), strict=True))
                for declaration, array in zip(
                    input_contract, feeds.values(), strict=True
                ):
                    _check_bound_sample_tensor(array, declaration, bindings, bind=True)
                if expected and len(expected) != len(output_contract):
                    raise ValueError(
                        "Reference output count differs from SOURCE declaration"
                    )
                for declaration, array in zip(
                    output_contract if expected else (), expected, strict=True
                ):
                    _check_bound_sample_tensor(array, declaration, bindings, bind=False)
            except Exception as exc:
                sample_checks.extend(
                    (
                        PrecisionCheck(
                            "sample-input",
                            CheckStatus.FAILED,
                            f"SOURCE shape contract: {exc}",
                        ),
                        PrecisionCheck(
                            "numerical-acceptance",
                            CheckStatus.UNVERIFIED,
                            "Sample or reference violates the declared SOURCE contract; execution was not attempted.",
                        ),
                    )
                )
                evidence.append(
                    SourceSampleEvidence(
                        index,
                        input_records,
                        references,
                        (),
                        sample.criteria,
                        tuple(sample_checks),
                    )
                )
                continue
        if not expected:
            sample_checks.append(
                PrecisionCheck(
                    "numerical-acceptance",
                    CheckStatus.UNVERIFIED,
                    "No independent reference outputs supplied; sample was not executed.",
                )
            )
            evidence.append(
                SourceSampleEvidence(
                    index,
                    input_records,
                    references,
                    (),
                    sample.criteria,
                    tuple(sample_checks),
                )
            )
            continue
        if session is None:
            try:
                # Optional runtime loading follows the SOURCE input/reference
                # gate, so invalid-only samples construct no execution session.
                import onnxruntime as ort  # type: ignore[import-untyped]

                level = (
                    ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                    if optimization == "disabled"
                    else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                options = ort.SessionOptions()
                options.intra_op_num_threads = 1
                options.inter_op_num_threads = 1
                options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                options.graph_optimization_level = level
                target = CpuSampleTarget(
                    ort.__version__, level.name, platform.machine(), ()
                )
                session = ort.InferenceSession(
                    blob, options, providers=["CPUExecutionProvider"]
                )
                session.disable_fallback()
                providers = tuple(session.get_providers())
                provider_options = tuple(
                    (
                        provider,
                        tuple(
                            sorted(
                                (str(key), str(value)) for key, value in values.items()
                            )
                        ),
                    )
                    for provider, values in sorted(
                        session.get_provider_options().items()
                    )
                )
                target = CpuSampleTarget(
                    ort.__version__,
                    level.name,
                    platform.machine(),
                    providers,
                    provider_options,
                )
                if providers != ("CPUExecutionProvider",):
                    raise RuntimeError(
                        f"Expected only CPUExecutionProvider, received {providers}"
                    )
            except Exception as exc:
                sample_checks.extend(
                    (
                        PrecisionCheck("runtime-load", CheckStatus.FAILED, str(exc)),
                        PrecisionCheck(
                            "numerical-acceptance",
                            CheckStatus.UNVERIFIED,
                            "The runtime could not load the artifact.",
                        ),
                    )
                )
                evidence.append(
                    SourceSampleEvidence(
                        index,
                        input_records,
                        references,
                        (),
                        sample.criteria,
                        tuple(sample_checks),
                    )
                )
                checks.append(
                    PrecisionCheck("runtime-load", CheckStatus.FAILED, str(exc))
                )
                return result()
        try:
            names = tuple(spec.name for spec in session.get_inputs())
            if batch_boundaries is not None:
                if names != tuple(item.name for item in batch_boundaries[0]):
                    raise ValueError(
                        "Runtime input order differs from checked SOURCE artifact"
                    )
                # Reuse the already bound protobuf-ordered feeds.
            elif named:
                if set(values) != set(names):
                    raise ValueError(
                        f"Sample input names {tuple(values)} do not match artifact inputs {names}"
                    )
                feeds = values
            else:
                feeds = dict(zip(names, values.values(), strict=True))
            outputs = tuple(np.asarray(value) for value in session.run(None, feeds))
            output_records = tuple(
                _array_evidence(str(i), value) for i, value in enumerate(outputs)
            )
            sample_checks.append(
                PrecisionCheck(
                    "runtime-execution",
                    CheckStatus.VALIDATED,
                    "Executed using the recorded CPU session settings.",
                )
            )
        except Exception as exc:
            sample_checks.append(
                PrecisionCheck("runtime-execution", CheckStatus.FAILED, str(exc))
            )
            sample_checks.append(
                PrecisionCheck(
                    "numerical-acceptance",
                    CheckStatus.UNVERIFIED,
                    "Execution did not produce comparable outputs.",
                )
            )
            evidence.append(
                SourceSampleEvidence(
                    index,
                    input_records,
                    references,
                    output_records,
                    sample.criteria,
                    tuple(sample_checks),
                )
            )
            continue
        if batch_boundaries is not None:
            try:
                if len(outputs) != len(batch_boundaries[1]):
                    raise ValueError(
                        "Actual output count differs from SOURCE declaration"
                    )
                for declaration, array in zip(
                    batch_boundaries[1], outputs, strict=True
                ):
                    _check_bound_sample_tensor(array, declaration, bindings, bind=False)
            except Exception as exc:
                sample_checks.extend(
                    (
                        PrecisionCheck(
                            "sample-output",
                            CheckStatus.FAILED,
                            f"SOURCE shape contract: {exc}",
                        ),
                        PrecisionCheck(
                            "numerical-acceptance",
                            CheckStatus.UNVERIFIED,
                            "Actual outputs violate the declared SOURCE contract.",
                        ),
                    )
                )
                evidence.append(
                    SourceSampleEvidence(
                        index,
                        input_records,
                        references,
                        output_records,
                        sample.criteria,
                        tuple(sample_checks),
                    )
                )
                continue
        failures = []
        if len(outputs) != len(expected):
            failures.append(
                f"Output count differs: actual {len(outputs)}, reference {len(expected)}"
            )
        else:
            for output_index, (actual, reference) in enumerate(
                zip(outputs, expected, strict=True)
            ):
                failure = _compare(actual, reference, sample.criteria)
                if failure is not None:
                    failures.append(f"Output {output_index}: {failure}")
        sample_checks.append(
            PrecisionCheck(
                "numerical-acceptance",
                CheckStatus.FAILED if failures else CheckStatus.VALIDATED,
                (
                    "; ".join(failures)
                    if failures
                    else "Supplied output dtypes, shapes, and values meet the declared criteria."
                ),
            )
        )
        evidence.append(
            SourceSampleEvidence(
                index,
                input_records,
                references,
                output_records,
                sample.criteria,
                tuple(sample_checks),
            )
        )
    return result()
