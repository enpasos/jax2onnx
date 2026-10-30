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
import ctypes
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import sys
from tempfile import TemporaryDirectory
from typing import Any, Literal, NoReturn, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import onnx
import onnx_ir as ir
from google.protobuf.unknown_fields import UnknownFieldSet  # type: ignore[import-untyped]


from jax2onnx._source_capability import (
    check_source_schema,
    exception_detail,
    runtime_load_detail,
)
from jax2onnx.converter.source_deployment import (
    DEPLOYMENT_METADATA_KEY,
    DeploymentTarget,
    DeploymentRequirements,
    RuntimeTargetObservation,
    deployment_requirements_from_json,
    require_runtime_target,
    _deployment_graph_scope,
    _deployment_function_scope,
    _deployment_child_scope,
)


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
    deployment_requirements: DeploymentRequirements | None = None

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
        if (
            self.deployment_requirements is not None
            and type(self.deployment_requirements) is not DeploymentRequirements
        ):
            raise PrecisionReportError("Invalid deployment requirements record")
        if self.deployment_requirements is not None and any(
            site.matrix_output is None
            or site.matrix_tag is None
            or site.matrix_scope is None
            for site in self.deployment_requirements.matrix_precision
        ):
            raise PrecisionReportError("Unbound matrix deployment requirement")
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
    data = asdict(report)
    if report.deployment_requirements is None:
        data.pop("deployment_requirements")
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


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
            deployment_requirements=deployment_requirements_from_json(
                None
                if data.get("deployment_requirements") is None
                else json.dumps(data["deployment_requirements"])
            ),
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
    deployment_observation: RuntimeTargetObservation | None = None


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


@dataclass(frozen=True)
class _SourceArtifactFingerprint:
    projection_sha256: str
    tensors: tuple[tuple[str, int, str], ...]


@dataclass(frozen=True)
class _SourceFileBinding:
    requested: str
    resolved: str
    identity: tuple[int, int, int, int, int]
    offset: int
    length: int
    sha256: str


@dataclass(frozen=True)
class _SourceArtifactSnapshot:
    fingerprint: _SourceArtifactFingerprint
    files: tuple[_SourceFileBinding, ...]


_TENSOR_STORAGE_FIELDS = frozenset(
    {
        "raw_data",
        "float_data",
        "int32_data",
        "string_data",
        "int64_data",
        "double_data",
        "uint64_data",
    }
)
_SOURCE_HASH_CHUNK = 1024 * 1024


def _source_file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("SOURCE serialization integrity requires regular local files")
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _source_file_binding(
    requested: Path,
    *,
    root: Path,
    offset: int = 0,
    length: int | None = None,
) -> _SourceFileBinding:
    """Hash one referenced range; allow stable explicit parent symlinks."""
    requested = requested.absolute()
    resolved = requested.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("SOURCE external tensor escapes the model directory")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        identity = _source_file_identity(os.fstat(stream.fileno()))
        count = identity[2] - offset if length is None else length
        if offset < 0 or count < 0 or offset + count > identity[2]:
            raise ValueError("SOURCE external tensor range is outside its payload")
        stream.seek(offset)
        remaining = count
        while remaining:
            chunk = stream.read(min(remaining, _SOURCE_HASH_CHUNK))
            if not chunk:
                raise ValueError("SOURCE external tensor payload is truncated")
            digest.update(chunk)
            remaining -= len(chunk)
        if _source_file_identity(os.fstat(stream.fileno())) != identity:
            raise ValueError("SOURCE artifact changed while its payload was read")
    binding = _SourceFileBinding(
        str(requested), str(resolved), identity, offset, count, digest.hexdigest()
    )
    _source_recheck_file(binding)
    return binding


def _source_recheck_file(binding: _SourceFileBinding) -> None:
    requested = Path(binding.requested)
    if (
        str(requested.resolve(strict=True)) != binding.resolved
        or _source_file_identity(requested.stat()) != binding.identity
    ):
        raise ValueError("SOURCE artifact path or file changed during verification")


def _source_artifact_fingerprint(
    proto: onnx.ModelProto | onnx.TensorProto,
    *,
    base: Path | None = None,
    root: Path | None = None,
    files: list[_SourceFileBinding] | None = None,
) -> _SourceArtifactFingerprint:
    """Compact protobuf projection plus ordered, exact tensor-storage digests.

    This compares serialization with the final audited IR via the official IR
    codec. It is not an independent reconstruction of SOURCE semantics. Typed
    protobuf storage stays typed; only external raw storage is normalized to
    inline raw storage. No numeric conversion can canonicalize tensor bits.
    """
    payloads: list[tuple[str, int, str]] = []

    def project(message: Any, target: Any) -> None:
        # Unknown fields cannot originate in the reviewed codec and must not be
        # silently dropped by this descriptor-based projection.
        if UnknownFieldSet(message):
            raise ValueError("SOURCE artifact contains unknown protobuf fields")
        tensor = isinstance(message, onnx.TensorProto)
        omitted: set[str] = set()
        if tensor:
            omitted.update(_TENSOR_STORAGE_FIELDS)
            omitted.update(("external_data", "data_location"))
            typed = any(
                len(getattr(message, name))
                for name in _TENSOR_STORAGE_FIELDS - {"raw_data"}
            )
            external = onnx.external_data_helper.uses_external_data(message)
            if external:
                if base is None or root is None or files is None:
                    raise ValueError(
                        "SOURCE external tensor requires a saved model path"
                    )
                if typed or message.HasField("raw_data"):
                    raise ValueError(
                        "SOURCE tensor has ambiguous external/inline storage"
                    )
                entries = list(message.external_data)
                if any(UnknownFieldSet(entry) for entry in entries):
                    raise ValueError(
                        "SOURCE external-data entry contains unknown protobuf fields"
                    )
                values = {entry.key: entry.value for entry in entries}
                if len(values) != len(entries):
                    raise ValueError("SOURCE tensor has duplicate external-data keys")
                if set(values) - {"location", "offset", "length"}:
                    raise ValueError(
                        "SOURCE tensor has unrecognized external-data keys"
                    )
                location = Path(values.get("location", ""))
                if (
                    not values.get("location")
                    or location.is_absolute()
                    or ".." in location.parts
                ):
                    raise ValueError(
                        "SOURCE external tensor needs a relative local path"
                    )
                payload_path = base / location
                binding = _source_file_binding(
                    payload_path,
                    root=root,
                    offset=int(values.get("offset", "0")),
                    length=int(values["length"]) if "length" in values else None,
                )
                files.append(binding)
                payloads.append(("raw", binding.length, binding.sha256))
            elif (
                message.external_data
                or message.data_location != onnx.TensorProto.DEFAULT
            ):
                raise ValueError(
                    "SOURCE tensor has inconsistent external storage metadata"
                )
            elif message.HasField("raw_data"):
                if typed:
                    raise ValueError("SOURCE tensor has ambiguous raw/typed storage")
                raw = message.raw_data
                payloads.append(("raw", len(raw), hashlib.sha256(raw).hexdigest()))
                del raw
            else:
                # Copy at most this current tensor. CopyFrom preserves float
                # storage words, including NaN payloads and negative zero.
                storage = onnx.TensorProto()
                storage.CopyFrom(message)
                for descriptor in message.DESCRIPTOR.fields:
                    if descriptor.name not in _TENSOR_STORAGE_FIELDS:
                        storage.ClearField(descriptor.name)
                raw = storage.SerializeToString(deterministic=True)
                payloads.append(("typed", len(raw), hashlib.sha256(raw).hexdigest()))
                del raw, storage
        # Tensor ListFields() would materialize omitted raw_data a second time.
        # Inspect descriptors first so skipped payload fields are never fetched.
        for descriptor in message.DESCRIPTOR.fields:
            if descriptor.name in omitted:
                continue
            if not descriptor.is_repeated and not message.HasField(descriptor.name):
                continue
            value = getattr(message, descriptor.name)
            if descriptor.is_repeated and not value:
                continue
            if descriptor.message_type is not None:
                destination = getattr(target, descriptor.name)
                if descriptor.is_repeated:
                    for child in value:
                        project(child, destination.add())
                else:
                    project(value, destination)
                    destination.SetInParent()
            elif descriptor.is_repeated:
                getattr(target, descriptor.name).extend(value)
            else:
                setattr(target, descriptor.name, value)

    projection = type(proto)()
    project(proto, projection)
    return _SourceArtifactFingerprint(
        hashlib.sha256(projection.SerializeToString(deterministic=True)).hexdigest(),
        tuple(payloads),
    )


def _source_artifact_snapshot(
    path: str | os.PathLike[str],
) -> tuple[onnx.ModelProto, _SourceArtifactSnapshot]:
    """Parse and hash the same local header, then stream referenced ranges."""
    requested = Path(path).absolute()
    root = requested.parent.resolve(strict=True)
    resolved = requested.resolve(strict=True)
    with resolved.open("rb") as stream:
        identity = _source_file_identity(os.fstat(stream.fileno()))
        header = stream.read()
        if _source_file_identity(os.fstat(stream.fileno())) != identity:
            raise ValueError("SOURCE model changed while its header was read")
    binding = _SourceFileBinding(
        str(requested),
        str(resolved),
        identity,
        0,
        len(header),
        hashlib.sha256(header).hexdigest(),
    )
    proto = onnx.load_model_from_string(header)
    del header
    files = [binding]
    fingerprint = _source_artifact_fingerprint(
        proto, base=requested.parent, root=root, files=files
    )
    for item in files:
        _source_recheck_file(item)
    return proto, _SourceArtifactSnapshot(fingerprint, tuple(files))


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


def _bfloat16_cpu_array(value: Any) -> np.ndarray[Any, Any]:
    """Copy actual CPU BF16 storage while its runtime owner remains alive."""
    if (
        not value.is_tensor()
        or value.device_name() != "cpu"
        or value.element_type() != onnx.TensorProto.BFLOAT16
    ):
        raise ValueError("Expected a CPU BFLOAT16 tensor for sample readback")
    shape = tuple(value.shape())
    size = value.tensor_size_in_bytes()
    if (
        any(type(dim) is not int or dim < 0 for dim in shape)
        or type(size) is not int
        or not 0 <= size <= sys.maxsize
        or size != math.prod(shape) * 2
    ):
        raise ValueError("Invalid BFLOAT16 tensor shape or byte count")
    if size:
        address = value.data_ptr()
        if type(address) is not int or address <= 0:
            raise ValueError("Nonempty BFLOAT16 tensor has no valid data pointer")
        # The local OrtValue owner stays alive until the independent bytes copy
        # completes. Shape and byte count come only from that actual CPU tensor.
        raw = ctypes.string_at(address, size)
    else:
        raw = b""  # Empty tensors need neither a pointer query nor a memory read.
    dtype = onnx.helper.tensor_dtype_to_np_dtype(onnx.TensorProto.BFLOAT16)
    copied: np.ndarray[Any, Any] = (
        np.frombuffer(raw, dtype=dtype).reshape(shape).copy(order="C")
    )
    return copied


def _run_cpu_samples(
    session: Any,
    feeds: Mapping[str, np.ndarray[Any, Any]],
    model: onnx.ModelProto,
) -> tuple[np.ndarray[Any, Any], ...]:
    """Acquire physical outputs once, without NumPy's unsupported BF16 marshal."""
    bf16 = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(onnx.TensorProto.BFLOAT16))
    if not any(array.dtype == bf16 for array in feeds.values()) and not any(
        port.type.tensor_type.elem_type == onnx.TensorProto.BFLOAT16
        for port in model.graph.output
    ):
        return tuple(np.asarray(value) for value in session.run(None, feeds))

    import onnxruntime as ort  # type: ignore[import-untyped]

    # copy(order="C") preserves scalar rank, unlike ascontiguousarray. Explicit
    # owners retain both physical input storage and OrtValues through execution.
    owners = {name: array.copy(order="C") for name, array in feeds.items()}
    runtime_feeds = {
        name: (
            ort.OrtValue.ortvalue_from_numpy_with_onnx_type(
                array.view(np.uint16), onnx.TensorProto.BFLOAT16
            )
            if array.dtype == bf16
            else ort.OrtValue.ortvalue_from_numpy(array)
        )
        for name, array in owners.items()
    }
    actual = session.run_with_ort_values(None, runtime_feeds)
    outputs = []
    for value, port in zip(actual, model.graph.output, strict=True):
        element_type = port.type.tensor_type.elem_type
        if value.element_type() != element_type:
            raise ValueError("Runtime output dtype differs from the physical artifact")
        outputs.append(
            _bfloat16_cpu_array(value)
            if element_type == onnx.TensorProto.BFLOAT16
            else np.asarray(value.numpy())
        )
    return tuple(outputs)


def _source_batch_boundaries(
    model: onnx.ModelProto,
) -> tuple[tuple[TensorProperties, ...], tuple[TensorProperties, ...]] | None:
    """Check the finite audited symbolic signature against its source report."""
    report = get_precision_report(model)
    if report is None or report.profile != SOURCE_PRECISION_PROFILE:
        return None
    # Public inputs retain leading B and positive static tails. Native-owned
    # outputs may retain that input-derived B on any axis, with static zeros.
    properties = tuple(
        (item, False)
        for entry in report.inputs
        for item in (entry.supplied, entry.normalized, entry.traced, entry.exported)
    ) + tuple(
        (item, True) for item in (*report.traced_outputs, *report.exported_outputs)
    )
    artifact_symbolic = any(
        dim.HasField("dim_param")
        for value in (*model.graph.input, *model.graph.output)
        for dim in value.type.tensor_type.shape.dim
    )
    if not artifact_symbolic and not any(
        isinstance(dim, str) for item, _ in properties for dim in item.shape
    ):
        return None
    symbol: str | None = None
    for item, output in properties:
        labels = [dim for dim in item.shape if isinstance(dim, str)]
        if labels:
            if (
                len(labels) != 1
                or (not output and item.shape[0] != labels[0])
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
                    not isinstance(dim, int)
                    or isinstance(dim, bool)
                    or dim < (0 if output else 1)
                    for dim in item.shape
                    if not isinstance(dim, str)
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

    def preserves_representation(
        logical: TensorProperties, physical: TensorProperties, role: str, index: int
    ) -> bool:
        if logical.dtype != physical.dtype:
            return False
        if logical.shape == physical.shape:
            return True
        # This is the existing explicit image boundary adapter. Logical source
        # declarations remain NHWC; the artifact and feeds use declared NCHW.
        # Require its exact report provenance as well as the fixed permutation.
        return (
            len(logical.shape) == 4
            and physical.shape == tuple(logical.shape[axis] for axis in (0, 3, 1, 2))
            and (
                # The exporter formats native DimExpr tuples, whose symbolic
                # extents have unquoted reprs; report shapes use string labels.
                f"{role} {index}: logical NHWC ({', '.join(map(str, logical.shape))}) "
                f"uses external NCHW storage ({', '.join(map(str, physical.shape))})"
            )
            in report.adaptations
        )

    for index, entry in enumerate(report.inputs):
        if (
            entry.index != index
            or any(
                item.shape != entry.traced.shape
                or (item.dtype is not None and item.dtype != entry.traced.dtype)
                for item in (entry.supplied, entry.normalized, entry.traced)
            )
            or not preserves_representation(
                entry.traced, entry.exported, "input", index
            )
        ):
            raise PrecisionReportError(
                "SOURCE input report does not preserve its declared batch contract"
            )
    if len(report.traced_outputs) != len(outputs) or any(
        not preserves_representation(traced, exported, "output", index)
        for index, (traced, exported) in enumerate(
            zip(report.traced_outputs, outputs, strict=True)
        )
    ):
        raise PrecisionReportError(
            "SOURCE output report does not preserve its declared batch contract"
        )
    if not any(symbol in item.shape for item in inputs):
        raise PrecisionReportError("SOURCE symbolic output has no input batch binding")
    return inputs, outputs


def _source_static_input_boundaries(
    model: onnx.ModelProto,
) -> tuple[TensorProperties, ...] | None:
    """Bind audited static feeds to physical artifact ports, including complex storage."""
    report = get_precision_report(model)
    if report is None or report.profile != SOURCE_PRECISION_PROFILE:
        return None
    declarations = tuple(entry.exported for entry in report.inputs)
    if len(model.graph.input) != len(declarations):
        raise PrecisionReportError("SOURCE input report count differs from artifact")
    for index, (value, declaration, entry) in enumerate(
        zip(model.graph.input, declarations, report.inputs, strict=True)
    ):
        tensor = value.type.tensor_type
        if not tensor.HasField("shape") or any(
            not dim.HasField("dim_value") for dim in tensor.shape.dim
        ):
            raise PrecisionReportError(
                "SOURCE static input requires concrete artifact dimensions"
            )
        shape = tuple(int(dim.dim_value) for dim in tensor.shape.dim)
        dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor.elem_type)).name
        if entry.index != index or (value.name, shape, dtype) != (
            declaration.name,
            declaration.shape,
            declaration.dtype,
        ):
            raise PrecisionReportError(
                "SOURCE exported input report name, order, dtype or shape differs from artifact"
            )
    return declarations


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


def _artifact_deployment_requirements(
    model: onnx.ModelProto,
) -> DeploymentRequirements | None:
    properties = {item.key: item.value for item in model.metadata_props}
    declared = deployment_requirements_from_json(
        properties.get(DEPLOYMENT_METADATA_KEY)
    )
    report = get_precision_report(model)
    recorded = None if report is None else report.deployment_requirements
    if declared != recorded:
        raise PrecisionReportError(
            "Artifact deployment declaration differs from its precision report"
        )
    return declared


def _configure_deployment_session(
    options: Any, requirements: DeploymentRequirements, optimized_path: Path
) -> None:
    """Configure declared settings explicitly; these are not runtime evidence."""
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    for name, value in requirements.target.session_config:
        options.add_session_config_entry(name, value)
    options.optimized_model_filepath = str(optimized_path)


def _deployment_graph_boundary(
    original: onnx.ModelProto,
    optimized: onnx.ModelProto,
    requirements: DeploymentRequirements,
) -> str:
    """Bind actual qualified producers and each runtime Cast's dataflow edges."""
    import onnx.inliner

    def unfinished(detail: str) -> NoReturn:
        raise RuntimeError("Deployment optimizer support remains unfinished: " + detail)

    source: dict[str, tuple[str, onnx.NodeProto]] = {}

    def raw(body: Any, scope: str) -> None:
        for node in body.node:
            if (
                re.fullmatch(r"jax2onnx_src_[0-9]{6}", node.name) is None
                or node.name in source
            ):
                unfinished("missing or ambiguous original source-node tags")
            source[node.name] = (scope, node)
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    raw(
                        attribute.g,
                        _deployment_child_scope(scope, node.name, attribute.name, 0),
                    )
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for index, child in enumerate(attribute.graphs):
                        raw(
                            child,
                            _deployment_child_scope(
                                scope, node.name, attribute.name, index
                            ),
                        )

    raw(original.graph, _deployment_graph_scope(original.graph.name))
    for function in original.functions:
        raw(
            function,
            _deployment_function_scope(
                function.domain, function.name, function.overload
            ),
        )
    for site in requirements.matrix_precision:
        owner = source.get(site.matrix_tag or "")
        if owner is None or owner[0] != site.matrix_scope:
            unfinished("qualified matrix has no exact original scope/tag binding")
        node = owner[1]
        if (
            node.op_type != "MatMul"
            or node.domain not in ("", "ai.onnx")
            or site.matrix_output not in node.output
        ):
            unfinished("qualified source producer is not MatMul")
        # Layout-only descendants may carry the logical equation result.
        if site.output not in node.output:
            scoped = {
                name: item
                for scope, item in source.values()
                if scope == owner[0]
                for name in item.output
            }
            name = site.output
            seen = set()
            while name not in node.output:
                if name in seen:
                    unfinished("cyclic source result provenance")
                seen.add(name)
                item = scoped.get(name)
                if item is None or item.op_type not in {
                    "Identity",
                    "Transpose",
                    "Reshape",
                }:
                    unfinished(
                        "qualified result is not a layout-only matrix descendant"
                    )
                name = item.input[0]

    def role(node: onnx.NodeProto) -> str:
        candidates: list[str] = []
        for tag in re.findall(r"jax2onnx_src_[0-9]{6}", node.name):
            if tag not in source:
                continue
            original_node = source[tag][1]
            if (
                (original_node.op_type, original_node.domain)
                == (node.op_type, node.domain)
                or original_node.op_type == "MatMul"
                and original_node.domain in ("", "ai.onnx")
                and node.op_type == "FusedMatMul"
                and node.domain == "com.microsoft"
            ):
                candidates.append(tag)
        if len(set(candidates)) != 1:
            unfinished("optimizer source-role correspondence is missing or ambiguous")
        return candidates[0]

    def tensor_key(tensor: onnx.TensorProto) -> str:
        array = onnx.numpy_helper.to_array(tensor)
        header = repr((array.dtype.str, tuple(array.shape))).encode()
        return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()

    def direct_constant_tensor(node: onnx.NodeProto) -> onnx.TensorProto | None:
        if (
            node.op_type != "Constant"
            or node.domain not in ("", "ai.onnx")
            or node.overload
            or node.input
            or len(node.output) != 1
            or not node.output[0]
            or len(node.attribute) != 1
        ):
            return None
        attribute = node.attribute[0]
        if (
            attribute.name != "value"
            or attribute.type != onnx.AttributeProto.TENSOR
            or attribute.ref_attr_name
        ):
            return None
        return attribute.t

    def constant_origin(node: onnx.NodeProto) -> tuple[Any, ...]:
        current = direct_constant_tensor(node)
        if current is None:
            unfinished("static Constant definition has an unproved schema")
        tag = role(node)
        definition_scope, definition = source[tag]
        declared = direct_constant_tensor(definition)
        if declared is None:
            unfinished("static Constant source definition has an unproved schema")
        key = (current.data_type, tuple(current.dims), tensor_key(current))
        if key != (declared.data_type, tuple(declared.dims), tensor_key(declared)):
            unfinished("static Constant differs from exact original typed storage")
        return ("source-constant", definition_scope, tag, 0, *key)

    original_flat = onnx.inliner.inline_local_functions(original)
    original_constants = {
        tensor_key(tensor) for tensor in original_flat.graph.initializer
    }

    def contract(model: onnx.ModelProto) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        flat = onnx.shape_inference.infer_shapes(
            onnx.inliner.inline_local_functions(model), strict_mode=True
        )
        matrices: list[tuple[Any, ...]] = []
        casts: list[tuple[Any, ...]] = []

        def graph(
            body: onnx.GraphProto,
            inherited_anchors: Mapping[str, tuple[Any, ...]],
            scope: tuple[str, ...],
            inherited_types: Mapping[str, int],
            inherited_shapes: Mapping[str, tuple[Any, ...]],
            inherited_constants: Mapping[str, onnx.NodeProto],
        ) -> None:
            types = dict(inherited_types)
            types.update(
                (value.name, value.type.tensor_type.elem_type)
                for value in (*body.input, *body.output, *body.value_info)
                if value.type.HasField("tensor_type")
            )
            types.update((value.name, value.data_type) for value in body.initializer)
            shapes = dict(inherited_shapes)
            shapes.update(
                (
                    value.name,
                    tuple(
                        dim.dim_value if dim.HasField("dim_value") else dim.dim_param
                        for dim in value.type.tensor_type.shape.dim
                    ),
                )
                for value in (*body.input, *body.output, *body.value_info)
                if value.type.HasField("tensor_type")
                and value.type.tensor_type.HasField("shape")
            )
            producers = {
                name: (node, index)
                for node in body.node
                for index, name in enumerate(node.output)
            }
            local_definitions = (
                {value.name for value in body.input}
                | {value.name for value in body.initializer}
                | set(producers)
            )
            # Only direct definitions cross scopes. Local definitions shadow
            # inherited candidates; output/value_info declarations do not.
            static_constants = {
                name: definition
                for name, definition in inherited_constants.items()
                if name not in local_definitions
            }
            static_constants.update(
                (name, node)
                for name, (node, _) in producers.items()
                if direct_constant_tensor(node) is not None
            )
            consumers: dict[str, list[tuple[onnx.NodeProto, int]]] = {}
            for node in body.node:
                for index, name in enumerate(node.input):
                    consumers.setdefault(name, []).append((node, index))
            constant_keys = {
                tensor.name: tensor_key(tensor) for tensor in body.initializer
            }
            anchors_cache: dict[str, tuple[Any, ...]] = dict(inherited_anchors)
            for index, value in enumerate(body.input):
                anchors_cache[value.name] = (
                    ("input", scope, index, value.name if not scope else ""),
                )
            for name, key in constant_keys.items():
                # New wholly constant optimizer subgraphs confer no dynamic provenance.
                anchors_cache[name] = (
                    (("initializer", key),) if key in original_constants else ()
                )

            def anchors(
                name: str, active: frozenset[str] = frozenset()
            ) -> tuple[Any, ...]:
                if name in anchors_cache:
                    return anchors_cache[name]
                if name in active:
                    unfinished("cyclic optimized value provenance")
                item = producers.get(name)
                if item is None:
                    unfinished("unbound optimized value provenance")
                node, _ = item

                def free_names(nested: onnx.GraphProto) -> set[str]:
                    local = (
                        {value.name for value in nested.input}
                        | {value.name for value in nested.initializer}
                        | {output for inner in nested.node for output in inner.output}
                    )
                    used = {
                        argument
                        for inner in nested.node
                        for argument in inner.input
                        if argument
                    }
                    for inner in nested.node:
                        for attribute in inner.attribute:
                            if attribute.type == onnx.AttributeProto.GRAPH:
                                used.update(free_names(attribute.g))
                            elif attribute.type == onnx.AttributeProto.GRAPHS:
                                for child in attribute.graphs:
                                    used.update(free_names(child))
                    return used - local

                arguments = set(argument for argument in node.input if argument)
                for attribute in node.attribute:
                    if attribute.type == onnx.AttributeProto.GRAPH:
                        arguments.update(free_names(attribute.g))
                    elif attribute.type == onnx.AttributeProto.GRAPHS:
                        for child in attribute.graphs:
                            arguments.update(free_names(child))
                result = tuple(
                    sorted(
                        set(
                            anchor
                            for argument in arguments
                            for anchor in anchors(argument, active | {name})
                        )
                    )
                )
                anchors_cache[name] = result
                return result

            def origin(name: str) -> tuple[Any, ...]:
                item = producers.get(name)
                while item is not None and item[0].op_type in {
                    "Identity",
                    "Transpose",
                    "Reshape",
                }:
                    name = item[0].input[0]
                    item = producers.get(name)
                if name in static_constants:
                    # Resolve raw role/storage lazily, only for an inspected
                    # origin; unused optimizer shape Constants stay unrelated.
                    return constant_origin(static_constants[name])
                if item is None:
                    return ("boundary", anchors(name))
                node, index = item
                return ("producer", role(node), index, anchors(name))

            def consumer_edges(
                name: str, active: frozenset[str] = frozenset()
            ) -> tuple[Any, ...]:
                if name in active:
                    unfinished("cyclic optimized consumer provenance")
                edges: list[tuple[Any, ...]] = []
                for node, index in consumers.get(name, ()):
                    if node.op_type in {"Identity", "Transpose", "Reshape"}:
                        edges.extend(consumer_edges(node.output[0], active | {name}))
                    else:
                        edges.append(
                            (
                                "consumer",
                                role(node),
                                index,
                                tuple(anchors(output) for output in node.output),
                            )
                        )
                edges.extend(
                    ("output", scope, index)
                    for index, value in enumerate(body.output)
                    if value.name == name
                )
                return tuple(sorted(edges))

            def layout(name: str, transpose: bool = False) -> tuple[Any, ...]:
                dimensions = shapes.get(name)
                item = producers.get(name)
                steps: list[tuple[Any, ...]] = []
                while item is not None and item[0].op_type in {
                    "Identity",
                    "Transpose",
                    "Reshape",
                }:
                    node, _ = item
                    if node.op_type == "Transpose":
                        attributes = {
                            item.name: onnx.helper.get_attribute_value(item)
                            for item in node.attribute
                        }
                        shape = shapes.get(node.input[0])
                        if shape is None:
                            unfinished("transpose has unknown rank")
                        steps.append(
                            (
                                "transpose",
                                tuple(
                                    attributes.get(
                                        "perm", tuple(reversed(range(len(shape))))
                                    )
                                ),
                            )
                        )
                    elif node.op_type == "Reshape":
                        steps.append(("reshape", shapes.get(name)))
                    name = node.input[0]
                    item = producers.get(name)
                if transpose:
                    if dimensions is None or len(dimensions) < 2:
                        unfinished("FusedMatMul transpose has unknown rank")
                    permutation = tuple(range(len(dimensions) - 2)) + (
                        len(dimensions) - 1,
                        len(dimensions) - 2,
                    )
                    steps.insert(0, ("transpose", permutation))
                return (origin(name), tuple(reversed(steps)))

            for node in body.node:
                if node.op_type == "Gemm":
                    unfinished(
                        "Gemm bias/scale boundary needs separate actual inspection"
                    )
                if node.domain == "com.microsoft" and node.op_type != "FusedMatMul":
                    unfinished("unreviewed fused operator " + node.op_type)
                if node.op_type in {"MatMul", "FusedMatMul"}:
                    tag = role(node)
                    qualified = any(
                        site.matrix_tag == tag for site in requirements.matrix_precision
                    )
                    if qualified and any(
                        types.get(name) != onnx.TensorProto.FLOAT
                        for name in (*node.input, *node.output)
                    ):
                        unfinished(
                            "qualified matrix kernel types differ from full FP32"
                        )
                    attributes = {
                        item.name: onnx.helper.get_attribute_value(item)
                        for item in node.attribute
                    }
                    if node.op_type == "FusedMatMul":
                        if (
                            set(attributes)
                            - {
                                "alpha",
                                "transA",
                                "transB",
                                "transBatchA",
                                "transBatchB",
                            }
                            or attributes.get("alpha", 1.0) != 1.0
                            or attributes.get("transBatchA", 0) != 0
                            or attributes.get("transBatchB", 0) != 0
                            or any(
                                attributes.get(name, 0) not in (0, 1)
                                for name in ("transA", "transB")
                            )
                        ):
                            unfinished(
                                "FusedMatMul correspondence exceeds last-axis transpose with alpha1"
                            )
                    matrices.append(
                        (
                            scope,
                            tag,
                            layout(node.input[0], bool(attributes.get("transA", 0))),
                            layout(node.input[1], bool(attributes.get("transB", 0))),
                            tuple(consumer_edges(output) for output in node.output),
                        )
                    )
                if node.op_type == "Cast" and anchors(node.output[0]):
                    if (
                        types.get(node.input[0]) is None
                        or types.get(node.output[0]) is None
                    ):
                        unfinished("runtime Cast boundary has unknown types")
                    casts.append(
                        (
                            scope,
                            role(node),
                            types[node.input[0]],
                            types[node.output[0]],
                            origin(node.input[0]),
                            consumer_edges(node.output[0]),
                        )
                    )
                for attribute in node.attribute:
                    if attribute.type in (
                        onnx.AttributeProto.GRAPH,
                        onnx.AttributeProto.GRAPHS,
                    ):
                        children = (
                            (attribute.g,)
                            if attribute.type == onnx.AttributeProto.GRAPH
                            else attribute.graphs
                        )
                        for index, nested in enumerate(children):
                            nested_scope = scope + (
                                role(node),
                                attribute.name,
                                str(index),
                            )
                            captured = {
                                name: anchors(name)
                                for name in (*producers, *anchors_cache)
                            }
                            graph(
                                nested,
                                captured,
                                nested_scope,
                                types,
                                shapes,
                                static_constants,
                            )

        graph(flat.graph, {}, (), {}, {}, {})
        if len(set(matrices)) != len(matrices) or len(set(casts)) != len(casts):
            unfinished("repeated source role has ambiguous value/edge provenance")
        return tuple(sorted(matrices)), tuple(sorted(casts))

    from jax2onnx._deployment_fusion_bridge import (
        contains_fusions,
        prove_deployment_fusions,
    )
    from jax2onnx._deployment_fusions import FusionUnfinished

    before = contract(original)
    if not contains_fusions(optimized) and before == contract(optimized):
        return "matmul-transpose-and-original-casts-v1"
    roles = {
        tag: (node.op_type, node.domain, scope) for tag, (scope, node) in source.items()
    }
    try:
        inspected, fusion_kinds = prove_deployment_fusions(original, optimized, roles)
    except FusionUnfinished as exc:
        unfinished(str(exc))
    after = contract(inspected)
    if before != after:
        unfinished(
            "a qualified matrix or original runtime Cast changed its bound dataflow path"
        )
    return (
        "matmul-gemm-quickgelu-and-original-casts-v2"
        if fusion_kinds
        else "matmul-transpose-and-original-casts-v1"
    )


def _observe_deployment_session(
    model: onnx.ModelProto,
    session: Any,
    ort: Any,
    optimized_path: Path,
) -> RuntimeTargetObservation:
    """Observe loaded runtime, relevant hardware, session and optimized artifact."""
    from importlib.metadata import version

    required = _artifact_deployment_requirements(model)
    if required is None:
        raise ValueError("Artifact has no deployment declaration")
    distribution_version = version("onnxruntime")
    if platform.system() != "Linux":
        raise ValueError("Actual deployment system is outside the reviewed envelope")
    blocks = []
    for block in Path("/proc/cpuinfo").read_text().split("\n\n"):
        entries = dict(
            (name.strip(), value.strip())
            for line in block.splitlines()
            if ":" in line
            for name, value in (line.split(":", 1),)
        )
        if "processor" in entries:
            blocks.append(entries)
    if not blocks or any(
        not {"vendor_id", "model name", "flags"} <= set(block) for block in blocks
    ):
        raise ValueError("Actual CPU identity or ISA is unknown")
    identities = {(block["vendor_id"], block["model name"]) for block in blocks}
    if len(identities) != 1:
        raise ValueError("Actual CPU identity is not uniform")
    vendor, cpu_model = identities.pop()
    flags = [set(block["flags"].split()) for block in blocks]
    relevant = set.intersection(*flags) & {"sse2", "avx", "avx2", "fma"}
    relevant.update(
        flag
        for values in flags
        for flag in values
        if flag.startswith(("avx512", "amx")) or "bf16" in flag
    )
    libraries = tuple(
        sorted(
            (path.name, hashlib.sha256(path.read_bytes()).hexdigest())
            for path in (Path(ort.__file__).parent / "capi").glob("*.so*")
            if path.is_file()
        )
    )
    options = session.get_session_options()
    provider_options = tuple(
        (name, tuple(sorted((str(key), str(value)) for key, value in values.items())))
        for name, values in sorted(session.get_provider_options().items())
    )
    target = DeploymentTarget(
        runtime="onnxruntime",
        runtime_version=ort.__version__,
        runtime_build=ort.get_build_info(),
        system=platform.system(),
        architecture=platform.machine(),
        cpu_vendor=vendor,
        cpu_model=cpu_model,
        cpu_features=tuple(sorted(relevant)),
        libraries=libraries,
        providers=tuple(session.get_providers()),
        provider_options=provider_options,
        session_config=tuple(
            (name, options.get_session_config_entry(name))
            for name, _ in required.target.session_config
        ),
    )
    # The optimized artifact is retained before its boundary is evaluated.
    optimized_digest = None
    boundary = None
    boundary_error = None
    try:
        optimized_bytes = optimized_path.read_bytes()
        optimized_digest = hashlib.sha256(optimized_bytes).hexdigest()
        optimized = onnx.load_model(optimized_path)
        boundary = _deployment_graph_boundary(model, optimized, required)
    except Exception as exc:
        boundary_error = f"{type(exc).__name__}: {exc}"
    actual = RuntimeTargetObservation(
        target=target,
        optimization=options.graph_optimization_level.name,
        execution_mode=options.execution_mode.name,
        intra_op_threads=options.intra_op_num_threads,
        inter_op_threads=options.inter_op_num_threads,
        distribution_version=distribution_version,
        optimized_graph_sha256=optimized_digest,
        boundary_contract=boundary,
        boundary_error=boundary_error,
    )
    return actual


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
        check_source_schema(blob)
    except Exception as exc:
        checks.append(
            PrecisionCheck("onnx-schema", CheckStatus.FAILED, exception_detail(exc))
        )
        return result()
    checks.append(
        PrecisionCheck(
            "onnx-schema", CheckStatus.VALIDATED, "ONNX schema and shape checks passed."
        )
    )
    try:
        deployment_requirements = _artifact_deployment_requirements(model)
        batch_boundaries = _source_batch_boundaries(model)
        static_inputs = (
            _source_static_input_boundaries(model) if batch_boundaries is None else None
        )
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
        input_contract = (
            batch_boundaries[0] if batch_boundaries is not None else static_inputs
        )
        if input_contract is not None:
            output_contract = (
                batch_boundaries[1] if batch_boundaries is not None else ()
            )
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
                if (
                    batch_boundaries is not None
                    and expected
                    and len(expected) != len(output_contract)
                ):
                    raise ValueError(
                        "Reference output count differs from SOURCE declaration"
                    )
                for declaration, array in zip(
                    (
                        output_contract
                        if expected and batch_boundaries is not None
                        else ()
                    ),
                    expected if batch_boundaries is not None else (),
                    strict=True,
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
            loaded_runtime = None
            try:
                # Optional runtime loading follows the SOURCE input/reference
                # gate, so invalid-only samples construct no execution session.
                import onnxruntime as ort

                loaded_runtime = ort
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
                deployment_directory = (
                    TemporaryDirectory(prefix="jax2onnx-deployment-")
                    if deployment_requirements is not None
                    else None
                )
                observation = None
                try:
                    optimized_path = None
                    if (
                        deployment_directory is not None
                        and deployment_requirements is not None
                    ):
                        optimized_path = (
                            Path(deployment_directory.name) / "optimized.onnx"
                        )
                        _configure_deployment_session(
                            options, deployment_requirements, optimized_path
                        )
                    session = ort.InferenceSession(
                        blob, options, providers=["CPUExecutionProvider"]
                    )
                    session.disable_fallback()
                    if optimized_path is not None:
                        observation = _observe_deployment_session(
                            model, session, ort, optimized_path
                        )
                finally:
                    if deployment_directory is not None:
                        deployment_directory.cleanup()
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
                    session_config=(
                        ()
                        if deployment_requirements is None
                        else deployment_requirements.target.session_config
                    ),
                    deployment_observation=observation,
                )
                if deployment_requirements is not None:
                    assert observation is not None
                    require_runtime_target(deployment_requirements, observation)
                if providers != ("CPUExecutionProvider",):
                    raise RuntimeError(
                        f"Expected only CPUExecutionProvider, received {providers}"
                    )
            except Exception as exc:
                detail = runtime_load_detail(
                    exc,
                    blob,
                    loaded_runtime,
                    providers_requested=("CPUExecutionProvider",),
                )
                sample_checks.extend(
                    (
                        PrecisionCheck("runtime-load", CheckStatus.FAILED, detail),
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
                    PrecisionCheck("runtime-load", CheckStatus.FAILED, detail)
                )
                return result()
        try:
            names = tuple(spec.name for spec in session.get_inputs())
            if input_contract is not None:
                if names != tuple(item.name for item in input_contract):
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
            outputs = _run_cpu_samples(session, feeds, model)
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
