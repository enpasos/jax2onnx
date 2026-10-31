# tests/extra_tests/converter/source_precision_raw_replay.py

"""Native capture and exact recipe adapters for the full raw migration schedule.

This is replay tooling. It does not change product admission or select criteria.
The old finite corpus replay continues to use its original preparation path.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, fields, is_dataclass
import builtins as _native_builtins
import dis
import functools
import hashlib
import json
from numbers import Integral
from pathlib import Path
import types
import sys
from typing import Any
import typing as _native_typing

import jax
from jax import export as jax_export
import numpy as np
import onnx
import onnx_ir as ir

from jax2onnx.precision import CheckStatus, PrecisionCheck, get_precision_report
from jax2onnx._compat.jax import (
    Array as _NATIVE_UNION_ARRAY,
    Literal as JaxLiteral,
    canonical_code_fingerprint,
    is_selected_prng_key_dtype,
    selected_prng_key_words,
    selected_native_capture,
    selected_native_capture_member,
    native_transformed_function_role,
    native_transformed_closure_state,
    selected_native_transformed_function,
    NativeTransformedExecutionError,
    native_transformed_execution_context,
)
from jax2onnx.converter.source_representation import (
    native_shape_contract,
    representation_for_aval,
)
from jax2onnx.converter.source_structure import ConstantContract, StructureResult
from scripts.source_precision_inventory import canonical_contract
from scripts._source_inventory_epoch import validate_replay_manifest
from tests.extra_tests.converter.source_precision_support import (
    PRESERVE_FACTORY_DTYPE,
    PreparedSourceCase,
    SourceCaseResult,
    SourcePreparationError,
    SourceSettings,
    _argument,
    _criteria,
    _export_prepared,
    _properties,
    _require_preserved_dtype,
    _typed_input_spec,
    source_configuration,
    value_evidence,
)


class _ReplayReferenceWitness:
    """Retain only edges already read by the existing source-state traversal."""

    def __init__(self) -> None:
        self.edges: list[tuple[tuple[str, ...], object]] = []
        self.selected: dict[tuple[str, ...], str] = {}
        self.native_references: list[object] = []

    def take(self, path: tuple[str, ...], value: object) -> None:
        self.edges.append((path, value))

    def document(self) -> Any:
        return {
            "selected_roles": [
                [list(path), role] for path, role in self.selected.items()
            ],
            "original_edges": [[list(path), id(value)] for path, value in self.edges],
        }

    def require_same(self, observed: _ReplayReferenceWitness) -> None:
        if self.selected != observed.selected or len(self.edges) != len(observed.edges):
            raise SourcePreparationError(
                "Actual selected native wrapper or original source edges changed",
                category="replay-lane-gap",
            )
        for (path, original), (current_path, current) in zip(
            self.edges, observed.edges, strict=True
        ):
            if path != current_path or original is not current:
                raise SourcePreparationError(
                    "Actual selected native wrapper source identity/alias changed",
                    category="replay-lane-gap",
                )


class _StructureBindingGap(ValueError):
    """Artifact/source evidence is unavailable without changing export admission."""


class _StructureBindingMismatch(ValueError):
    """Complete observed binding evidence contradicts the frozen source."""


def _structure_digest(payload: bytes) -> dict[str, Any]:
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def _structure_file_digest(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            count += len(block)
    return {"sha256": digest.hexdigest(), "bytes": count}


def _structure_external_storage(message: Any) -> bool:
    if isinstance(message, onnx.TensorProto):
        return bool(
            message.data_location == onnx.TensorProto.EXTERNAL or message.external_data
        )
    for descriptor, child in message.ListFields():
        if descriptor.message_type is not None:
            children = child if descriptor.is_repeated else (child,)
            if any(_structure_external_storage(value) for value in children):
                return True
    return False


def _structure_model_digest(value: Any) -> tuple[dict[str, Any], bool]:
    if isinstance(value, ir.Model):
        proto = ir.to_proto(value)
    elif isinstance(value, onnx.ModelProto):
        proto = value
    else:
        raise _StructureBindingGap("return-normalization-unavailable")
    external = _structure_external_storage(proto)
    payload = proto.SerializeToString(deterministic=True)
    digest = _structure_digest(payload)
    # Do not overlap the next model serialization with these temporary bytes.
    del payload
    return digest, external


def _structure_pending(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            (key == "unresolved_parameters" and bool(item))
            or (str(key).startswith("unresolved") and key != "unresolved_parameters")
            or (key == "var" and item == "unbound")
            or _structure_pending(item)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list)):
        return any(_structure_pending(item) for item in value)
    return False


def _structure_compact_manifest(value: Any) -> Any:
    """Persist typed topology and payload digests, never duplicate word payloads."""
    if isinstance(value, dict):
        payload = value.pop("bytes_hex", None)
        if payload is not None:
            value["payload_sha256"] = hashlib.sha256(bytes.fromhex(payload)).hexdigest()
        for key in tuple(value):
            if key in {"literal", "const_payloads"}:
                payload = json.dumps(
                    value[key], sort_keys=True, separators=(",", ":")
                ).encode()
                value[key] = {"canonical_payload": _structure_digest(payload)}
                del payload
            else:
                value[key] = _structure_compact_manifest(value[key])
    elif isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _structure_compact_manifest(item)
    elif isinstance(value, tuple):
        return [_structure_compact_manifest(item) for item in value]
    return value


def _structure_trace_manifest(closed: Any) -> tuple[dict[str, Any], str, bool]:
    from tests.extra_tests.converter.source_precision_diagnostics import (
        typed_trace_manifest,
    )

    manifest = typed_trace_manifest(closed)
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(payload).hexdigest()
    del payload
    return manifest, digest, _structure_pending(manifest)


def _structure_body(closed: Any) -> Any:
    # JAX 0.11 may return itself from .jaxpr; unwrap once, never recursively.
    return getattr(closed, "jaxpr", closed)


def _structure_trace_storage(closed: Any) -> list[dict[str, Any]]:
    body = _structure_body(closed)
    constants = tuple(getattr(closed, "consts", ()))
    if len(constants) != len(body.constvars):
        raise _StructureBindingGap("closed-constant-storage-unavailable")
    evidence = []

    def take(role: str, var: Any, value: Any) -> None:
        # Match the existing shared literal observer's typed storage conversion.
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(value, dtype=var.aval.dtype)
            representation = representation_for_aval(var.aval)
            logical = asdict(ConstantContract.from_array(array))
            physical = asdict(
                ConstantContract.from_array(representation.encode_array(array))
            )
        evidence.append(
            {
                "role": role,
                "logical_dtype": np.dtype(var.aval.dtype).name,
                "weak_type": bool(getattr(var.aval, "weak_type", False)),
                "logical": logical,
                "encoding": representation.encoding,
                "physical": physical,
            }
        )

    for index, (var, value) in enumerate(zip(body.constvars, constants, strict=True)):
        take(f"constant:{index}", var, value)
    for index, eqn in enumerate(body.eqns):
        for port, var in enumerate(eqn.invars):
            if isinstance(var, JaxLiteral):
                take(f"equation:{index}:input:{port}", var, var.val)
    for index, var in enumerate(body.outvars):
        if isinstance(var, JaxLiteral):
            take(f"output:{index}", var, var.val)
    return evidence


def _structure_flat_parameters(value: Any) -> None:
    if value is None or type(value) in (bool, int, str) or isinstance(value, np.dtype):
        return
    if type(value) in (tuple, list):
        for item in value:
            _structure_flat_parameters(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _structure_flat_parameters(item)
        return
    # In particular, nested Jaxprs and unowned parameter objects are not proved
    # merely because their printed names or partial JSON documents agree.
    raise _StructureBindingGap("nested-or-unowned-source-parameter")


def _structure_flat_relation(original: Any, actual: Any) -> None:
    left, right = _structure_body(original), _structure_body(actual)
    if left.effects or right.effects:
        raise _StructureBindingGap("source-effects-relation-unverified")
    if len(left.eqns) != len(right.eqns):
        raise _StructureBindingMismatch("source-equation-count-differs")
    pairs = [(left.constvars, right.constvars), (left.invars, right.invars)]
    pairs.append((left.outvars, right.outvars))
    for a, b in zip(left.eqns, right.eqns, strict=True):
        if a.primitive is not b.primitive:
            raise _StructureBindingMismatch("native-source-primitive-differs")
        pairs.extend(((a.invars, b.invars), (a.outvars, b.outvars)))
    for a, b in zip(left.eqns, right.eqns, strict=True):
        _structure_flat_parameters(a.params)
        _structure_flat_parameters(b.params)
    for left_vars, right_vars in pairs:
        if len(left_vars) != len(right_vars):
            raise _StructureBindingMismatch("source-port-arity-differs")
        for a, b in zip(left_vars, right_vars, strict=True):
            if (
                np.dtype(a.aval.dtype) != np.dtype(b.aval.dtype)
                or bool(getattr(a.aval, "weak_type", False))
                != bool(getattr(b.aval, "weak_type", False))
                or isinstance(a, JaxLiteral) != isinstance(b, JaxLiteral)
            ):
                raise _StructureBindingMismatch("source-port-type-differs")
            try:
                a_shape = native_shape_contract(a.aval.shape)
                b_shape = native_shape_contract(b.aval.shape)
            except (TypeError, ValueError) as exc:
                raise _StructureBindingGap("native-source-shape-unverified") from exc
            if (a_shape.symbol is None) != (b_shape.symbol is None):
                raise _StructureBindingMismatch("source-symbolic-port-differs")
            if a_shape.symbol is not None and b_shape.symbol is not None:
                if a_shape.symbol[1] is not b_shape.symbol[1]:
                    raise _StructureBindingGap("cross-trace-native-scope-unverified")
            if (
                a_shape.shape != b_shape.shape
                or a_shape.symbol_axis != b_shape.symbol_axis
            ):
                raise _StructureBindingMismatch("source-port-shape-differs")


def _structure_source_evidence(inputs: Any, references: Any) -> str:
    payload = json.dumps(
        {
            "inputs": [asdict(item) for item in inputs],
            "references": [asdict(item) for item in references],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class RawReplayCapture:
    """Write each native result/returned artifact before subsequent checks."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        self.records: dict[str, Any] = {
            role: {"status": "not-attempted"}
            for role in (
                "eager",
                "fresh-jit",
                "concrete-typed-trace",
                "declared-typed-trace",
            )
        }
        self.records["product_admission"] = "not-attempted"
        self.records["raw_callback"] = {"status": "not-executed"}
        self.model: onnx.ModelProto | None = None
        self.facade_value: Any = None
        self.records["facade_return"] = {"status": "not-attempted"}
        self.native_values: dict[str, Any] = {}
        self.native_errors: dict[str, Exception] = {}
        self.gaps: list[dict[str, str]] = []
        self.target: Any = None
        self.target_state: Any = None
        self.target_witness: _ReplayReferenceWitness | None = None
        self.preparation_complete = False
        self.records["native_outcomes"] = {}
        self.records["replay_readiness"] = {"status": "unverified", "gaps": self.gaps}
        self.records["source_structure"] = {
            "status": "unverified",
            "collector_calls": 0,
            "ir_result": None,
            "source_binding": {"status": "unverified", "reasons": []},
            "artifact_binding": {"status": "unverified", "reasons": []},
        }

    def _structure_note(self, section: str, status: CheckStatus, reason: str) -> None:
        try:
            binding = self.records["source_structure"][section]
            # Later unavailable evidence must not erase a known contradiction.
            if binding.get("status") != "failed":
                # Every nonempty reason denotes required evidence. An unrelated
                # successful subcheck cannot fill a previously recorded gap.
                binding["status"] = (
                    "unverified"
                    if status is CheckStatus.VALIDATED and binding.get("reasons")
                    else status.value
                )
            reasons = binding.setdefault("reasons", [])
            if reason and reason not in reasons and len(reasons) < 16:
                reasons.append(reason)
        except Exception:
            pass

    def _structure_error(self, section: str, phase: str, exc: Exception) -> None:
        known = type(exc) is _StructureBindingMismatch
        reason = (
            str(exc)
            if type(exc) in (_StructureBindingMismatch, _StructureBindingGap)
            else phase + "-unreadable"
        )
        self._structure_note(
            section, CheckStatus.FAILED if known else CheckStatus.UNVERIFIED, reason
        )

    def freeze_source_structure(
        self, closed: Any, trace_json: str, inputs: Any, references: Any
    ) -> None:
        """Freeze exact prepared words now, before any later export or callback."""
        try:
            binding = self.records["source_structure"]["source_binding"]
            binding["prepared_trace_sha256"] = hashlib.sha256(
                trace_json.encode()
            ).hexdigest()
            binding["prepared_trace_pending"] = _structure_pending(
                json.loads(trace_json)
            )
            binding["prepared_source_evidence_sha256"] = _structure_source_evidence(
                inputs, references
            )
            binding["prepared_storage"] = _structure_trace_storage(closed)
        except Exception as exc:
            self._structure_error("source_binding", "prepared-source", exc)

    def source_structure_collector(self, prepared: PreparedSourceCase) -> Any:
        """One invocation-local callback; persisted records contain metadata only."""

        def collect(trace: Any, final_ir: ir.Model, result: Any, returned: Any) -> None:
            try:
                record = self.records["source_structure"]
                record["collector_calls"] += 1
                if record["collector_calls"] != 1:
                    self._structure_note(
                        "artifact_binding",
                        CheckStatus.UNVERIFIED,
                        "collector-duplicate",
                    )
                    return
                if type(result) is StructureResult:
                    record["ir_result"] = {
                        "status": result.status,
                        "expected_equations": result.expected_equations,
                        "checked_equations": result.checked_equations,
                        "reasons": list(result.reasons),
                    }
                else:
                    self._structure_note(
                        "artifact_binding",
                        CheckStatus.UNVERIFIED,
                        "ir-result-unavailable",
                    )
                self._collect_structure_source(prepared, trace)
                try:
                    binding = record["artifact_binding"]
                    candidate, external = _structure_model_digest(final_ir)
                    binding["checked_ir_proto"] = candidate
                    binding["return_kind"] = (
                        "proto"
                        if isinstance(returned, onnx.ModelProto)
                        else (
                            "ir"
                            if isinstance(returned, ir.Model)
                            else "file" if isinstance(returned, str) else "unknown"
                        )
                    )
                    if binding["return_kind"] in {"file", "unknown"}:
                        raise _StructureBindingGap(
                            "file-or-unknown-return-binding-unverified"
                        )
                    actual, returned_external = _structure_model_digest(returned)
                    binding["actual_return"] = actual
                    binding["external_storage"] = external or returned_external
                    if candidate != actual:
                        raise _StructureBindingMismatch(
                            "checked-ir-and-returned-artifact-differ"
                        )
                    if binding["external_storage"]:
                        raise _StructureBindingGap(
                            "external-payload-binding-unverified"
                        )
                    self._structure_note("artifact_binding", CheckStatus.VALIDATED, "")
                except Exception as exc:
                    self._structure_error("artifact_binding", "returned-artifact", exc)
            except Exception as exc:
                self._structure_error("artifact_binding", "collector", exc)

        return collect

    def _collect_structure_source(
        self, prepared: PreparedSourceCase, trace: Any
    ) -> None:
        manifest = None
        try:
            binding = self.records["source_structure"]["source_binding"]
            manifest, actual_sha, actual_pending = _structure_trace_manifest(
                trace.closed_jaxpr
            )
            binding["actual_trace_sha256"] = actual_sha
            _, fresh_sha, fresh_pending = _structure_trace_manifest(
                prepared.original_jaxpr
            )
            binding["fresh_prepared_trace_sha256"] = fresh_sha
            if self.target is not prepared.fn:
                raise _StructureBindingMismatch("prepared-target-binding-differs")
            expected_evidence = binding.get("prepared_source_evidence_sha256")
            if expected_evidence is None:
                raise _StructureBindingGap("prepared-source-evidence-unavailable")
            if (
                _structure_source_evidence(
                    prepared.input_evidence, prepared.reference_evidence
                )
                != expected_evidence
            ):
                raise _StructureBindingMismatch("prepared-source-evidence-differs")
            fresh_storage = _structure_trace_storage(prepared.original_jaxpr)
            actual_storage = _structure_trace_storage(trace.closed_jaxpr)
            binding["fresh_prepared_storage"] = fresh_storage
            binding["actual_storage"] = actual_storage
            if "prepared_storage" not in binding:
                raise _StructureBindingGap("prepared-source-storage-unavailable")
            if (
                fresh_storage != binding["prepared_storage"]
                or actual_storage != binding["prepared_storage"]
            ):
                raise _StructureBindingMismatch(
                    "source-constant-or-literal-storage-differs"
                )
            _structure_flat_relation(prepared.original_jaxpr, trace.closed_jaxpr)
            if (
                binding.get("prepared_trace_pending", True)
                or fresh_pending
                or actual_pending
            ):
                raise _StructureBindingGap("unresolved-source-trace-binding")
            expected_sha = binding.get("prepared_trace_sha256")
            if expected_sha is None:
                raise _StructureBindingGap("prepared-source-trace-unavailable")
            if (
                hashlib.sha256(prepared.original_trace_json.encode()).hexdigest()
                != expected_sha
                or fresh_sha != expected_sha
                or actual_sha != expected_sha
            ):
                raise _StructureBindingMismatch("complete-source-trace-differs")
            binding["native_primitive_relation"] = "identical-flat-sequence"
            binding["native_shape_relation"] = "static-or-identical-native-scope"
            self._structure_note("source_binding", CheckStatus.VALIDATED, "")
        except Exception as exc:
            self._structure_error("source_binding", "export-source", exc)
        # A sink failure is only this bridge's gap, never a source-readiness gap.
        if manifest is not None:
            try:
                binding = self.records["source_structure"]["source_binding"]
                compact = _structure_compact_manifest(manifest)
                payload = json.dumps(
                    {
                        "canonical_typed_trace_sha256": binding["actual_trace_sha256"],
                        "manifest": compact,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                path = self.directory / "source-export-typed-trace.json"
                with path.open("xb") as stream:
                    stream.write(payload)
                binding["export_trace_manifest"] = {
                    "path": str(path),
                    **_structure_digest(payload),
                }
                del payload
            except Exception as exc:
                self._structure_error("source_binding", "export-trace-sink", exc)

    def finalize_source_structure(self, normalized: Any) -> None:
        """Fence real returned objects and saved bytes after the raw callback."""
        try:
            record = self.records["source_structure"]
            binding = record["artifact_binding"]
            if record["collector_calls"] != 1:
                raise _StructureBindingGap("collector-missing-or-duplicate")
            if binding.get("return_kind") not in {"proto", "ir"} or binding.get(
                "external_storage"
            ):
                raise _StructureBindingGap("file-or-external-bundle-binding-unverified")
            candidate = binding.get("checked_ir_proto")
            if candidate is None:
                raise _StructureBindingGap("checked-ir-digest-unavailable")
            normalized_return = binding.get("normalized_return")
            if normalized_return is None:
                raise _StructureBindingGap("normalized-return-evidence-unavailable")
            if normalized_return != candidate:
                raise _StructureBindingMismatch(
                    "normalized-return-and-checked-ir-differ"
                )
            # Each serialization is released before beginning the next one.
            for role, value in (
                ("after_callback", normalized),
                ("facade_after_callback", self.facade_value),
            ):
                try:
                    digest, external = _structure_model_digest(value)
                    binding[role] = digest
                    if digest != candidate or external:
                        raise _StructureBindingMismatch(
                            "post-callback-artifact-mutation"
                        )
                except Exception as exc:
                    self._structure_error("artifact_binding", role, exc)
            artifact = self.records.get("artifact", {})
            if artifact.get("status") != "captured":
                raise _StructureBindingGap("saved-artifact-unavailable")
            if artifact.get("sha256") != candidate["sha256"]:
                raise _StructureBindingMismatch("saved-artifact-and-checked-ir-differ")
            path = Path(artifact["path"])
            if path != self.directory / "returned.onnx" or path.is_symlink():
                raise _StructureBindingMismatch("saved-artifact-path-differs")
            saved = _structure_file_digest(path)
            binding["saved_artifact"] = saved
            if saved != candidate:
                raise _StructureBindingMismatch("post-callback-saved-artifact-mutation")
            binding["fenced_after_callback"] = True
        except Exception as exc:
            self._structure_error("artifact_binding", "artifact-fence", exc)
        try:
            trace = self.records["source_structure"]["source_binding"].get(
                "export_trace_manifest"
            )
            if trace is not None:
                actual_trace_file = _structure_file_digest(Path(trace["path"]))
                if actual_trace_file != {
                    key: trace[key] for key in ("sha256", "bytes")
                }:
                    raise _StructureBindingMismatch(
                        "post-callback-export-trace-mutation"
                    )
        except Exception as exc:
            self._structure_error("source_binding", "export-trace-fence", exc)
        self.source_structure_status()

    def source_structure_status(self) -> CheckStatus:
        try:
            record = self.records["source_structure"]
            source, artifact = record["source_binding"], record["artifact_binding"]
            if "failed" in (source["status"], artifact["status"]):
                status = CheckStatus.FAILED
            elif (
                record["collector_calls"] != 1
                or source["status"] != "validated"
                or artifact["status"] != "validated"
                or not artifact.get("fenced_after_callback")
                or self.readiness is not CheckStatus.VALIDATED
                or record["ir_result"] is None
            ):
                status = CheckStatus.UNVERIFIED
            else:
                status = CheckStatus(record["ir_result"]["status"])
            record["status"] = status.value
            return status
        except Exception:
            return CheckStatus.UNVERIFIED

    def gap(self, role: str, exc: Exception) -> None:
        self.failure(role, exc)
        self.gaps.append({"role": role, "detail": str(exc)})
        self.records["replay_readiness"] = {"status": "unverified", "gaps": self.gaps}

    @property
    def readiness(self) -> CheckStatus:
        status = (
            CheckStatus.VALIDATED
            if self.preparation_complete and not self.gaps
            else CheckStatus.UNVERIFIED
        )
        self.records["replay_readiness"] = {"status": status.value, "gaps": self.gaps}
        return status

    def _record_context_gap(self, role: str, error: Exception) -> None:
        key = role + "-native-context"
        try:
            self.gap(key, error)
        finally:
            # A failed evidence reporter cannot turn context drift into readiness.
            if not any(gap["role"] == key for gap in self.gaps):
                self.gaps.append({"role": key, "detail": str(error)})
            self.records["replay_readiness"] = {
                "status": "unverified",
                "gaps": self.gaps,
            }

    def execution_context(self, role: str) -> AbstractContextManager[None]:
        if (
            self.target_witness is None
            or "vmap_f" not in self.target_witness.selected.values()
        ):
            return nullcontext()
        return native_transformed_execution_context(
            on_error=lambda error: self._record_context_gap(role, error)
        )

    def invoke(self, role: str, call: Any) -> Any:
        # Retain the actual object/exception before boundary cleanup or evidence.
        result = None
        try:
            with self.execution_context(role):
                try:
                    result = call()
                except Exception as exc:
                    self.native_errors[role] = exc
                    self.records["native_outcomes"][role] = {
                        "status": "raised",
                        "exception": f"{type(exc).__module__}.{type(exc).__qualname__}",
                        "detail": str(exc),
                    }
                else:
                    self.native_values[role] = result
                    self.records["native_outcomes"][role] = {"status": "returned"}
        except NativeTransformedExecutionError:
            # Entry failed before the source call; the boundary recorded a gap.
            pass
        return result

    def save_arrays(self, role: str, values: Any) -> tuple[np.ndarray, ...] | None:
        try:
            return self.arrays(role, values)
        except Exception as exc:
            self.gap(role, exc)
            return None

    def save_document(self, role: str, document: Any) -> None:
        try:
            self.document(role, document)
        except Exception as exc:
            self.gap(role, exc)

    def bind_target(self, target: Any) -> None:
        self.target = target
        try:
            witness = _ReplayReferenceWitness()
            self.target_state = _capture_actual_target_state(target, witness=witness)
            if witness.selected:
                self.target_witness = witness
        except Exception as exc:
            self.gap("instantiated-target-state", exc)
            raise SourcePreparationError(
                "Actual instantiated target needs an exact state capture adapter",
                category="replay-lane-gap",
            ) from exc
        self.save_document("instantiated-target-state", self.target_state)
        if self.target_witness is not None:
            self.save_document(
                "instantiated-target-references", self.target_witness.document()
            )
        if (
            self.records.get("instantiated-target-state", {}).get("status")
            != "captured"
        ):
            raise SourcePreparationError(
                "Actual target state capture is unavailable", category="replay-lane-gap"
            )

    def check_target(self, role: str) -> bool:
        try:
            witness = (
                _ReplayReferenceWitness() if self.target_witness is not None else None
            )
            observed = _capture_actual_target_state(self.target, witness=witness)
            if self.target_witness is not None:
                assert witness is not None
                self.target_witness.require_same(witness)
            if observed != self.target_state:
                self.save_document(role + "-changed-state", observed)
                raise SourcePreparationError(
                    "Actual instantiated target changed weights/state; an exact reset/capture schedule is required",
                    category="replay-lane-gap",
                )
        except Exception as exc:
            self.gap(role + "-state", exc)
            return False
        self.records[role + "-state"] = {"status": "unchanged"}
        return True

    def require_stable_target(self, role: str) -> None:
        if not self.check_target(role):
            raise SourcePreparationError(
                "Actual instantiated target state is not preserved; later source invocations were not attempted",
                category="replay-lane-gap",
            )

    def arrays(self, role: str, values: Any) -> tuple[np.ndarray, ...]:
        payloads = []
        properties = []
        for value in values:
            if is_selected_prng_key_dtype(getattr(value, "dtype", None)):
                array = np.array(selected_prng_key_words(value), copy=True)
                properties.append(
                    {
                        "dtype": value.dtype.name,
                        "shape": list(value.shape),
                        "weak_type": bool(getattr(value, "weak_type", False)),
                        "encoding": "threefry-key-words",
                        "physical_dtype": array.dtype.name,
                        "physical_shape": list(array.shape),
                    }
                )
            else:
                array = np.array(value, copy=True)
                properties.append(
                    {
                        "dtype": array.dtype.name,
                        "shape": list(array.shape),
                        "weak_type": bool(getattr(value, "weak_type", False)),
                    }
                )
            payloads.append(array)
        arrays = tuple(payloads)
        path = self.directory / (role + ".npz")
        with path.open("xb") as stream:
            np.savez(stream, **{f"value_{i}": v for i, v in enumerate(arrays)})
        self.records[role] = {
            "status": "captured",
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "properties": properties,
        }
        for array in arrays:
            array.setflags(write=False)
        return arrays

    def document(self, role: str, document: Any) -> None:
        path = self.directory / (role + ".json")
        payload = json.dumps(document, sort_keys=True, separators=(",", ":"))
        with path.open("x") as stream:
            stream.write(payload + "\n")
        self.records[role] = {
            "status": "captured",
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def failure(self, role: str, exc: Exception) -> None:
        self.records[role] = {
            "status": "failed",
            "exception": f"{type(exc).__module__}.{type(exc).__qualname__}",
            "detail": str(exc),
        }
        if type(exc) is SourcePreparationError and "replay_diagnostic" in vars(exc):
            self.records[role]["diagnostic"] = vars(exc)["replay_diagnostic"]

    def observe_facade_return(self, value: Any) -> None:
        # Successful ordinary return is observed before any normalization/read.
        self.facade_value = value
        kind = (
            "proto"
            if isinstance(value, onnx.ModelProto)
            else (
                "ir"
                if isinstance(value, ir.Model)
                else "file" if isinstance(value, str) else "unknown"
            )
        )
        self.records["facade_return"] = {
            "status": "returned",
            "kind": kind,
            "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
        }
        self.records["product_admission"] = "returned-artifact"
        if isinstance(value, str):
            self.records["facade_return"]["path"] = value
            files = []
            self.records["facade_return"]["files"] = files
            for path in (Path(value), Path(value + ".data")):
                if path != Path(value) and not path.exists():
                    continue
                try:
                    with path.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    files.append(
                        {
                            "path": str(path),
                            "bytes": path.stat().st_size,
                            "sha256": digest,
                        }
                    )
                except Exception as exc:
                    self.gap("facade-return-file-evidence", exc)

    def artifact(self, model: onnx.ModelProto) -> None:
        # The actual facade return is retained even if its evidence sink fails.
        self.model = model
        try:
            path = self.directory / "returned.onnx"
            with path.open("xb") as stream:
                stream.write(model.SerializeToString(deterministic=True))
            self.records["artifact"] = {
                "status": "captured",
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        except Exception as exc:
            self.gap("artifact", exc)
        try:
            artifact = self.records.get("artifact", {})
            if artifact.get("status") == "captured":
                # This is the existing serializer's actual normalized return,
                # before the raw callback. Do not serialize another model here.
                self.records["source_structure"]["artifact_binding"][
                    "normalized_return"
                ] = {
                    "sha256": artifact["sha256"],
                    "bytes": Path(artifact["path"]).stat().st_size,
                }
        except Exception as exc:
            self._structure_error("artifact_binding", "normalized-return", exc)


def structured_source_error(exc: Exception) -> dict[str, Any]:
    """Retain owning SourcePrecisionError fields without parsing its message."""
    return {
        name: getattr(exc, name)
        for name in (
            "phase",
            "category",
            "location",
            "primitive",
            "requested_dtypes",
            "precision_params",
            "opset",
            "requirement",
            "action",
        )
        if hasattr(exc, name)
    }


def _selected_native_state(value: Any) -> Any:
    try:
        return selected_native_capture(value)
    except (TypeError, ValueError) as exc:
        raise SourcePreparationError(
            f"Selected native callable state needs an exact adapter: {exc}",
            category="replay-lane-gap",
        ) from exc


def _capture_recipe(value: Any, *, location: str = "callable", depth: int = 0) -> Any:
    """Capture only registered recipe containers and bounded dataclass objects.

    No object is rebuilt, copied into a different model or replaced for export.
    Unknown mutable state, identity tokens and unsupported fields stay lane gaps.
    """
    if depth > 48:
        raise SourcePreparationError(
            "Original recipe nesting needs an adapter", category="replay-lane-gap"
        )
    native = _selected_native_state(value)
    if native is not None:
        return native
    unresolved: list[str] = []
    canonical = canonical_contract(value, unresolved=unresolved)
    if not unresolved:
        return canonical

    def child(item: Any, suffix: str) -> Any:
        return _capture_recipe(item, location=location + "." + suffix, depth=depth + 1)

    typename = f"{type(value).__module__}.{type(value).__qualname__}"
    if (
        is_dataclass(value)
        and not isinstance(value, type)
        and (
            typename.startswith("equinox.nn.")
            or typename
            in {
                "jax2onnx.plugins.examples.eqx.gpt_oss.GPTOSSConfig",
                "jax2onnx.plugins.examples.nnx.gpt_oss_flax.GPTOSSConfig",
            }
        )
    ):
        return {
            "original_type": typename,
            "fields": {
                f.name: child(getattr(value, f.name), f.name) for f in fields(value)
            },
        }
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise SourcePreparationError(
                "Opaque recipe mapping keys need an adapter", category="replay-lane-gap"
            )
        return {key: child(item, key) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return {
            "container": type(value).__name__,
            "values": [child(item, str(i)) for i, item in enumerate(value)],
        }
    if isinstance(value, functools.partial):
        return {
            "partial": child(value.func, "func"),
            "args": child(value.args, "args"),
            "kwargs": child(value.keywords, "kwargs"),
        }
    if typename == "jax2onnx.plugins.plugin_system._ConstructAndCall":
        return {
            "factory": child(value._ctor, "ctor"),
            "dtype": child(value._dtype, "dtype"),
            "args": child(value._args, "args"),
            "kwargs": child(value._kwargs, "kwargs"),
        }
    if isinstance(value, types.FunctionType):
        result = {
            "callable": f"{value.__module__}.{value.__qualname__}",
            "defaults": child(value.__defaults__, "defaults"),
            "kwdefaults": child(value.__kwdefaults__, "kwdefaults"),
        }
        result["captures"] = {
            name: child(cell.cell_contents, "capture." + name)
            for name, cell in zip(
                value.__code__.co_freevars, value.__closure__ or (), strict=True
            )
        }
        return result
    raise SourcePreparationError(
        f"No exact original-object capture adapter for {location}: {typename}",
        category="replay-lane-gap",
    )


_LIBRARY_STATE_NAMESPACES = (
    "jax",
    "jaxlib",
    "numpy",
    "ml_dtypes",
    "flax",
    "equinox",
    "math",
    "operator",
    "functools",
)


def _library_namespace(name: str) -> bool:
    return any(
        name == prefix or name.startswith(prefix + ".")
        for prefix in _LIBRARY_STATE_NAMESPACES
    )


def _imported_executable_binding(value: Any) -> dict[str, str] | None:
    """Recognize loaded module API functions, never models or bound owners.

    A namespace is only a lookup boundary. Exact executable types and identity
    at the defining module's top-level export are both required. Classes and
    owned methods always need actual state capture or an explicit lane gap.
    """
    owner = getattr(value, "__self__", None)
    if owner is not None and not isinstance(owner, types.ModuleType):
        return None
    executable_types = {
        types.FunctionType,
        types.BuiltinFunctionType,
        np.ufunc,
        type(np.sum),
        type(jax.numpy.tanh),
        jax.custom_jvp,
        jax.custom_vjp,
    }
    if type(value) not in executable_types:
        return None
    module_name = getattr(value, "__module__", None) or type(value).__module__
    if not _library_namespace(module_name):
        return None
    name = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if not isinstance(name, str) or "." in name:
        return None  # Class/static methods are not top-level API bindings.
    module = sys.modules.get(module_name)
    if not isinstance(module, types.ModuleType) or vars(module).get(name) is not value:
        return None
    return {
        "library_binding": module_name + "." + name,
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
    }


_JAX_SCALAR_META = type(jax.numpy.int32)
_JAX_SCALAR_DTYPE_ALIASES = tuple(
    (name, vars(jax.numpy)[name], vars(vars(jax.numpy)[name])["dtype"])
    for name in (
        "bool_",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "bfloat16",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    )
    if name in vars(jax.numpy)
)


def _jax_scalar_dtype_binding(value: Any) -> dict[str, str] | None:
    """Capture only original loaded JAX scalar alias objects and exact dtypes.

    JAX's scalar metaclass compares equal to the NumPy scalar type; identity
    checks and a separate alias record prevent that equality from erasing a
    changed binding. Unknown class/static data remains outside this adapter.
    """
    public = sys.modules.get("jax.numpy")
    defining = sys.modules.get("jax._src.numpy.scalar_types")
    if (
        type(value) is not _JAX_SCALAR_META
        or public is not jax.numpy
        or not isinstance(defining, types.ModuleType)
        or vars(defining).get("_ScalarMeta") is not _JAX_SCALAR_META
    ):
        return None
    for name, original, original_dtype in _JAX_SCALAR_DTYPE_ALIASES:
        if value is not original:
            continue
        canonical = np.dtype("bool" if name == "bool_" else name)
        if (
            vars(public).get(name) is not value
            or vars(defining).get(name) is not value
            or value.__module__ != "jax.numpy"
            or value.__name__ != canonical.type.__name__
            or value.__qualname__ != canonical.type.__name__
            or set(vars(value))
            - {
                "dtype",
                "__module__",
                "__doc__",
                "__dict__",
                "__weakref__",
                "__qualname__",
            }
        ):
            return None
        dtype = vars(value).get("dtype")
        if (
            dtype is not original_dtype
            or type(dtype) is not type(canonical)
            or dtype.type is not canonical.type
            or dtype.str != canonical.str
            or dtype.metadata is not None
            or np.dtype(value).type is not canonical.type
        ):
            return None
        return {
            "jax_scalar_alias": "jax.numpy." + name,
            "metaclass": f"{type(value).__module__}.{type(value).__qualname__}",
            "dtype": canonical.name,
            "dtype_str": canonical.str,
            "numpy_scalar_type": f"{canonical.type.__module__}.{canonical.type.__qualname__}",
        }
    return None


def _capture_actual_target_state(
    target: Any, *, witness: _ReplayReferenceWitness | None = None
) -> Any:
    """Require an actual callable adapter; recursive type tokens are metadata."""
    if isinstance(target, type):
        binding = _jax_scalar_dtype_binding(target)
        if binding is not None:
            return binding
        raise SourcePreparationError(
            "An actual class callable needs an instantiated-target adapter",
            category="replay-lane-gap",
        )
    return _capture_target_state(target, witness=witness)


def _diagnostic_class_symbol(value: type) -> dict[str, str | None]:
    """Read original symbols for identity lookup; bound them before recording."""
    descriptors = vars(type)
    module = descriptors["__module__"].__get__(value)
    qualname = descriptors["__qualname__"].__get__(value)
    return {
        "module": module if type(module) is str else None,
        "qualname": qualname if type(qualname) is str else None,
    }


def _diagnostic_text(value: Any, *, limit: int = 96) -> Any:
    """Bound native strings; retain exact length/digest for truncated labels."""
    if type(value) is not str:
        return None
    if len(value) <= limit:
        return value
    digest = hashlib.sha256()
    for index in range(0, len(value), 4096):
        digest.update(value[index : index + 4096].encode("utf-8", "surrogatepass"))
    return {
        "truncated": True,
        "characters": len(value),
        "prefix": value[: limit - 16],
        "suffix": value[-16:],
        "sha256_utf8_surrogatepass": digest.hexdigest(),
    }


def _diagnostic_module_namespace(value: Any) -> dict[str, Any] | None:
    """Inspect true native ancestry, never an instance's __class__ property."""
    ancestry = vars(type)["__mro__"].__get__(type(value))
    if any(base is types.ModuleType for base in ancestry[:64]):
        return vars(types.ModuleType)["__dict__"].__get__(value)
    return None


def _class_state_diagnostic(
    value: type, *, location: tuple[str, ...], function: Any
) -> dict[str, Any]:
    """Record at most 32 KiB of ASCII JSON for the existing class rejection.

    Labels retain at most 96 characters; path labels retain 64. At most twelve
    path segments survive (first four/last eight if truncated). Original symbols
    alone determine binding identity; truncated labels never become lookup keys.
    """
    symbol = _diagnostic_class_symbol(value)
    module_name, qualname = symbol["module"], symbol["qualname"]
    module = sys.modules.get(module_name) if module_name is not None else None
    if module_name is None or qualname is None:
        binding = "symbol-unavailable"
    elif "." in qualname:
        binding = "not-top-level"
    elif module is None:
        binding = "module-not-loaded"
    else:
        namespace = _diagnostic_module_namespace(module)
        binding = (
            "module-dictionary-unavailable"
            if namespace is None
            else (
                "missing"
                if qualname not in namespace
                else "exact" if namespace[qualname] is value else "different"
            )
        )
    callable_state = None
    if type(function) is types.FunctionType:
        callable_state = {
            "module": _diagnostic_text(function.__module__),
            "qualname": _diagnostic_text(function.__qualname__),
            "code_sha256": _diagnostic_text(
                canonical_code_fingerprint(function.__code__)
            ),
        }
    retained = location if len(location) <= 12 else location[:4] + location[-8:]
    diagnostic = {
        "kind": "unsupported-function-class-state",
        "class": {name: _diagnostic_text(text) for name, text in symbol.items()},
        "metaclass": {
            name: _diagnostic_text(text)
            for name, text in _diagnostic_class_symbol(type(value)).items()
        },
        "callable": callable_state,
        "path": [_diagnostic_text(segment, limit=64) for segment in retained],
        "defining_module_binding": binding,
    }
    if len(location) > 12:
        diagnostic["path_truncation"] = {
            "total_segments": len(location),
            "retained_prefix": 4,
            "retained_suffix": 8,
        }
    payload = json.dumps(diagnostic, ensure_ascii=True, separators=(",", ":"))
    if len(payload.encode("ascii")) > 32768:
        return {
            "kind": "unsupported-function-class-state",
            "unavailable": "diagnostic-payload-limit",
        }
    return diagnostic


def _function_owned_data(
    value: Any,
    child: Any,
    *,
    depth: int = 0,
    location: tuple[str, ...] = ("root",),
    function: Any = None,
    witness: _ReplayReferenceWitness | None = None,
) -> Any:
    if witness is not None:
        witness.take(location, value)
    native = _selected_native_state(value)
    if native is not None:
        return native
    if depth > 48:
        raise SourcePreparationError(
            "Referenced function data nesting needs an exact adapter",
            category="replay-lane-gap",
        )
    if isinstance(value, types.ModuleType):
        raise SourcePreparationError(
            "Referenced function data contains module state without an exact adapter",
            category="replay-lane-gap",
        )
    if isinstance(value, type) and value not in {bool, int, float, complex, str, bytes}:
        alias = _jax_scalar_dtype_binding(value)
        if alias is not None:
            return alias
        try:
            scalar_type = np.dtype(value).type
        except (TypeError, ValueError):
            scalar_type = None
        if scalar_type is not value:
            error = SourcePreparationError(
                "Referenced function class/static state needs an exact adapter",
                category="replay-lane-gap",
            )
            try:
                vars(error)["replay_diagnostic"] = _class_state_diagnostic(
                    value, location=location, function=function
                )
            except Exception:
                # Diagnostic enrichment must never replace the original gap.
                vars(error)["replay_diagnostic"] = {
                    "kind": "unsupported-function-class-state",
                    "unavailable": "diagnostic-facts-unavailable",
                }
            raise error
    if (
        isinstance(value, types.BuiltinFunctionType)
        and getattr(value, "__self__", None) is not None
        and not isinstance(value.__self__, types.ModuleType)
    ):
        raise SourcePreparationError(
            "Owned native method needs an exact owner/state adapter",
            category="replay-lane-gap",
        )
    if isinstance(value, Mapping):
        return {
            "mapping": sorted(
                [
                    [
                        child(
                            key,
                            location=location + (f"mapping[{index}].key",),
                            function=function,
                        ),
                        _function_owned_data(
                            item,
                            child,
                            depth=depth + 1,
                            location=location + (f"mapping[{index}].value",),
                            function=function,
                            witness=witness,
                        ),
                    ]
                    for index, (key, item) in enumerate(value.items())
                ],
                key=lambda pair: json.dumps(pair[0], sort_keys=True),
            )
        }
    if isinstance(value, (tuple, list)):
        return {
            "container": type(value).__name__,
            "values": [
                _function_owned_data(
                    item,
                    child,
                    depth=depth + 1,
                    location=location + (f"item[{index}]",),
                    function=function,
                    witness=witness,
                )
                for index, item in enumerate(value)
            ],
        }
    return child(value, location=location, function=function)


def _function_global_state(
    fn: Any,
    child: Any,
    *,
    location: tuple[str, ...] = ("root",),
    native_implementation: Any = None,
    witness: _ReplayReferenceWitness | None = None,
) -> dict[str, Any]:
    """Bind referenced data; retain explicit boundaries for executable libraries.

    Direct model data and function attributes use the real state adapter. Library
    namespace attributes are resolved only along bytecode-visible access chains;
    arbitrary module escape or custom class/static dependencies remain gaps.
    """
    if native_implementation is not None:
        return {"selected_native_implementation": native_implementation}
    accesses: dict[str, set[tuple[str, ...]]] = {}

    def inspect_code(code: Any) -> None:
        instructions = list(dis.get_instructions(code))
        for index, instruction in enumerate(instructions):
            if instruction.opname not in {"LOAD_GLOBAL", "LOAD_NAME"}:
                continue
            name = instruction.argval
            if name not in fn.__globals__:
                continue  # Native builtin lookup, not mutable module model data.
            path = []
            for following in instructions[index + 1 :]:
                if following.opname not in {"LOAD_ATTR", "LOAD_METHOD"}:
                    break
                path.append(following.argval)
            accesses.setdefault(name, set()).add(tuple(path))
        for constant in code.co_consts:
            if isinstance(constant, types.CodeType):
                inspect_code(constant)

    def dependency(
        value: Any, *, location: tuple[str, ...], retain: bool = True
    ) -> Any:
        if witness is not None and retain:
            witness.take(location, value)
        native = _selected_native_state(value)
        if native is not None:
            return native
        binding = _imported_executable_binding(value)
        if binding is not None:
            # Imported implementation is owned by the environment/source pins.
            # Model instances and owned methods cannot enter this branch.
            return binding
        return _function_owned_data(
            value,
            child,
            location=location,
            function=fn,
            witness=witness if retain else None,
        )

    inspect_code(fn.__code__)
    result = {}
    for name, paths in sorted(accesses.items()):
        value = fn.__globals__[name]
        if witness is not None:
            witness.take(location + ("global:" + name,), value)
        if not isinstance(value, types.ModuleType):
            result[name] = dependency(value, location=location + ("global:" + name,))
            continue
        if not _library_namespace(value.__name__):
            raise SourcePreparationError(
                f"Referenced module state needs an exact adapter: {value.__name__}",
                category="replay-lane-gap",
            )
        selected = {}
        for path in sorted(paths):
            if not path:
                raise SourcePreparationError(
                    f"Referenced library module {value.__name__} is passed without a bounded attribute path",
                    category="replay-lane-gap",
                )
            member = value
            retained_member = True
            for index, attribute in enumerate(path):
                if isinstance(member, types.ModuleType) and attribute in vars(member):
                    member = vars(member)[attribute]
                    if witness is not None:
                        witness.take(
                            location
                            + ("global:" + name,)
                            + tuple("attribute:" + item for item in path[: index + 1]),
                            member,
                        )
                    continue
                try:
                    native_member = selected_native_capture_member(member, attribute)
                except (TypeError, ValueError) as exc:
                    raise SourcePreparationError(
                        f"Referenced native member changed: {name}.{'.'.join(path)}: {exc}",
                        category="replay-lane-gap",
                    ) from exc
                if native_member is None:
                    raise SourcePreparationError(
                        f"Referenced attribute path needs an exact state adapter: {name}.{'.'.join(path)}",
                        category="replay-lane-gap",
                    )
                if witness is not None and type(native_member) is types.MethodType:
                    witness.take(
                        location + ("global:" + name, "native-owner:" + attribute),
                        native_member.__self__,
                    )
                    witness.take(
                        location + ("global:" + name, "native-method:" + attribute),
                        native_member.__func__,
                    )
                    retained_member = False
                member = native_member
            selected[".".join(path)] = dependency(
                member,
                location=location
                + ("global:" + name,)
                + tuple("attribute:" + attribute for attribute in path),
                retain=retained_member,
            )
        result[name] = {"library_namespace": value.__name__, "attributes": selected}
    return result


def _native_union_class_symbol(value: type[Any]) -> tuple[str, str]:
    return (
        vars(type)["__module__"].__get__(value, type(value)),
        vars(type)["__qualname__"].__get__(value, type(value)),
    )


_NATIVE_UNION_TYPE = types.UnionType
_NATIVE_UNION_ARGS = vars(_NATIVE_UNION_TYPE)["__args__"]
_NATIVE_UNION_ARRAY_SYMBOL = _native_union_class_symbol(_NATIVE_UNION_ARRAY)
_NATIVE_UNION_ARRAY_MODULE = jax
_NATIVE_UNION_ARRAY_META = type(_NATIVE_UNION_ARRAY)
_NATIVE_UNION_ARRAY_META_SYMBOL = _native_union_class_symbol(_NATIVE_UNION_ARRAY_META)
_NATIVE_UNION_ARRAY_META_MODULE = sys.modules[_NATIVE_UNION_ARRAY_META_SYMBOL[0]]


def _capture_native_union(
    value: Any,
    *,
    location: tuple[str, ...],
    witness: _ReplayReferenceWitness | None,
) -> dict[str, Any]:
    """Snapshot only the measured native annotation, never construct a union."""
    if type(value) is not _NATIVE_UNION_TYPE:
        raise SourcePreparationError(
            "Union annotation requires the exact native UnionType",
            category="replay-lane-gap",
        )
    arguments = _NATIVE_UNION_ARGS.__get__(value, _NATIVE_UNION_TYPE)
    if type(arguments) is not tuple or len(arguments) != 2:
        raise SourcePreparationError(
            "Native union arguments need a separately measured replay adapter",
            category="replay-lane-gap",
        )
    array_module = _NATIVE_UNION_ARRAY_MODULE
    meta_module = _NATIVE_UNION_ARRAY_META_MODULE
    if (
        type(array_module) is not types.ModuleType
        or type(meta_module) is not types.ModuleType
    ):
        raise SourcePreparationError(
            "Native union Array module identity changed", category="replay-lane-gap"
        )
    namespace = vars(types.ModuleType)["__dict__"]
    array_namespace = namespace.__get__(array_module, types.ModuleType)
    meta_namespace = namespace.__get__(meta_module, types.ModuleType)
    if (
        jax is not array_module
        or sys.modules.get("jax") is not array_module
        or array_namespace.get("Array") is not _NATIVE_UNION_ARRAY
        or type(_NATIVE_UNION_ARRAY) is not _NATIVE_UNION_ARRAY_META
        or _NATIVE_UNION_ARRAY_SYMBOL
        not in (
            ("jax", "Array"),
            ("jax", "jaxlib._jax.Array"),
        )
        or _native_union_class_symbol(_NATIVE_UNION_ARRAY) != _NATIVE_UNION_ARRAY_SYMBOL
        or _native_union_class_symbol(_NATIVE_UNION_ARRAY_META)
        != _NATIVE_UNION_ARRAY_META_SYMBOL
        or sys.modules.get(_NATIVE_UNION_ARRAY_META_SYMBOL[0]) is not meta_module
        or meta_namespace.get(_NATIVE_UNION_ARRAY_META_SYMBOL[1])
        is not _NATIVE_UNION_ARRAY_META
    ):
        raise SourcePreparationError(
            "Native union Array class or metaclass binding changed",
            category="replay-lane-gap",
        )
    leaves = []
    for index, argument in enumerate(arguments):
        if argument is _NATIVE_UNION_ARRAY:
            leaf = {
                "kind": "native-jax-array-class",
                "symbol": ["jax", "Array"],
                "metaclass": list(_NATIVE_UNION_ARRAY_META_SYMBOL),
            }
        elif argument is type(None):
            leaf = {"kind": "native-none-type-class"}
        else:
            raise SourcePreparationError(
                "Native union member needs a separately measured replay adapter",
                category="replay-lane-gap",
            )
        leaves.append(leaf)
        if witness is not None:
            witness.take(location + (f"union-arg[{index}]",), argument)
    if witness is not None:
        witness.selected[location] = "native-union-annotation"
        for name, original in (
            ("union", value),
            ("union-args", arguments),
            ("array-module", array_module),
            ("array-module-dict", array_namespace),
            ("array-class", _NATIVE_UNION_ARRAY),
            ("array-metaclass", _NATIVE_UNION_ARRAY_META),
            ("array-metaclass-module", meta_module),
            ("array-metaclass-module-dict", meta_namespace),
        ):
            witness.take(location + (name,), original)
    return {"kind": "native-union-type", "args": leaves}


class _NativeTypingAnnotationBoundary:
    """Only the measured native int/Literal annotation and its stdlib storage."""

    _IMMUTABLE_UNION_FIELDS = (
        "__repr__",
        "__hash__",
        "__getattribute__",
        "__lt__",
        "__le__",
        "__eq__",
        "__ne__",
        "__gt__",
        "__ge__",
        "__or__",
        "__ror__",
        "__getitem__",
        "__mro_entries__",
        "__class_getitem__",
        "__args__",
        "__name__",
        "__qualname__",
        "__origin__",
        "__parameters__",
        "__doc__",
    )

    def __init__(self) -> None:
        self.type = type
        self.module_descriptor = vars(types.ModuleType)["__dict__"]
        self.type_descriptors = {
            name: vars(type)[name]
            for name in ("__dict__", "__mro__", "__module__", "__qualname__")
        }
        self.modules = (
            ("typing", _native_typing),
            ("builtins", _native_builtins),
            ("types", types),
            ("functools", functools),
        )
        self.namespaces = {
            name: self.module_descriptor.__get__(module, types.ModuleType)
            for name, module in self.modules
        }
        self.variant = sys.version_info[:2]
        self.immutable_union = self.variant == (3, 14)
        self.global_literal_cache = self.variant in ((3, 13), (3, 14))
        self.function_type_parameters = self.variant in ((3, 13), (3, 14))
        if self.variant not in ((3, 11), (3, 12), (3, 13), (3, 14)):
            # Keep unrelated replay imports usable. Any selected alias on an
            # unmeasured interpreter still needs its exact storage adapter.
            self.union_class = self.namespaces["typing"].get("_UnionGenericAlias")
            self.literal_class = self.namespaces["typing"].get("_LiteralGenericAlias")
            return
        self.builtin_types = tuple(
            (name, self.namespaces["builtins"][name])
            for name in ("type", "int", "bool", "str", "tuple", "list", "dict")
            + (("object",) if self.immutable_union else ())
        )
        self.native_int = type(0)
        literal_origin_name = (
            "_TypedCacheSpecialForm"
            if self.global_literal_cache
            else "_LiteralSpecialForm"
        )
        self.classes = tuple(
            (name, self.namespaces["typing"][name])
            for name in (
                ("Union",) if self.immutable_union else ("_UnionGenericAlias",)
            )
            + (
                "_LiteralGenericAlias",
                "_GenericAlias",
                "_BaseGenericAlias",
                "_Final",
                "_NotIterable",
                "_SpecialForm",
                literal_origin_name,
            )
        )
        classes = dict(self.classes)
        self.union_class = classes[
            "Union" if self.immutable_union else "_UnionGenericAlias"
        ]
        self.literal_class = classes["_LiteralGenericAlias"]
        self.special_class = classes["_SpecialForm"]
        self.literal_origin_class = classes[literal_origin_name]
        self.alias_dictionaries = (
            ()
            if self.immutable_union
            else (
                (
                    self.union_class,
                    self.union_class,
                    vars(self.union_class)["__dict__"],
                ),
            )
        ) + (
            (
                self.literal_class,
                classes["_BaseGenericAlias"],
                vars(classes["_BaseGenericAlias"])["__dict__"],
            ),
        )
        self.union_descriptors = (
            {
                name: vars(self.union_class)[name]
                for name in ("__origin__", "__args__", "__parameters__")
            }
            if self.immutable_union
            else {}
        )
        self.union_layout_descriptors = (
            {name: vars(type)[name] for name in ("__flags__", "__dictoffset__")}
            if self.immutable_union
            else {}
        )
        self.literal_dictionary = vars(self.literal_origin_class)["__dict__"]
        self.origin_slots = {
            name: vars(self.special_class)[name]
            for name in ("__doc__", "_name", "_getitem")
        }
        self.function_type = types.FunctionType
        self.function_descriptors = {
            name: vars(types.FunctionType)[name]
            for name in (
                "__name__",
                "__module__",
                "__qualname__",
                "__doc__",
                "__code__",
                "__globals__",
                "__defaults__",
                "__kwdefaults__",
                "__closure__",
                "__dict__",
                "__annotations__",
            )
            + (("__type_params__",) if self.function_type_parameters else ())
            + (("__annotate__",) if self.immutable_union else ())
        }
        self.cell_type = types.CellType
        self.cell_contents = vars(types.CellType)["cell_contents"]
        self.origins = tuple(
            (name, self.namespaces["typing"][name]) for name in ("Union", "Literal")
        )
        self.native_types = tuple(
            (name, self.namespaces["types"][name])
            for name in (
                "ModuleType",
                "FunctionType",
                "CodeType",
                "CellType",
                "GetSetDescriptorType",
                "MemberDescriptorType",
            )
            + (("UnionType",) if self.immutable_union else ())
        )
        self.cache_type = self.namespaces["functools"]["_lru_cache_wrapper"]
        self.cache_dictionary = vars(self.cache_type)["__dict__"]
        if self.global_literal_cache:
            # The selected wrappers read this exact global dictionary. Retain
            # only their selected key/value edges, not unrelated cache state.
            self.typing_caches = self.namespaces["typing"].get("_caches")
            if self.type(self.typing_caches) is not dict:
                self.fail("native cache dictionary")
        self.alias_fields = (
            (
                "_inst",
                "_name",
                "__origin__",
                "__slots__",
                "__args__",
                "__parameters__",
            )
            + (("_paramspec_tvars",) if self.variant == (3, 11) else ())
            + ("__module__",)
        )
        self.baseline_state, self.baseline_edges = self.read_boundary()

    @staticmethod
    def fail(detail: str) -> Any:
        raise SourcePreparationError(
            "Native typing annotation " + detail + " needs an exact adapter",
            category="replay-lane-gap",
        )

    def dictionary(self, value: Any, keys: tuple[str, ...]) -> dict[str, Any]:
        if self.type(value) is not dict:
            self.fail("dictionary type")
        actual = tuple(value)
        if any(self.type(key) is not str for key in actual) or actual != keys:
            self.fail("ordered storage fields")
        return value

    def scalar(self, value: Any) -> Any:
        kind = self.type(value)
        if value is None:
            return {"none": True}
        if kind is bool:
            return {"bool": value}
        if kind is self.native_int:
            return {"int": value}
        if kind is str:
            return {"str": value}
        self.fail("native scalar metadata")

    def read_boundary(self) -> tuple[dict[str, Any], list[tuple[str, object]]]:
        edges: list[tuple[str, object]] = []

        def take(role: str, value: object) -> None:
            edges.append((role, value))

        for name, module in self.modules:
            if sys.modules.get(name) is not module:
                self.fail("module binding")
            namespace = self.module_descriptor.__get__(module, self.type(module))
            if namespace is not self.namespaces[name]:
                self.fail("module dictionary")
            take("module:" + name, module)
            take("namespace:" + name, namespace)
        if (
            _native_typing is not self.modules[0][1]
            or _native_builtins is not self.modules[1][1]
            or types is not self.modules[2][1]
            or functools is not self.modules[3][1]
        ):
            self.fail("import binding")
        for name, original in self.builtin_types:
            if self.namespaces["builtins"].get(name) is not original:
                self.fail("native builtin binding")
            take("builtin:" + name, original)
        if self.namespaces["builtins"].get("int") is not self.native_int:
            self.fail("native int binding")
        for name, original in self.native_types:
            if self.namespaces["types"].get(name) is not original:
                self.fail("native descriptor class binding")
            take("types:" + name, original)
        type_dictionary = self.type_descriptors["__dict__"].__get__(
            self.type, self.type
        )
        for name, descriptor in self.type_descriptors.items():
            if type_dictionary.get(name) is not descriptor:
                self.fail("native type descriptor")
            take("type-descriptor:" + name, descriptor)
        if vars(types.ModuleType).get("__dict__") is not self.module_descriptor:
            self.fail("native module descriptor")
        take("module-dictionary-descriptor", self.module_descriptor)
        class_state = []
        for name, original in self.classes:
            if (
                self.namespaces["typing"].get(name) is not original
                or self.type(original) is not self.type
            ):
                self.fail("native typing class binding")
            namespace = self.type_descriptors["__dict__"].__get__(original, self.type)
            symbol = [
                self.type_descriptors[field].__get__(original, self.type)
                for field in ("__module__", "__qualname__")
            ]
            if any(self.type(item) is not str for item in symbol) or symbol != [
                "typing",
                name,
            ]:
                self.fail("native class symbol")
            take("class:" + name, original)
            fields = []
            for key, member in namespace.items():
                if self.type(key) is not str:
                    self.fail("native class field")
                fields.append(key)
                take("class:" + name + ":" + key, member)
            mro = self.type_descriptors["__mro__"].__get__(original, self.type)
            take("class:" + name + ":mro", mro)
            class_state.append([symbol, fields])
        immutable_union_state = None
        if self.immutable_union:
            namespace = self.type_descriptors["__dict__"].__get__(
                self.union_class, self.type
            )
            mro = self.type_descriptors["__mro__"].__get__(self.union_class, self.type)
            if (
                self.union_class is not _NATIVE_UNION_TYPE
                or self.namespaces["types"].get("UnionType") is not self.union_class
                or tuple(namespace) != self._IMMUTABLE_UNION_FIELDS
                or self.type(mro) is not tuple
                or len(mro) != 2
                or mro[0] is not self.union_class
                or mro[1] is not self.namespaces["builtins"]["object"]
            ):
                self.fail("native immutable Union class layout")
            layout = {}
            for name, descriptor in self.union_layout_descriptors.items():
                if vars(type).get(name) is not descriptor:
                    self.fail("native Union type layout descriptor")
                item = descriptor.__get__(self.union_class, self.type)
                if self.type(item) is not self.native_int:
                    self.fail("native Union type layout scalar")
                layout[name] = item
                take("union-layout-descriptor:" + name, descriptor)
            flags = layout["__flags__"]
            if (
                not flags & (1 << 8)
                or flags & ((1 << 9) | (1 << 10))
                or layout["__dictoffset__"] != 0
            ):
                self.fail("native immutable Union flags or dictionary")
            for name, descriptor in self.union_descriptors.items():
                expected = (
                    types.MemberDescriptorType
                    if name == "__args__"
                    else types.GetSetDescriptorType
                )
                if (
                    namespace.get(name) is not descriptor
                    or self.type(descriptor) is not expected
                ):
                    self.fail("native immutable Union storage descriptor")
                take("union-storage-descriptor:" + name, descriptor)
            immutable_union_state = {
                "symbol": ["typing", "Union"],
                "namespace_fields": list(self._IMMUTABLE_UNION_FIELDS),
                "storage_fields": list(self.union_descriptors),
                "immutable": True,
                "instance_dictionary": "absent",
            }
        for cls, owner, descriptor in self.alias_dictionaries:
            if (
                self.type_descriptors["__dict__"]
                .__get__(owner, self.type)
                .get("__dict__")
                is not descriptor
            ):
                self.fail("native alias dictionary descriptor")
            take(
                "alias-descriptor:"
                + self.type_descriptors["__qualname__"].__get__(cls, self.type),
                descriptor,
            )
        if (
            vars(self.literal_origin_class).get("__dict__")
            is not self.literal_dictionary
        ):
            self.fail("native Literal dictionary descriptor")
        take("literal-dictionary-descriptor", self.literal_dictionary)
        for name, descriptor in self.origin_slots.items():
            if vars(self.special_class).get(name) is not descriptor:
                self.fail("native origin slot descriptor")
            take("origin-slot:" + name, descriptor)
        for name, descriptor in self.function_descriptors.items():
            if vars(self.function_type).get(name) is not descriptor:
                self.fail("native function descriptor")
            take("function-descriptor:" + name, descriptor)
        if vars(self.cell_type).get("cell_contents") is not self.cell_contents:
            self.fail("native cell descriptor")
        take("cell-contents-descriptor", self.cell_contents)
        if (
            self.namespaces["functools"].get("_lru_cache_wrapper")
            is not self.cache_type
        ):
            self.fail("native Literal cache class")
        if vars(self.cache_type).get("__dict__") is not self.cache_dictionary:
            self.fail("native Literal cache descriptor")
        take("cache-class", self.cache_type)
        take("cache-dictionary-descriptor", self.cache_dictionary)

        if self.global_literal_cache:
            if self.namespaces["typing"].get("_caches") is not self.typing_caches:
                self.fail("native cache dictionary binding")
            take("typing-cache-dictionary", self.typing_caches)

        def function(
            value: Any, role: str, mode: str, *, typed: bool = True
        ) -> dict[str, Any]:
            if self.type(value) is not self.function_type:
                self.fail("native getitem function")
            take(role, value)
            if self.immutable_union:
                annotate = self.function_descriptors["__annotate__"].__get__(
                    value, self.function_type
                )
                if annotate is not None:
                    self.fail("native getitem lazy annotation callback")
            fields = {
                name: descriptor.__get__(value, self.function_type)
                for name, descriptor in self.function_descriptors.items()
            }
            expected_name = (
                "<lambda>"
                if mode == "parameters"
                else (
                    "__getitem__"
                    if mode in ("union-method", "union-method-body", "literal-method")
                    else ("Union" if mode == "union" else "Literal")
                )
            )
            expected_module = "functools" if mode == "parameters" else "typing"
            if (
                self.type(fields["__name__"]) is not str
                or self.type(fields["__module__"]) is not str
                or fields["__name__"] != expected_name
                or fields["__module__"] != expected_module
            ):
                self.fail("native getitem symbol")
            if mode in ("union-method", "union-method-body", "literal-method"):
                expected_qualname = (
                    "_TypedCacheSpecialForm.__getitem__"
                    if mode == "literal-method"
                    else "_SpecialForm.__getitem__"
                )
                if (
                    self.type(fields["__qualname__"]) is not str
                    or fields["__qualname__"] != expected_qualname
                ):
                    self.fail("native getitem qualified symbol")
            if fields["__globals__"] is not self.namespaces[expected_module]:
                self.fail("native getitem globals")
            code = fields["__code__"]
            if self.type(code) is not types.CodeType:
                self.fail("native getitem code")
            metadata = {}
            for key in ("__name__", "__module__", "__qualname__", "__doc__"):
                metadata[key] = self.scalar(fields[key])
            for key in ("__defaults__", "__kwdefaults__"):
                if fields[key] is not None:
                    self.fail("native getitem defaults")
                metadata[key] = {"none": True}
            self.dictionary(fields["__annotations__"], ())
            if self.immutable_union:
                metadata["__annotate__"] = {"none": True}
            if self.function_type_parameters:
                parameters = fields["__type_params__"]
                if self.type(parameters) is not tuple or parameters:
                    self.fail("native function type parameters")
                metadata["__type_params__"] = []
            attributes = self.dictionary(
                fields["__dict__"],
                ("__wrapped__",) if mode in ("literal", "union-method") else (),
            )
            for key, item in fields.items():
                take(role + ":" + key, item)
            metadata["code_sha256"] = canonical_code_fingerprint(code)
            metadata["function_dictionary_keys"] = list(attributes)
            closure = fields["__closure__"]
            if mode in ("union", "body", "union-method-body", "literal-method"):
                if closure is not None:
                    self.fail("native getitem closure")
                metadata["closure"] = None
            else:
                expected_freevars = (
                    ("maxsize", "typed")
                    if mode == "parameters"
                    else (("cached", "func") if self.variant == (3, 11) else ("func",))
                )
                if (
                    self.type(closure) is not tuple
                    or code.co_freevars != expected_freevars
                    or len(closure) != len(expected_freevars)
                ):
                    self.fail("native getitem closure schema")
                values = []
                for index, cell in enumerate(closure):
                    if self.type(cell) is not self.cell_type:
                        self.fail("native closure cell")
                    item = self.cell_contents.__get__(cell, self.cell_type)
                    take(role + f":cell[{index}]", cell)
                    take(role + f":cell-value[{index}]", item)
                    values.append(item)
                if mode == "parameters":
                    if (
                        self.type(values[0]) is not self.native_int
                        or values[0] != 128
                        or values[1] is not typed
                    ):
                        self.fail("native Literal cache parameters")
                    metadata["closure"] = [self.scalar(item) for item in values]
                else:
                    body = values[-1]
                    if attributes["__wrapped__"] is not body:
                        self.fail("native Literal wrapped function")
                    take(role + ":wrapped", body)
                    body_mode = (
                        "union-method-body" if mode == "union-method" else "body"
                    )
                    metadata["body"] = function(body, role + ":body", body_mode)
                    if self.variant == (3, 11):
                        metadata["cache"] = cache(values[0], body, role + ":cache")
                    elif self.global_literal_cache:
                        # The wrapper's actual globals and closure key are
                        # already pinned above; never invoke the cache itself.
                        for key, selected in dict.items(self.typing_caches):
                            if key is body:
                                break
                        else:
                            self.fail("native selected cache entry")
                        take(role + ":cache-key", body)
                        take(role + ":cache-value", selected)
                        metadata["cache"] = cache(
                            selected,
                            body,
                            role + ":cache",
                            typed=mode != "union-method",
                        )
                    metadata["closure"] = list(expected_freevars)
            return metadata

        def cache(
            value: Any, body: Any, role: str, *, typed: bool = True
        ) -> dict[str, Any]:
            if self.type(value) is not self.cache_type:
                self.fail("native Literal cache wrapper")
            take(role, value)
            namespace = self.dictionary(
                self.cache_dictionary.__get__(value, self.cache_type),
                (
                    "cache_parameters",
                    "__module__",
                    "__name__",
                    "__qualname__",
                    "__doc__",
                    "__annotate__" if self.immutable_union else "__annotations__",
                )
                + (("__type_params__",) if self.function_type_parameters else ())
                + ("__wrapped__",),
            )
            take(role + ":dictionary", namespace)
            if namespace["__wrapped__"] is not body:
                self.fail("native Literal cache wrapped function")
            result = {}
            for key, item in namespace.items():
                take(role + ":field:" + key, item)
                if key == "cache_parameters":
                    result[key] = function(
                        item, role + ":parameters", "parameters", typed=typed
                    )
                elif key == "__wrapped__":
                    result[key] = (
                        "native-selected-body"
                        if self.global_literal_cache
                        else "native-Literal-body"
                    )
                elif key == "__annotate__":
                    if item is not None:
                        self.fail("native cache lazy annotation callback")
                    result[key] = {"none": True}
                elif key == "__annotations__":
                    self.dictionary(item, ())
                    result[key] = []
                elif key == "__type_params__":
                    if self.type(item) is not tuple or item:
                        self.fail("native cache type parameters")
                    result[key] = []
                else:
                    result[key] = self.scalar(item)
            return result

        origins = []
        for name, origin in self.origins:
            if name == "Union" and self.immutable_union:
                if (
                    self.namespaces["typing"].get("Union") is not self.union_class
                    or origin is not self.union_class
                ):
                    self.fail("native public immutable Union binding")
                take("origin:Union", origin)
                origins.append(["Union", {"native_class": ["typing", "Union"]}])
                continue
            expected_class = (
                self.special_class if name == "Union" else self.literal_origin_class
            )
            if (
                self.namespaces["typing"].get(name) is not origin
                or self.type(origin) is not expected_class
            ):
                self.fail("native public origin binding")
            take("origin:" + name, origin)
            native_name = self.origin_slots["_name"].__get__(origin, expected_class)
            if self.type(native_name) is not str or native_name != name:
                self.fail("native origin name")
            take("origin:" + name + ":name", native_name)
            if name == "Union":
                doc = self.origin_slots["__doc__"].__get__(origin, expected_class)
                if self.type(doc) is not str:
                    self.fail("native Union doc slot")
                storage = {"doc_slot": self.scalar(doc), "dictionary": "absent"}
                take("origin:" + name + ":doc", doc)
            else:
                try:
                    self.origin_slots["__doc__"].__get__(origin, expected_class)
                except AttributeError:
                    pass
                else:
                    self.fail("native Literal inherited doc slot must be absent")
                namespace = self.dictionary(
                    self.literal_dictionary.__get__(origin, expected_class),
                    ("__doc__",),
                )
                doc = namespace["__doc__"]
                if self.type(doc) is not str:
                    self.fail("native Literal doc dictionary")
                storage = {
                    "doc_slot": "absent",
                    "dictionary": [["__doc__", self.scalar(doc)]],
                }
                take("origin:" + name + ":dictionary", namespace)
                take("origin:" + name + ":doc", doc)
            getitem = self.origin_slots["_getitem"].__get__(origin, expected_class)
            storage["getitem"] = function(
                getitem, "origin:" + name + ":getitem", name.lower()
            )
            if self.global_literal_cache:
                method = vars(expected_class).get("__getitem__")
                storage["method"] = function(
                    method,
                    "origin:" + name + ":method",
                    "union-method" if name == "Union" else "literal-method",
                )
            origins.append([name, storage])
        state: dict[str, Any] = {
            "python": list(self.variant),
            "classes": class_state,
            "origins": origins,
        }
        if immutable_union_state is not None:
            state["immutable_union"] = immutable_union_state
        return state, edges

    def capture(
        self,
        value: Any,
        *,
        location: tuple[str, ...],
        witness: _ReplayReferenceWitness | None,
    ) -> dict[str, Any]:
        if self.variant not in ((3, 11), (3, 12), (3, 13), (3, 14)):
            self.fail("unmeasured interpreter alias storage")
        state, edges = self.read_boundary()
        if state != self.baseline_state or len(edges) != len(self.baseline_edges):
            self.fail("native implementation metadata changed")
        for (role, original), (current_role, current) in zip(
            self.baseline_edges, edges, strict=True
        ):
            if role != current_role or original is not current:
                self.fail("native implementation identity changed")
        if witness is not None:
            witness.selected[location] = "native-typing-annotation"
            for role, original in edges:
                witness.take(location + ("native-typing:" + role,), original)
        return {
            "kind": "native-typing-annotation",
            "alias": self.alias(value, location, witness),
            "native_boundary": state,
        }

    def alias(
        self,
        value: Any,
        location: tuple[str, ...],
        witness: _ReplayReferenceWitness | None,
    ) -> dict[str, Any]:
        cls = self.type(value)
        if cls is self.union_class:
            if self.immutable_union:
                return self.immutable_alias(value, location, witness)
            kind, origin = "union", self.origins[0][1]
        elif cls is self.literal_class:
            kind, origin = "literal", self.origins[1][1]
        else:
            self.fail("exact alias class")
        descriptor = next(
            descriptor
            for alias_class, _, descriptor in self.alias_dictionaries
            if alias_class is cls
        )
        namespace = self.dictionary(descriptor.__get__(value, cls), self.alias_fields)
        if (
            namespace["_inst"] is not True
            or namespace["_name"] is not None
            or namespace["__slots__"] is not None
            or namespace["__origin__"] is not origin
        ):
            self.fail("alias flags or origin")
        parameters = namespace["__parameters__"]
        if self.type(parameters) is not tuple or len(parameters) != 0:
            self.fail("alias parameters")
        if self.variant == (3, 11) and namespace["_paramspec_tvars"] is not False:
            self.fail("native3.11 alias paramspec flag")
        if (
            self.type(namespace["__module__"]) is not str
            or namespace["__module__"] != "typing"
        ):
            self.fail("alias module")
        arguments = namespace["__args__"]
        if self.type(arguments) is not tuple or len(arguments) != (
            2 if kind == "union" else 1
        ):
            self.fail("ordered alias arguments")
        if witness is not None:
            witness.take(location + ("typing-alias",), value)
            witness.take(location + ("typing-dictionary",), namespace)
            for key, item in namespace.items():
                witness.take(location + ("typing-key:" + key,), key)
                witness.take(location + ("typing-field:" + key,), item)
        leaves = self.arguments(arguments, kind, location, witness)
        fields = []
        for key, item in namespace.items():
            if key == "__origin__":
                encoded = {"native-origin": kind}
            elif key == "__args__":
                encoded = leaves
            elif key == "__parameters__":
                encoded = []
            else:
                encoded = self.scalar(item)
            fields.append([key, encoded])
        return {"kind": kind, "fields": fields}

    def matches(self, value: Any) -> bool:
        cls = self.type(value)
        if cls is self.literal_class:
            return True
        if cls is not self.union_class:
            return False
        if not self.immutable_union:
            return True
        # In 3.14 typing.Union and Array|None share the same immutable C type.
        # Read only its pinned member storage to select the bounded grammar;
        # no parameter computation, constructor, equality or foreign callbacks.
        if (
            _NATIVE_UNION_TYPE is not self.union_class
            or _NATIVE_UNION_ARGS is not self.union_descriptors["__args__"]
            or self.type_descriptors["__dict__"]
            .__get__(self.union_class, self.type)
            .get("__args__")
            is not _NATIVE_UNION_ARGS
        ):
            self.fail("native immutable Union selector descriptor")
        arguments = _NATIVE_UNION_ARGS.__get__(value, _NATIVE_UNION_TYPE)
        return (
            self.type(arguments) is tuple
            and len(arguments) == 2
            and (
                (
                    arguments[0] is self.native_int
                    and self.type(arguments[1]) is self.literal_class
                )
                or (
                    self.type(arguments[0]) is self.literal_class
                    and arguments[1] is self.native_int
                )
            )
        )

    def immutable_alias(
        self,
        value: Any,
        location: tuple[str, ...],
        witness: _ReplayReferenceWitness | None,
    ) -> dict[str, Any]:
        arguments = self.union_descriptors["__args__"].__get__(value, self.union_class)
        if self.type(arguments) is not tuple or len(arguments) != 2:
            self.fail("ordered immutable Union arguments")
        # Validate the complete actual Literal namespace before touching the
        # C parameter getset, which can consult its member annotations.
        leaves = self.arguments(arguments, "union", location, witness)
        origin = self.union_descriptors["__origin__"].__get__(value, self.union_class)
        parameters = self.union_descriptors["__parameters__"].__get__(
            value, self.union_class
        )
        if origin is not self.union_class:
            self.fail("immutable Union origin")
        if self.type(parameters) is not tuple or parameters:
            self.fail("immutable Union parameters")
        if witness is not None:
            witness.take(location + ("typing-alias",), value)
            witness.take(location + ("typing-union-class",), self.union_class)
            for name, item in (
                ("__origin__", origin),
                ("__args__", arguments),
                ("__parameters__", parameters),
            ):
                witness.take(location + ("typing-field:" + name,), item)
                witness.take(
                    location + ("typing-union-descriptor:" + name,),
                    self.union_descriptors[name],
                )
        return {
            "kind": "union",
            "storage": "native-immutable-union",
            "fields": [
                ["__origin__", {"native-origin": "union"}],
                ["__args__", leaves],
                ["__parameters__", []],
            ],
        }

    def arguments(
        self,
        arguments: tuple[Any, ...],
        kind: str,
        location: tuple[str, ...],
        witness: _ReplayReferenceWitness | None,
    ) -> list[dict[str, Any]]:
        leaves = []
        for index, argument in enumerate(arguments):
            if witness is not None:
                witness.take(location + (f"typing-arg[{index}]",), argument)
            if kind == "literal":
                if self.type(argument) is not str or argument != "scalar":
                    self.fail("measured Literal string")
                leaves.append({"kind": "literal-string", "value": argument})
            elif argument is self.native_int:
                leaves.append({"kind": "native-int-class"})
            elif self.type(argument) is self.literal_class:
                leaves.append(
                    self.alias(argument, location + (f"typing-arg[{index}]",), witness)
                )
            else:
                self.fail("measured Union member")
        if kind == "union" and [leaf["kind"] for leaf in leaves] not in (
            ["literal", "native-int-class"],
            ["native-int-class", "literal"],
        ):
            self.fail("exact Union grammar")
        return leaves


_NATIVE_TYPING_BOUNDARY = _NativeTypingAnnotationBoundary()


def _capture_target_state(
    value: Any,
    *,
    depth: int = 0,
    location: tuple[str, ...] = ("root",),
    function: Any = None,
    witness: _ReplayReferenceWitness | None = None,
) -> Any:
    """Snapshot the actual callable, with bounded Eqx/NNX state adapters.

    The snapshot never constructs another model or resets unknown mutable state.
    Unsupported fields remain explicit replay gaps, including new graph layouts.
    """
    if depth > 48:
        raise SourcePreparationError(
            "Target state nesting needs an exact adapter", category="replay-lane-gap"
        )
    if witness is not None:
        witness.take(location, value)

    def child(
        item: Any,
        *segments: str,
        location: tuple[str, ...] = location,
        function: Any = function,
        retain: bool = True,
    ) -> Any:
        return _capture_target_state(
            item,
            depth=depth + 1,
            location=location + segments,
            function=function,
            witness=witness if retain else None,
        )

    if _NATIVE_TYPING_BOUNDARY.matches(value):
        return _NATIVE_TYPING_BOUNDARY.capture(
            value, location=location, witness=witness
        )

    if type(value) is _NATIVE_UNION_TYPE:
        return _capture_native_union(value, location=location, witness=witness)

    native = _selected_native_state(value)
    if native is not None:
        return native
    if is_selected_prng_key_dtype(getattr(value, "dtype", None)):
        words = np.asarray(selected_prng_key_words(value))
        return {
            "dtype": value.dtype.name,
            "shape": list(value.shape),
            "weak_type": bool(getattr(value, "weak_type", False)),
            "key_words_sha256": hashlib.sha256(words.tobytes()).hexdigest(),
        }
    if isinstance(value, jax.tree_util.PyTreeDef):
        return {
            "pytree_node": child(value.node_data(), "pytree-node", retain=False),
            "children": [
                child(tree, f"pytree-child[{index}]", retain=False)
                for index, tree in enumerate(value.children())
            ],
        }
    if isinstance(value, types.MethodType):
        return {
            "bound_target": child(value.__self__, "bound-target"),
            "method": child(value.__func__, "method"),
        }
    if isinstance(value, types.FunctionType):
        try:
            role = native_transformed_function_role(value)
            selected = selected_native_transformed_function(value)
        except (TypeError, ValueError) as exc:
            raise SourcePreparationError(
                "Selected native transformed-function capture needs an exact adapter: "
                + str(exc),
                category="replay-lane-gap",
            ) from exc
        if witness is not None:
            for name, item in (
                ("code", value.__code__),
                ("globals", value.__globals__),
                ("closure", value.__closure__),
            ):
                witness.take(location + (name,), item)
            for name, cell in zip(
                value.__code__.co_freevars, value.__closure__ or (), strict=True
            ):
                witness.take(location + ("cell:" + name,), cell)
                witness.take(location + ("cell-content:" + name,), cell.cell_contents)
            if role is not None:
                witness.selected[location] = role
                assert selected is not None
                witness.native_references.extend(selected.retained)

        def capture_cell(name: str, cell: Any) -> Any:
            try:
                contextual = native_transformed_closure_state(
                    value, name, cell.cell_contents
                )
            except (TypeError, ValueError) as exc:
                raise SourcePreparationError(
                    str(exc), category="replay-lane-gap"
                ) from exc
            if contextual is not None:
                return contextual
            return _function_owned_data(
                cell.cell_contents,
                child,
                location=location + ("capture:" + name,),
                function=value,
                witness=witness,
            )

        result = {
            "callable": f"{value.__module__}.{value.__qualname__}",
            "code_sha256": canonical_code_fingerprint(value.__code__),
            "defaults": _function_owned_data(
                value.__defaults__,
                child,
                location=location + ("defaults",),
                function=value,
                witness=witness,
            ),
            "kwdefaults": _function_owned_data(
                value.__kwdefaults__,
                child,
                location=location + ("kwdefaults",),
                function=value,
                witness=witness,
            ),
            "attributes": _function_owned_data(
                value.__dict__,
                child,
                location=location + ("attributes",),
                function=value,
                witness=witness,
            ),
            "referenced_globals": _function_global_state(
                value,
                child,
                location=location,
                native_implementation=(
                    selected.document if selected is not None else None
                ),
                witness=witness,
            ),
            "captures": {
                name: capture_cell(name, cell)
                for name, cell in zip(
                    value.__code__.co_freevars, value.__closure__ or (), strict=True
                )
            },
        }
        if selected is not None:
            result["metadata"] = {
                name: child(item, "metadata:" + name)
                for name, item in (
                    ("name", value.__name__),
                    ("module", value.__module__),
                    ("qualname", value.__qualname__),
                    ("doc", value.__doc__),
                    ("annotations", value.__annotations__),
                )
            }
        return result
    if isinstance(value, functools.partial):
        return {
            "partial": child(value.func, "partial-function"),
            "args": child(value.args, "partial-args"),
            "kwargs": child(value.keywords, "partial-kwargs"),
            "attributes": _function_owned_data(
                vars(value),
                child,
                location=location + ("partial-attributes",),
                function=function,
                witness=witness,
            ),
        }
    if getattr(value, "__jax2onnx_factory__", False):
        raise SourcePreparationError(
            "An actual target cannot be verified through its constructor recipe",
            category="replay-lane-gap",
        )
    # NNX split reads both static graph configuration and all Variable state.
    # It retains the original target object; no merge/reconstruction is used.
    from flax import nnx

    if isinstance(value, nnx.Module):
        graph, state = nnx.split(value)
        return {
            "target_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "graph": child(graph, "nnx-graph", retain=False),
            "state": child(state, "nnx-state", retain=False),
        }
    if isinstance(value, nnx.Variable):
        raw = value.get_raw_value() if hasattr(value, "get_raw_value") else value.value
        metadata = value.get_metadata()
        return {
            "variable_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": child(raw, "nnx-variable-value"),
            "metadata": child(metadata, "nnx-variable-metadata", retain=False),
        }
    typename = f"{type(value).__module__}.{type(value).__qualname__}"
    if (
        is_dataclass(value)
        and not isinstance(value, type)
        and (
            typename.startswith(
                (
                    "equinox.nn.",
                    "jax2onnx.plugins.examples.eqx.",
                    "flax.nnx.graph.",
                    "flax.nnx.graphlib.",
                )
            )
            or typename in {"jax2onnx.plugins.examples.nnx.gpt_oss_flax.GPTOSSConfig"}
        )
    ):
        return {
            "state_type": typename,
            "fields": {
                field.name: child(getattr(value, field.name), "field:" + field.name)
                for field in fields(value)
            },
        }
    graph_fields = {
        "NodeRef": ("index",),
        "NodeDef": ("type", "index", "outer_index", "num_attributes", "metadata"),
        "VariableDef": ("type", "index", "outer_index", "metadata", "array_refdef"),
        "ArrayRefDef": ("index", "outer_index"),
    }
    if type(value).__module__ == "flaxlib" and type(value).__name__ in graph_fields:
        return {
            "state_type": typename,
            "fields": {
                name: child(getattr(value, name), "field:" + name)
                for name in graph_fields[type(value).__name__]
            },
        }
    if isinstance(value, Mapping):
        return {
            "mapping": sorted(
                [
                    [
                        child(key, f"mapping[{index}].key"),
                        child(item, f"mapping[{index}].value"),
                    ]
                    for index, (key, item) in enumerate(value.items())
                ],
                key=lambda pair: json.dumps(pair[0], sort_keys=True),
            )
        }
    if isinstance(value, (list, tuple)):
        return {
            "container": type(value).__name__,
            "values": [
                child(item, f"item[{index}]") for index, item in enumerate(value)
            ],
        }
    unresolved: list[str] = []
    result = canonical_contract(value, unresolved=unresolved)
    if unresolved:
        raise SourcePreparationError(
            f"No exact instantiated state adapter for {typename}",
            category="replay-lane-gap",
        )
    return result


def replay_adapter(
    row: Any,
    manifest: Mapping[str, Any],
    *,
    _inventory_sha256: str | None = None,
) -> Mapping[str, Any]:
    validate_replay_manifest(manifest, _inventory_sha256)
    adapter = manifest.get("rows", {}).get(f"{row.origin}::{row.ordinal}", {})
    if adapter and adapter.get("contract_sha256") != row.contract_sha256:
        raise SourcePreparationError(
            "Replay adapter does not bind this exact raw contract",
            category="replay-lane-gap",
        )
    if row.unresolved_recipes and not adapter.get("capture_original_object"):
        raise SourcePreparationError(
            "Opaque original recipe has no registered capture adapter",
            category="replay-lane-gap",
        )
    return adapter


def _native_specs(specs: Any) -> tuple[Any, ...]:
    # One shared native scope for the exact original string declarations. Existing
    # symbolic objects keep their original scope and are never string-recreated.
    scope = jax_export.SymbolicScope()
    symbols: dict[str, Any] = {}
    normalized = []
    for spec in specs:
        shape = []
        for dim in spec.shape:
            if isinstance(dim, str):
                if dim not in symbols:
                    parsed = jax_export.symbolic_shape(dim, scope=scope)
                    if len(parsed) != 1:
                        raise SourcePreparationError(
                            "Symbolic declaration is not one dimension",
                            category="replay-lane-gap",
                        )
                    symbols[dim] = parsed[0]
                dim = symbols[dim]
            shape.append(dim)
        normalized.append(
            jax.ShapeDtypeStruct(tuple(shape), spec.dtype, weak_type=spec.weak_type)
        )
    return tuple(normalized)


def _concrete_shapes(
    specs: Any, bindings: Mapping[str, Any]
) -> tuple[tuple[int, ...], ...]:
    shapes = []
    for spec in specs:
        concrete = []
        for dim in spec.shape:
            if isinstance(dim, Integral) and not isinstance(dim, bool):
                extent = int(dim)
            elif isinstance(dim, str) and dim in bindings:
                extent = bindings[dim]
            else:
                raise SourcePreparationError(
                    f"Original symbolic dimension {dim!s} needs an explicit exact sample binding",
                    category="replay-lane-gap",
                )
            if not isinstance(extent, int) or isinstance(extent, bool) or extent < 0:
                raise SourcePreparationError(
                    "Sample binding must be a nonnegative integer",
                    category="replay-lane-gap",
                )
            concrete.append(extent)
        shapes.append(tuple(concrete))
    return tuple(shapes)


def _generated_inputs(shapes: Any, specs: Any, scenario: Any) -> tuple[np.ndarray, ...]:
    digest = hashlib.sha256(
        f"{scenario.provenance.origin}\0{scenario.provenance.ordinal}".encode()
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    generated = []
    for shape, spec in zip(shapes, specs, strict=True):
        dtype = np.dtype(spec.dtype)
        if dtype.kind == "f" or dtype.name == "bfloat16":
            data = rng.standard_normal(shape) * 0.25
        elif dtype.kind in "iu":
            data = rng.integers(0, 5, size=shape)
        elif dtype.kind == "b":
            data = rng.random(shape) > 0.5
        else:
            raise SourcePreparationError(
                f"No reviewed raw sample domain for {dtype}; supply original typed input_values",
                category="replay-lane-gap",
            )
        generated.append(np.asarray(data, dtype=dtype))
    return tuple(generated)


def _physical_arrays(values: Any, indices: Any) -> tuple[np.ndarray, ...]:
    selected = set(indices or ())
    return tuple(
        representation_for_aval(
            jax.ShapeDtypeStruct(value.shape, value.dtype),
            boundary_layout="NCHW" if i in selected else None,
        ).encode_array(value)
        for i, value in enumerate(values)
    )


def _save_physical(
    capture: RawReplayCapture, role: str, values: Any, indices: Any
) -> None:
    try:
        capture.arrays(role, _physical_arrays(values, indices))
    except Exception as exc:
        # A replay adapter failure must not erase a successful native attempt or
        # prevent ordinary product admission from checking its own real options.
        capture.gap(role, exc)


def prepare_raw_source_case(
    metadata: Mapping[str, Any],
    scenario: Any,
    *,
    adapter: Mapping[str, Any],
    capture: RawReplayCapture,
    settings_observer: Any = None,
) -> PreparedSourceCase:
    """Attempt native eager/JIT/trace before any SOURCE shape/profile validator."""
    from tests.extra_tests.converter.source_precision_diagnostics import (
        typed_trace_manifest,
    )

    identity = tuple(
        str(metadata.get(key, "default"))
        for key in ("context", "component", "testcase")
    )
    with source_configuration(scenario):
        settings = SourceSettings.observe()
        if settings_observer is not None:
            settings_observer(settings)
        declared = tuple(scenario.input_specs or ())
        params = dict(metadata.get("input_params") or {})
        names = tuple(params)
        named_specs = []
        for value in params.values():
            spec = (
                value
                if isinstance(value, jax.ShapeDtypeStruct)
                else _typed_input_spec(value)
            )
            if spec is None:
                spec = jax.eval_shape(lambda x: x, value)
            named_specs.append(spec)
        all_declarations = (*declared, *named_specs)
        for spec in all_declarations:
            _require_preserved_dtype(spec.dtype)
        bindings = (adapter.get("samples") or [{}])[0]
        concrete = _concrete_shapes(all_declarations, bindings)
        supplied = scenario.input_values
        if supplied is None:
            supplied = _generated_inputs(concrete[: len(declared)], declared, scenario)
        if len(supplied) != len(declared):
            raise SourcePreparationError(
                "Original input value/declaration arity differs"
            )
        arguments = tuple(
            _argument(value, spec)
            for value, spec in zip(supplied, declared, strict=True)
        )
        named = {
            name: _argument(params[name], spec)
            for name, spec in zip(names, named_specs, strict=True)
        }
        all_arguments = (*arguments, *named.values())
        if tuple(tuple(value.shape) for value in all_arguments) != concrete:
            raise SourcePreparationError(
                "Supplied input bytes do not match the explicit original sample binding",
                category="replay-lane-gap",
            )
        capture.save_arrays("inputs", all_arguments)
        _save_physical(
            capture, "physical-feeds", all_arguments, metadata.get("inputs_as_nchw")
        )
        capture.save_document(
            "sample-bindings",
            {
                "primary": bindings,
                "additional_native_samples": adapter.get("samples", [])[1:],
                "declarations": [
                    _properties(spec).__dict__ for spec in all_declarations
                ],
            },
        )
        target = metadata.get("callable")
        recipe = (
            _capture_recipe(target)
            if adapter.get("capture_original_object")
            else canonical_contract(target, unresolved=[])
        )
        capture.save_document("original-recipe", recipe)
        if getattr(target, "__jax2onnx_factory__", False):
            if scenario.model_dtype is not PRESERVE_FACTORY_DTYPE:
                raise SourcePreparationError(
                    "Raw replay must retain original factory dtype",
                    category="replay-lane-gap",
                )
            target = target.instantiate()
        if not callable(target):
            raise SourcePreparationError(
                "Original recipe did not produce a callable", category="replay-lane-gap"
            )
        capture.bind_target(target)

        def original(*values: Any) -> Any:
            return target(
                *values[: len(arguments)],
                **dict(zip(names, values[len(arguments) :], strict=True)),
            )

        outputs_by_role = {}
        trees = {}

        def native_outputs(role: str, values: Any) -> None:
            result = capture.invoke(
                role,
                lambda: jax.device_get(
                    (original if role.endswith("eager") else jax.jit(original))(*values)
                ),
            )
            if role in capture.native_values:
                try:
                    leaves, tree = jax.tree_util.tree_flatten(result)
                    # Native return status is already retained independently of
                    # flattening, NumPy conversion, encoding and file adapters.
                    outputs_by_role[role] = tuple(leaves)
                    trees[role] = str(tree)
                    capture.save_arrays(role, leaves)
                    _save_physical(
                        capture,
                        role + "-physical-reference",
                        leaves,
                        metadata.get("outputs_as_nchw"),
                    )
                except Exception as exc:
                    capture.gap(role + "-capture", exc)
            capture.require_stable_target(role)

        for role in ("eager", "fresh-jit"):
            native_outputs(role, all_arguments)
        for index, sample_bindings in enumerate(adapter.get("samples", [])[1:], 1):
            label = f"native-sample-{index}"
            if scenario.input_values is not None:
                capture.records[label] = {
                    "status": "unavailable",
                    "detail": "Original supplied input bytes fix the sample shape; no alternate bytes are invented.",
                }
                capture.gap(
                    label + "-capture",
                    SourcePreparationError(
                        "Exact alternate supplied-input bytes are unavailable",
                        category="replay-lane-gap",
                    ),
                )
                continue
            sample_shapes = _concrete_shapes(all_declarations, sample_bindings)
            if sample_shapes[len(declared) :] != concrete[len(declared) :]:
                capture.gap(
                    label + "-capture",
                    SourcePreparationError(
                        "Original named input bytes require an exact alternate parameter adapter",
                        category="replay-lane-gap",
                    ),
                )
                continue
            values = _generated_inputs(
                sample_shapes[: len(declared)], declared, scenario
            )
            sample_args = (
                *(
                    _argument(value, spec)
                    for value, spec in zip(values, declared, strict=True)
                ),
                *named.values(),
            )
            capture.save_arrays(label + "-inputs", sample_args)
            for role in ("eager", "fresh-jit"):
                native_outputs(label + "-" + role, sample_args)

        # Each actual tracing attempt is independent from the other attempt and
        # from serialization. A const/trace adapter cannot invent native failure.
        concrete_closed = capture.invoke(
            "concrete-typed-trace", lambda: jax.make_jaxpr(original)(*all_arguments)
        )
        capture.require_stable_target("concrete-typed-trace")
        if "concrete-typed-trace" in capture.native_values:
            try:
                capture.document(
                    "concrete-typed-trace", typed_trace_manifest(concrete_closed)
                )
            except Exception as exc:
                capture.gap("concrete-typed-trace", exc)
            capture.save_arrays("captured-constants", concrete_closed.consts)
        all_specs = None
        try:
            all_specs = _native_specs(all_declarations)
        except Exception as exc:
            capture.gap("declared-input-specs", exc)
        closed = None
        trace = None
        if all_specs is not None:
            closed = capture.invoke(
                "declared-typed-trace", lambda: jax.make_jaxpr(original)(*all_specs)
            )
            capture.require_stable_target("declared-typed-trace")
            if "declared-typed-trace" in capture.native_values:
                try:
                    trace = typed_trace_manifest(closed)
                    capture.document("declared-typed-trace", trace)
                except Exception as exc:
                    capture.gap("declared-typed-trace", exc)
        if capture.native_errors:
            raise SourcePreparationError(
                "Original native invocation/tracing raised; actual outcomes and independent captures retained",
                category="native-source-failure",
            )
        if (
            closed is None
            or trace is None
            or all_specs is None
            or "eager" not in outputs_by_role
            or "fresh-jit" not in outputs_by_role
        ):
            raise SourcePreparationError(
                "Native returns/traces need a supported evidence adapter; successful native observations are retained",
                category="replay-lane-gap",
            )
        if trees["eager"] != trees["fresh-jit"]:
            raise SourcePreparationError(
                "Original eager/fresh-JIT trees differ",
                category="native-source-failure",
            )
        traced_inputs = tuple(_properties(var.aval) for var in closed.jaxpr.invars)
        traced_outputs = tuple(_properties(var.aval) for var in closed.jaxpr.outvars)
        try:
            reference = tuple(
                np.array(value, copy=True) for value in outputs_by_role["eager"]
            )
            input_evidence = tuple(
                value_evidence(value, prop)
                for value, prop in zip(all_arguments, traced_inputs, strict=True)
            )
            reference_evidence = tuple(
                value_evidence(value, prop)
                for value, prop in zip(reference, traced_outputs, strict=True)
            )
            original_trace_json = json.dumps(
                trace, sort_keys=True, separators=(",", ":")
            )
        except Exception as exc:
            capture.gap("prepared-evidence", exc)
            raise SourcePreparationError(
                "Actual native values need a supported prepared-evidence adapter",
                category="replay-lane-gap",
            ) from exc
        if len(reference) != len(traced_outputs) or any(
            value.dtype.name != prop.dtype
            for value, prop in zip(reference, traced_outputs, strict=True)
        ):
            raise SourcePreparationError(
                "Original eager/typed trace dtype or output arity differs",
                category="native-source-failure",
            )
        for value in reference:
            value.setflags(write=False)
        if (
            adapter.get("capture_original_object")
            and _capture_recipe(metadata.get("callable")) != recipe
        ):
            capture.gap(
                "original-recipe-state",
                SourcePreparationError(
                    "Original registered constructor/config recipe changed during native preparation",
                    category="replay-lane-gap",
                ),
            )
            raise SourcePreparationError(
                "Original registered recipe state is not preserved",
                category="replay-lane-gap",
            )
        if SourceSettings.observe() != settings:
            raise SourcePreparationError(
                "Original callable changed caller JAX settings during preparation"
            )
        capture.require_stable_target("native-preparation-complete")
        capture.freeze_source_structure(
            closed, original_trace_json, input_evidence, reference_evidence
        )
        capture.preparation_complete = True
        capture.readiness
        return PreparedSourceCase(
            identity,
            scenario,
            target,
            dict(metadata),
            all_specs[: len(arguments)],
            arguments,
            traced_inputs,
            traced_outputs,
            reference,
            trees["eager"],
            _criteria(metadata, scenario),
            settings,
            original_trace_json,
            input_evidence,
            reference_evidence,
            closed,
            dict(zip(names, all_specs[len(arguments) :], strict=True)),
            named,
        )


def run_raw_source_case(
    prepared: PreparedSourceCase, *, capture: RawReplayCapture
) -> SourceCaseResult:
    """Ordinary product observations stay separate from replay adapter readiness."""
    model = None
    schema = CheckStatus.UNVERIFIED
    callback_status = "not-executed"
    settings = None
    error = None
    stage = "source-export"
    product_attempted = False
    state_ok = capture.target is prepared.fn and capture.check_target(
        "before-product-export"
    )
    if not state_ok:
        capture.gap(
            "before-product-export",
            SourcePreparationError(
                "Prepared callable identity/state differs from the actual captured target",
                category="replay-lane-gap",
            ),
        )
    else:
        try:
            with capture.execution_context("product-export"):
                try:
                    with source_configuration(prepared.scenario):
                        settings = SourceSettings.observe()
                        product_attempted = True
                        model = _export_prepared(
                            prepared,
                            artifact_observer=capture.artifact,
                            facade_return_observer=capture.observe_facade_return,
                            source_structure_collector=capture.source_structure_collector(
                                prepared
                            ),
                            output_directory=capture.directory,
                        )
                    schema = CheckStatus.VALIDATED
                except Exception as exc:
                    error = exc
                    if (
                        capture.records["facade_return"]["status"] == "returned"
                        and getattr(exc, "category", None) == "replay-lane-gap"
                    ):
                        capture.gap("facade-return-normalization", exc)
                    else:
                        capture.failure("source-export", exc)
                        capture.records["source_error"] = structured_source_error(exc)
                    if isinstance(
                        exc,
                        (
                            onnx.checker.ValidationError,
                            onnx.shape_inference.InferenceError,
                        ),
                    ):
                        schema = CheckStatus.FAILED
        except NativeTransformedExecutionError:
            # The actual export result/error was recorded before boundary cleanup.
            pass
        finally:
            # A changed target does not erase a real returned artifact/checker
            # outcome. The native model is never reset or reconstructed here.
            capture.check_target("after-product-export")
    model = model if model is not None else capture.model
    report = get_precision_report(model) if model is not None else None
    lowering = (
        next(
            (check.status for check in report.checks if check.category == "lowering"),
            CheckStatus.UNVERIFIED,
        )
        if report
        else CheckStatus.UNVERIFIED
    )
    capture.records["product_admission"] = (
        "returned-artifact"
        if capture.records["facade_return"]["status"] == "returned" or model is not None
        else "rejected" if product_attempted else "not-attempted"
    )
    capture.records["schema_observation"] = schema.value
    capture.records["product_audit_status"] = (
        lowering
        if report
        else (
            CheckStatus.UNVERIFIED
            if error is None
            or getattr(error, "category", None)
            in {"profile-exclusion", "replay-lane-gap"}
            else CheckStatus.FAILED
        )
    )
    if model is not None and error is None:
        available = {}
        for role in (
            "physical-feeds",
            "eager-physical-reference",
            "fresh-jit-physical-reference",
        ):
            record = capture.records.get(role, {})
            if record.get("status") == "captured" and Path(record["path"]).is_file():
                available[role] = {"path": record["path"], "sha256": record["sha256"]}
        capture.save_document(
            "physical-port-bindings",
            {
                "inputs": [port.name for port in model.graph.input],
                "outputs": [port.name for port in model.graph.output],
                "captures": available,
                "original_output_path": prepared.metadata.get("output_path"),
                "original_recording_path": prepared.metadata.get(
                    "record_primitive_calls_file"
                ),
                "actual_destinations": [
                    str(path)
                    for path in sorted(capture.directory.glob("facade-return.onnx*"))
                ],
            },
        )
        stage = "raw-callback"
        callback = prepared.metadata.get("post_check_onnx_graph")
        callback_status = "absent" if callback is None else "error"
        callback_attempted = False
        try:
            with capture.execution_context("raw-callback"):
                callback_attempted = True
                try:
                    if callback is not None:
                        callback_status = "passed" if callback(model) else "failed"
                    if callback_status == "failed":
                        raise AssertionError("Raw metadata structural callback failed")
                except Exception as exc:
                    error = exc
                    capture.failure("raw-callback", exc)
        except NativeTransformedExecutionError:
            if not callback_attempted:
                callback_status = "not-executed"
    if model is not None:
        capture.check_target("after-raw-callback")
    capture.records["raw_callback"] = {
        "status": callback_status,
        "detail": str(error) if stage == "raw-callback" and error is not None else "",
    }
    capture.finalize_source_structure(model)
    readiness = capture.readiness
    category = (
        "replay-lane-gap"
        if capture.gaps
        else (
            "raw-callback-failure"
            if stage == "raw-callback" and error
            else (
                getattr(error, "category", "implementation-gap")
                if error
                else "audited-profile"
            )
        )
    )
    status = (
        CheckStatus.UNVERIFIED
        if capture.gaps
        else (
            CheckStatus.UNVERIFIED
            if getattr(error, "category", None) == "profile-exclusion"
            else CheckStatus.FAILED if error else lowering
        )
    )
    detail = (
        "; ".join(gap["role"] + ": " + gap["detail"] for gap in capture.gaps)
        if capture.gaps
        else (
            str(error)
            if error
            else "Product export/schema checked; raw callback recorded separately. Runtime/numerics unverified."
        )
    )
    return SourceCaseResult(
        (
            "replay-capture-pending"
            if readiness is CheckStatus.UNVERIFIED and capture.gaps
            else getattr(error, "phase", stage) if error else "artifact-checked"
        ),
        category,
        PrecisionCheck("source-audit", status, detail),
        prepared,
        model,
        export_settings=settings,
        schema=schema,
        raw_callback_status=callback_status,
    )
