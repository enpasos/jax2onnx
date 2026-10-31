# jax2onnx/_source_capability.py

"""Describe an observed SOURCE compatibility failure without changing its result.

Classification follows the real checker/session attempt. Registry metadata does
not establish runtime support, and the runtime is never imported during export.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import re
import sys
from types import ModuleType
from typing import Any, Literal

import onnx


_PREFIX = "jax2onnx.source-capability.v1:"
_SCHEMA_ERROR = re.compile(
    r"\[ShapeInferenceError\]\s*\(op_type:(?P<operator>\w+), "
    r"node name: (?P<node>[^)]+)\): (?P<port>\w+) typestr: "
    r"(?P<parameter>\w+), has unsupported type: (?P<actual>tensor\(\w+\))"
)
_RUNTIME_ERROR = re.compile(
    r"ONNX Runtime only \*guarantees\* support for models stamped with official "
    r"released onnx opset versions\. Opset (?P<requested>\d+) is under development "
    r"and support for this is limited\..*?Current official support for domain "
    r"(?P<domain>[\w.]+) is till opset (?P<supported>\d+)\.",
    re.DOTALL,
)
_ACTION = "Select an ONNX schema/runtime installation compatible with the explicit request; the artifact dtype and opset were not changed."


@dataclass(frozen=True)
class CapabilityDiagnostic:
    protocol: Literal[1]
    boundary: Literal["onnx-schema", "runtime-load"]
    artifact_sha256: str
    opset_imports: tuple[tuple[str, int], ...]
    cause_type: str
    cause_message: str
    context: Mapping[str, Any]
    action: str = _ACTION

    def note(self) -> str:
        return _PREFIX + json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def _domain(domain: str) -> str:
    return "" if domain == "ai.onnx" else domain


def _imports(imports: Iterable[Any]) -> tuple[tuple[str, int], ...]:
    result: dict[str, int] = {}
    for item in imports:
        domain = _domain(item.domain)
        if domain in result or item.version < 1:
            raise ValueError("Ambiguous or invalid opset imports")
        result[domain] = item.version
    return tuple(sorted(result.items()))


def _cause_type(cause: Exception) -> str:
    return f"{type(cause).__module__}.{type(cause).__qualname__}"


def _positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def diagnostic_from_detail(
    detail: str,
    *,
    artifact_sha256: str | None = None,
    providers_requested: tuple[str, ...] | None = None,
) -> CapabilityDiagnostic | None:
    """Read only this exact versioned failure, optionally checking its binding."""
    raw, separator, payload = detail.rpartition("\n" + _PREFIX)
    if not separator:
        return None
    try:
        data = json.loads(payload)
        if not isinstance(data, dict) or set(data) != {
            "protocol",
            "boundary",
            "artifact_sha256",
            "opset_imports",
            "cause_type",
            "cause_message",
            "context",
            "action",
        }:
            return None
        if type(data["protocol"]) is not int or data["protocol"] != 1:
            return None
        digest = data["artifact_sha256"]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            return None
        if artifact_sha256 is not None and digest != artifact_sha256:
            return None
        if data["cause_message"] != raw or not raw or data["action"] != _ACTION:
            return None
        if not isinstance(data["cause_type"], str) or "." not in data["cause_type"]:
            return None
        imports = data["opset_imports"]
        if not isinstance(imports, list) or not imports:
            return None
        parsed: list[tuple[str, int]] = []
        for item in imports:
            if not isinstance(item, list) or len(item) != 2:
                return None
            domain, version = item
            if (
                not isinstance(domain, str)
                or _domain(domain) != domain
                or not _positive_int(version)
            ):
                return None
            parsed.append((domain, version))
        if parsed != sorted(set(parsed)) or len({item[0] for item in parsed}) != len(
            parsed
        ):
            return None
        context = data["context"]
        if not isinstance(context, dict):
            return None
        boundary = data["boundary"]
        if boundary == "onnx-schema":
            if data["cause_type"] not in {
                f"{kind.__module__}.{kind.__qualname__}"
                for kind in (
                    onnx.checker.ValidationError,
                    onnx.shape_inference.InferenceError,
                )
            }:
                return None
            if providers_requested is not None or set(context) != {
                "scope",
                "domain",
                "operator",
                "node",
                "port",
                "input_index",
                "value",
                "type_parameter",
                "actual_type",
                "schema_since_version",
                "allowed_types",
                "requested_opset",
                "onnx_version",
            }:
                return None
            if not all(
                isinstance(context[key], str) and context[key]
                for key in (
                    "scope",
                    "operator",
                    "node",
                    "port",
                    "value",
                    "type_parameter",
                    "actual_type",
                    "onnx_version",
                )
            ) or not isinstance(context["domain"], str):
                return None
            if type(context["input_index"]) is not int or context["input_index"] < 0:
                return None
            if (
                not _positive_int(context["schema_since_version"])
                or context["schema_since_version"] > context["requested_opset"]
            ):
                return None
            allowed = context["allowed_types"]
            if (
                not isinstance(allowed, list)
                or not allowed
                or not all(isinstance(item, str) for item in allowed)
            ):
                return None
            match = _SCHEMA_ERROR.fullmatch(raw)
            if (
                match is None
                or any(
                    context[key] != match.group(group)
                    for key, group in (
                        ("operator", "operator"),
                        ("node", "node"),
                        ("port", "port"),
                        ("type_parameter", "parameter"),
                        ("actual_type", "actual"),
                    )
                )
                or context["actual_type"] in allowed
            ):
                return None
        elif boundary == "runtime-load":
            state = sys.modules.get("onnxruntime.capi.onnxruntime_pybind11_state")
            fail_type = getattr(state, "Fail", None)
            if (
                not isinstance(fail_type, type)
                or data["cause_type"]
                != f"{fail_type.__module__}.{fail_type.__qualname__}"
            ):
                return None
            if set(context) != {
                "domain",
                "requested_opset",
                "supported_opset",
                "runtime_version",
                "runtime_build",
                "providers_requested",
            }:
                return None
            if (
                not isinstance(context["domain"], str)
                or not isinstance(context["runtime_version"], str)
                or not context["runtime_version"]
            ):
                return None
            if context["runtime_build"] is not None and not isinstance(
                context["runtime_build"], str
            ):
                return None
            if context["providers_requested"] != ["CPUExecutionProvider"]:
                return None
            if (
                providers_requested is not None
                and tuple(context["providers_requested"]) != providers_requested
            ):
                return None
            match = _RUNTIME_ERROR.search(raw)
            if match is None or not _positive_int(context["supported_opset"]):
                return None
            if (
                context["domain"] != _domain(match.group("domain"))
                or context["requested_opset"] != int(match.group("requested"))
                or context["supported_opset"] != int(match.group("supported"))
            ):
                return None
            if context["requested_opset"] <= context["supported_opset"]:
                return None
        else:
            return None
        if (
            not _positive_int(context["requested_opset"])
            or (context["domain"], context["requested_opset"]) not in parsed
        ):
            return None
        return CapabilityDiagnostic(
            1, boundary, digest, tuple(parsed), data["cause_type"], raw, context
        )
    except (KeyError, TypeError, ValueError):
        return None


def exception_detail(cause: Exception) -> str:
    """Retain our checked note when a text-only audit records the same attempt."""
    raw = str(cause)
    for note in getattr(cause, "__notes__", ()):
        if not isinstance(note, str) or not note.startswith(_PREFIX):
            continue
        detail = raw + "\n" + note
        diagnostic = diagnostic_from_detail(detail)
        if diagnostic is not None and diagnostic.cause_type == _cause_type(cause):
            return detail
    return raw


def _tensor_types(graph: Any, inherited: Mapping[str, str]) -> dict[str, str]:
    result = dict(inherited)

    def add(name: str, elem_type: int) -> None:
        if not name or not elem_type:
            return
        dtype = f"tensor({onnx.TensorProto.DataType.Name(elem_type).lower()})"
        if name in result and result[name] != dtype:
            raise ValueError("Conflicting tensor metadata")
        result[name] = dtype

    # FunctionProto has string formals; it may also carry typed value_info.
    for value in (*graph.input, *graph.output, *graph.value_info):
        if not isinstance(value, str) and value.type.HasField("tensor_type"):
            add(value.name, value.type.tensor_type.elem_type)
    if isinstance(graph, onnx.GraphProto):
        for value in graph.initializer:
            add(value.name, value.data_type)
    for node in graph.node:
        if node.op_type == "Constant" and not node.domain and len(node.output) == 1:
            for attribute in node.attribute:
                if (
                    attribute.name == "value"
                    and attribute.type == onnx.AttributeProto.TENSOR
                ):
                    add(node.output[0], attribute.t.data_type)
    return result


def _schema_mismatches(
    graph: Any,
    imports: tuple[tuple[str, int], ...],
    scope: str,
    inherited: Mapping[str, str],
    match: re.Match[str],
) -> list[dict[str, Any]]:
    types = _tensor_types(graph, inherited)
    result: list[dict[str, Any]] = []
    for index, node in enumerate(graph.node):
        if node.op_type == match.group("operator") and node.name == match.group("node"):
            domain = _domain(node.domain)
            requested = dict(imports).get(domain)
            if requested is not None:
                try:
                    schema = onnx.defs.get_schema(node.op_type, requested, domain)
                except onnx.defs.SchemaError:
                    schema = None
                if schema is not None and schema.domain == domain:
                    for input_index, formal in enumerate(schema.inputs):
                        if any(
                            item.option
                            == onnx.defs.OpSchema.FormalParameterOption.Variadic
                            for item in schema.inputs[: input_index + 1]
                        ):
                            continue
                        if (
                            formal.name != match.group("port")
                            or formal.type_str != match.group("parameter")
                            or input_index >= len(node.input)
                        ):
                            continue
                        actual = types.get(node.input[input_index])
                        constraints = [
                            item
                            for item in schema.type_constraints
                            if item.type_param_str == formal.type_str
                        ]
                        if len(constraints) != 1:
                            continue
                        allowed = tuple(constraints[0].allowed_type_strs)
                        if actual == match.group("actual") and actual not in allowed:
                            result.append(
                                {
                                    "scope": scope,
                                    "domain": domain,
                                    "operator": node.op_type,
                                    "node": node.name,
                                    "port": formal.name,
                                    "input_index": input_index,
                                    "value": node.input[input_index],
                                    "type_parameter": formal.type_str,
                                    "actual_type": actual,
                                    "schema_since_version": schema.since_version,
                                    "allowed_types": list(allowed),
                                    "requested_opset": requested,
                                    "onnx_version": onnx.__version__,
                                }
                            )
        for attribute in node.attribute:
            children = (
                (attribute.g,)
                if attribute.type == onnx.AttributeProto.GRAPH
                else (
                    attribute.graphs
                    if attribute.type == onnx.AttributeProto.GRAPHS
                    else ()
                )
            )
            for ordinal, child in enumerate(children):
                result.extend(
                    _schema_mismatches(
                        child,
                        imports,
                        f"{scope}/{index}:{attribute.name}:{ordinal}",
                        types,
                        match,
                    )
                )
    return result


def check_source_schema(model: onnx.ModelProto | bytes) -> None:
    """Run the real checker once; annotate only a demonstrated typed-port gap."""
    blob = model if isinstance(model, bytes) else model.SerializeToString()
    try:
        onnx.checker.check_model(blob, full_check=True)
    except (onnx.checker.ValidationError, onnx.shape_inference.InferenceError) as cause:
        match = _SCHEMA_ERROR.fullmatch(str(cause))
        if match is not None:
            try:
                frozen = onnx.load_model_from_string(blob)
                imports = _imports(frozen.opset_import)
                mismatches = _schema_mismatches(
                    frozen.graph, imports, "graph", {}, match
                )
                for function in frozen.functions:
                    function_imports = _imports(function.opset_import)
                    mismatches.extend(
                        _schema_mismatches(
                            function,
                            function_imports,
                            f"function:{function.domain}:{function.name}:{function.overload}",
                            {},
                            match,
                        )
                    )
                if len(mismatches) == 1:
                    context = mismatches[0]
                    # Local function imports are recorded too, without replacing
                    # an ambiguous root target with an inferred target.
                    if (context["domain"], context["requested_opset"]) in imports:
                        diagnostic = CapabilityDiagnostic(
                            1,
                            "onnx-schema",
                            hashlib.sha256(blob).hexdigest(),
                            imports,
                            _cause_type(cause),
                            str(cause),
                            context,
                        )
                        detail = str(cause) + "\n" + diagnostic.note()
                        if diagnostic_from_detail(detail) is not None:
                            cause.add_note(diagnostic.note())
            except Exception:
                # Diagnosis must never replace the actual checker exception.
                pass
        raise


def runtime_load_detail(
    cause: Exception,
    blob: bytes,
    runtime: ModuleType | None,
    *,
    providers_requested: Sequence[str],
) -> str:
    """Describe an actual ORT unsupported-opset failure, with no extra attempt."""
    raw = str(cause)
    if runtime is None or tuple(providers_requested) != ("CPUExecutionProvider",):
        return raw
    # This import is reachable only after the actual optional runtime import and
    # session failure. Class identity keeps similarly worded unrelated errors out.
    try:
        from onnxruntime.capi.onnxruntime_pybind11_state import Fail  # type: ignore[import-untyped]
    except ImportError:
        return raw

    if type(cause) is not Fail:
        return raw
    match = _RUNTIME_ERROR.search(raw)
    if match is None:
        return raw
    try:
        imports = _imports(onnx.load_model_from_string(blob).opset_import)
        requested, supported = int(match.group("requested")), int(
            match.group("supported")
        )
        domain = _domain(match.group("domain"))
        if (domain, requested) not in imports or requested <= supported:
            return raw
        build_getter = getattr(runtime, "get_build_info", None)
        build = str(build_getter()) if callable(build_getter) else None
        diagnostic = CapabilityDiagnostic(
            1,
            "runtime-load",
            hashlib.sha256(blob).hexdigest(),
            imports,
            _cause_type(cause),
            raw,
            {
                "domain": domain,
                "requested_opset": requested,
                "supported_opset": supported,
                "runtime_version": runtime.__version__,
                "runtime_build": build,
                "providers_requested": list(providers_requested),
            },
        )
        detail = raw + "\n" + diagnostic.note()
        return (
            detail
            if diagnostic_from_detail(
                detail,
                artifact_sha256=hashlib.sha256(blob).hexdigest(),
                providers_requested=("CPUExecutionProvider",),
            )
            is not None
            else raw
        )
    except Exception:
        # Optional diagnostic metadata cannot replace the actual load failure.
        return raw
