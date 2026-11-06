# jax2onnx/parameter_metadata.py

"""Caller-supplied annotations for final exported parameter names.

Annotations describe root-graph inputs and initializers. They do not change the
graph, establish aliases, or provide gradient or training support.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

import onnx
import onnx_ir as ir

_METADATA_KEY = "pkg.jax2onnx.parameter_metadata"
_Model: TypeAlias = ir.Model | onnx.ModelProto


@dataclass(frozen=True)
class ParameterMetadata:
    """An explicit annotation, with unknown trainability represented by ``None``.

    ``source_path`` is an optional, informational caller-supplied path. Matching
    paths do not establish shared values or tied parameters.
    """

    trainable: bool | None = None
    source_path: str | None = None

    def __post_init__(self) -> None:
        _validate_fields(self.trainable, self.source_path)


def annotate_parameters(
    model: ir.Model | onnx.ModelProto,
    parameters: Mapping[str, ParameterMetadata],
) -> None:
    """Merge annotations into ``model`` in place after complete validation.

    Keys must exactly name final root-graph inputs or initializers. Each supplied
    record replaces the previous record for that name, including its source
    path. Other annotations and unrelated model metadata are preserved. An empty
    mapping validates the model and existing manifest without adding metadata.

    No roles, trainability, source paths, aliases, or dtype eligibility are
    inferred. Malformed or stale existing annotations are rejected. Invalid
    caller types raise ``TypeError``; invalid names or manifests raise
    ``ValueError``. Validation failures leave the model unchanged.
    """
    names = _root_parameter_names(model)
    annotations = _read_manifest(model, names)
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping of names to ParameterMetadata")
    for name, metadata in parameters.items():
        _validate_name(name)
        if name not in names:
            raise ValueError(f"Unknown root-graph input or initializer: {name!r}")
        if not isinstance(metadata, ParameterMetadata):
            raise TypeError(f"Annotation for {name!r} must be ParameterMetadata")
        _validate_fields(metadata.trainable, metadata.source_path)
        annotations[name] = metadata
    if not parameters:
        return

    records: dict[str, dict[str, bool | str | None]] = {}
    for name, metadata in annotations.items():
        record: dict[str, bool | str | None] = {"trainable": metadata.trainable}
        if metadata.source_path is not None:
            record["source_path"] = metadata.source_path
        records[name] = record
    payload = json.dumps(
        {"version": 1, "parameters": records},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if isinstance(model, ir.Model):
        model.metadata_props[_METADATA_KEY] = payload
    else:
        for prop in model.metadata_props:
            if prop.key == _METADATA_KEY:
                prop.value = payload
                return
        model.metadata_props.add(key=_METADATA_KEY, value=payload)


def read_parameter_metadata(
    model: ir.Model | onnx.ModelProto,
) -> dict[str, ParameterMetadata]:
    """Read and validate annotations, returning an empty dict when absent.

    Unannotated values have unknown roles and trainability. Only root-graph input
    and initializer names are accepted; stale names and unsupported manifest
    versions raise ``ValueError``. The graph and model metadata are unchanged.
    """
    return _read_manifest(model, _root_parameter_names(model))


def _validate_fields(trainable: object, source_path: object) -> None:
    if trainable is not None and type(trainable) is not bool:
        raise TypeError("trainable must be bool or None")
    if source_path is not None:
        if not isinstance(source_path, str):
            raise TypeError("source_path must be a string or None")
        if not source_path:
            raise ValueError("source_path must be nonempty when supplied")


def _validate_name(name: object) -> None:
    if not isinstance(name, str):
        raise TypeError("Parameter names must be strings")
    if not name:
        raise ValueError("Parameter names must be nonempty")


def _root_parameter_names(model: _Model) -> set[str]:
    if isinstance(model, ir.Model):
        inputs: dict[str, ir.Value] = {}
        initializers: dict[str, ir.Value] = {}
        for values, by_name in (
            (model.graph.inputs, inputs),
            (model.graph.initializers.values(), initializers),
        ):
            for value in values:
                if value.name:
                    if value.name in by_name:
                        raise ValueError(f"Ambiguous root-graph name: {value.name!r}")
                    by_name[value.name] = value
        for name in inputs.keys() & initializers.keys():
            if inputs[name] is not initializers[name]:
                raise ValueError(f"Ambiguous root-graph name: {name!r}")
        names = set(inputs) | set(initializers)
        definitions = names.copy()
        for node in model.graph:
            for value in node.outputs:
                if value.name:
                    _add_definition(definitions, value.name)
        return names
    if isinstance(model, onnx.ModelProto):
        inputs_set: set[str] = set()
        initializers_set: set[str] = set()
        for value in model.graph.input:
            if value.name:
                _add_definition(inputs_set, value.name)
        for tensor in model.graph.initializer:
            if tensor.name:
                _add_definition(initializers_set, tensor.name)
        for sparse_tensor in model.graph.sparse_initializer:
            if sparse_tensor.values.name:
                _add_definition(initializers_set, sparse_tensor.values.name)
        names = inputs_set | initializers_set
        definitions = names.copy()
        for proto_node in model.graph.node:
            for name in proto_node.output:
                if name:
                    _add_definition(definitions, name)
        return names
    raise TypeError("model must be an onnx_ir.Model or onnx.ModelProto")


def _add_definition(names: set[str], name: str) -> None:
    if name in names:
        raise ValueError(f"Ambiguous root-graph name: {name!r}")
    names.add(name)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def _read_manifest(model: _Model, names: set[str]) -> dict[str, ParameterMetadata]:
    if isinstance(model, ir.Model):
        if _METADATA_KEY not in model.metadata_props:
            return {}
        payload = model.metadata_props[_METADATA_KEY]
    else:
        reserved = [prop for prop in model.metadata_props if prop.key == _METADATA_KEY]
        if len(reserved) > 1:
            raise ValueError(f"Duplicate model metadata key: {_METADATA_KEY!r}")
        if not reserved:
            return {}
        payload = reserved[0].value
    if not isinstance(payload, str):
        raise ValueError("Parameter metadata must be a JSON string")
    try:
        manifest = json.loads(
            payload, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except ValueError as exc:
        raise ValueError(f"Invalid parameter metadata JSON: {exc}") from exc
    if not isinstance(manifest, dict) or set(manifest) != {"version", "parameters"}:
        raise ValueError("Parameter metadata must contain only version and parameters")
    if type(manifest["version"]) is not int or manifest["version"] != 1:
        raise ValueError("Unsupported parameter metadata version; expected integer 1")
    records = manifest["parameters"]
    if not isinstance(records, dict):
        raise ValueError("Parameter metadata parameters must be an object")
    annotations: dict[str, ParameterMetadata] = {}
    for name, record in records.items():
        if not name or name not in names:
            raise ValueError(f"Unknown root-graph input or initializer: {name!r}")
        if not isinstance(record, dict) or set(record) not in (
            {"trainable"},
            {"trainable", "source_path"},
        ):
            raise ValueError(
                f"Annotation for {name!r} requires trainable and optional source_path"
            )
        trainable = record["trainable"]
        source_path = record.get("source_path")
        if "source_path" in record and source_path is None:
            raise ValueError(f"source_path for {name!r} must be a nonempty string")
        try:
            annotations[name] = ParameterMetadata(trainable, source_path)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid annotation for {name!r}: {exc}") from exc
    return annotations
