# jax2onnx/user_interface.py

"""
User-facing API for converting JAX functions and models to ONNX.

This module provides the primary interface for exporting JAX/Flax models to the
ONNX format. It supports dynamic shapes, runtime parameters, file-oriented export
profiles, and numerical validation against ONNX Runtime.

Key Functions:

- **to_onnx**: Convert a JAX function or Flax module to an ONNX model.
- **onnx_function**: Decorator to mark a function or class as an ONNX function node.
- **allclose**: Validate numerical equivalence between JAX and ONNX Runtime outputs.
- **allclose_onnxruntime_web**: Validate ONNX Runtime Web/WASM output against
  Python ONNX Runtime CPU output.

Example:
    ```python
    import jax.numpy as jnp

    from jax2onnx import to_onnx


    def my_model(x):
        return jnp.sin(x)


    to_onnx(
        my_model,
        inputs=[("B", 10)],
        return_mode="file",
        output_path="model.onnx",
    )

    to_onnx(
        my_model,
        inputs=[("B", 10)],
        return_mode="file",
        output_path="model.web.onnx",
        export_mode="web",
    )
    ```
"""

import logging
from functools import partial
from dataclasses import dataclass, replace
import importlib
import json
import os
import re
import subprocess
import tempfile
import warnings
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
    runtime_checkable,
)

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
from onnx_ir.traversal import RecursiveGraphIterator
from jax import core

from jax2onnx._compat.jax import preserve_caller_configuration
from jax2onnx._source_capability import check_source_schema
from jax2onnx.converter.conversion_api import (
    InputSpec,
    ShapeDimSpec,
    ShapeTupleSpec,
    _TraceResult,
    _source_input_values,
    _source_named_input_spec,
    _convert_to_ir,
    _force_jax_x64,
    _to_ir_dtype_from_np,
    to_onnx as to_onnx_impl,
)
from jax2onnx.converter.ir_postprocess import postprocess_ir_model
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_deployment import (
    DEPLOYMENT_METADATA_KEY,
    DeploymentTarget,
    DeploymentRequirements,
    deployment_requirements_from_json,
    _deployment_graph_scope,
    _deployment_function_scope,
    _deployment_child_scope,
)
from jax2onnx.converter.source_deployment_bindings import (
    owned_graphs,
    resolve_matrix_bindings,
    update_matrix_bindings,
    refresh_matrix_names,
)
from jax2onnx.converter.normalization_options import (
    NormalizationMode,
    normalize_normalization_mode,
    source_normalization_adaptations,
)
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    validate_source_input_dtype,
    validate_source_jaxpr,
)
from jax2onnx.converter.source_representation import (
    normalized_shape_dimensions,
    representation_for_aval,
    source_shape_contract,
    validate_source_shapes,
)
from jax2onnx.ir_utils import iter_ir_functions
from jax2onnx.converter.source_structure import (
    StructureResult,
    validate_source_structure,
)
from jax2onnx.precision import (
    CheckStatus,
    PrecisionCheck,
    PrecisionInputSummary,
    PrecisionReport,
    PrecisionWarning,
    SOURCE_DIAGNOSTIC_PROFILE,
    SOURCE_PRECISION_PROFILE,
    TensorProperties,
    attach_precision_report,
    get_precision_report,
    _SourceArtifactFingerprint,
    _SourceArtifactSnapshot,
    _source_artifact_fingerprint,
    _source_artifact_snapshot,
)
from jax2onnx.plugins.plugin_system import onnx_function as onnx_function_impl

_LOGGER = logging.getLogger(__name__)

ReturnMode = Literal["proto", "ir", "file"]
_SourceStructureCollector = Callable[
    [_TraceResult, ir.Model, StructureResult, object], None
]
_VALID_RETURN_MODES = {"proto", "ir", "file"}
ExportMode = Literal["standard", "web"]
_VALID_EXPORT_MODES = {"standard", "web"}


PathLikeStr = Union[str, os.PathLike[str]]
OnnxFunctionTarget = Union[Callable[..., Any], type[Any]]
OnnxFunctionDecorator = Callable[[OnnxFunctionTarget], OnnxFunctionTarget]

_temporary_x64 = _force_jax_x64


@dataclass(frozen=True)
class _SourceTensorGuard:
    tensor: ir.TensorProtocol
    fingerprint: _SourceArtifactFingerprint
    name: str | None
    initializer: bool
    proto_name_present: bool | None


def _freeze_source_ir(
    model: ir.Model,
) -> tuple[ir.Model, tuple[_SourceTensorGuard, ...]]:
    """Freeze topology and small metadata; guard shared tensor storage exactly.

    Model.clone() deliberately shares types, non-graph attributes and tensors.
    The official leaf codec detaches types/shapes and small attributes. Tensor
    attributes get new wrappers while tensor objects stay shared; their exact
    compact guards must match on both sides of reference serialization.
    """
    frozen = model.clone()
    tensors: dict[int, ir.TensorProtocol] = {}
    initializer_tensors: set[int] = set()
    values: set[ir.Value] = set()
    graphs: set[ir.Graph] = set()

    def freeze_value(value: ir.Value) -> None:
        if value in values:
            return
        values.add(value)
        detached = ir.from_proto(ir.to_proto(value))
        value.type, value.shape = detached.type, detached.shape
        # This is the one .meta entry consumed by the official graph codec.
        # Other .meta entries retain only the clone's ordinary shallow ownership.
        if "quant_parameter_tensor_names" in value.meta:
            value.meta["quant_parameter_tensor_names"] = dict(
                value.meta["quant_parameter_tensor_names"]
            )
        if value.const_value is not None:
            tensors[id(value.const_value)] = value.const_value

    def freeze_attributes(attributes: Any) -> None:
        for key, attribute in tuple(attributes.items()):
            if attribute.value is None and not attribute.is_ref():
                attributes[key] = ir.Attr(
                    attribute.name,
                    attribute.type,
                    None,
                    doc_string=attribute.doc_string,
                )
            elif attribute.is_ref():
                attributes[key] = ir.from_proto(ir.to_proto(attribute))
            elif attribute.type is ir.AttributeType.GRAPH:
                freeze_graph(attribute.as_graph())
            elif attribute.type is ir.AttributeType.GRAPHS:
                for graph in attribute.as_graphs():
                    freeze_graph(graph)
            elif attribute.type in (
                ir.AttributeType.TENSOR,
                ir.AttributeType.TENSORS,
            ):
                payloads = (
                    (attribute.as_tensor(),)
                    if attribute.type is ir.AttributeType.TENSOR
                    else tuple(attribute.as_tensors())
                )
                for tensor in payloads:
                    tensors[id(tensor)] = tensor
                attributes[key] = ir.Attr(
                    attribute.name,
                    attribute.type,
                    (
                        payloads[0]
                        if attribute.type is ir.AttributeType.TENSOR
                        else payloads
                    ),
                    doc_string=attribute.doc_string,
                )
            else:
                attributes[key] = ir.from_proto(ir.to_proto(attribute))

    def freeze_graph(graph: ir.Graph) -> None:
        if graph in graphs:
            return
        graphs.add(graph)
        for value in graph.initializers.values():
            if value.const_value is not None:
                initializer_tensors.add(id(value.const_value))
        for value in (*graph.inputs, *graph.outputs, *graph.initializers.values()):
            freeze_value(value)
        for node in graph:
            for operand in (*node.inputs, *node.outputs):
                if operand is not None:
                    freeze_value(operand)
            freeze_attributes(node.attributes)

    freeze_graph(frozen.graph)
    for function in iter_ir_functions(frozen.functions):
        freeze_graph(function.graph)
        freeze_attributes(function.attributes)
    guards = []
    for tensor in tensors.values():
        proto = ir.to_proto(tensor)
        guards.append(
            _SourceTensorGuard(
                tensor,
                _source_artifact_fingerprint(proto),
                tensor.name,
                id(tensor) in initializer_tensors,
                (
                    proto.HasField("name")
                    if isinstance(tensor, ir.TensorProtoTensor)
                    else None
                ),
            )
        )
        del proto
    return frozen, tuple(guards)


def _recheck_source_tensors(
    guards: tuple[_SourceTensorGuard, ...],
) -> None:
    for guard in guards:
        proto = ir.to_proto(guard.tensor)
        if _source_artifact_fingerprint(proto) != guard.fingerprint:
            raise RuntimeError(
                "SOURCE serialization integrity: "
                "shared tensor changed after audit capture"
            )
        del proto


def _restore_source_initializer_names(guards: tuple[_SourceTensorGuard, ...]) -> None:
    # Only the reference codec's initializer-name side effect is scoped. Actual
    # serialization keeps its established behavior. No payload/other field is
    # restored, and a codec or restoration exception still fails the export.
    for guard in guards:
        if guard.initializer:
            if guard.proto_name_present is False:
                assert isinstance(guard.tensor, ir.TensorProtoTensor)
                guard.tensor.raw.ClearField("name")
            else:
                guard.tensor.name = guard.name


def _normalize_return_mode(value: str) -> ReturnMode:
    mode = value.lower().strip()
    if mode not in _VALID_RETURN_MODES:
        raise ValueError(
            f"Unsupported return_mode '{value}'. Expected one of: {sorted(_VALID_RETURN_MODES)}"
        )
    return cast(ReturnMode, mode)


def _normalize_export_mode(value: str) -> ExportMode:
    mode = value.lower().strip()
    if mode not in _VALID_EXPORT_MODES:
        raise ValueError(
            f"Unsupported export_mode '{value}'. Expected one of: {sorted(_VALID_EXPORT_MODES)}"
        )
    return cast(ExportMode, mode)


def _normalize_normalization_mode(value: str) -> NormalizationMode:
    return normalize_normalization_mode(value)


@runtime_checkable
class _SupportsShapeAndDtype(Protocol):
    shape: Tuple[Any, ...]
    dtype: Any


UserInputSpec = Union[
    jax.ShapeDtypeStruct,
    core.ShapedArray,
    _SupportsShapeAndDtype,
    Sequence[ShapeDimSpec],
]


def _normalize_shape_tuple(candidate: Sequence[Any]) -> ShapeTupleSpec:
    dims: list[ShapeDimSpec] = []
    for dim in candidate:
        if isinstance(dim, (int, np.integer)):
            dims.append(int(dim))
        elif isinstance(dim, str):
            dims.append(dim)
        else:
            raise TypeError(
                "Shape tuples may only contain int or str entries; "
                f"received {type(dim)}."
            )
    return tuple(dims)


def _normalize_input_specs(
    raw_inputs: Sequence[UserInputSpec],
    *,
    precision_policy: PrecisionPolicy = PrecisionPolicy.LEGACY,
) -> List[InputSpec]:
    normalized: List[InputSpec] = []
    for item in raw_inputs:
        if isinstance(item, jax.ShapeDtypeStruct):
            dims = tuple(
                int(dim) if isinstance(dim, np.integer) else dim
                for dim in tuple(item.shape)
            )
            normalized.append(
                jax.ShapeDtypeStruct(
                    dims,
                    item.dtype,
                    weak_type=(
                        item.weak_type
                        if precision_policy is PrecisionPolicy.SOURCE
                        else False
                    ),
                )
            )
            continue
        if isinstance(item, core.ShapedArray):
            dims = tuple(
                int(dim) if isinstance(dim, np.integer) else dim for dim in item.shape
            )
            normalized.append(
                jax.ShapeDtypeStruct(
                    dims,
                    item.dtype,
                    weak_type=(
                        item.weak_type
                        if precision_policy is PrecisionPolicy.SOURCE
                        else False
                    ),
                )
            )
            continue
        if isinstance(item, _SupportsShapeAndDtype):
            dims = tuple(
                int(dim) if isinstance(dim, np.integer) else dim
                for dim in tuple(item.shape)
            )
            normalized.append(
                jax.ShapeDtypeStruct(
                    dims,
                    item.dtype,
                    weak_type=(
                        bool(getattr(item, "weak_type", False))
                        if precision_policy is PrecisionPolicy.SOURCE
                        else False
                    ),
                )
            )
            continue
        if isinstance(item, (list, tuple)):
            shape = _normalize_shape_tuple(item)
            normalized.append(
                jax.ShapeDtypeStruct(shape, np.float32)
                if precision_policy is PrecisionPolicy.SOURCE
                else shape
            )
            continue
        raise TypeError(
            "Invalid 'inputs' entry. Expected a jax.ShapeDtypeStruct, "
            "jax.core.ShapedArray, array-like object, or shape tuple/list. "
            f"Received {type(item)}."
        )
    return normalized


def _maybe_int_dim(dim: Any) -> int:
    return int(dim)


def _materialize_input_params_on_ir(
    model: ir.Model, param_map: Mapping[str, object]
) -> None:
    if not param_map:
        return

    graph = model.graph
    existing_inputs = {
        value.name for value in graph.inputs if getattr(value, "name", None)
    }
    initializer_names = {name for name in graph.initializers.keys() if name}

    referenced_names: set[str] = set()
    value_by_name: Dict[str, ir.Value] = {}

    def _record(value: Optional[ir.Value]) -> None:
        if value is None:
            return
        name = getattr(value, "name", None)
        if not name:
            return
        referenced_names.add(name)
        value_by_name.setdefault(name, value)

    for node in graph:
        for val in node.inputs:
            _record(val)
    for output in graph.outputs:
        _record(output)

    for name, python_value in param_map.items():
        if not name:
            continue
        if name in existing_inputs or name in initializer_names:
            continue
        if name not in referenced_names:
            continue

        existing_value = value_by_name.get(name)
        if existing_value is not None:
            graph.inputs.append(existing_value)
            existing_inputs.add(name)
            continue

        arr = np.asarray(python_value)
        try:
            ir_dtype = _to_ir_dtype_from_np(arr.dtype)
        except Exception:
            ir_dtype = None
        if ir_dtype is None:
            continue

        shape = tuple(_maybe_int_dim(dim) for dim in arr.shape)
        new_value = ir.Value(
            name=name,
            type=ir.TensorType(ir_dtype),
            shape=ir.Shape(shape),
        )
        graph.inputs.append(new_value)
        existing_inputs.add(name)


_POSITIONAL_INPUT_NAME_RE = re.compile(r"^in_(\d+)(?:_nchw)?$")


def _normalize_io_names(
    names: Optional[Sequence[str]], *, kind: str
) -> Optional[List[str]]:
    if names is None:
        return None
    if isinstance(names, (str, bytes)):
        raise TypeError(f"{kind} must be a sequence of strings, not a single string.")

    normalized: List[str] = []
    seen: set[str] = set()
    for idx, entry in enumerate(names):
        if not isinstance(entry, str):
            raise TypeError(f"{kind}[{idx}] must be a string, received {type(entry)}.")
        if not entry.strip():
            raise ValueError(f"{kind}[{idx}] must be a non-empty string.")
        if entry in seen:
            raise ValueError(f"{kind} entries must be unique; duplicate '{entry}'.")
        seen.add(entry)
        normalized.append(entry)
    return normalized


def _top_graph_value_map(graph: ir.Graph) -> Dict[str, ir.Value]:
    """Return named values from the top graph only."""

    return ir.convenience.create_value_mapping(graph, include_subgraphs=False)


def _unique_values(values: Sequence[ir.Value]) -> List[ir.Value]:
    unique: List[ir.Value] = []
    seen: set[int] = set()
    for value in values:
        vid = id(value)
        if vid in seen:
            continue
        seen.add(vid)
        unique.append(value)
    return unique


def _resolve_positional_inputs(
    graph: ir.Graph, positional_input_count: int
) -> List[ir.Value]:
    if positional_input_count == 0:
        return []

    indexed: Dict[int, ir.Value] = {}
    for value in graph.inputs:
        name = getattr(value, "name", None)
        if not isinstance(name, str):
            continue
        match = _POSITIONAL_INPUT_NAME_RE.fullmatch(name)
        if match is None:
            continue
        idx = int(match.group(1))
        indexed.setdefault(idx, value)

    if all(idx in indexed for idx in range(positional_input_count)):
        return [indexed[idx] for idx in range(positional_input_count)]

    fallback = list(graph.inputs)[:positional_input_count]
    if len(fallback) != positional_input_count:
        raise ValueError(
            "Cannot map custom input_names to graph inputs: positional inputs were pruned."
        )
    return fallback


def _apply_custom_io_names_on_ir(
    model: ir.Model,
    *,
    input_names: Optional[Sequence[str]],
    output_names: Optional[Sequence[str]],
    positional_input_count: int,
    input_values: Optional[Sequence[ir.Value]] = None,
) -> None:
    if not input_names and not output_names:
        return

    graph = model.graph
    rename_pairs: List[tuple[ir.Value, str]] = []

    if input_names is not None:
        positional_inputs = (
            list(input_values)
            if input_values is not None
            else _resolve_positional_inputs(graph, positional_input_count)
        )
        if len(positional_inputs) != len(input_names):
            raise ValueError(
                f"input_names length ({len(input_names)}) does not match positional graph inputs ({len(positional_inputs)})."
            )
        rename_pairs.extend(zip(positional_inputs, input_names))

    if output_names is not None:
        graph_outputs = list(graph.outputs)
        if len(graph_outputs) != len(output_names):
            raise ValueError(
                f"output_names length ({len(output_names)}) does not match graph outputs ({len(graph_outputs)})."
            )
        rename_pairs.extend(zip(graph_outputs, output_names))

    if not rename_pairs:
        return

    target_by_value: Dict[int, str] = {}
    for value, target in rename_pairs:
        vid = id(value)
        existing = target_by_value.get(vid)
        if existing is None:
            target_by_value[vid] = target
            continue
        if existing != target:
            raise ValueError(
                f"Conflicting custom names for one value: '{existing}' vs '{target}'."
            )

    targets = list(target_by_value.values())
    if len(set(targets)) != len(targets):
        raise ValueError("Custom input/output names must be globally unique.")

    renamed_ids = set(target_by_value.keys())
    occupied_by_other: set[str] = set()
    for value in _top_graph_value_map(graph).values():
        name = getattr(value, "name", None)
        if not name:
            continue
        if id(value) in renamed_ids:
            continue
        occupied_by_other.add(name)

    collisions = sorted(name for name in targets if name in occupied_by_other)
    if collisions:
        joined = ", ".join(collisions)
        raise ValueError(
            f"Custom names collide with existing graph value names: {joined}."
        )

    rename_values = _unique_values(
        [value for value, _ in rename_pairs if getattr(value, "name", None) is not None]
    )
    target_names = [target_by_value[id(value)] for value in rename_values]
    ir.convenience.rename_values(rename_values, target_names)
    if DEPLOYMENT_METADATA_KEY in model.metadata_props:
        refresh_matrix_names(model)


def _finalize_source_deployment(model: ir.Model) -> None:
    """Bind qualified equations to the final owned IR producer and source scope."""
    payload = model.metadata_props.get(DEPLOYMENT_METADATA_KEY)
    requirements = deployment_requirements_from_json(payload)
    if requirements is None:
        return
    requirements, values = resolve_matrix_bindings(model)
    index = 0
    for graph in owned_graphs(model):
        for node in graph:
            node.name = f"jax2onnx_src_{index:06d}"
            index += 1
    scopes: dict[int, str] = {}

    def visit(graph: ir.Graph, scope: str) -> None:
        if id(graph) in scopes:
            raise ValueError("Deployment Graph has ambiguous lexical ownership")
        scopes[id(graph)] = scope
        for node in graph:
            tag = node.name
            if tag is None:
                raise ValueError("Deployment source node lost its tag")
            for attribute in node.attributes.values():
                if attribute.type is ir.AttributeType.GRAPH:
                    visit(
                        attribute.as_graph(),
                        _deployment_child_scope(scope, tag, attribute.name, 0),
                    )
                elif attribute.type is ir.AttributeType.GRAPHS:
                    for position, child in enumerate(attribute.as_graphs()):
                        visit(
                            child,
                            _deployment_child_scope(
                                scope, tag, attribute.name, position
                            ),
                        )

    visit(model.graph, _deployment_graph_scope(str(model.graph.name)))
    for function in iter_ir_functions(model.functions):
        visit(
            function.graph,
            _deployment_function_scope(
                function.domain, function.name, function.overload
            ),
        )
    sites = []
    for site, (logical, matrix) in zip(
        requirements.matrix_precision, values, strict=True
    ):
        matrix_graph, producer = matrix.graph, matrix.producer()
        if (
            matrix_graph is None
            or producer is None
            or logical.graph is not matrix_graph
        ):
            raise ValueError("Deployment logical matrix result lost its actual scope")
        descendant = logical
        seen: set[int] = set()
        while descendant is not matrix:
            if id(descendant) in seen:
                raise ValueError("Deployment logical result has cyclic provenance")
            seen.add(id(descendant))
            descendant_node = descendant.producer()
            if (
                descendant.graph is not matrix_graph
                or descendant_node is None
                or descendant_node.graph is not matrix_graph
                or descendant_node.domain not in ("", "ai.onnx")
                or descendant_node.op_type not in {"Identity", "Transpose", "Reshape"}
                or not descendant_node.inputs
                or descendant_node.inputs[0] is None
            ):
                raise ValueError(
                    "Deployment logical result is not a layout-only matrix descendant"
                )
            descendant = descendant_node.inputs[0]
        if producer.op_type != "MatMul" or producer.domain not in ("", "ai.onnx"):
            raise ValueError(
                "Deployment lowering remains unfinished for a non-MatMul precision site"
            )
        if any(
            value is None or value.dtype is not ir.DataType.FLOAT
            for value in (*producer.inputs, *producer.outputs)
        ):
            raise ValueError("Deployment source matrix producer is not exact FLOAT")
        port = matrix.index()
        if port is None or producer.outputs[port] is not matrix:
            raise ValueError("Deployment source matrix lost its actual output port")
        if not logical.name or not matrix.name:
            raise ValueError("Deployment source matrix lost its final name")
        sites.append(
            replace(
                site,
                output=logical.name,
                matrix_output=matrix.name,
                matrix_scope=scopes[id(matrix_graph)],
                matrix_tag=producer.name,
            )
        )
    update_matrix_bindings(
        model, DeploymentRequirements(requirements.target, tuple(sites))
    )


def _source_migration_gap(option: str, opset: int) -> None:
    raise SourcePrecisionError(
        f"{option} is an internal migration gap in this audited milestone",
        phase="input",
        category="profile-exclusion",
        location="export options",
        opset=opset,
        requirement="the current private source-static-v1 admission scope",
        action="omit this option for the internal comparison; broader existing functionality still requires migration",
    )


def _source_input_properties(
    value: UserInputSpec, *, index: int, opset: int
) -> TensorProperties:
    shape_only = isinstance(value, (tuple, list))
    typed = cast(_SupportsShapeAndDtype, value)
    shape = tuple(value) if isinstance(value, (tuple, list)) else tuple(typed.shape)
    dtype = None if shape_only else np.dtype(typed.dtype).name
    try:
        normalized_shape, _ = source_shape_contract(shape)
    except ValueError as exc:
        raise SourcePrecisionError(
            f"input shape {shape} is outside this internal audited milestone",
            phase="input",
            category="profile-exclusion",
            location=f"input {index}",
            requested_dtypes=() if dtype is None else (dtype,),
            opset=opset,
            requirement=str(exc),
            action="use static inputs or the audited shared leading rank-one through rank-four batch declaration; broader symbolic migration remains pending",
        ) from exc
    return TensorProperties(
        None,
        normalized_shape,
        dtype,
        None if shape_only else bool(getattr(value, "weak_type", False)),
    )


def _validate_source_inputs(
    inputs: Sequence[InputSpec], opset: int, *, diagnostic: bool = False
) -> None:
    for index, spec in enumerate(inputs):
        assert isinstance(spec, jax.ShapeDtypeStruct)
        validate_source_input_dtype(spec.dtype, index=index, opset=opset)
    if opset not in (16, 18, 21, 23, 24, 26, 27):
        _source_migration_gap(f"opset={opset}", opset)
    for index, spec in enumerate(inputs):
        assert isinstance(spec, jax.ShapeDtypeStruct)
        dtype = np.dtype(spec.dtype).name
        shape = tuple(spec.shape)
        if opset == 16 and dtype not in {
            "float16",
            "float32",
            "float64",
            "bool",
            "int32",
            "int64",
        }:
            _source_migration_gap(f"opset=16 input dtype={dtype}", opset)
        requirement = "ordinary Boolean/integer/floating static or shared leading-batch inputs, or static complex64/complex128 inputs"
        action = "use typed inputs within the finite audited shape contract; symbolic complex shapes remain pending"
        if not diagnostic and dtype not in {
            "float16",
            "bfloat16",
            "float32",
            "float64",
            "bool",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "complex64",
            "complex128",
        }:
            raise SourcePrecisionError(
                f"input shape {shape} and dtype {dtype} are outside the internal milestone",
                phase="input",
                category="profile-exclusion",
                location=f"input {index}",
                requested_dtypes=(dtype,),
                opset=opset,
                requirement=requirement,
                action=action,
            )
        if (
            not diagnostic
            and dtype in {"complex64", "complex128"}
            and source_shape_contract(shape)[1] is not None
        ):
            raise SourcePrecisionError(
                "symbolic complex inputs are outside the internal milestone",
                phase="input",
                category="profile-exclusion",
                location=f"input {index}",
                requested_dtypes=(dtype,),
                opset=opset,
                requirement="static complex input shapes",
                action="use static complex values; symbolic complex migration remains pending",
            )
    try:
        validate_source_shapes(
            [tuple(cast(jax.ShapeDtypeStruct, spec).shape) for spec in inputs]
        )
    except ValueError as exc:
        raise SourcePrecisionError(
            str(exc),
            phase="input",
            category="profile-exclusion",
            location="inputs",
            opset=opset,
            requirement="static shapes or one shared atomic leading batch variable in rank two/three with positive static tails",
            action="retain one unconstrained native scope and the same batch variable; broader symbolic migration remains pending",
        ) from exc


@contextmanager
def _capture_source_warnings(captured: list[PrecisionWarning]) -> Iterator[None]:
    # Existing warning filters apply once. Replay precisely the emitted warnings
    # after restoring the caller's warning machinery, including on failure.
    observed = []
    try:
        with warnings.catch_warnings(record=True) as observed:
            yield
    finally:
        for item in observed:
            captured.append(PrecisionWarning(item.category.__name__, str(item.message)))
            warnings.showwarning(
                item.message, item.category, item.filename, item.lineno, line=item.line
            )


def _separate_source_output_aliases(
    model: ir.Model,
    input_names: Optional[Sequence[str]],
    output_names: Optional[Sequence[str]],
) -> None:
    if output_names is None:
        return
    graph = model.graph
    assigned = {
        id(value): input_names[index] if input_names is not None else value.name
        for index, value in enumerate(graph.inputs)
    }
    occupied = set(_top_graph_value_map(graph)) | set(output_names)
    if input_names is not None:
        occupied.update(input_names)
    for index, (value, target) in enumerate(
        zip(tuple(graph.outputs), output_names, strict=True)
    ):
        prior = assigned.get(id(value))
        if prior is not None and prior != target:
            candidate = f"source_output_{index}"
            suffix = 0
            while candidate in occupied:
                suffix += 1
                candidate = f"source_output_{index}_{suffix}"
            occupied.add(candidate)
            output = ir.Value(name=candidate, type=value.type, shape=value.shape)
            graph.append(
                ir.Node(
                    "",
                    "Identity",
                    [value],
                    outputs=[output],
                    name=f"{candidate}_identity",
                )
            )
            graph.outputs[index] = output
            value = output
        assigned[id(value)] = target


def _source_boundary_failure(
    label: str, index: int, expected: Any, opset: int, *, nchw: bool = False
) -> None:
    representation = representation_for_aval(
        expected, boundary_layout="NCHW" if nchw else None
    )
    raise SourcePrecisionError(
        "exported boundary dtype or shape differs from the actual source trace",
        phase="finalization",
        category="lowering-defect",
        location=f"{label} {index}",
        requested_dtypes=(np.dtype(expected.dtype).name,),
        opset=opset,
        requirement=(
            f"logical dtype {expected.dtype} and shape {tuple(expected.shape)} "
            f"represented by dtype {representation.physical_dtype} and shape "
            f"{representation.physical_shape}"
        ),
        action="report this exporter defect with the model and input specifications",
    )


def _assert_source_boundaries(
    model: ir.Model | onnx.ModelProto,
    trace: _TraceResult,
    opset: int,
    *,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
) -> None:
    ir_model = isinstance(model, ir.Model)
    graph_inputs: Sequence[Any]
    graph_outputs: Sequence[Any]
    if isinstance(model, ir.Model):
        graph_inputs, graph_outputs = model.graph.inputs, model.graph.outputs
    else:
        graph_inputs, graph_outputs = model.graph.input, model.graph.output
    for label, variables, values, names, layouts in (
        ("input", trace.jaxpr.invars, graph_inputs, input_names, trace.inputs_as_nchw),
        (
            "output",
            trace.jaxpr.outvars,
            graph_outputs,
            output_names,
            trace.outputs_as_nchw,
        ),
    ):
        if len(variables) != len(values):
            raise SourcePrecisionError(
                "exported boundary arity differs from the actual source trace",
                phase="finalization",
                category="lowering-defect",
                location=label,
                opset=opset,
                requirement=f"{len(variables)} values",
                action="report this exporter defect with a minimal reproduction",
            )
        if names is not None and tuple(value.name for value in values) != tuple(names):
            raise SourcePrecisionError(
                "exported boundary names/order differ from the finalized source bindings",
                phase="finalization",
                category="lowering-defect",
                location=label,
                opset=opset,
                requirement=f"ordered names {tuple(names)!r}",
                action="report this exporter defect with a minimal reproduction",
            )
        for index, (var, value) in enumerate(zip(variables, values, strict=True)):
            expected = var.aval
            representation = representation_for_aval(
                expected, boundary_layout="NCHW" if index in layouts else None
            )
            dtype = representation.ir_dtype
            shape = representation.physical_shape
            if ir_model:
                if not representation.matches_ir_value(value):
                    _source_boundary_failure(
                        label, index, expected, opset, nchw=index in layouts
                    )
            else:
                tensor = value.type.tensor_type
                expected_shape = normalized_shape_dimensions(shape)
                actual_dims = tuple(tensor.shape.dim)
                if (
                    tensor.elem_type != int(dtype)
                    or not tensor.HasField("shape")
                    or len(actual_dims) != len(expected_shape)
                    or any(
                        (
                            (
                                not dim.HasField("dim_value")
                                or dim.dim_value != expected_dim
                            )
                            if isinstance(expected_dim, int)
                            else (
                                not dim.HasField("dim_param")
                                or dim.dim_param != expected_dim
                            )
                        )
                        for dim, expected_dim in zip(actual_dims, expected_shape)
                    )
                ):
                    _source_boundary_failure(
                        label, index, expected, opset, nchw=index in layouts
                    )


def _source_value_properties(value: ir.Value) -> TensorProperties:
    assert value.shape is not None and value.dtype is not None
    return TensorProperties(
        value.name,
        normalized_shape_dimensions(tuple(value.shape)),
        value.dtype.numpy().name,
        None,
    )


def _source_aval_properties(aval: Any) -> TensorProperties:
    return TensorProperties(
        None,
        normalized_shape_dimensions(aval.shape),
        representation_for_aval(aval).logical_dtype.name,
        bool(aval.weak_type),
    )


def _source_adaptations(model: ir.Model) -> tuple[str, ...]:
    graphs = [model.graph, *(fn.graph for fn in iter_ir_functions(model.functions))]
    for graph in graphs:
        for node in RecursiveGraphIterator(graph):
            if node.op_type != "MatMul" or len(node.inputs) != 2:
                continue
            casts = [
                value.producer() if value is not None else None for value in node.inputs
            ]
            if all(
                cast_node is not None
                and cast_node.op_type == "Cast"
                and cast_node.inputs[0] is not None
                and cast_node.inputs[0].dtype == ir.DataType.FLOAT16
                and cast_node.outputs[0].dtype == ir.DataType.FLOAT
                for cast_node in casts
            ):
                return (
                    "dot_general: FP16 operands widened locally to FP32 before MatMul; execution cost unmeasured",
                )
    return ()


def _build_source_report(
    model: ir.Model,
    trace: _TraceResult,
    supplied: Sequence[TensorProperties],
    captured: Sequence[PrecisionWarning],
    *,
    caller_x64: bool,
    caller_precision: str | None,
    caller_dtype_promotion: str,
    caller_rank_promotion: str,
    opset: int,
    diagnostic: bool,
    named_supplied: Mapping[str, TensorProperties],
    normalization_mode: NormalizationMode,
) -> PrecisionReport:
    representations = []
    for role, variables, layouts in (
        ("input", trace.jaxpr.invars, trace.inputs_as_nchw),
        ("output", trace.jaxpr.outvars, trace.outputs_as_nchw),
    ):
        for index, var in enumerate(variables):
            representation = representation_for_aval(
                var.aval, boundary_layout="NCHW" if index in layouts else None
            )
            if representation.encoding == "complex-real-pair":
                representations.append(
                    f"{role} {index}: {representation.logical_dtype} "
                    f"{representation.logical_shape} represented by "
                    f"{representation.physical_dtype} {representation.physical_shape} "
                    "real/imaginary channels"
                )
            if index in layouts:
                pair_note = (
                    "; complex pair axis unchanged"
                    if representation.encoding == "complex-real-pair"
                    else ""
                )
                representations.append(
                    f"{role} {index}: logical NHWC {representation.logical_shape} "
                    f"uses external NCHW storage {representation.physical_shape}{pair_note}"
                )
    return PrecisionReport(
        exporter_version=model.producer_version or "unknown",
        jax_version=jax.__version__,
        opset=opset,
        caller_x64_enabled=caller_x64,
        caller_default_matmul_precision=caller_precision,
        caller_numpy_dtype_promotion=caller_dtype_promotion,
        caller_numpy_rank_promotion=caller_rank_promotion,
        deployment_requirements=deployment_requirements_from_json(
            model.metadata_props.get(DEPLOYMENT_METADATA_KEY)
        ),
        inputs=tuple(
            PrecisionInputSummary(
                binding.trace_index,
                (
                    supplied[binding.positional_index]
                    if binding.positional_index is not None
                    else named_supplied[cast(str, binding.keyword_name)]
                ),
                _source_aval_properties(binding.spec),
                _source_aval_properties(binding.var.aval),
                _source_value_properties(value),
            )
            for binding, value in zip(
                trace.source_inputs,
                _source_input_values(trace, model.graph.inputs),
                strict=True,
            )
        ),
        traced_outputs=tuple(
            _source_aval_properties(var.aval) for var in trace.jaxpr.outvars
        ),
        exported_outputs=tuple(
            _source_value_properties(value) for value in model.graph.outputs
        ),
        warnings=tuple(captured),
        adaptations=(
            *_source_adaptations(model),
            *source_normalization_adaptations(
                normalization_mode, diagnostic=diagnostic
            ),
            *representations,
        ),
        profile=SOURCE_DIAGNOSTIC_PROFILE if diagnostic else SOURCE_PRECISION_PROFILE,
        checks=(
            PrecisionCheck(
                "representation",
                CheckStatus.VALIDATED,
                "Full ONNX checker passed; final boundaries match the declared logical-to-physical representation of this source trace.",
            ),
            PrecisionCheck(
                "lowering",
                CheckStatus.UNVERIFIED if diagnostic else CheckStatus.VALIDATED,
                (
                    "Diagnostic callback and strict representation checks passed; lowering audit_pending."
                    if diagnostic
                    else "Private audited profile, strict optimizer, and source result dtype checks passed; no runtime or numerical claim."
                ),
            ),
            PrecisionCheck(
                "runtime-execution",
                CheckStatus.UNVERIFIED,
                "No runtime execution was requested by export.",
            ),
            PrecisionCheck(
                "numerical-acceptance",
                CheckStatus.UNVERIFIED,
                "No independent samples and numerical criteria were supplied to export.",
            ),
        ),
    )


if TYPE_CHECKING:

    @overload
    def to_onnx(
        fn: Callable[..., Any],
        inputs: Sequence[UserInputSpec],
        input_params: Optional[Mapping[str, object]] = ...,
        model_name: str = ...,
        opset: int = ...,
        *,
        enable_double_precision: bool = ...,
        record_primitive_calls_file: Optional[str] = ...,
        return_mode: Literal["proto"] = ...,
        output_path: None = ...,
        inputs_as_nchw: Optional[Sequence[int]] = ...,
        outputs_as_nchw: Optional[Sequence[int]] = ...,
        input_names: Optional[Sequence[str]] = ...,
        output_names: Optional[Sequence[str]] = ...,
        export_mode: ExportMode = ...,
        normalization_mode: NormalizationMode = ...,
    ) -> onnx.ModelProto: ...

    @overload
    def to_onnx(
        fn: Callable[..., Any],
        inputs: Sequence[UserInputSpec],
        input_params: Optional[Mapping[str, object]] = ...,
        model_name: str = ...,
        opset: int = ...,
        *,
        enable_double_precision: bool = ...,
        record_primitive_calls_file: Optional[str] = ...,
        return_mode: Literal["ir"],
        output_path: Optional[PathLikeStr] = ...,
        inputs_as_nchw: Optional[Sequence[int]] = ...,
        outputs_as_nchw: Optional[Sequence[int]] = ...,
        input_names: Optional[Sequence[str]] = ...,
        output_names: Optional[Sequence[str]] = ...,
        export_mode: ExportMode = ...,
        normalization_mode: NormalizationMode = ...,
    ) -> ir.Model: ...

    @overload
    def to_onnx(
        fn: Callable[..., Any],
        inputs: Sequence[UserInputSpec],
        input_params: Optional[Mapping[str, object]] = ...,
        model_name: str = ...,
        opset: int = ...,
        *,
        enable_double_precision: bool = ...,
        record_primitive_calls_file: Optional[str] = ...,
        return_mode: Literal["file"],
        output_path: PathLikeStr,
        inputs_as_nchw: Optional[Sequence[int]] = ...,
        outputs_as_nchw: Optional[Sequence[int]] = ...,
        input_names: Optional[Sequence[str]] = ...,
        output_names: Optional[Sequence[str]] = ...,
        export_mode: ExportMode = ...,
        normalization_mode: NormalizationMode = ...,
    ) -> str: ...


def to_onnx(
    fn: Callable[..., Any],
    inputs: Sequence[UserInputSpec],
    input_params: Optional[Mapping[str, object]] = None,
    model_name: str = "jax_model",
    opset: int = 23,
    *,  # All arguments after this must be keyword-only
    enable_double_precision: bool = False,
    record_primitive_calls_file: Optional[str] = None,
    return_mode: ReturnMode = "proto",
    output_path: Optional[PathLikeStr] = None,
    inputs_as_nchw: Optional[Sequence[int]] = None,
    outputs_as_nchw: Optional[Sequence[int]] = None,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    export_mode: ExportMode = "standard",
    normalization_mode: NormalizationMode = "auto",
) -> Union[onnx.ModelProto, ir.Model, str]:
    """
    Converts a JAX function or model into an ONNX model.

    This function serves as the main entry point for converting JAX/Flax models to ONNX format.
    It supports dynamic shapes and additional runtime parameters.

    Args:
        fn: The JAX function or Flax module to convert.
        inputs: Sequence of input specifications. Each entry may be:
            * a `jax.ShapeDtypeStruct` (or `jax.core.ShapedArray`);
            * any array-like object exposing `.shape` and `.dtype`
              (e.g. `jax.Array`, `np.ndarray`);
            * a tuple/list of ints/strs describing the desired shape.
              Example: `[('B', 128), (1, 10)]` implies two inputs, the first with a dynamic batch dimension 'B' and fixed size 128.
        input_params: Optional mapping of string keys to runtime parameters that
            should be exposed as inputs in the ONNX model rather than baked into
            the export (e.g. `"deterministic"` flags).
        model_name: Name to give the ONNX model. Defaults to "jax_model".
        opset: ONNX opset version to target. Defaults to 23.
        enable_double_precision: If True, export tensors as tensor(double). Defaults to False (use tensor(float)).
        record_primitive_calls_file: Optional path to a file. If provided,
            details of each JAX primitive encountered during conversion will be
            recorded to this file. This log can be used by developers to manually
            create new test cases. Defaults to None (disabled).
        return_mode: Output mode. `"proto"` (default) returns an ONNX ModelProto,
            `"ir"` returns the intermediate onnx_ir.Model, and `"file"`
            serialises directly to disk.
        output_path: Destination path (str or PathLike) required when `return_mode` is
            `"file"`. Ignored otherwise.
        inputs_as_nchw: Optional sequence of input indices (0-based) that should be treated as NCHW layout.
            If specified for an input, jax2onnx assumes the external input is NCHW and will automatically
            transpose it to NHWC before feeding it to the JAX graph (which typically expects NHWC for images).
            This allows the exported ONNX model to accept NCHW inputs while preserving correct graph semantics.
        outputs_as_nchw: Optional sequence of output indices (0-based) that should be treated as NCHW layout.
            If specified for an output, jax2onnx assumes the external output should be NCHW and will automatically
            transpose the NHWC output derived from JAX graph to NCHW before returning it.
        input_names: Optional sequence of names for positional model inputs.
            When provided, names are assigned by positional index (0-based) after conversion.
            These names do not apply to entries supplied through `input_params`.
        output_names: Optional sequence of names for model outputs (0-based).
            Names are assigned after conversion in output order.
        export_mode: Serialization profile. `"standard"` preserves the existing
            file behavior, spilling large initializers into `.onnx.data` sidecars
            when needed. `"web"` writes a single self-contained `.onnx` file for
            browser/WASM deployment via `onnxruntime-web`. This only affects
            `return_mode="file"`; `"proto"` and `"ir"` return values are unchanged.
        normalization_mode: Export policy for normalization plugins that offer
            both native ONNX operators and explicit graphs. `"auto"` (default)
            follows a two-step policy: choose the most reproducibly accurate
            implementation in a defined test environment and fix its acceptance
            bounds; then permit a native replacement only if it meets those
            bounds. Export does not benchmark each model. The current selection
            is explicit for Flax NNX and Linen GroupNorm, and Equinox, Flax
            NNX, and Linen RMSNorm and LayerNorm. Fixed
            comparative bounds currently cover selected LayerNorm cases,
            not GroupNorm or RMSNorm. `"prefer_native"` requests a native operator
            when eligible (`LayerNormalization` from opset 17, fast-variance
            `GroupNormalization` from 21, `RMSNormalization` from 23), falling
            back to the explicit graph otherwise. Eligibility is determined
            by the plugin and opset, without a per-model accuracy check. Linen
            LayerNorm with use_fast_variance=False traces JAX in every mode.
            `"force_decomposed"` selects the explicit graph. Runtime graph optimization is separate from
            the exported representation. See the user guide for tested limits.

    Returns:
        * If `return_mode="proto"` (default): Returns an `onnx.ModelProto` object.
        * If `return_mode="ir"`: Returns an `onnx_ir.Model` object (intermediate representation).
        * If `return_mode="file"`: Returns the string path to the saved file.

    Raises:
        ValueError: If `return_mode` is "file" but `output_path` is not provided.
        ValueError: If `return_mode` is invalid.
        TypeError: If `input_params` keys are not strings.

    Example:
        ```python
        import jax.numpy as jnp

        from jax2onnx import to_onnx


        def linear(x, w, b):
            return jnp.dot(x, w) + b


        input_specs = [
            ("B", 32),  # x: [Batch, 32]
            (32, 10),  # w: [32, 10]
            (10,),  # b: [10]
        ]

        to_onnx(
            linear,
            inputs=input_specs,
            model_name="linear_model",
            return_mode="file",
            output_path="linear_model.onnx",
        )
        ```
    """

    return _to_onnx(
        fn,
        inputs,
        input_params,
        model_name,
        opset,
        enable_double_precision=enable_double_precision,
        record_primitive_calls_file=record_primitive_calls_file,
        return_mode=return_mode,
        output_path=output_path,
        inputs_as_nchw=inputs_as_nchw,
        outputs_as_nchw=outputs_as_nchw,
        input_names=input_names,
        output_names=output_names,
        export_mode=export_mode,
        normalization_mode=normalization_mode,
    )


def _to_onnx(
    fn: Callable[..., Any],
    inputs: Sequence[UserInputSpec],
    input_params: Optional[Mapping[str, object]] = None,
    model_name: str = "jax_model",
    opset: int = 23,
    *,
    enable_double_precision: bool = False,
    record_primitive_calls_file: Optional[str] = None,
    return_mode: ReturnMode = "proto",
    output_path: Optional[PathLikeStr] = None,
    inputs_as_nchw: Optional[Sequence[int]] = None,
    outputs_as_nchw: Optional[Sequence[int]] = None,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    export_mode: ExportMode = "standard",
    normalization_mode: NormalizationMode = "auto",
    precision_policy: PrecisionPolicy = PrecisionPolicy.LEGACY,
    diagnostic_jaxpr_validator: Callable[[Any, int], None] | None = None,
    deployment_target: DeploymentTarget | None = None,
    source_structure_collector: _SourceStructureCollector | None = None,
) -> Union[onnx.ModelProto, ir.Model, str]:
    """One facade pipeline; SOURCE is a private, bounded migration experiment."""
    source = precision_policy is PrecisionPolicy.SOURCE
    with preserve_caller_configuration() if source else nullcontext():
        diagnostic = diagnostic_jaxpr_validator is not None
        if deployment_target is not None:
            if not source or type(deployment_target) is not DeploymentTarget:
                raise ValueError(
                    "Deployment target requires a typed private SOURCE declaration"
                )
        if diagnostic and not source:
            raise ValueError("Diagnostic validation requires the private SOURCE policy")
        if diagnostic and not callable(diagnostic_jaxpr_validator):
            raise TypeError("diagnostic_jaxpr_validator must be callable")
        caller_x64 = bool(jax.config.x64_enabled)
        caller_precision = jax.config.jax_default_matmul_precision
        caller_dtype_promotion = str(jax.config.jax_numpy_dtype_promotion)
        caller_rank_promotion = str(jax.config.jax_numpy_rank_promotion)
        captured_warnings: list[PrecisionWarning] = []
        supplied: tuple[TensorProperties, ...] = ()
        trace_result: _TraceResult | None = None

        def observe_trace(trace: _TraceResult) -> None:
            nonlocal trace_result
            trace_result = trace

        _LOGGER.info(
            f"Converting JAX function to ONNX model with parameters: "
            f"model_name={model_name}, opset={opset}, input_shapes={inputs}, "
            f"input_params={input_params}, "
            f"enable_double_precision={enable_double_precision}, "
            f"record_primitive_calls_file={record_primitive_calls_file}, "
            f"return_mode={return_mode}, output_path={output_path}, "
            f"inputs_as_nchw={inputs_as_nchw}, outputs_as_nchw={outputs_as_nchw}, "
            f"input_names={input_names}, output_names={output_names}, "
            f"export_mode={export_mode}, normalization_mode={normalization_mode}"
        )

        # Determine the nature of the 'inputs' argument to prepare for to_onnx_impl
        normalized_mode = _normalize_return_mode(return_mode)
        normalized_export_mode = _normalize_export_mode(export_mode)
        normalized_normalization_mode = _normalize_normalization_mode(
            normalization_mode
        )
        if source:
            if enable_double_precision:
                _source_migration_gap("enable_double_precision", opset)
            for item in inputs:
                if not isinstance(item, (_SupportsShapeAndDtype, tuple, list)):
                    _source_migration_gap(
                        f"array-like input type {type(item).__name__}", opset
                    )
            supplied = tuple(
                _source_input_properties(item, index=index, opset=opset)
                for index, item in enumerate(inputs)
            )

        file_path: Optional[str] = None
        if normalized_mode == "file":
            if output_path is None:
                raise ValueError(
                    "`output_path` must be provided when return_mode is 'file'."
                )
            file_path = os.fspath(output_path)

        normalized_inputs: List[InputSpec] = []
        if inputs:
            normalized_inputs = _normalize_input_specs(
                inputs, precision_policy=precision_policy
            )
        if source:
            _validate_source_inputs(normalized_inputs, opset, diagnostic=diagnostic)

        normalized_input_names = _normalize_io_names(input_names, kind="input_names")
        normalized_output_names = _normalize_io_names(output_names, kind="output_names")

        if normalized_input_names is not None and len(normalized_input_names) != len(
            normalized_inputs
        ):
            raise ValueError(
                f"input_names length ({len(normalized_input_names)}) must match "
                f"the number of positional inputs ({len(normalized_inputs)})."
            )

        param_map: Dict[str, object] = {}
        if input_params:
            for key, value in input_params.items():
                if not isinstance(key, str):
                    raise TypeError(
                        "input_params must use string keys; "
                        f"received key of type {type(key)}."
                    )
                param_map[key] = value

        if source:
            named_specs = [
                _source_named_input_spec(name, value, opset=opset)
                for name, value in param_map.items()
            ]
            _validate_source_inputs(
                [*normalized_inputs, *named_specs], opset, diagnostic=diagnostic
            )

        if normalized_input_names is not None:
            collisions = sorted(
                set(normalized_input_names).intersection(param_map.keys())
            )
            if collisions:
                names = ", ".join(collisions)
                raise ValueError(
                    f"input_names collide with names reserved by input_params: {names}."
                )

        if normalized_output_names is not None:
            collisions = sorted(
                set(normalized_output_names).intersection(param_map.keys())
            )
            if collisions:
                names = ", ".join(collisions)
                raise ValueError(
                    f"output_names collide with names reserved by input_params: {names}."
                )

        conversion = _convert_to_ir if source else to_onnx_impl
        policy_options: dict[str, Any] = {}
        if source:
            policy_options = {
                "precision_policy": precision_policy,
                "jaxpr_validator": (
                    diagnostic_jaxpr_validator
                    if diagnostic_jaxpr_validator is not None
                    else partial(
                        validate_source_jaxpr, deployment_target=deployment_target
                    )
                ),
                "trace_observer": observe_trace,
                "deployment_target": deployment_target,
            }
        config_scope = (
            preserve_caller_configuration()
            if source
            else _temporary_x64(enable_double_precision)
        )
        warning_scope = (
            _capture_source_warnings(captured_warnings) if source else nullcontext()
        )
        with config_scope, warning_scope:
            result = conversion(
                fn=fn,
                inputs=normalized_inputs,
                input_params=param_map,
                model_name=model_name,
                opset=opset,
                enable_double_precision=enable_double_precision,
                record_primitive_calls_file=record_primitive_calls_file,
                protective_clone=(normalized_mode == "ir"),
                inputs_as_nchw=inputs_as_nchw,
                outputs_as_nchw=outputs_as_nchw,
                input_names=normalized_input_names,
                output_names=normalized_output_names,
                normalization_mode=normalized_normalization_mode,
                **policy_options,
            )

            postprocess_ir_model(
                result,
                promote_to_double=enable_double_precision,
            )

        def _save_model_proto(
            model_proto: onnx.ModelProto,
            dest: str,
            *,
            mode: ExportMode,
            external_threshold: int = 1_048_576,  # 1 MB default before spilling to .data
        ) -> str:
            dest_dir = os.path.dirname(dest)
            if dest_dir:
                os.makedirs(dest_dir, exist_ok=True)
            data_location = os.path.basename(dest) + ".data"
            data_path = os.path.join(dest_dir or ".", data_location)

            if mode == "web":
                onnx.save_model(model_proto, dest, save_as_external_data=False)
                # A previous standard export to the same path may have left a sidecar.
                try:
                    if os.path.exists(data_path):
                        os.remove(data_path)
                except OSError:
                    pass
                return dest

            # Function tensor attributes stay inline across supported ONNX versions.
            # Some versions include them even when convert_attribute is False.
            function_snapshots: list[tuple[onnx.FunctionProto, onnx.FunctionProto]] = []
            for function in model_proto.functions:
                snapshot = onnx.FunctionProto()
                snapshot.CopyFrom(function)
                function_snapshots.append((function, snapshot))
            try:
                onnx.convert_model_to_external_data(
                    model_proto,
                    all_tensors_to_one_file=True,
                    location=data_location,
                    size_threshold=external_threshold,
                    convert_attribute=False,
                )
            finally:
                for function, snapshot in function_snapshots:
                    function.CopyFrom(snapshot)
            onnx.save_model(model_proto, dest, save_as_external_data=False)
            # Only keep the .data sidecar if the export actually referenced external data.
            if not any(init.external_data for init in model_proto.graph.initializer):
                # No external payloads; remove an empty sidecar if one was produced.
                try:
                    if os.path.exists(data_path) and os.path.getsize(data_path) == 0:
                        os.remove(data_path)
                except OSError:
                    pass
            return dest

        source_values: tuple[ir.Value, ...] | None = None
        final_input_names = normalized_input_names
        named_supplied: dict[str, TensorProperties] = {}
        if source:
            assert trace_result is not None
            expected_roles = [
                *((index, None) for index in range(len(normalized_inputs))),
                *((None, name) for name in param_map),
            ]
            if [
                (binding.positional_index, binding.keyword_name)
                for binding in trace_result.source_inputs
            ] != expected_roles:
                raise RuntimeError(
                    "SOURCE input roles/names/order differ from the real call bindings"
                )
            source_values = _source_input_values(
                trace_result, result.graph.inputs, require_internal_names=True
            )
            final_input_names = []
            for binding, value in zip(
                trace_result.source_inputs, source_values, strict=True
            ):
                if binding.positional_index is not None:
                    final_input_names.append(
                        normalized_input_names[binding.positional_index]
                        if normalized_input_names is not None
                        else cast(str, value.name)
                    )
                else:
                    name = cast(str, binding.keyword_name)
                    final_input_names.append(name)
                    named_supplied[name] = TensorProperties(
                        name,
                        normalized_shape_dimensions(binding.spec.shape),
                        np.dtype(binding.spec.dtype).name,
                        bool(binding.spec.weak_type),
                    )
            _separate_source_output_aliases(
                result, final_input_names, normalized_output_names
            )
        else:
            _materialize_input_params_on_ir(result, param_map)
        _apply_custom_io_names_on_ir(
            result,
            input_names=final_input_names,
            output_names=normalized_output_names,
            positional_input_count=len(normalized_inputs),
            input_values=source_values,
        )
        model_proto: onnx.ModelProto | None = None
        source_structure_result: StructureResult | None = None
        source_fingerprint: _SourceArtifactFingerprint | None = None
        source_ir_fingerprint: _SourceArtifactFingerprint | None = None
        if source:
            _finalize_source_deployment(result)
            assert trace_result is not None
            source_output_names = tuple(
                cast(str, value.name) for value in result.graph.outputs
            )
            boundary_names = {
                "input_names": final_input_names,
                "output_names": source_output_names,
            }
            _assert_source_boundaries(result, trace_result, opset, **boundary_names)
            report = _build_source_report(
                result,
                trace_result,
                supplied,
                captured_warnings,
                caller_x64=caller_x64,
                caller_precision=(
                    str(caller_precision) if caller_precision is not None else None
                ),
                caller_dtype_promotion=caller_dtype_promotion,
                caller_rank_promotion=caller_rank_promotion,
                opset=opset,
                diagnostic=diagnostic,
                named_supplied=named_supplied,
                normalization_mode=normalized_normalization_mode,
            )
            attach_precision_report(result, report)

            # Missing semantic rules remain UNVERIFIED. The mandatory boundary
            # below only checks serialization integrity relative to this IR.
            frozen_ir, tensor_guards = _freeze_source_ir(result)
            source_structure_result = validate_source_structure(result)

            # Freeze compact references BEFORE constructing the returned proto.
            # IR return also needs the state after ordinary codec name updates.
            # All passes use the official codec; identical permanent codec faults
            # remain outside this serialization-integrity checkpoint.
            _recheck_source_tensors(tensor_guards)
            try:
                reference_proto = ir.to_proto(frozen_ir)
                source_fingerprint = _source_artifact_fingerprint(reference_proto)
                del reference_proto
                if normalized_mode == "ir":
                    reference_proto = ir.to_proto(frozen_ir)
                    source_ir_fingerprint = _source_artifact_fingerprint(
                        reference_proto
                    )
                    del reference_proto
            finally:
                _restore_source_initializer_names(tensor_guards)
            _recheck_source_tensors(tensor_guards)
            del frozen_ir, tensor_guards
            if validate_source_structure(result) != source_structure_result:
                raise RuntimeError(
                    "SOURCE serialization integrity: IR changed after its audit"
                )
            model_proto = ir.to_proto(result)
            if _source_artifact_fingerprint(model_proto) != source_fingerprint:
                raise RuntimeError(
                    "SOURCE serialization integrity: protobuf differs from audited IR"
                )
            _assert_source_boundaries(
                model_proto, trace_result, opset, **boundary_names
            )
            check_source_schema(model_proto)
            if get_precision_report(model_proto) != report:
                raise RuntimeError(
                    "Source precision metadata failed its serialization roundtrip"
                )

        def observe_source_return(value: object) -> None:
            if (
                source
                and not diagnostic
                and source_structure_collector is not None
                and trace_result is not None
                and source_structure_result is not None
            ):
                # Private evidence cannot change ordinary export success.
                try:
                    source_structure_collector(
                        trace_result, result, source_structure_result, value
                    )
                except Exception:
                    pass

        def recheck_source_ir() -> None:
            if source and validate_source_structure(result) != source_structure_result:
                raise RuntimeError(
                    "SOURCE serialization integrity: IR changed after its audit"
                )

        def source_snapshot(
            path: str,
        ) -> tuple[onnx.ModelProto, _SourceArtifactSnapshot]:
            try:
                return _source_artifact_snapshot(path)
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    "SOURCE serialization integrity: "
                    f"could not verify saved artifact: {exc}"
                ) from exc

        if normalized_mode == "ir":
            observe_source_return(result)
            if source:
                recheck_source_ir()
                # Release the ordinary serialization before checking the actual
                # returned IR again; retain no second complete proto or weights.
                model_proto = None
                current_proto = ir.to_proto(result)
                if _source_artifact_fingerprint(current_proto) != source_ir_fingerprint:
                    raise RuntimeError(
                        "SOURCE serialization integrity: returned IR changed"
                    )
            return result

        if model_proto is None:
            model_proto = ir.to_proto(result)
        if normalized_mode == "file":
            assert file_path is not None
            saved_path = _save_model_proto(
                model_proto, file_path, mode=normalized_export_mode
            )
            if source:
                # The writer has changed this proto's external storage. Release
                # it before parsing the actual saved artifact for verification.
                del model_proto
                assert trace_result is not None
                reloaded, saved_snapshot = source_snapshot(saved_path)
                if saved_snapshot.fingerprint != source_fingerprint:
                    raise RuntimeError(
                        "SOURCE serialization integrity: saved artifact differs from audited IR"
                    )
                _assert_source_boundaries(
                    reloaded, trace_result, opset, **boundary_names
                )
                onnx.checker.check_model(saved_path, full_check=True)
                if get_precision_report(reloaded) != report:
                    raise RuntimeError(
                        "Saved source precision metadata failed its roundtrip"
                    )
                del reloaded
            observe_source_return(saved_path)
            if source:
                reloaded, current_snapshot = source_snapshot(saved_path)
                if current_snapshot != saved_snapshot:
                    raise RuntimeError(
                        "SOURCE serialization integrity: saved artifact changed after verification"
                    )
                del reloaded
            return saved_path
        observe_source_return(model_proto)
        if source:
            if _source_artifact_fingerprint(model_proto) != source_fingerprint:
                raise RuntimeError(
                    "SOURCE serialization integrity: returned protobuf changed"
                )
        return model_proto


@overload
def onnx_function(
    target: None = ...,
    *,
    unique: bool = ...,
    namespace: Optional[str] = ...,
    name: Optional[str] = ...,
    type: Optional[str] = ...,  # noqa: A002 - user-facing keyword
) -> OnnxFunctionDecorator: ...


@overload
def onnx_function(
    target: OnnxFunctionTarget,
    *,
    unique: bool = ...,
    namespace: Optional[str] = ...,
    name: Optional[str] = ...,
    type: Optional[str] = ...,  # noqa: A002 - user-facing keyword
) -> OnnxFunctionTarget: ...


def onnx_function(
    target: Optional[OnnxFunctionTarget] = None,
    *,
    unique: bool = False,
    namespace: Optional[str] = None,
    name: Optional[str] = None,
    type: Optional[str] = None,  # noqa: A002 - user-facing keyword
) -> Union[OnnxFunctionTarget, OnnxFunctionDecorator]:
    """
    Decorator to mark a function or class as an ONNX function.

    This decorator is used to indicate that a function or class should be converted to
    an ONNX function node when included in a model. It allows the function to be traced
    and exported as a reusable component with its own namespace in the ONNX graph.

    Args:
        target: The target function or class to decorate. When omitted, the decorator
            must be called with parentheses.
        unique: If True, reuse a single ONNX Function definition for all call sites
            that share the same callable type and captured parameters.
        namespace: Custom domain prefix for the emitted FunctionProto. Defaults to
            ``"custom"`` when omitted.
        name: Optional human-readable base name for the ONNX function. When set,
            this overrides the callable's Python name for the function `op_type`
            and FunctionProto name; the domain still derives from ``namespace``.
        type: Alias for ``name``; preferred keyword for setting the function
            `op_type`/display name in ONNX. If both ``name`` and ``type`` are
            supplied, ``type`` takes precedence.

    Raises:
        ValueError: If the same target is decorated again with a conflicting
            namespace or display-name override.

    Returns:
        The decorated function or class with ONNX function capabilities.

    Example:
        ```python
        import jax.numpy as jnp
        from flax import nnx

        from jax2onnx import onnx_function


        @onnx_function
        def my_custom_op(x, y):
            return jnp.sin(x) * y


        @onnx_function
        class MLPBlock(nnx.Module):
            def __init__(self, features, rngs):
                self.dense = nnx.Linear(features, rngs=rngs)
                self.activation = nnx.relu

            def __call__(self, x):
                return self.activation(self.dense(x))
        ```
    """

    return cast(
        Union[OnnxFunctionTarget, OnnxFunctionDecorator],
        onnx_function_impl(
            target, unique=unique, namespace=namespace, name=name, type=type
        ),
    )


def allclose(
    fn: Callable[..., Any],
    onnx_model_path: str,
    inputs: Sequence[Any],
    input_params: Optional[Dict[str, Any]] = None,
    rtol: float = 1e-3,
    atol: float = 1e-5,
    *,
    enable_double_precision: bool = False,
    inputs_as_nchw: Optional[Sequence[int]] = None,
    outputs_as_nchw: Optional[Sequence[int]] = None,
) -> Tuple[bool, str]:
    """
    Checks if JAX and ONNX Runtime outputs remain numerically close.

    Args:
        fn: JAX callable to compare against the exported ONNX model.
        onnx_model_path: Path to a serialized model that ORT can execute.
        inputs: Concrete input arrays (or shape tuples, which will be sampled).
        input_params: Optional keyword arguments applied to both call sites.
        rtol: Relative tolerance for floating-point comparisons.
        atol: Absolute tolerance for floating-point comparisons.
        enable_double_precision: Temporarily enable `jax_enable_x64` while running the
            comparison. Defaults to False.

    Returns:
        `(is_match, message)` where `is_match` indicates success and `message`
        provides context when a mismatch occurs.

    Example:
        ```python
        import jax.numpy as jnp

        from jax2onnx import allclose, to_onnx


        def my_func(x):
            return jnp.sin(x)


        model_path = to_onnx(
            my_func,
            inputs=[("B", 10)],
            return_mode="file",
            output_path="my_model.onnx",
        )

        validation_inputs = [(5, 10)]
        is_match, msg = allclose(
            my_func,
            model_path,
            inputs=validation_inputs,
            atol=1e-5,
        )

        assert is_match, f"Validation failed: {msg}"
        ```
    """

    _LOGGER.info(
        "Comparing JAX and ONNX outputs (path=%s, rtol=%s, atol=%s)",
        onnx_model_path,
        rtol,
        atol,
    )

    xs = _validation_inputs_to_arrays(inputs)

    params = dict(input_params or {})
    with _temporary_x64(enable_double_precision):
        with jax.default_matmul_precision("float32"):
            return _run_allclose(
                fn,
                onnx_model_path,
                xs,
                params,
                rtol=rtol,
                atol=atol,
                inputs_as_nchw=inputs_as_nchw,
                outputs_as_nchw=outputs_as_nchw,
            )


def allclose_onnxruntime_web(
    onnx_model_path: str,
    inputs: Sequence[Any],
    input_params: Optional[Dict[str, Any]] = None,
    rtol: float = 1e-3,
    atol: float = 1e-5,
    *,
    inputs_as_nchw: Optional[Sequence[int]] = None,
    node_command: str = "node",
    runner: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Validate that `onnxruntime-web/wasm` matches Python ONNX Runtime CPU output.

    This helper is intended for deployment-readiness checks. It executes the
    serialized model once with Python `onnxruntime` on `CPUExecutionProvider`,
    then executes the same model and feeds through `onnxruntime-web/wasm`.
    The default runner is Node.js; pass `runner="chrome"` or set
    `JAX2ONNX_ONNXRUNTIME_WEB_RUNNER=chrome` to validate in a real browser via
    Playwright/Chromium.

    Args:
        onnx_model_path: Path to a serialized ONNX model, typically produced with
            `to_onnx(..., return_mode="file", export_mode="web")`.
        inputs: Concrete input arrays (or shape tuples, which will be sampled).
        input_params: Optional named ONNX inputs that correspond to runtime
            parameters materialized during export.
        rtol: Relative tolerance for output comparison.
        atol: Absolute tolerance for output comparison.
        inputs_as_nchw: Optional sequence of input indices that should be
            transposed from NHWC validation arrays to NCHW ONNX feeds.
        node_command: Node.js executable used to run the Web/WASM validator.
        runner: Optional runtime selector. Use `"node"` for Node.js WASM
            validation or `"chrome"`/`"browser"` for Playwright Chromium.

    Returns:
        `(is_match, message)` where `is_match` indicates success and `message`
        contains the Web runner result or failure details.

    Example:
        ```python
        import numpy as np

        from jax2onnx import allclose_onnxruntime_web


        is_match, message = allclose_onnxruntime_web(
            "model.web.onnx",
            inputs=[np.zeros((2, 10), dtype=np.float32)],
            rtol=1e-5,
            atol=1e-5,
            runner="node",
        )

        assert is_match, message
        ```
    """

    xs = _validation_inputs_to_arrays(inputs)
    params = dict(input_params or {})

    try:
        ort = cast(Any, importlib.import_module("onnxruntime"))
    except ImportError as exc:
        return False, f"onnxruntime is required for CPU reference execution: {exc}"

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_options.enable_mem_pattern = False
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1

    try:
        session = ort.InferenceSession(
            onnx_model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:  # pragma: no cover - defensive aid
        return False, f"Failed to create CPU ONNX Runtime session: {exc}"

    ort_xs = list(xs)
    if inputs_as_nchw:
        nhwc_to_nchw = [0, 3, 1, 2]
        for idx in inputs_as_nchw:
            if 0 <= idx < len(ort_xs):
                val = ort_xs[idx]
                if hasattr(val, "ndim") and val.ndim == 4:
                    ort_xs[idx] = np.transpose(val, nhwc_to_nchw)

    try:
        ort_inputs = _build_ort_inputs(session, ort_xs, params)
        expected_outputs = session.run(None, ort_inputs)
    except Exception as exc:  # pragma: no cover - defensive aid
        return False, f"Failed to run CPU ONNX Runtime reference: {exc}"

    output_names = [output_meta.name for output_meta in session.get_outputs()]
    if len(output_names) != len(expected_outputs):
        return (
            False,
            f"Output metadata mismatch (names={len(output_names)} outputs={len(expected_outputs)})",
        )

    try:
        spec = {
            "modelPath": str(Path(onnx_model_path).resolve()),
            "rtol": float(rtol),
            "atol": float(atol),
            "inputs": [
                _array_to_onnxruntime_web_tensor(name, value)
                for name, value in ort_inputs.items()
            ],
            "outputs": [
                _array_to_onnxruntime_web_tensor(name, value)
                for name, value in zip(output_names, expected_outputs, strict=True)
            ],
        }
    except TypeError as exc:
        return False, str(exc)

    configured_runner = (
        runner
        if runner is not None
        else os.getenv("JAX2ONNX_ONNXRUNTIME_WEB_RUNNER", "node")
    )
    runner_name = configured_runner.strip().lower()
    if runner_name == "node":
        script_name = "validate_onnxruntime_web.mjs"
    elif runner_name in {"browser", "chrome"}:
        script_name = "validate_onnxruntime_web_chrome.mjs"
        runner_name = "chrome"
    else:
        return (
            False,
            "Unsupported onnxruntime-web runner "
            f"'{runner_name}'. Use 'node' or 'chrome'.",
        )

    script_path = Path(__file__).resolve().parent.parent / "scripts" / script_name
    if not script_path.is_file():
        return (
            False,
            f"onnxruntime-web/{runner_name} validation script not found: {script_path}",
        )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", encoding="utf-8", delete=False
    ) as tmp:
        spec_path = Path(tmp.name)
        json.dump(spec, tmp, allow_nan=False)

    try:
        completed = subprocess.run(
            [node_command, str(script_path), str(spec_path)],
            cwd=str(script_path.parent.parent),
            text=True,
            capture_output=True,
            check=False,
            env=os.environ.copy(),
        )
    except FileNotFoundError:
        return False, f"Node.js command not found: {node_command}"
    finally:
        try:
            spec_path.unlink()
        except OSError:
            pass

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        details = stderr or stdout or f"exit code {completed.returncode}"
        return False, f"onnxruntime-web/{runner_name} validation failed: {details}"

    summary = (
        completed.stdout.strip() or f"onnxruntime-web/{runner_name} outputs match."
    )
    return True, summary


def _validation_inputs_to_arrays(inputs: Sequence[Any]) -> List[Any]:
    def _is_shape(x: Any) -> bool:
        return isinstance(x, (tuple, list)) and all(
            isinstance(dim, (int, str)) for dim in x
        )

    if all(_is_shape(x) for x in inputs):
        return [
            np.random.rand(*[d if isinstance(d, int) else 2 for d in shape]).astype(
                np.float32
            )
            for shape in inputs
        ]
    return list(inputs)


def _array_to_onnxruntime_web_tensor(name: str, value: Any) -> Dict[str, Any]:
    arr = np.asarray(value)
    tensor_type = _onnxruntime_web_tensor_type(arr.dtype)
    if arr.dtype == np.dtype(np.float16):
        data: list[Any] = (
            np.ascontiguousarray(arr).view(np.uint16).reshape(-1).astype(int).tolist()
        )
    else:
        data = [_json_safe_scalar(item, arr.dtype) for item in arr.reshape(-1)]
    return {
        "name": name,
        "type": tensor_type,
        "dims": [int(dim) for dim in arr.shape],
        "data": data,
    }


def _onnxruntime_web_tensor_type(dtype: np.dtype[Any]) -> str:
    np_dtype = np.dtype(dtype)
    dtype_map = {
        np.dtype(np.float16): "float16",
        np.dtype(np.float32): "float32",
        np.dtype(np.float64): "float64",
        np.dtype(np.uint8): "uint8",
        np.dtype(np.int8): "int8",
        np.dtype(np.uint16): "uint16",
        np.dtype(np.int16): "int16",
        np.dtype(np.uint32): "uint32",
        np.dtype(np.int32): "int32",
        np.dtype(np.uint64): "uint64",
        np.dtype(np.int64): "int64",
        np.dtype(np.bool_): "bool",
    }
    mapped = dtype_map.get(np_dtype)
    if mapped is not None:
        return mapped
    if np_dtype.kind in ("U", "S"):
        return "string"
    raise TypeError(f"onnxruntime-web validation does not support dtype {np_dtype}.")


def _json_safe_scalar(value: Any, dtype: np.dtype[Any]) -> Any:
    np_dtype = np.dtype(dtype)
    if np_dtype.kind == "b":
        return bool(value)
    if np_dtype.kind in ("i", "u"):
        as_int = int(value)
        if np_dtype.itemsize >= 8:
            return str(as_int)
        return as_int
    if np_dtype.kind == "f":
        as_float = float(value)
        if np.isnan(as_float):
            return "NaN"
        if np.isposinf(as_float):
            return "Infinity"
        if np.isneginf(as_float):
            return "-Infinity"
        return as_float
    if np_dtype.kind in ("U", "S"):
        return str(value)
    raise TypeError(f"Cannot serialize dtype {np_dtype} for onnxruntime-web.")


def _run_allclose(
    fn: Callable[..., Any],
    model_path: str,
    xs: List[Any],
    params: Dict[str, Any],
    *,
    rtol: float,
    atol: float,
    inputs_as_nchw: Optional[Sequence[int]] = None,
    outputs_as_nchw: Optional[Sequence[int]] = None,
) -> Tuple[bool, str]:
    ort = cast(Any, importlib.import_module("onnxruntime"))

    # Use single-threaded ORT to reduce nondeterminism across runs.
    sess_options = ort.SessionOptions()
    # Disable ORT graph rewrites (e.g., QuickGelu fusion) that may not support
    # float64 and can break validation for otherwise valid models.
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    # Avoid buffer re-use across dynamic shapes (Range/If) that can trigger ORT
    # shape mismatch errors during validation.
    sess_options.enable_mem_pattern = False

    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    # Prepare ORT inputs (handling potential NCHW transposition)
    ort_xs = list(xs)
    if inputs_as_nchw:
        nhwc_to_nchw = [0, 3, 1, 2]
        for idx in inputs_as_nchw:
            if 0 <= idx < len(ort_xs):
                # Transpose input to NCHW for ORT
                # Assumes input is 4D (NHWC)
                val = ort_xs[idx]
                if hasattr(val, "ndim") and val.ndim == 4:
                    ort_xs[idx] = np.transpose(val, nhwc_to_nchw)

    ort_inputs = _build_ort_inputs(session, ort_xs, params)
    ort_outputs = session.run(None, ort_inputs)

    jax_args = [_to_jax_array(x) for x in xs]
    jax_kwargs = {k: _to_jax_kwarg(v) for k, v in params.items()}

    try:
        jax_result = fn(*jax_args, **jax_kwargs)
    except Exception as exc:  # pragma: no cover - defensive aid
        return False, f"Failed to evaluate JAX function: {exc}"

    jax_host = jax.device_get(jax_result)
    jax_flat, _ = jax.tree_util.tree_flatten(jax_host)
    jax_outputs = [_to_numpy_output(val) for val in jax_flat]

    if len(jax_outputs) != len(ort_outputs):
        return (
            False,
            f"Output count mismatch (JAX={len(jax_outputs)} vs ORT={len(ort_outputs)})",
        )

    for idx, (expected, got) in enumerate(zip(jax_outputs, ort_outputs)):
        expected_arr = np.asarray(expected)
        got_arr = np.asarray(got)

        # Handle NCHW output transposition if needed
        if outputs_as_nchw and idx in outputs_as_nchw:
            # ORT output is NCHW, JAX is NHWC. Transpose ORT back to NHWC.
            nchw_to_nhwc = [0, 2, 3, 1]
            if got_arr.ndim == 4:
                got_arr = np.transpose(got_arr, nchw_to_nhwc)

        if np.issubdtype(expected_arr.dtype, np.complexfloating) and (
            np.issubdtype(got_arr.dtype, np.floating)
            or got_arr.dtype.name == "bfloat16"
        ):
            if (
                got_arr.ndim == expected_arr.ndim + 1
                and got_arr.shape[:-1] == expected_arr.shape
                and got_arr.shape[-1] == 2
            ):
                got_unpacked = (
                    got_arr.astype(np.float32)
                    if got_arr.dtype.name == "bfloat16"
                    else got_arr
                )
                got_arr = got_unpacked[..., 0] + 1j * got_unpacked[..., 1]
                got_arr = got_arr.astype(expected_arr.dtype, copy=False)

        if expected_arr.shape != got_arr.shape:
            return (
                False,
                "Output {} shape mismatch (JAX {} vs ORT {})".format(
                    idx, expected_arr.shape, got_arr.shape
                ),
            )

        if _is_floating_dtype(expected_arr) or _is_floating_dtype(got_arr):
            expected_cmp = (
                expected_arr.astype(np.float32)
                if expected_arr.dtype.name == "bfloat16"
                else expected_arr
            )
            got_cmp = (
                got_arr.astype(np.float32)
                if got_arr.dtype.name == "bfloat16"
                else got_arr
            )
            if not np.issubdtype(got_cmp.dtype, np.complexfloating) or np.issubdtype(
                expected_cmp.dtype, np.complexfloating
            ):
                got_cmp = got_cmp.astype(expected_cmp.dtype, copy=False)
            if not np.allclose(
                expected_cmp,
                got_cmp,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            ):
                diff = np.abs(expected_cmp - got_cmp)
                max_diff = float(diff.max()) if diff.size else 0.0
                return (
                    False,
                    f"Output {idx} mismatch (max abs diff {max_diff}, rtol={rtol}, atol={atol})",
                )
        else:
            if not np.array_equal(
                expected_arr, got_arr.astype(expected_arr.dtype, copy=False)
            ):
                return (False, f"Output {idx} mismatch (non-floating tensors differ)")

    return True, "Outputs match within tolerance."


def _build_ort_inputs(
    session: Any,
    xs: List[Any],
    params: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    feed: Dict[str, np.ndarray] = {}
    positional_iter = iter(xs)

    for input_meta in session.get_inputs():
        name = input_meta.name
        if name in params:
            feed[name] = _to_numpy_input(params[name], input_meta)
        else:
            try:
                value = next(positional_iter)
            except StopIteration as exc:  # pragma: no cover - defensive
                raise ValueError(
                    f"Not enough positional inputs provided for ORT (missing value for '{name}')"
                ) from exc
            feed[name] = _to_numpy_input(value, input_meta)

    try:
        next(positional_iter)
    except StopIteration:
        pass
    else:  # pragma: no cover - defensive
        raise ValueError("Too many positional inputs supplied for ORT session")

    return feed


def _to_numpy_input(value: Any, input_meta: Any) -> np.ndarray:
    arr = cast(np.ndarray, np.asarray(value))
    expected_dtype = getattr(input_meta, "type", None)

    if isinstance(expected_dtype, str) and expected_dtype.startswith("tensor("):
        dtype_name = expected_dtype[len("tensor(") : -1]
        bfloat16_type = jnp.dtype("bfloat16").type
        dtype_map = {
            "float": np.float32,
            "double": np.float64,
            "float16": np.float16,
            "bf16": bfloat16_type,
            "bfloat16": bfloat16_type,
            "int64": np.int64,
            "int32": np.int32,
            "int16": np.int16,
            "int8": np.int8,
            "uint64": np.uint64,
            "uint32": np.uint32,
            "uint16": np.uint16,
            "uint8": np.uint8,
            "bool": np.bool_,
        }
        target_dtype = dtype_map.get(dtype_name)
        if target_dtype is not None:
            if np.issubdtype(arr.dtype, np.complexfloating) and (
                np.issubdtype(target_dtype, np.floating)
                or np.dtype(target_dtype).name == "bfloat16"
            ):
                expected_shape = getattr(input_meta, "shape", None)
                if expected_shape is not None and len(expected_shape) == arr.ndim + 1:
                    last_dim = expected_shape[-1]
                    last_dim_as_int: int | None
                    try:
                        last_dim_as_int = int(last_dim)
                    except (TypeError, ValueError):
                        last_dim_as_int = None
                    if last_dim_as_int is not None and last_dim_as_int != 2:
                        raise ValueError(
                            "Expected trailing dimension of size 2 to pack complex input."
                        )
                    packed = np.stack([arr.real, arr.imag], axis=-1).astype(
                        target_dtype
                    )
                    return cast(np.ndarray, packed)
                raise ValueError(
                    "Cannot map complex input to expected real tensor without trailing dimension of size 2."
                )
            if arr.dtype != target_dtype:
                arr = arr.astype(target_dtype)

    return arr


def _to_jax_array(value: Any) -> jax.Array:
    return cast(jax.Array, jnp.asarray(value))


def _to_jax_kwarg(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return jnp.asarray(value)
    if isinstance(value, (list, tuple)):
        return jnp.asarray(value)
    return value


def _to_numpy_output(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return cast(np.ndarray, np.asarray(value))


def _is_floating_dtype(arr: np.ndarray) -> bool:
    return (
        arr.dtype.name == "bfloat16"
        or np.issubdtype(arr.dtype, np.floating)
        or np.issubdtype(arr.dtype, np.complexfloating)
    )
