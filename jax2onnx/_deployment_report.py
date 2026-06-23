# jax2onnx/_deployment_report.py

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

import onnx


ShapeDim: TypeAlias = int | str | None
ModelSource: TypeAlias = onnx.ModelProto | str | os.PathLike[str]


@dataclass(frozen=True)
class CheckSummary:
    """Pass/fail status for one deployment-readiness check."""

    ok: bool
    message: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "message": self.message}


@dataclass(frozen=True)
class OpsetSummary:
    """ONNX opset import used by the model."""

    domain: str
    version: int

    def to_dict(self) -> dict[str, object]:
        return {"domain": self.domain, "version": self.version}


@dataclass(frozen=True)
class TensorSummary:
    """Name, dtype, and shape summary for a public tensor or initializer."""

    name: str
    dtype: str
    shape: tuple[ShapeDim, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
        }


@dataclass(frozen=True)
class OperatorSummary:
    """Count of one operator type in the model, including subgraphs."""

    domain: str
    op_type: str
    count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "domain": self.domain,
            "op_type": self.op_type,
            "count": self.count,
        }


@dataclass(frozen=True)
class DeploymentReadinessReport:
    """Serializable deployment summary for an exported ONNX model."""

    model_name: str
    producer_name: str
    ir_version: int
    opsets: tuple[OpsetSummary, ...]
    inputs: tuple[TensorSummary, ...]
    outputs: tuple[TensorSummary, ...]
    initializers: tuple[TensorSummary, ...]
    operators: tuple[OperatorSummary, ...]
    checker: CheckSummary
    shape_inference: CheckSummary
    warnings: tuple[str, ...]

    @property
    def is_ready(self) -> bool:
        return self.checker.ok and self.shape_inference.ok

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "producer_name": self.producer_name,
            "ir_version": self.ir_version,
            "opsets": [opset.to_dict() for opset in self.opsets],
            "inputs": [value.to_dict() for value in self.inputs],
            "outputs": [value.to_dict() for value in self.outputs],
            "initializers": [value.to_dict() for value in self.initializers],
            "operators": [operator.to_dict() for operator in self.operators],
            "checker": self.checker.to_dict(),
            "shape_inference": self.shape_inference.to_dict(),
            "warnings": list(self.warnings),
            "is_ready": self.is_ready,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def deployment_readiness_report(
    model: ModelSource,
    *,
    strict_shape_inference: bool = True,
) -> DeploymentReadinessReport:
    """Build a deployment-readiness report for a ModelProto or ONNX file path."""

    model_proto = _load_model(model)
    checker = _run_checker(model_proto)
    inferred_model, shape_inference = _run_shape_inference(
        model_proto,
        strict_shape_inference=strict_shape_inference,
    )
    summary_model = inferred_model if inferred_model is not None else model_proto

    inputs = tuple(_summarize_value_info(value) for value in summary_model.graph.input)
    outputs = tuple(
        _summarize_value_info(value) for value in summary_model.graph.output
    )
    initializers = tuple(
        _summarize_initializer(initializer)
        for initializer in summary_model.graph.initializer
    )
    operators = _summarize_operators(model_proto)
    warnings = _collect_warnings(inputs=inputs, outputs=outputs)

    return DeploymentReadinessReport(
        model_name=summary_model.graph.name,
        producer_name=summary_model.producer_name,
        ir_version=summary_model.ir_version,
        opsets=tuple(
            OpsetSummary(
                domain=opset.domain or "ai.onnx",
                version=opset.version,
            )
            for opset in summary_model.opset_import
        ),
        inputs=inputs,
        outputs=outputs,
        initializers=initializers,
        operators=operators,
        checker=checker,
        shape_inference=shape_inference,
        warnings=warnings,
    )


def _load_model(model: ModelSource) -> onnx.ModelProto:
    if isinstance(model, onnx.ModelProto):
        return model
    return onnx.load_model(os.fspath(model))


def _run_checker(model: onnx.ModelProto) -> CheckSummary:
    try:
        onnx.checker.check_model(model)
    except Exception as exc:
        return CheckSummary(ok=False, message=str(exc))
    return CheckSummary(ok=True)


def _run_shape_inference(
    model: onnx.ModelProto,
    *,
    strict_shape_inference: bool,
) -> tuple[onnx.ModelProto | None, CheckSummary]:
    try:
        inferred = onnx.shape_inference.infer_shapes(
            model,
            strict_mode=strict_shape_inference,
        )
    except Exception as exc:
        return None, CheckSummary(ok=False, message=str(exc))
    return inferred, CheckSummary(ok=True)


def _summarize_value_info(value_info: onnx.ValueInfoProto) -> TensorSummary:
    type_proto = value_info.type
    if not type_proto.HasField("tensor_type"):
        return TensorSummary(
            name=value_info.name,
            dtype="NON_TENSOR",
            shape=(),
        )

    tensor_type = type_proto.tensor_type
    dtype = _tensor_dtype_name(tensor_type.elem_type)
    if not tensor_type.HasField("shape"):
        return TensorSummary(
            name=value_info.name,
            dtype=dtype,
            shape=(),
        )

    dims: list[ShapeDim] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append(None)

    return TensorSummary(
        name=value_info.name,
        dtype=dtype,
        shape=tuple(dims),
    )


def _summarize_initializer(initializer: onnx.TensorProto) -> TensorSummary:
    return TensorSummary(
        name=initializer.name,
        dtype=_tensor_dtype_name(initializer.data_type),
        shape=tuple(int(dim) for dim in initializer.dims),
    )


def _tensor_dtype_name(elem_type: int) -> str:
    if elem_type == onnx.TensorProto.UNDEFINED:
        return "UNDEFINED"
    try:
        return cast(str, onnx.TensorProto.DataType.Name(elem_type))
    except ValueError:
        return f"UNKNOWN_{elem_type}"


def _summarize_operators(model: onnx.ModelProto) -> tuple[OperatorSummary, ...]:
    counts: Counter[tuple[str, str]] = Counter()
    for node in _iter_nodes(model.graph):
        domain = node.domain or "ai.onnx"
        counts[(domain, node.op_type)] += 1

    return tuple(
        OperatorSummary(domain=domain, op_type=op_type, count=count)
        for (domain, op_type), count in sorted(counts.items())
    )


def _iter_nodes(graph: onnx.GraphProto) -> tuple[onnx.NodeProto, ...]:
    nodes: list[onnx.NodeProto] = []
    for node in graph.node:
        nodes.append(node)
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                nodes.extend(_iter_nodes(attribute.g))
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    nodes.extend(_iter_nodes(subgraph))
    return tuple(nodes)


def _collect_warnings(
    *,
    inputs: tuple[TensorSummary, ...],
    outputs: tuple[TensorSummary, ...],
) -> tuple[str, ...]:
    warnings: list[str] = []
    for role, values in (("input", inputs), ("output", outputs)):
        for value in values:
            for axis, dim in enumerate(value.shape):
                if dim is None:
                    warnings.append(
                        f"{role} '{value.name}' has unknown dimension at axis {axis}."
                    )
                elif isinstance(dim, str):
                    warnings.append(
                        f"{role} '{value.name}' has symbolic dimension '{dim}' "
                        f"at axis {axis}."
                    )
    return tuple(warnings)


def write_deployment_readiness_report(
    model: ModelSource,
    output_path: str | os.PathLike[str],
    *,
    strict_shape_inference: bool = True,
    indent: int = 2,
) -> DeploymentReadinessReport:
    """Write a deployment-readiness report JSON file and return the report."""

    report = deployment_readiness_report(
        model,
        strict_shape_inference=strict_shape_inference,
    )
    Path(output_path).write_text(report.to_json(indent=indent) + "\n", encoding="utf-8")
    return report
