# jax2onnx/converter/source_deployment_bindings.py

"""Private, clone-preserved identities for qualified SOURCE matrix sites.

Public IR ``meta`` is shallow-copied by Graph.clone and is never serialized.
The markers identify real definitions; names are read only after resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Iterator, Sequence

import onnx_ir as ir

from jax2onnx.ir_utils import iter_ir_functions
from .source_deployment import (
    DEPLOYMENT_METADATA_KEY,
    DeploymentRequirements,
    DeploymentTarget,
    MatrixPrecisionRequirement,
    deployment_requirements_from_json,
    deployment_requirements_to_json,
)

_GRAPH: str = "jax2onnx.deployment.owner"
_PRODUCER: str = "jax2onnx.deployment.producer"
_VALUES: str = "jax2onnx.deployment.values"
_REGISTRY: str = "jax2onnx.deployment.bindings"


@dataclass(frozen=True, eq=False)
class _ValueWitness:
    owner: object
    producer: object
    port: int


@dataclass(frozen=True)
class MatrixSiteBinding:
    requirement: MatrixPrecisionRequirement
    logical: _ValueWitness
    matrix: _ValueWitness


@dataclass
class _BindingRegistry:
    bindings: tuple[MatrixSiteBinding, ...]
    expected: DeploymentRequirements


def _witness(value: ir.Value) -> _ValueWitness:
    graph, producer, port = value.graph, value.producer(), value.index()
    if (
        graph is None
        or producer is None
        or producer.graph is not graph
        or not any(node is producer for node in graph)
        or type(port) is not int
        or port < 0
        or port >= len(producer.outputs)
        or producer.outputs[port] is not value
    ):
        raise ValueError("Deployment binding requires an actual owned producer port")
    owner = graph.meta.setdefault(_GRAPH, object())
    origin = producer.meta.setdefault(_PRODUCER, object())
    if type(owner) is not object or type(origin) is not object:
        raise ValueError("Deployment owner marker is invalid")
    witness = _ValueWitness(owner, origin, port)
    previous = value.meta.get(_VALUES, ())
    if type(previous) is not tuple or any(
        type(item) is not _ValueWitness for item in previous
    ):
        raise ValueError("Deployment Value markers are invalid")
    value.meta[_VALUES] = (*previous, witness)
    return witness


def record_matrix_site(
    requirement: MatrixPrecisionRequirement, logical: ir.Value, matrix: ir.Value
) -> MatrixSiteBinding:
    if logical.graph is not matrix.graph:
        raise ValueError("Deployment matrix and logical result have different owners")
    logical_witness = _witness(logical)
    return MatrixSiteBinding(
        requirement,
        logical_witness,
        logical_witness if logical is matrix else _witness(matrix),
    )


def owned_graphs(model: ir.Model) -> Iterator[ir.Graph]:
    """Visit actual lexical Graph definitions, rejecting multiple containment."""
    seen: set[int] = set()

    def visit(graph: ir.Graph) -> Iterator[ir.Graph]:
        if id(graph) in seen:
            raise ValueError("Deployment Graph has ambiguous lexical containment")
        seen.add(id(graph))
        yield graph
        for node in graph:
            if node.graph is not graph:
                raise ValueError("Deployment node has a stale Graph owner")
            for attribute in node.attributes.values():
                if attribute.type is ir.AttributeType.GRAPH:
                    yield from visit(attribute.as_graph())
                elif attribute.type is ir.AttributeType.GRAPHS:
                    for child in attribute.as_graphs():
                        yield from visit(child)

    yield from visit(model.graph)
    for function in iter_ir_functions(model.functions):
        yield from visit(function.graph)


def _resolve(
    model: ir.Model, bindings: Sequence[MatrixSiteBinding]
) -> tuple[tuple[ir.Value, ir.Value], ...]:
    wanted = {
        witness for binding in bindings for witness in (binding.logical, binding.matrix)
    }
    found: dict[_ValueWitness, list[ir.Value]] = {witness: [] for witness in wanted}
    for graph in owned_graphs(model):
        values = [*graph.inputs, *graph.initializers.values()]
        for node in graph:
            for port, value in enumerate(node.outputs):
                if (
                    value.graph is not graph
                    or value.producer() is not node
                    or value.index() != port
                ):
                    raise ValueError("Deployment final producer port is not owned")
                values.append(value)
        for value in values:
            markers = value.meta.get(_VALUES, ())
            if type(markers) is not tuple or any(
                type(item) is not _ValueWitness for item in markers
            ):
                raise ValueError("Deployment final Value markers are invalid")
            for witness in markers:
                if witness not in wanted:
                    continue
                producer = value.producer()
                if (
                    value.graph is not graph
                    or graph.meta.get(_GRAPH) is not witness.owner
                    or producer is None
                    or producer.graph is not graph
                    or producer.meta.get(_PRODUCER) is not witness.producer
                    or value.index() != witness.port
                    or producer.outputs[witness.port] is not value
                ):
                    raise ValueError(
                        "Deployment Value marker has a stale or wrong owner/port"
                    )
                found[witness].append(value)
    if any(len(values) != 1 for values in found.values()):
        raise ValueError(
            "Deployment Value identity is missing or ambiguous after optimization"
        )
    result = tuple(
        (found[binding.logical][0], found[binding.matrix][0]) for binding in bindings
    )
    if any(logical.graph is not matrix.graph for logical, matrix in result):
        raise ValueError("Deployment final logical result crosses a Graph owner")
    return result


def _named_requirements(
    requirements: DeploymentRequirements, values: Sequence[tuple[ir.Value, ir.Value]]
) -> DeploymentRequirements:
    sites = []
    for site, (logical, matrix) in zip(
        requirements.matrix_precision, values, strict=True
    ):
        if not logical.name or not matrix.name:
            raise ValueError("Deployment qualified result lost its final IR name")
        sites.append(replace(site, output=logical.name, matrix_output=matrix.name))
    return DeploymentRequirements(requirements.target, tuple(sites))


def attach_matrix_bindings(
    model: ir.Model, target: DeploymentTarget, bindings: Sequence[MatrixSiteBinding]
) -> None:
    if _REGISTRY in model.meta or DEPLOYMENT_METADATA_KEY in model.metadata_props:
        raise ValueError("Deployment binding registry was already attached")
    ordered = tuple(bindings)
    requirements = DeploymentRequirements(
        target, tuple(binding.requirement for binding in ordered)
    )
    requirements = _named_requirements(requirements, _resolve(model, ordered))
    model.meta[_REGISTRY] = _BindingRegistry(ordered, requirements)
    model.metadata_props[DEPLOYMENT_METADATA_KEY] = deployment_requirements_to_json(
        requirements
    )


def resolve_matrix_bindings(
    model: ir.Model,
) -> tuple[DeploymentRequirements, tuple[tuple[ir.Value, ir.Value], ...]]:
    registry = model.meta.get(_REGISTRY)
    requirements = deployment_requirements_from_json(
        model.metadata_props.get(DEPLOYMENT_METADATA_KEY)
    )
    if type(registry) is not _BindingRegistry or requirements != registry.expected:
        raise ValueError("Deployment metadata differs from its actual binding registry")
    if len(registry.bindings) != len(registry.expected.matrix_precision):
        raise ValueError("Deployment binding registry cardinality changed")
    for binding, site in zip(
        registry.bindings, registry.expected.matrix_precision, strict=True
    ):
        original = binding.requirement
        if (
            replace(
                site,
                output=original.output,
                matrix_output=original.matrix_output,
                matrix_scope=original.matrix_scope,
                matrix_tag=original.matrix_tag,
            )
            != original
        ):
            raise ValueError(
                "Deployment binding registry requirement semantics changed"
            )
    return registry.expected, _resolve(model, registry.bindings)


def update_matrix_bindings(
    model: ir.Model, requirements: DeploymentRequirements
) -> None:
    previous, values = resolve_matrix_bindings(model)
    expected = _named_requirements(previous, values)
    # Finalization may add only the verified producer scope/tag to each site.
    if requirements.target != expected.target or len(
        requirements.matrix_precision
    ) != len(expected.matrix_precision):
        raise ValueError("Deployment binding update changed its target or cardinality")
    for site, wanted in zip(
        requirements.matrix_precision, expected.matrix_precision, strict=True
    ):
        if (
            replace(
                site, matrix_scope=wanted.matrix_scope, matrix_tag=wanted.matrix_tag
            )
            != wanted
        ):
            raise ValueError("Deployment binding update changed requirement semantics")
    registry = model.meta[_REGISTRY]
    assert type(registry) is _BindingRegistry
    registry.expected = requirements
    model.metadata_props[DEPLOYMENT_METADATA_KEY] = deployment_requirements_to_json(
        requirements
    )


def refresh_matrix_names(model: ir.Model) -> None:
    requirements, values = resolve_matrix_bindings(model)
    update_matrix_bindings(model, _named_requirements(requirements, values))
