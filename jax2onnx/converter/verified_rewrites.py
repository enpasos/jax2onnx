# jax2onnx/converter/verified_rewrites.py

"""Shared optimizer rules with independently checkable source/use rewrites.

Candidate discovery never grants permission to mutate the graph. Every export
rechecks ``can_replace_node`` before mutation. Annotated exports additionally
verify each substitution against the metadata registry's sealed topology.
"""

from __future__ import annotations

from collections.abc import Iterator
from math import prod
import struct
from typing import TYPE_CHECKING, cast

import numpy as np
import onnx_ir as ir

from jax2onnx.ir_utils import const_value_to_numpy

from .initializer_metadata import is_annotated

if TYPE_CHECKING:
    from .initializer_metadata import InitializerMetadataRegistry


# Only deterministic, standard operators with one ordinary tensor result are
# admitted. In particular, no Constant, random, training, control-flow, custom,
# or local-Function operations can enter this table.
_CSE_OPS: frozenset[str] = frozenset(
    {
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Identity",
        "Cast",
        "Reshape",
        "Transpose",
        "Unsqueeze",
        "Squeeze",
    }
)


def _standard(node: ir.Node, op_type: str) -> bool:
    return node.domain == "" and not node.overload and node.op_type == op_type


def _static_shape(value: ir.Value) -> tuple[int, ...] | None:
    if value.shape is None:
        return None
    dims = tuple(value.shape)
    if any(not isinstance(dim, int) or dim < 0 for dim in dims):
        return None
    return cast(tuple[int, ...], dims)


def _same_description(first: ir.Value, second: ir.Value) -> bool:
    return (
        first.type == second.type
        and first.shape == second.shape
        and dict(first.metadata_props) == dict(second.metadata_props)
        and first.doc_string == second.doc_string
    )


def _discardable(node: ir.Node) -> bool:
    return (
        node.graph is not None
        and len(node.outputs) == 1
        and not node.outputs[0].is_graph_output()
        and not node.outputs[0].is_initializer()
        and node.outputs[0].const_value is None
    )


def _identity_input(node: ir.Node) -> ir.Value | None:
    if not _standard(node, "Identity") or node.attributes or len(node.inputs) != 1:
        return None
    source = node.inputs[0]
    if (
        source is None
        or len(node.outputs) != 1
        or node.metadata_props
        or node.doc_string
        or not _same_description(source, node.outputs[0])
    ):
        return None
    return source


def _shape_operand(value: ir.Value, graph: ir.Graph) -> tuple[int, ...] | None:
    tensor = value.const_value
    if (
        value.graph is not graph
        or value.is_graph_input()
        or tensor is None
        or tensor.metadata_props
        or is_annotated(value)
        or value.dtype != ir.DataType.INT64
    ):
        return None
    if not value.is_initializer():
        producer = value.producer()
        if (
            producer is None
            or producer.graph is not graph
            or not _standard(producer, "Constant")
            or producer.inputs
            or set(producer.attributes) != {"value"}
        ):
            return None
        attribute = producer.attributes["value"]
        if (
            attribute.is_ref()
            or attribute.type != ir.AttributeType.TENSOR
            or attribute.as_tensor() is not tensor
        ):
            return None
    array = const_value_to_numpy(value)
    if array is None or array.dtype != np.dtype(np.int64) or array.ndim != 1:
        return None
    if value.shape != ir.Shape(array.shape):
        return None
    return tuple(int(dim) for dim in array)


def _reshape_source(node: ir.Node) -> ir.Value | None:
    """Verify a static Reshape, including its exact input/output descriptions."""
    graph = node.graph
    if (
        graph is None
        or not _standard(node, "Reshape")
        or len(node.inputs) != 2
        or len(node.outputs) != 1
        or set(node.attributes) - {"allowzero"}
        or node.metadata_props
        or node.doc_string
    ):
        return None
    source, shape_value = node.inputs
    if source is None or shape_value is None:
        return None
    source_shape = _static_shape(source)
    target = _shape_operand(shape_value, graph)
    if source_shape is None or target is None:
        return None
    allowzero = 0
    if "allowzero" in node.attributes:
        attr = node.attributes["allowzero"]
        if attr.is_ref() or attr.type != ir.AttributeType.INT:
            return None
        allowzero = attr.as_int()
        if allowzero not in (0, 1):
            return None
    if any(dim < -1 for dim in target) or target.count(-1) > 1:
        return None
    resolved = list(target)
    for axis, dim in enumerate(resolved):
        if dim == 0 and not allowzero:
            if axis >= len(source_shape):
                return None
            resolved[axis] = source_shape[axis]
    if -1 in resolved:
        known_size = prod(dim for dim in resolved if dim != -1)
        if known_size == 0 or prod(source_shape) % known_size:
            return None
        resolved[resolved.index(-1)] = prod(source_shape) // known_size
    if prod(resolved) != prod(source_shape):
        return None
    output = node.outputs[0]
    if (
        source.dtype is None
        or source.dtype != output.dtype
        or _static_shape(output) != tuple(resolved)
        or output.metadata_props
        or output.doc_string
    ):
        return None
    return source


def _identity_replacement(node: ir.Node) -> ir.Value | None:
    source = _identity_input(node)
    if source is not None:
        return source
    if _standard(node, "Cast") and len(node.inputs) == 1:
        source = node.inputs[0]
        attr = node.attributes.get("to")
        if (
            source is not None
            and set(node.attributes) == {"to"}
            and attr is not None
            and not attr.is_ref()
            and attr.type == ir.AttributeType.INT
            and source.dtype is not None
            and int(source.dtype) == attr.as_int()
            and not node.metadata_props
            and not node.doc_string
            and _same_description(source, node.outputs[0])
        ):
            return source
    source = _reshape_source(node)
    if source is None:
        return None
    if _same_description(source, node.outputs[0]):
        return source

    # Only shape changes and exact Identity nodes may occur between the two
    # Reshapes. Do not commute arithmetic, casts, or broadcasting through them.
    current = source
    consumer = node
    seen: set[ir.Value] = set()
    while current not in seen:
        seen.add(current)
        # Removed nodes can retain input references. Only attached consumers
        # count, while uses in another attached graph still forbid this fold.
        uses = tuple(use for use in current.uses() if use.node.graph is not None)
        producer = current.producer()
        if (
            current.is_graph_output()
            or len(uses) != 1
            or uses[0].node is not consumer
            or uses[0].idx != 0
            or producer is None
            or producer.graph is not node.graph
        ):
            return None
        identity_source = _identity_input(producer)
        if identity_source is not None:
            current, consumer = identity_source, producer
            continue
        original = _reshape_source(producer)
        if original is not None and _same_description(original, node.outputs[0]):
            return original
        return None
    return None


def _text_bytes(value: str | bytes) -> bytes:
    return value if isinstance(value, bytes) else value.encode("utf-8")


def _attributes_key(node: ir.Node) -> tuple[object, ...] | None:
    attributes: list[object] = []
    for name, attr in sorted(node.attributes.items()):
        if attr.is_ref():
            return None
        if attr.type == ir.AttributeType.INT:
            value: object = attr.as_int()
        elif attr.type == ir.AttributeType.INTS:
            value = tuple(attr.as_ints())
        elif attr.type == ir.AttributeType.FLOAT:
            value = struct.pack("!d", attr.as_float())
        elif attr.type == ir.AttributeType.FLOATS:
            value = tuple(struct.pack("!d", item) for item in attr.as_floats())
        elif attr.type == ir.AttributeType.STRING:
            value = _text_bytes(cast(str | bytes, attr.value))
        elif attr.type == ir.AttributeType.STRINGS:
            value = tuple(
                _text_bytes(item) for item in cast(tuple[str | bytes, ...], attr.value)
            )
        else:
            return None
        attributes.append((name, attr.type, value))
    return tuple(attributes)


def _cse_key(node: ir.Node) -> tuple[object, ...] | None:
    graph = node.graph
    if (
        graph is None
        or node.domain
        or node.overload
        or node.op_type not in _CSE_OPS
        or len(node.outputs) != 1
        or node.outputs[0].const_value is not None
    ):
        return None
    attributes = _attributes_key(node)
    if attributes is None:
        return None
    output = node.outputs[0]
    return (
        node.op_type,
        node.version,
        graph.opset_imports.get(""),
        tuple(node.inputs),
        attributes,
        str(output.type),
        str(output.shape),
        tuple(sorted(node.metadata_props.items())),
        node.doc_string,
        tuple(sorted(output.metadata_props.items())),
        output.doc_string,
    )


def can_replace_node(node: ir.Node, replacement: ir.Value) -> bool:
    """Independently prove a local, storage-free identity or exact CSE rule."""
    graph = node.graph
    if (
        graph is None
        or not _discardable(node)
        or replacement is node.outputs[0]
        or replacement.graph is not graph
    ):
        return False
    if _identity_replacement(node) is replacement:
        return True
    producer = replacement.producer()
    if producer is None or producer.graph is not graph:
        return False
    key = _cse_key(node)
    if key is None or key != _cse_key(producer):
        return False
    if not _same_description(node.outputs[0], replacement):
        return False
    # The representative must dominate the removed node in this graph.
    for preceding in graph:
        if preceding is producer:
            return True
        if preceding is node:
            return False
    return False


def _graphs(model: ir.Model) -> Iterator[ir.Graph]:
    seen: set[ir.Graph] = set()
    pending = [model.graph, *(function.graph for function in model.functions.values())]
    while pending:
        graph = pending.pop()
        if graph not in seen:
            seen.add(graph)
            yield graph
            pending.extend(graph.subgraphs())


def _replace_verified_node(
    model: ir.Model,
    node: ir.Node,
    replacement: ir.Value,
    registry: InitializerMetadataRegistry | None,
) -> bool:
    if registry is not None:
        return registry.replace_uses_with_verified_value(model, node, replacement)
    if not can_replace_node(node, replacement):
        return False
    graph = node.graph
    assert graph is not None
    ir.convenience.replace_all_uses_with(
        node.outputs[0], replacement, replace_graph_outputs=True
    )
    graph.remove(node, safe=True)
    return True


def simplify_graph(
    model: ir.Model, registry: InitializerMetadataRegistry | None = None
) -> None:
    """Apply the same proved simplifications with or without annotations."""
    if registry is not None:
        registry.validate_ir(model, stage="annotated optimizer input")
    changed = True
    while changed:
        changed = False
        for graph in _graphs(model):
            representatives: dict[tuple[object, ...], ir.Value] = {}
            for node in list(graph):
                if not _discardable(node):
                    continue
                replacement = _identity_replacement(node)
                key = _cse_key(node)
                if replacement is None and key is not None:
                    replacement = representatives.get(key)
                if replacement is not None and _replace_verified_node(
                    model, node, replacement, registry
                ):
                    changed = True
                elif key is not None:
                    representatives.setdefault(key, node.outputs[0])
