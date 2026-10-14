# jax2onnx/converter/optimizer_graph_utils.py

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, TypeAlias

import onnx_ir as ir

NodeList: TypeAlias = List[ir.Node]
NodeSeq: TypeAlias = Sequence[ir.Node]
ValueList: TypeAlias = List[ir.Value]
ValueSeq: TypeAlias = Sequence[ir.Value]
InputValue: TypeAlias = ir.Value | None
InputValueList: TypeAlias = List[InputValue]


def _v_name(v: ir.Value | str | None) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        return v or None
    return v.name or None


def node_in_graph(graph: ir.Graph, node: ir.Node) -> bool:
    """Check current ownership, independently of iteration snapshots or names."""
    return node.graph is graph


def producer_in_graph(graph: ir.Graph, value: ir.Value | None) -> ir.Node | None:
    """Return a value's producer only when it belongs to this graph scope."""
    if value is None:
        return None
    producer = value.producer()
    return producer if producer is not None and producer.graph is graph else None


def consumers_in_graph(graph: ir.Graph, value: ir.Value | None) -> list[ir.Node]:
    """Return current consumers in this scope, including captured outer values."""
    if value is None:
        return []
    return [consumer for consumer in value.consumers() if consumer.graph is graph]


def _node_outputs(n: ir.Node) -> ValueList:
    return list(n.outputs)


def _node_output(n: ir.Node) -> Optional[ir.Value]:
    outs = _node_outputs(n)
    return outs[0] if outs else None


def _node_inputs(n: ir.Node) -> InputValueList:
    return list(n.inputs)


def _set_node_inputs(n: ir.Node, new_ins: Sequence[InputValue]) -> None:
    for idx, val in enumerate(new_ins):
        n.replace_input_with(idx, val)


def _has_input_name_or_obj(
    node: ir.Node, name: Optional[str], obj: Optional[ir.Value]
) -> bool:
    """Legacy match for detached snapshots that may resolve values by name."""
    target_name = name or _v_name(obj)
    return any(
        (obj is not None and value is obj)
        or (target_name is not None and _v_name(value) == target_name)
        for value in node.inputs
    )


def legacy_node_in_sequence(node: ir.Node, nodes: Sequence[ir.Node]) -> bool:
    """Check identity membership in a detached sequence, including stale snapshots."""
    return any(existing is node for existing in nodes)


def legacy_consumer_nodes(
    nodes: Sequence[ir.Node], value_or_name: ir.Value | str | None
) -> list[ir.Node]:
    """Resolve consumers in a detached sequence, retaining historical name fallback."""
    if value_or_name is None:
        return []
    if isinstance(value_or_name, ir.Value):
        node_ids = {id(node) for node in nodes}
        linked = [
            consumer
            for consumer in value_or_name.consumers()
            if id(consumer) in node_ids
        ]
        if linked:
            return linked
    target_name = _v_name(value_or_name)
    target_value = value_or_name if isinstance(value_or_name, ir.Value) else None
    return [
        node
        for node in nodes
        if _has_input_name_or_obj(node, target_name, target_value)
    ]


def legacy_producer_node(
    nodes: Sequence[ir.Node], value_or_name: ir.Value | str | None
) -> ir.Node | None:
    """Resolve a producer in a detached sequence, retaining historical name fallback."""
    if value_or_name is None:
        return None
    if isinstance(value_or_name, ir.Value):
        producer = value_or_name.producer()
        if producer is not None:
            if legacy_node_in_sequence(producer, nodes):
                return producer
        elif value_or_name.graph is nodes and (
            value_or_name.is_graph_input() or value_or_name.is_initializer()
        ):
            return None
    target_name = _v_name(value_or_name)
    for node in nodes:
        for output in node.outputs:
            if output is value_or_name or (
                target_name is not None and _v_name(output) == target_name
            ):
                return node
    return None


# Compatibility imports for callers that still supply lists or raw value names.
# New optimizer passes use the graph-native functions above.
_node_in_sequence: Callable[[ir.Node, Sequence[ir.Node]], bool] = (
    legacy_node_in_sequence
)
_consumer_nodes: Callable[[Sequence[ir.Node], ir.Value | str | None], list[ir.Node]] = (
    legacy_consumer_nodes
)
_producer_node: Callable[[Sequence[ir.Node], ir.Value | str | None], ir.Node | None] = (
    legacy_producer_node
)
