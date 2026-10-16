# tests/extra_tests/framework/test_optimizer_graph_utils.py

from __future__ import annotations

import onnx_ir as ir
import pytest

from jax2onnx.converter.ir_optimizations import remove_redundant_transpose_pairs_ir
from jax2onnx.converter.optimizer_graph_utils import (
    consumers_in_graph,
    legacy_consumer_nodes,
    legacy_node_in_sequence,
    legacy_producer_node,
    node_in_graph,
    producer_in_graph,
)


def test_native_discovery_distinguishes_captures_and_shadowed_names() -> None:
    parent_input = ir.Value(name="parent_input")
    parent_node = ir.Node("", "Neg", [parent_input], num_outputs=1)
    captured = parent_node.outputs[0]
    captured.name = "shared"
    parent_graph = ir.Graph([parent_input], [captured], nodes=[parent_node])

    capture_node = ir.Node("", "Identity", [captured], num_outputs=1)
    capture_graph = ir.Graph([], list(capture_node.outputs), nodes=[capture_node])

    shadow_input = ir.Value(name="shared")
    shadow_node = ir.Node("", "Neg", [shadow_input], num_outputs=1)
    shadow_node.outputs[0].name = "shared_result"
    shadow_graph = ir.Graph(
        [shadow_input], list(shadow_node.outputs), nodes=[shadow_node]
    )

    assert captured.graph is parent_graph
    assert producer_in_graph(parent_graph, captured) is parent_node
    assert producer_in_graph(capture_graph, captured) is None
    assert consumers_in_graph(capture_graph, captured) == [capture_node]
    assert consumers_in_graph(parent_graph, captured) == []
    assert consumers_in_graph(shadow_graph, captured) == []
    assert consumers_in_graph(shadow_graph, shadow_input) == [shadow_node]

    # A disconnected reference with a matching name is intentionally resolved
    # only by the compatibility API, never by the native graph API.
    disconnected = ir.Value(name="shared_result")
    assert producer_in_graph(shadow_graph, disconnected) is None
    assert legacy_producer_node(list(shadow_graph), disconnected) is shadow_node
    assert legacy_consumer_nodes(list(shadow_graph), captured) == [shadow_node]


def test_native_discovery_excludes_removed_producers_and_consumers() -> None:
    x = ir.Value(name="x")
    producer = ir.Node("", "Neg", [x], num_outputs=1)
    value = producer.outputs[0]
    consumer = ir.Node("", "Relu", [value], num_outputs=1)
    graph = ir.Graph([x], [value], nodes=[producer, consumer])

    assert producer_in_graph(graph, value) is producer
    assert consumers_in_graph(graph, value) == [consumer]
    graph.remove(consumer)
    assert consumers_in_graph(graph, value) == []
    graph.remove(producer)

    # Graph-output ownership and producer references survive removal in IR.
    assert value.graph is graph
    assert value.producer() is producer
    assert producer_in_graph(graph, value) is None
    assert not node_in_graph(graph, producer)
    assert consumers_in_graph(graph, x) == []


def test_snapshot_membership_does_not_follow_live_graph_mutations() -> None:
    x = ir.Value(name="x")
    first = ir.Node("", "Neg", [x], num_outputs=1)
    middle = ir.Node("", "Abs", [first.outputs[0]], num_outputs=1)
    last = ir.Node("", "Relu", [middle.outputs[0]], num_outputs=1)
    graph = ir.Graph([x], list(last.outputs), nodes=[first, middle, last])
    snapshot = list(graph)
    graph.remove(middle)
    replacement = ir.Node("", "Identity", [first.outputs[0]], num_outputs=1)
    graph.insert_before(last, replacement)

    assert len(graph) == len(snapshot)
    assert graph[0] is snapshot[0]
    assert graph[-1] is snapshot[-1]
    assert node_in_graph(graph, replacement)
    assert not node_in_graph(graph, middle)
    assert not legacy_node_in_sequence(replacement, snapshot)
    assert legacy_node_in_sequence(middle, snapshot)
    assert producer_in_graph(graph, replacement.outputs[0]) is replacement
    assert legacy_producer_node(snapshot, replacement.outputs[0]) is None
    assert legacy_producer_node(snapshot, middle.outputs[0]) is middle
    assert consumers_in_graph(graph, first.outputs[0]) == [replacement]
    assert legacy_consumer_nodes(snapshot, first.outputs[0]) == [middle]


def test_legacy_discovery_preserves_raw_names_and_detached_values() -> None:
    x = ir.Value(name="x")
    producer = ir.Node("", "Neg", [x], num_outputs=1)
    value = producer.outputs[0]
    value.name = "produced"
    consumer = ir.Node("", "Add", [value, value, None], num_outputs=1)
    nodes = [producer, consumer]

    assert legacy_producer_node(nodes, "produced") is producer
    assert legacy_producer_node(nodes, ir.Value(name="produced")) is producer
    assert legacy_consumer_nodes(nodes, "produced") == [consumer]
    assert legacy_consumer_nodes(nodes, ir.Value(name="produced")) == [consumer]
    assert legacy_consumer_nodes(nodes, value) == [consumer]
    assert legacy_producer_node(nodes, "") is None
    assert legacy_consumer_nodes(nodes, "") == []
    assert legacy_producer_node(nodes, None) is None
    assert legacy_consumer_nodes(nodes, None) == []


def test_native_discovery_handles_missing_inputs_and_initializers() -> None:
    x = ir.Value(name="x")
    weight = ir.Value(name="weight", const_value=ir.tensor([1.0]))
    node = ir.Node("", "Add", [x, weight, None], num_outputs=1)
    graph = ir.Graph([x], list(node.outputs), nodes=[node])
    graph.register_initializer(weight)

    assert producer_in_graph(graph, x) is None
    assert producer_in_graph(graph, weight) is None
    assert producer_in_graph(graph, None) is None
    assert consumers_in_graph(graph, weight) == [node]
    assert consumers_in_graph(graph, None) == []


@pytest.mark.parametrize("initializer", [False, True])
def test_legacy_live_graph_preserves_foreign_input_name_fallback(
    initializer: bool,
) -> None:
    source = ir.Value(name="source")
    produced = ir.Value(name="shared")
    producer = ir.Node("", "Identity", [source], outputs=[produced])
    graph = ir.Graph([source], [produced], nodes=[producer])
    foreign = ir.Value(name="shared")
    other = ir.Graph([] if initializer else [foreign], [], nodes=[])
    if initializer:
        foreign.const_value = ir.tensor([1.0])
        other.register_initializer(foreign)

    assert foreign.graph is other
    assert producer_in_graph(graph, foreign) is None
    assert legacy_producer_node(graph, foreign) is producer
    assert legacy_producer_node(list(graph), foreign) is producer
    assert legacy_producer_node(other, foreign) is None


def test_transpose_folding_does_not_cross_a_captured_graph_boundary() -> None:
    x = ir.Value(name="x", shape=ir.Shape([2, 3]))
    outer_transpose = ir.Node(
        "",
        "Transpose",
        [x],
        attributes={"perm": ir.AttrInt64s("perm", [1, 0])},
        num_outputs=1,
    )
    captured = outer_transpose.outputs[0]
    captured.shape = ir.Shape([3, 2])
    parent = ir.Graph([x], [captured], nodes=[outer_transpose])
    inner_transpose = ir.Node(
        "",
        "Transpose",
        [captured],
        attributes={"perm": ir.AttrInt64s("perm", [1, 0])},
        num_outputs=1,
    )
    inner_transpose.outputs[0].shape = ir.Shape([2, 3])
    child = ir.Graph([], list(inner_transpose.outputs), nodes=[inner_transpose])

    remove_redundant_transpose_pairs_ir(child)

    assert list(parent) == [outer_transpose]
    assert list(child) == [inner_transpose]
    assert inner_transpose.inputs[0] is captured
    assert child.outputs[0] is inner_transpose.outputs[0]
