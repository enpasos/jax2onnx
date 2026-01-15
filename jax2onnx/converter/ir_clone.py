# jax2onnx/converter/ir_clone.py

from __future__ import annotations

import onnx_ir as ir
from onnx_ir import _cloner


class _PreservingCloner(_cloner.Cloner):
    def __init__(
        self,
        *,
        graph: ir.Graph,
        value_map: dict[ir.Value, ir.Value | None],
    ) -> None:
        super().__init__(
            attr_map={},
            value_map=value_map,
            metadata_props={},
        )
        self._local_values = self._collect_local_values(graph)

    def _collect_local_values(self, gr: ir.Graph) -> set[ir.Value]:
        local: set[ir.Value] = set()
        local.update(gr.inputs)
        local.update(gr.outputs)
        local.update(gr.initializers.values())
        for node in gr:
            local.update(node.outputs)
            for attr in node.attributes.values():
                if attr.type == ir.AttributeType.GRAPH:
                    local.update(self._collect_local_values(attr.as_graph()))
                elif attr.type == ir.AttributeType.GRAPHS:
                    for g in attr.as_graphs():
                        local.update(self._collect_local_values(g))
        return local

    def clone_value(self, value: ir.Value) -> ir.Value:
        if value in self._value_map:
            known = self._value_map[value]
            assert known is not None
            return known

        # If the value is not local to the graph being cloned, it is an outer-scope value.
        # We must preserve it (i.e. map it to itself) rather than cloning it as a new input.
        if value not in self._local_values:
            self._value_map[value] = value
            return value

        new_value = super().clone_value(value)
        if hasattr(value, "meta"):
            new_value.meta.update(value.meta)
        if hasattr(value, "metadata_props"):
            new_value.metadata_props.update(value.metadata_props)
        return new_value

    def clone_node(self, node: ir.Node) -> ir.Node:
        new_node = super().clone_node(node)
        new_node.meta.update(node.meta)

        # Super implementation creates fresh output values but only copies names.
        # We must copy type, shape, and metadata manually.
        for orig_out, new_out in zip(node.outputs, new_node.outputs):
            new_out.type = orig_out.type
            if orig_out.shape is not None:
                new_out.shape = orig_out.shape.copy()
            if hasattr(orig_out, "meta"):
                new_out.meta.update(orig_out.meta)
            if hasattr(orig_out, "metadata_props"):
                new_out.metadata_props.update(orig_out.metadata_props)

        return new_node

    def clone_graph(self, graph: ir.Graph | ir.GraphView) -> ir.Graph:
        new_graph = super().clone_graph(graph)
        if hasattr(graph, "meta"):
            new_graph.meta.update(graph.meta)
        return new_graph


def clone_graph(graph: ir.Graph) -> ir.Graph:
    """
    Create a detached copy of an ``onnx_ir.Graph``.

    This implementation uses the native ``onnx_ir._cloner.Cloner`` machinery
    but adds logic to preserve values referenced from outer scopes (e.g. captured
    variables in subgraphs) instead of turning them into new graph inputs.
    """
    cloner = _PreservingCloner(
        graph=graph,
        value_map={},
    )
    return cloner.clone_graph(graph)
