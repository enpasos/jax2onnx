# jax2onnx/converter/ir_clone.py

from __future__ import annotations

from typing import Dict

import onnx_ir as ir
from onnx_ir import Attr, AttributeType, RefAttr


def _assign_metadata(
    kwargs: Dict[str, object], metadata: dict[str, str]
) -> Dict[str, object]:
    kwargs["metadata_props"] = metadata or None
    return kwargs


def clone_graph(graph: ir.Graph) -> ir.Graph:
    """
    Create a detached copy of an ``onnx_ir.Graph``.

    This mirrors the functionality of ``_CopyReplace.clone_graph`` inside the
    ONNX IR inliner (see onnx/ir-py@main) and the discussion in
    onnx/ir-py#172 about upstreaming a first-class clone API. We keep a local
    helper so the converter can safely export graphs without mutating the
    builder state. Drop this file once the upstream library exposes an eager
    clone utility.
    """

    value_map: Dict[ir.Value, ir.Value] = {}

    def clone_value(value: ir.Value) -> ir.Value:
        if value in value_map:
            return value_map[value]
        metadata_props = dict(getattr(value, "metadata_props", {}))
        kwargs = dict(
            name=value.name,
            shape=value.shape,
            type=value.type,
            doc_string=value.doc_string,
            const_value=value.const_value,
        )
        cloned = ir.Value(**_assign_metadata(kwargs, metadata_props))
        try:
            cloned.meta.update(getattr(value, "meta", {}))
        except Exception:
            pass
        value_map[value] = cloned
        return cloned

    def clone_optional_value(value: ir.Value | None) -> ir.Value | None:
        if value is None:
            return None
        return clone_value(value)

    def clone_attr(attr: Attr) -> Attr:
        if attr.is_ref():
            ref_name = attr.ref_attr_name
            assert ref_name is not None, "Reference attribute must point to a name"
            return RefAttr(attr.name, ref_name, attr.type, doc_string=attr.doc_string)
        if attr.type == AttributeType.GRAPH:
            cloned_graph = clone_graph(attr.as_graph())
            return Attr(
                attr.name,
                AttributeType.GRAPH,
                cloned_graph,
                doc_string=attr.doc_string,
            )
        if attr.type == AttributeType.GRAPHS:
            cloned_graphs = [clone_graph(subgraph) for subgraph in attr.as_graphs()]
            return Attr(
                attr.name,
                AttributeType.GRAPHS,
                cloned_graphs,
                doc_string=attr.doc_string,
            )
        # Attributes are immutable containers in onnx_ir so reusing the objects is fine,
        # but we still clone simple containers to avoid sharing accidental references.
        value = attr.value
        if isinstance(value, dict):
            value = dict(value)
        elif isinstance(value, list):
            value = list(value)
        elif isinstance(value, tuple):
            value = tuple(value)
        return Attr(attr.name, attr.type, value, doc_string=attr.doc_string)

    def clone_node(node: ir.Node) -> ir.Node:
        inputs = [clone_optional_value(val) for val in node.inputs]
        outputs = [clone_value(val) for val in node.outputs]
        attributes = [clone_attr(attr) for attr in node.attributes.values()]
        metadata_props = dict(node.metadata_props)
        node_kwargs = _assign_metadata({}, metadata_props)
        cloned = ir.Node(
            node.domain,
            node.op_type,
            inputs,
            attributes,
            overload=node.overload,
            num_outputs=len(outputs),
            outputs=outputs,
            version=node.version,
            name=node.name,
            doc_string=node.doc_string,
            **node_kwargs,
        )
        cloned.meta.update(node.meta)
        return cloned

    input_values = [clone_value(value) for value in graph.inputs]
    output_values = [clone_value(value) for value in graph.outputs]
    cloned_nodes = [clone_node(node) for node in graph]
    initializer_values = [clone_value(value) for value in graph.initializers.values()]
    metadata_props = dict(graph.metadata_props)
    graph_kwargs = _assign_metadata({}, metadata_props)
    cloned_graph = ir.Graph(
        input_values,
        output_values,
        nodes=cloned_nodes,
        initializers=initializer_values,
        doc_string=graph.doc_string,
        opset_imports=dict(graph.opset_imports),
        name=graph.name,
        **graph_kwargs,
    )
    cloned_graph.meta.update(graph.meta)
    return cloned_graph
