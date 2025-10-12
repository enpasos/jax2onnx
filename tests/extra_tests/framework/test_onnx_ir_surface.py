# tests/extra_tests/framework/test_onnx_ir_surface.py

from __future__ import annotations

import onnx_ir as ir


def _public_attrs(obj: object) -> set[str]:
    return {name for name in dir(obj) if not name.startswith("_")}


EXPECTED_GRAPH_ATTRS = {
    "all_nodes",
    "append",
    "count",
    "doc_string",
    "extend",
    "index",
    "initializers",
    "inputs",
    "insert_after",
    "insert_before",
    "meta",
    "metadata_props",
    "name",
    "node",
    "num_nodes",
    "opset_imports",
    "outputs",
    "register_initializer",
    "remove",
    "sort",
    "subgraphs",
}

EXPECTED_NODE_ATTRS = {
    "append",
    "attributes",
    "doc_string",
    "domain",
    "graph",
    "inputs",
    "meta",
    "metadata_props",
    "name",
    "op_identifier",
    "op_type",
    "outputs",
    "overload",
    "prepend",
    "predecessors",
    "replace_input_with",
    "successors",
    "version",
}

EXPECTED_VALUE_ATTRS = {
    "const_value",
    "consumers",
    "doc_string",
    "dtype",
    "graph",
    "index",
    "is_graph_input",
    "is_graph_output",
    "is_initializer",
    "meta",
    "metadata_props",
    "name",
    "producer",
    "shape",
    "type",
    "uses",
}

EXPECTED_ATTR_ATTRS = {
    "as_float",
    "as_floats",
    "as_graph",
    "as_graphs",
    "as_int",
    "as_ints",
    "as_string",
    "as_strings",
    "as_tensor",
    "as_tensors",
    "doc_string",
    "is_ref",
    "meta",
    "name",
    "ref_attr_name",
    "type",
    "value",
}

EXPECTED_SHAPE_ATTRS = {
    "copy",
    "dims",
    "freeze",
    "frozen",
    "get_denotation",
    "has_unknown_dim",
    "is_dynamic",
    "is_static",
    "is_unknown_dim",
    "numpy",
    "rank",
    "set_denotation",
}

EXPECTED_TOP_LEVEL_EXPORTS = {
    "Attr",
    "Graph",
    "Model",
    "Node",
    "TensorProtocol",
    "Value",
    "convenience",
    "node",
    "tensor",
    "val",
}


def _assert_surface(expected: set[str], obj: object) -> None:
    actual = _public_attrs(obj)
    missing = expected - actual
    assert not missing, f"{obj!r} missing {sorted(missing)}"


def test_graph_surface_matches_stub() -> None:
    _assert_surface(EXPECTED_GRAPH_ATTRS, ir.Graph)


def test_node_surface_matches_stub() -> None:
    _assert_surface(EXPECTED_NODE_ATTRS, ir.Node)


def test_value_surface_matches_stub() -> None:
    _assert_surface(EXPECTED_VALUE_ATTRS, ir.Value)


def test_attr_surface_matches_stub() -> None:
    _assert_surface(EXPECTED_ATTR_ATTRS, ir.Attr)


def test_shape_surface_matches_stub() -> None:
    _assert_surface(EXPECTED_SHAPE_ATTRS, ir.Shape)


def test_convenience_surface_matches_stub() -> None:
    convenience_public = _public_attrs(ir.convenience)
    expected = {
        "annotations",
        "convert_attribute",
        "convert_attributes",
        "create_value_mapping",
        "get_const_tensor",
        "replace_all_uses_with",
        "replace_nodes_and_values",
    }
    missing = expected - convenience_public
    assert not missing, f"ir.convenience missing {sorted(missing)}"


def test_top_level_exports_match_expectations() -> None:
    exports = _public_attrs(ir)
    missing = EXPECTED_TOP_LEVEL_EXPORTS - exports
    assert not missing, f"onnx_ir missing top-level exports {sorted(missing)}"
