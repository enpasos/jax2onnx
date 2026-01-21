# tests/extra_tests/converter/test_ir_clone.py

import numpy as np
import onnx_ir as ir


def _tensor_value(name: str, array: np.ndarray) -> ir.Value:
    tensor = ir.tensor(array)
    return ir.Value(
        name=name,
        shape=ir.Shape(array.shape),
        type=ir.TensorType(ir.DataType.from_numpy(array.dtype)),
        const_value=tensor,
    )


def test_clone_graph_creates_independent_graph() -> None:
    input_val = ir.Value(
        name="input",
        shape=ir.Shape((1,)),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    weight = _tensor_value("weight", np.array([1.0], dtype=np.float32))
    output_val = ir.Value(
        name="output",
        shape=ir.Shape((1,)),
        type=ir.TensorType(ir.DataType.FLOAT),
    )

    add_node = ir.Node(
        "",
        "Add",
        [input_val, weight],
        (),
        outputs=[output_val],
        name="add_node",
        metadata_props={"kind": "main"},
    )
    add_node.meta["origin"] = "original"

    graph = ir.Graph(
        [input_val],
        [output_val],
        nodes=[add_node],
        initializers=[weight],
        name="main_graph",
        doc_string="primary graph",
        opset_imports={"": 18},
        metadata_props={"author": "tester"},
    )
    graph.meta["state"] = "original"

    cloned = graph.clone(allow_outer_scope_values=True)

    assert cloned is not graph
    assert cloned.name == graph.name
    assert cloned.doc_string == graph.doc_string
    assert cloned.metadata_props == graph.metadata_props
    assert cloned.opset_imports == graph.opset_imports

    assert cloned.inputs[0] is not graph.inputs[0]
    assert cloned.inputs[0].name == graph.inputs[0].name
    cloned.inputs[0].name = "modified_input"
    assert graph.inputs[0].name == "input"

    original_weight = graph.initializers["weight"]
    cloned_weight = cloned.initializers["weight"]
    assert cloned_weight is not original_weight
    cloned_weight.name = "cloned_weight"
    assert original_weight.name == "weight"

    original_node = list(graph)[0]
    cloned_node = list(cloned)[0]
    assert cloned_node is not original_node
    assert cloned_node.metadata_props == original_node.metadata_props
    cloned_node.metadata_props["kind"] = "clone"
    assert original_node.metadata_props["kind"] == "main"

    cloned.meta["state"] = "cloned"
    assert graph.meta["state"] == "original"


def test_clone_graph_copies_subgraph_attributes() -> None:
    cond = ir.Value(
        name="cond",
        shape=ir.Shape(()),
        type=ir.TensorType(ir.DataType.BOOL),
    )
    output_val = ir.Value(
        name="select",
        shape=ir.Shape(()),
        type=ir.TensorType(ir.DataType.FLOAT),
    )

    def make_branch(name: str, suffix: str) -> ir.Graph:
        branch_input = ir.Value(
            name=f"{name}_in",
            shape=ir.Shape(()),
            type=ir.TensorType(ir.DataType.FLOAT),
        )
        branch_output = ir.Value(
            name=f"{name}_out",
            shape=ir.Shape(()),
            type=ir.TensorType(ir.DataType.FLOAT),
        )
        identity_node = ir.Node(
            "",
            "Identity",
            [branch_input],
            (),
            outputs=[branch_output],
            name=f"{name}_identity",
        )
        branch_graph = ir.Graph(
            [branch_input],
            [branch_output],
            nodes=[identity_node],
            name=f"{name}_graph",
        )
        branch_graph.meta["branch"] = suffix
        return branch_graph

    then_graph = make_branch("then", "then")
    else_graph = make_branch("else", "else")

    if_node = ir.Node(
        "",
        "If",
        [cond],
        [
            ir.Attr("then_branch", ir.AttributeType.GRAPH, then_graph),
            ir.Attr("else_branch", ir.AttributeType.GRAPH, else_graph),
        ],
        outputs=[output_val],
        name="if_node",
    )

    graph = ir.Graph(
        [cond],
        [output_val],
        nodes=[if_node],
        name="if_graph",
    )

    cloned = graph.clone(allow_outer_scope_values=True)

    original_if = list(graph)[0]
    cloned_if = list(cloned)[0]

    orig_then = original_if.attributes["then_branch"].as_graph()
    cloned_then = cloned_if.attributes["then_branch"].as_graph()
    orig_else = original_if.attributes["else_branch"].as_graph()
    cloned_else = cloned_if.attributes["else_branch"].as_graph()

    assert cloned_then is not orig_then
    assert cloned_else is not orig_else
    assert cloned_then.name == orig_then.name
    assert cloned_else.name == orig_else.name
    assert cloned_then.meta["branch"] == "then"
    assert cloned_else.meta["branch"] == "else"

    cloned_then.meta["branch"] = "modified"
    assert orig_then.meta["branch"] == "then"
