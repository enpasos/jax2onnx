# tests/extra_tests/converter/test_subgraph_inputs.py

from __future__ import annotations

import onnx_ir as ir

from jax2onnx.converter.ir_builder import IRBuilder


def _scalar_value(name: str, dtype: ir.DataType) -> ir.Value:
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(()))


def _value_names(values) -> list[str]:
    if values is None:
        return []
    try:
        iterable = list(values)
    except Exception:
        iterable = values
    names = []
    for value in iterable:
        if isinstance(value, str):
            names.append(value)
        else:
            names.append(getattr(value, "name", ""))
    return names


def _graph_nodes(graph) -> list[ir.Node]:
    for attr in ("nodes", "_nodes", "node"):
        container = getattr(graph, attr, None)
        if container is None:
            continue
        try:
            return list(container)
        except Exception:
            pass
    return []


def test_intermediate_tensor_is_not_subgraph_input():
    builder = IRBuilder(opset=15, enable_double_precision=False)

    graph_input_name = "graph_input_A"
    intermediate_name = "intermediate_B_int32"
    graph_output_name = "graph_output_C"

    graph_input = _scalar_value(graph_input_name, ir.DataType.INT64)
    builder.inputs.append(graph_input)

    intermediate = _scalar_value(intermediate_name, ir.DataType.INT32)

    first_node = ir.Node(
        "",
        "Identity",
        [graph_input],
        (),
        outputs=[intermediate],
        name="node_identity_A_to_B",
    )
    builder.add_node_obj(first_node)

    graph_output = _scalar_value(graph_output_name, ir.DataType.INT32)

    second_node = ir.Node(
        "",
        "Identity",
        [intermediate],
        (),
        outputs=[graph_output],
        name="node_identity_B_to_C",
    )
    builder.add_node_obj(second_node)

    builder.outputs.append(graph_output)

    model = builder.to_ir_model(name="subgraph_input_test_model")
    graph = model.graph

    input_names = _value_names(
        getattr(graph, "inputs", None) or getattr(graph, "input", None)
    )
    output_names = _value_names(
        getattr(graph, "outputs", None) or getattr(graph, "output", None)
    )

    assert graph_input_name in input_names
    assert intermediate_name not in input_names
    assert output_names == [graph_output_name]
    assert len(input_names) == 1

    nodes = _graph_nodes(graph)
    node_names = [getattr(node, "name", "") for node in nodes]
    assert "node_identity_A_to_B" in node_names
    assert "node_identity_B_to_C" in node_names

    first_node_proto = next(
        node for node in nodes if node.name == "node_identity_A_to_B"
    )
    assert _value_names(
        getattr(first_node_proto, "inputs", None)
        or getattr(first_node_proto, "input", [])
    ) == [graph_input_name]
    assert _value_names(
        getattr(first_node_proto, "outputs", None)
        or getattr(first_node_proto, "output", [])
    ) == [intermediate_name]

    second_node_proto = next(
        node for node in nodes if node.name == "node_identity_B_to_C"
    )
    assert _value_names(
        getattr(second_node_proto, "inputs", None)
        or getattr(second_node_proto, "input", [])
    ) == [intermediate_name]
    assert _value_names(
        getattr(second_node_proto, "outputs", None)
        or getattr(second_node_proto, "output", [])
    ) == [graph_output_name]
