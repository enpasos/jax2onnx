# tests/extra_tests/converter/test_remove_redundant_reshapes.py

from __future__ import annotations

import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_optimizations import remove_redundant_reshape_pairs_ir


def _const_vector(name: str, values: list[int]) -> tuple[ir.Value, ir.Node]:
    array = np.asarray(values, dtype=np.int64)
    value = ir.Value(
        name=name,
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape(array.shape),
    )
    value.const_value = ir.tensor(array)
    node = ir.Node("", "Constant", [], (), outputs=[value], name=f"Const_{name}")
    return value, node


def _tensor_value(name: str, shape: tuple[int, ...]) -> ir.Value:
    return ir.Value(
        name=name,
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape(shape),
    )


def _node_list(graph) -> list[ir.Node]:
    for attr in ("nodes", "_nodes", "node"):
        container = getattr(graph, attr, None)
        if container is None:
            continue
        try:
            return list(container)
        except Exception:
            pass
    return []


def _graph_outputs(graph) -> list[ir.Value]:
    outputs = getattr(graph, "outputs", None) or getattr(graph, "output", None)
    if outputs is None:
        return []
    try:
        return list(outputs)
    except Exception:
        return []


def _value_name(value) -> str:
    if isinstance(value, str):
        return value
    return getattr(value, "name", "")


def _reshape_chain_graph() -> ir.Graph:
    input_val = _tensor_value("input", (2, 3))
    reshape1_out = _tensor_value("r1_out", (2, 3))
    gelu_out = _tensor_value("gelu_out", (2, 3))
    reshape2_out = _tensor_value("r2_out", (2, 3))

    shape1, const1 = _const_vector("shape1", [2, 3])
    shape2, const2 = _const_vector("shape2", [2, 3])

    reshape1 = ir.Node(
        "",
        "Reshape",
        [input_val, shape1],
        (),
        outputs=[reshape1_out],
        name="reshape1",
    )
    gelu = ir.Node(
        "",
        "Gelu",
        [reshape1_out],
        (),
        outputs=[gelu_out],
        name="gelu",
    )
    reshape2 = ir.Node(
        "",
        "Reshape",
        [gelu_out, shape2],
        (),
        outputs=[reshape2_out],
        name="reshape2",
    )

    return ir.Graph(
        inputs=[input_val],
        outputs=[reshape2_out],
        nodes=[const1, const2, reshape1, gelu, reshape2],
        name="test_graph",
    )


def test_remove_redundant_reshapes_ir():
    graph = _reshape_chain_graph()
    before = [node.op_type for node in _node_list(graph)]
    assert before == ["Constant", "Constant", "Reshape", "Gelu", "Reshape"]

    remove_redundant_reshape_pairs_ir(graph)

    after_nodes = _node_list(graph)
    after_types = [node.op_type for node in after_nodes]
    assert after_types == ["Constant", "Constant", "Gelu"]

    gelu_node = next(node for node in after_nodes if node.name == "gelu")
    gelu_inputs = getattr(gelu_node, "inputs", None) or getattr(gelu_node, "input", [])
    first_input = _value_name(gelu_inputs[0])
    assert first_input == "input"

    graph_outputs = [_value_name(value) for value in _graph_outputs(graph)]
    assert graph_outputs == ["gelu_out"]

    assert "Reshape" not in after_types


def test_reshape_pair_removal_refreshes_elementwise_rank_metadata() -> None:
    input_val = _tensor_value("input", (1, 8, 4, 4))
    reshape1_out = _tensor_value("r1_out", (1, 16, 8))
    identity_out = _tensor_value("identity_out", (1, 16, 8))
    reshape2_out = _tensor_value("r2_out", (1, 8, 4, 4))
    required_out = _tensor_value("required_out", (1, 16, 8))

    shape1, const1 = _const_vector("shape1", [1, 16, 8])
    shape2, const2 = _const_vector("shape2", [1, 8, 4, 4])
    shape3, const3 = _const_vector("shape3", [1, 16, 8])
    reshape1 = ir.Node(
        "", "Reshape", [input_val, shape1], (), outputs=[reshape1_out], name="reshape1"
    )
    identity = ir.Node(
        "", "Identity", [reshape1_out], (), outputs=[identity_out], name="identity"
    )
    reshape2 = ir.Node(
        "",
        "Reshape",
        [identity_out, shape2],
        (),
        outputs=[reshape2_out],
        name="reshape2",
    )
    required_reshape = ir.Node(
        "",
        "Reshape",
        [reshape2_out, shape3],
        (),
        outputs=[required_out],
        name="required_reshape",
    )
    graph = ir.Graph(
        inputs=[input_val],
        outputs=[required_out],
        nodes=[
            const1,
            const2,
            const3,
            reshape1,
            identity,
            reshape2,
            required_reshape,
        ],
        name="rank_refresh_graph",
    )

    remove_redundant_reshape_pairs_ir(graph)

    remaining = _node_list(graph)
    assert not any(node.name in {"reshape1", "reshape2"} for node in remaining)
    assert any(node.name == "required_reshape" for node in remaining)
    assert tuple(identity_out.shape) == (1, 8, 4, 4)
    assert _value_name(required_reshape.inputs[0]) == "identity_out"
