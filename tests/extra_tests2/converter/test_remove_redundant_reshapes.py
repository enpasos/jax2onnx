from __future__ import annotations

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as np_helper

from jax2onnx.converter.optimize_onnx_graph import remove_redundant_reshapes


def _create_test_model() -> onnx.ModelProto:
    input_tensor = oh.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 3])
    output_tensor = oh.make_tensor_value_info("r2_out", onnx.TensorProto.FLOAT, [2, 3])

    shape_arr = np.array([2, 3], dtype=np.int64)
    shape1_init = np_helper.from_array(shape_arr, name="shape1")
    shape2_init = np_helper.from_array(shape_arr, name="shape2")

    reshape1 = oh.make_node("Reshape", ["input", "shape1"], ["r1_out"], name="reshape1")
    gelu = oh.make_node("Gelu", ["r1_out"], ["gelu_out"], name="gelu")
    dropout = oh.make_node("Dropout", ["gelu_out"], ["dropout_out"], name="dropout")
    reshape2 = oh.make_node(
        "Reshape", ["dropout_out", "shape2"], ["r2_out"], name="reshape2"
    )

    graph = oh.make_graph(
        nodes=[reshape1, gelu, dropout, reshape2],
        name="test_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[shape1_init, shape2_init],
    )

    graph.value_info.extend([
        oh.make_tensor_value_info("r1_out", onnx.TensorProto.FLOAT, [2, 3]),
        oh.make_tensor_value_info("gelu_out", onnx.TensorProto.FLOAT, [2, 3]),
        oh.make_tensor_value_info("dropout_out", onnx.TensorProto.FLOAT, [2, 3]),
    ])

    return oh.make_model(graph)


def test_remove_redundant_reshapes_ir():
    model = _create_test_model()
    op_types_before = [node.op_type for node in model.graph.node]
    assert op_types_before == ["Reshape", "Gelu", "Dropout", "Reshape"]

    optimized_model = remove_redundant_reshapes(model)
    op_types_after = [node.op_type for node in optimized_model.graph.node]

    assert "Reshape" not in op_types_after
    assert "Gelu" in op_types_after
    assert "Dropout" in op_types_after

    gelu_node = next(node for node in optimized_model.graph.node if node.op_type == "Gelu")
    assert gelu_node.input[0] == "input"

    dropout_node = next(node for node in optimized_model.graph.node if node.op_type == "Dropout")
    assert dropout_node.output[0] == "r2_out"

    output_names = [out.name for out in optimized_model.graph.output]
    assert "r2_out" in output_names
