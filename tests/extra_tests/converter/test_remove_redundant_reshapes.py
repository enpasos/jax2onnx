import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as np_helper
import pytest

# Import the function under test.
from jax2onnx.converter.optimize_onnx_graph import (
    remove_redundant_reshapes,
)


@pytest.mark.order(-1)  # run *after* the models have been produced
def create_test_model() -> onnx.ModelProto:
    # Create an input and an output value_info.
    input_tensor = oh.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 3])
    output_tensor = oh.make_tensor_value_info("r2_out", onnx.TensorProto.FLOAT, [2, 3])

    # Create constant initializers for the shape inputs.
    # Both reshape nodes will use the same shape [2, 3].
    shape1_init = np_helper.from_array(np.array([2, 3], dtype=np.int64), name="shape1")
    shape2_init = np_helper.from_array(np.array([2, 3], dtype=np.int64), name="shape2")

    # Build the chain:
    #   Reshape1 -> Gelu -> Dropout -> Reshape2
    reshape1 = oh.make_node(
        "Reshape", inputs=["input", "shape1"], outputs=["r1_out"], name="reshape1"
    )
    gelu = oh.make_node("Gelu", inputs=["r1_out"], outputs=["gelu_out"], name="gelu")
    dropout = oh.make_node(
        "Dropout", inputs=["gelu_out"], outputs=["dropout_out"], name="dropout"
    )
    reshape2 = oh.make_node(
        "Reshape", inputs=["dropout_out", "shape2"], outputs=["r2_out"], name="reshape2"
    )

    # Assemble the graph.
    graph = oh.make_graph(
        nodes=[reshape1, gelu, dropout, reshape2],
        name="test_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[shape1_init, shape2_init],
    )

    # Provide value_info for intermediate tensors so that get_tensor_shape works.
    r1_vi = oh.make_tensor_value_info("r1_out", onnx.TensorProto.FLOAT, [2, 3])
    gelu_vi = oh.make_tensor_value_info("gelu_out", onnx.TensorProto.FLOAT, [2, 3])
    dropout_vi = oh.make_tensor_value_info(
        "dropout_out", onnx.TensorProto.FLOAT, [2, 3]
    )
    graph.value_info.extend([r1_vi, gelu_vi, dropout_vi])

    return oh.make_model(graph)


@pytest.mark.order(-1)  # run *after* the models have been produced
def test_remove_redundant_reshapes():
    # Create a model with the redundant reshape chain.
    model = create_test_model()

    # Check the unoptimized graph contains the expected chain.
    op_types_before = [node.op_type for node in model.graph.node]
    # Expected: Reshape, Gelu, Dropout, Reshape
    assert op_types_before == [
        "Reshape",
        "Gelu",
        "Dropout",
        "Reshape",
    ], f"Unexpected initial node chain: {op_types_before}"

    # Run the optimizer.
    optimized_model = remove_redundant_reshapes(model)

    # Get the op types from the optimized model.
    op_types_after = [node.op_type for node in optimized_model.graph.node]

    # Verify that no Reshape nodes remain.
    assert (
        "Reshape" not in op_types_after
    ), f"Expected redundant Reshape nodes to be removed, but found nodes: {op_types_after}"

    # Verify that the allowed elementwise nodes remain.
    assert "Gelu" in op_types_after, "Gelu node was removed but should be preserved."
    assert (
        "Dropout" in op_types_after
    ), "Dropout node was removed but should be preserved."

    # Verify that the Gelu node's input has been rewired to the original input.
    gelu_node = next(
        node for node in optimized_model.graph.node if node.op_type == "Gelu"
    )
    assert (
        gelu_node.input[0] == "input"
    ), f"Expected Gelu node's input to be 'input', but got {gelu_node.input[0]}"

    # Verify that the last allowed op (Dropout) now produces the output with the original name.
    dropout_node = next(
        node for node in optimized_model.graph.node if node.op_type == "Dropout"
    )
    assert (
        dropout_node.output[0] == "r2_out"
    ), f"Expected Dropout node's output to be renamed to 'r2_out', but got {dropout_node.output[0]}"

    # Finally, ensure the graph output remains.
    output_names = [out.name for out in optimized_model.graph.output]
    assert (
        "r2_out" in output_names
    ), f"Expected 'r2_out' to remain as a graph output, but got: {output_names}"
