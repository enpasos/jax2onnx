# repro_issue.py

import jax
import jax.numpy as jnp
from jax import random
import jax2onnx


def predict(input_tensor, kernel):
    # input_tensor is NHWC
    # kernel is HWIO
    return jax.lax.conv_general_dilated(
        input_tensor,  # lhs: NHWC
        kernel,  # rhs: HWIO
        (1, 1),  # window_strides
        "SAME",  # padding
        ("NHWC", "HWIO", "NHWC"),  # dimension_numbers
        feature_group_count=1,
    )


def repro():
    input_shape = (1, 32, 32, 3)
    # Kernel: 3x3, 3 input channels, 16 output channels
    kernel_shape = (3, 3, 3, 16)

    key = random.PRNGKey(0)
    k1, k2 = random.split(key)

    # We trace with abstract values (ShapeDtypeStruct) typically,
    # but jax2onnx.to_onnx accepts function and input_specs (or example inputs if wrapper handled it,
    # but strictly it uses inputs parameter which are ShapeDtypeStructs or similar).

    # Actually checking to_onnx signature:
    # def to_onnx(fn, inputs, input_params=None, ...)

    # Let's bind the kernel as a constant/param for simplicity,
    # so the function only takes input_tensor
    kernel_val = random.normal(k1, kernel_shape)

    def bound_predict(x):
        return predict(x, kernel_val)

    # Convert
    input_spec = jax.ShapeDtypeStruct(input_shape, jnp.float32)
    model = jax2onnx.to_onnx(
        fn=bound_predict,
        inputs=[input_spec],
        input_params=None,
        model_name="repro_conv",
        opset=17,
        enable_double_precision=False,
        record_primitive_calls_file=None,
    )

    print("Model converted successfully.")

    # Inspect the graph
    graph = model.graph
    print("Graph Inputs:")
    for inp in graph.inputs:
        print(
            f"  Name: {inp.name}, Type: {inp.type.tensor_type.elem_type}, Shape: {inp.type.tensor_type.shape}"
        )

    print("\nGraph Nodes:")
    for i, node in enumerate(graph.nodes):
        print(f"  Node {i}: {node.op_type} ({node.name})")
        print(f"    Inputs: {node.input}")
        print(f"    Outputs: {node.output}")
        if node.op_type == "Transpose":
            for attr in node.attribute:
                if attr.name == "perm":
                    print(f"    Perm: {attr.ints}")

    # Check for Transpose nodes at the beginning
    # Expectation:
    # Node 0: Transpose (NHWC -> NCHW)
    # Node 1: Conv (NCHW)
    # Node 2: Transpose (NCHW -> NHWC)

    first_node = graph.nodes[0]
    if first_node.op_type == "Transpose":
        print("\n[CONFIRMED] First node is Transpose.")
        perm = [int(i) for i in first_node.attribute[0].ints]
        if perm == [0, 3, 1, 2]:
            print("  Permutation is NHWC -> NCHW (0, 3, 1, 2).")
        else:
            print(f"  Permutation is {perm}.")
    else:
        print(f"\n[UNEXPECTED] First node is {first_node.op_type}, not Transpose.")


if __name__ == "__main__":
    repro()
