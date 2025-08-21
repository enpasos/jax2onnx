# file: jax2onnx/sandbox/nested_functions_example.py

import os

import numpy as np
import onnx
from onnx import TensorProto, helper

# === Define Function: LinearLayer ===
# Computes: Y = X @ W + B

# Define the inputs and outputs for the linear layer function.
linear_inputs = ["X", "W", "B"]  # Input tensors: X (data), W (weights), B (bias)
linear_outputs = ["Y"]  # Output tensor: Y (result of the linear transformation)

# Create ONNX nodes for the linear layer computation.
linear_nodes = [
    helper.make_node("MatMul", ["X", "W"], ["XM"]),  # Matrix multiplication: X @ W
    helper.make_node("Add", ["XM", "B"], ["Y"]),  # Add bias: XM + B
]

# Optional: Add shape annotations for debugging or visualization in tools like Netron.
linear_value_info = [
    helper.make_tensor_value_info(
        "XM", TensorProto.FLOAT, [1, 6]
    ),  # Intermediate tensor XM
    helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 6]),
]

linear_func = helper.make_function(
    domain="custom",
    fname="LinearLayer",
    inputs=linear_inputs,
    outputs=linear_outputs,
    nodes=linear_nodes,
    opset_imports=[helper.make_opsetid("", 14)],
    value_info=linear_value_info,
)

# === Define Function: LinearGelu ===
# Computes: Y = Gelu(LinearLayer(X, W, B))

gelu_inputs = ["X", "W", "B"]
gelu_outputs = ["Y"]
gelu_nodes = [
    helper.make_node("LinearLayer", ["X", "W", "B"], ["Z"], domain="custom"),
    helper.make_node("Gelu", ["Z"], ["Y"]),
]

gelu_value_info = [
    helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 6]),
    helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 6]),
]

gelu_func = helper.make_function(
    domain="custom",
    fname="LinearGelu",
    inputs=gelu_inputs,
    outputs=gelu_outputs,
    nodes=gelu_nodes,
    opset_imports=[helper.make_opsetid("", 14), helper.make_opsetid("custom", 1)],
    value_info=gelu_value_info,
)

# === Top-level model that uses LinearGelu ===

input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 6])

weight_tensor = helper.make_tensor_value_info("W", TensorProto.FLOAT, [4, 6])
bias_tensor = helper.make_tensor_value_info("B", TensorProto.FLOAT, [6])

# Parameter initializers
init_w = helper.make_tensor(
    "W", TensorProto.FLOAT, [4, 6], np.random.randn(4, 6).astype(np.float32).flatten()
)
init_b = helper.make_tensor(
    "B", TensorProto.FLOAT, [6], np.random.randn(6).astype(np.float32).flatten()
)

# Top-level node that uses the nested function
node = helper.make_node("LinearGelu", ["X", "W", "B"], ["Y"], domain="custom")

top_graph = helper.make_graph(
    nodes=[node],
    name="NestedFunctionGraph",
    inputs=[input_tensor, weight_tensor, bias_tensor],
    outputs=[output_tensor],
    initializer=[init_w, init_b],
)

# Assemble model
model = helper.make_model(
    top_graph,
    functions=[linear_func, gelu_func],
    opset_imports=[helper.make_opsetid("", 14), helper.make_opsetid("custom", 1)],
    ir_version=9,
    producer_name="nested-functions-demo",
)

# Save model
output_path = "./docs/onnx/sandbox/nested_function_example.onnx"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
onnx.save(model, output_path)
print(f"✅ Nested ONNX model saved to: {output_path}")
