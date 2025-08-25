# file: jax2onnx/sandbox/onnx_ir_01.py

import onnx_ir as ir
import numpy as np
import os

# --- Setup ---
# Use a consistent output path
output_dir = "tmp"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, "model.onnx")


# === Step 1: Define ALL values in the graph with full metadata ===
# Each ir.Value is created with its complete shape and type information.
print("Step 1: Defining all graph values with full shape info...")

# --- Graph Input ---
x = ir.Value(name="x", shape=ir.Shape(("B", 4)), type=ir.TensorType(ir.DataType.FLOAT))

# --- Initializers (Constants) ---
W_val = np.random.randn(2, 4).astype(np.float32)
W = ir.Value(
    name="W",
    type=ir.TensorType(ir.DataType.FLOAT),
    shape=ir.Shape(W_val.shape),
    const_value=ir.tensor(W_val),
)

b_val = np.random.randn(2).astype(np.float32)
b = ir.Value(
    name="b",
    type=ir.TensorType(ir.DataType.FLOAT),
    shape=ir.Shape(b_val.shape),
    const_value=ir.tensor(b_val),
)

# --- Intermediate Value ---
# Shape: ('B', 4) @ (4, 2) -> ('B', 2)
xw = ir.Value(
    name="xw", shape=ir.Shape(("B", 2)), type=ir.TensorType(ir.DataType.FLOAT)
)

# --- Final Graph Output ---
# Shape: ('B', 2) + (2,) -> ('B', 2)
y = ir.Value(name="y", shape=ir.Shape(("B", 2)), type=ir.TensorType(ir.DataType.FLOAT))


# === Step 2: Define the nodes using the pre-defined values ===
print("Step 2: Defining nodes manually...")
node1 = ir.node(op_type="MatMul", inputs=[x, W], outputs=[xw], attributes={"transB": 1})
node2 = ir.node(op_type="Add", inputs=[xw, b], outputs=[y])

# === Step 3: Construct the Graph and Model ===
print("Step 3: Building the Graph and Model objects...")
graph = ir.Graph(
    inputs=[x],
    outputs=[y],
    nodes=[node1, node2],
    initializers=[W, b],
    name="SimpleLinearGraph",
    opset_imports={"": 18},
)
model = ir.Model(graph, ir_version=10)


# === Step 4: Save the Model ===
print(f"Step 4: Saving the model to '{output_path}'...")
ir.save(model, output_path)

print("\n✅ Model saved successfully.")
