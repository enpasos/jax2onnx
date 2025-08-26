# file: jax2onnx/sandbox/onnx_ir_05.py

import onnx_ir as ir
import numpy as np
import os

# --- Setup ---
# Use a consistent output path
output_dir = "tmp"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, "tanh.onnx")


# === Minimal Tanh graph (matches the first IR test) ===
print("Building a 1-input Tanh graph...")
shape = (3,)  # mirrors tests/primitives2 tanh case
x = ir.Value(name="x0", shape=ir.Shape(shape), type=ir.TensorType(ir.DataType.FLOAT))
y = ir.Value(name="y0", shape=ir.Shape(shape), type=ir.TensorType(ir.DataType.FLOAT))
node = ir.node(op_type="Tanh", inputs=[x], outputs=[y])

# === Step 3: Construct the Graph and Model ===
graph = ir.Graph(
    inputs=[x], outputs=[y], nodes=[node], initializers=[], name="TanhGraph", opset_imports={"": 18}
)
model = ir.Model(graph, ir_version=10)

# === Step 4: Save the Model ===
print(f"Saving the model to '{output_path}'...")
ir.save(model, output_path)

print("\n✅ Model saved successfully.")
