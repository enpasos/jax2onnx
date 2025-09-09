# file: jax2onnx/sandbox/onnx_ir_06.py

import onnx_ir as ir
import os

# --- Setup ---
# Use a consistent output path
output_dir = "tmp"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, "custom_tanh.onnx")


# === Step 1: Define the Function Graph ===
print("Building the function graph for 'CustomTanh'...")

# Define the inputs and outputs for the function's internal graph.
# For function bodies, it's best to omit the shape and let it be
# inferred when the function is called.
func_x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT))
func_y = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT))


# Define the node(s) that make up the function's body
tanh_node = ir.node(op_type="Tanh", inputs=[func_x], outputs=[func_y])

# Create the graph for the function
function_graph = ir.Graph(
    inputs=[func_x],
    outputs=[func_y],
    nodes=[tanh_node],
    name="CustomTanhGraph",
    opset_imports={"": 18},
)

# === Step 2: Create the Function ===
print("Creating the 'CustomTanh' function...")
custom_tanh_function = ir.Function(
    domain="custom.domain",
    name="CustomTanh",
    graph=function_graph,
    attributes=[],
)


# === Step 3: Construct the Main Graph and Model ===
print("Building the main graph that calls 'CustomTanh'...")
shape = (3,)
x = ir.Value(name="main_x", shape=ir.Shape(shape), type=ir.TensorType(ir.DataType.FLOAT))
y = ir.Value(name="main_y", shape=ir.Shape(shape), type=ir.TensorType(ir.DataType.FLOAT))

# Create a node that calls our custom function
# Note the custom domain specified here.
custom_tanh_node = ir.node(
    op_type="CustomTanh",
    inputs=[x],
    outputs=[y],
    domain="custom.domain",
)

# Construct the main graph
main_graph = ir.Graph(
    inputs=[x],
    outputs=[y],
    nodes=[custom_tanh_node],
    initializers=[],
    name="MainGraph",
    # The main graph needs to import the custom domain
    opset_imports={"": 18, "custom.domain": 1},
)

# Create the model, including the function definition
model = ir.Model(
    main_graph,
    ir_version=10,
    functions=[custom_tanh_function],
)


# === Step 4: Save the Model ===
print(f"Saving the model to '{output_path}'...")
ir.save(model, output_path)

print("\n✅ Model with custom function saved successfully.")

# You can now inspect the saved ONNX model. It will contain the main graph
# with a call to your custom op, and the function definition will be stored
# in the model's `functions` field.