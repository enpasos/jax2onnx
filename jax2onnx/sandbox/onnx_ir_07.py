# file: onnx_ir_function_with_attribute.py

import onnx_ir as ir
import os

# --- Setup ---
output_dir = "tmp"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, "custom_leaky_relu.onnx")


# === Step 1: Define the Function Graph ===
print("Building the function graph for 'CustomLeakyRelu'...")

# Define inputs and outputs for the function's internal graph.
# Shapes are omitted as they are determined by the call site.
func_x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT))
func_y = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT))

# The node inside the function references an attribute that will be passed by the caller.
# This is done using `ir.RefAttr`, which links to an attribute defined on the function itself.
leaky_relu_node = ir.node(
    op_type="LeakyRelu",
    inputs=[func_x],
    outputs=[func_y],
    attributes={"alpha": ir.RefAttr("alpha", "alpha", ir.AttributeType.FLOAT)},
)

# Create the graph for the function
function_graph = ir.Graph(
    inputs=[func_x],
    outputs=[func_y],
    nodes=[leaky_relu_node],
    name="CustomLeakyReluGraph",
    opset_imports={"": 18},
)

# === Step 2: Create the Function with an Attribute Definition ===
print("Creating the 'CustomLeakyRelu' function with an 'alpha' attribute...")
custom_leaky_relu_function = ir.Function(
    domain="custom.domain",
    name="CustomLeakyRelu",
    graph=function_graph,
    # Declare the attributes that this function accepts.
    # Here, we declare 'alpha' without a default value.
    attributes=[ir.Attr("alpha", ir.AttributeType.FLOAT, None)],
)


# === Step 3: Construct the Main Graph and Model ===
print("Building the main graph that calls 'CustomLeakyRelu' with a specific alpha...")
shape = (3, 4)
x = ir.Value(name="main_x", shape=ir.Shape(shape), type=ir.TensorType(ir.DataType.FLOAT))
y = ir.Value(name="main_y", shape=ir.Shape(shape), type=ir.TensorType(ir.DataType.FLOAT))

# Create a node that calls our custom function.
# We provide a concrete value for the 'alpha' attribute here.
custom_leaky_relu_node = ir.node(
    op_type="CustomLeakyRelu",
    inputs=[x],
    outputs=[y],
    domain="custom.domain",
    attributes={"alpha": 0.1},  # Pass the attribute value here
)

# Construct the main graph
main_graph = ir.Graph(
    inputs=[x],
    outputs=[y],
    nodes=[custom_leaky_relu_node],
    initializers=[],
    name="MainGraph",
    opset_imports={"": 18, "custom.domain": 1},
)

# Create the model, including the function definition
model = ir.Model(
    main_graph,
    ir_version=10,
    functions=[custom_leaky_relu_function],
)


# === Step 4: Save the Model ===
print(f"Saving the model to '{output_path}'...")
ir.save(model, output_path)

print("\n✅ Model with attributed function saved successfully.")