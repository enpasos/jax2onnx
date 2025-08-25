import onnx_ir as ir
import numpy as np
import os

# --- Setup ---
output_dir = "tmp"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, "nested_functions_model.onnx")

# Define a custom domain for our functions
CUSTOM_DOMAIN = "my.custom.domain"

# === Step 1: Define the Inner Function (add_one) ===
print("Step 1: Defining the inner function 'add_one'...")

# --- Function Inputs/Outputs ---
add_one_input = ir.Value(
    name="x_inner", shape=ir.Shape(("B", 2)), type=ir.TensorType(ir.DataType.FLOAT)
)
add_one_output = ir.Value(
    name="y_inner", shape=ir.Shape(("B", 2)), type=ir.TensorType(ir.DataType.FLOAT)
)
one_const = ir.Value(
    name="one",
    const_value=ir.tensor(np.array([1.0], dtype=np.float32)),
    type=ir.TensorType(ir.DataType.FLOAT),
    shape=ir.Shape([1]),
)

# --- Function Body ---
add_one_node = ir.node(
    op_type="Add", inputs=[add_one_input, one_const], outputs=[add_one_output]
)

# --- Function Definition ---
# An ir.Function is essentially a graph with a name and domain.
add_one_func = ir.Function(
    domain=CUSTOM_DOMAIN,
    name="add_one",
    graph=ir.Graph(
        inputs=[add_one_input],
        outputs=[add_one_output],
        nodes=[add_one_node],
        initializers=[one_const],
    ),
    attributes=[],  # No function-level attributes for this simple case
)


# === Step 2: Define the Outer Function (add_two) ===
# This function will call the `add_one` function.
print("Step 2: Defining the outer function 'add_two'...")

# --- Function Inputs/Outputs ---
add_two_input = ir.Value(
    name="x_outer", shape=ir.Shape(("B", 2)), type=ir.TensorType(ir.DataType.FLOAT)
)
add_two_intermediate = ir.Value(
    name="intermediate", shape=ir.Shape(("B", 2)), type=ir.TensorType(ir.DataType.FLOAT)
)
add_two_output = ir.Value(
    name="y_outer", shape=ir.Shape(("B", 2)), type=ir.TensorType(ir.DataType.FLOAT)
)

# --- Function Body ---
# This node calls the 'add_one' function.
call_add_one_node1 = ir.node(
    op_type="add_one",  # Function name
    domain=CUSTOM_DOMAIN,  # Function domain
    inputs=[add_two_input],
    outputs=[add_two_intermediate],
)
# This node adds one again.
call_add_one_node2 = ir.node(
    op_type="add_one",
    domain=CUSTOM_DOMAIN,
    inputs=[add_two_intermediate],
    outputs=[add_two_output],
)

# --- Function Definition ---
add_two_func = ir.Function(
    domain=CUSTOM_DOMAIN,
    name="add_two",
    graph=ir.Graph(
        inputs=[add_two_input],
        outputs=[add_two_output],
        nodes=[call_add_one_node1, call_add_one_node2],
    ),
    attributes=[],
)


# === Step 3: Define the Main Graph ===
print("Step 3: Defining the main graph...")

# --- Main Graph Inputs/Outputs ---
main_input = ir.Value(
    name="X", shape=ir.Shape(("B", 2)), type=ir.TensorType(ir.DataType.FLOAT)
)
main_output = ir.Value(
    name="Y", shape=ir.Shape(("B", 2)), type=ir.TensorType(ir.DataType.FLOAT)
)

# --- Main Graph Node ---
# This node calls the 'add_two' function.
call_add_two_node = ir.node(
    op_type="add_two",  # Function name
    domain=CUSTOM_DOMAIN,  # Function domain
    inputs=[main_input],
    outputs=[main_output],
)


# === Step 4: Construct the Final Model ===
# The model contains the main graph and the definitions of all functions it uses.
print("Step 4: Building the final Model object...")
main_graph = ir.Graph(
    inputs=[main_input],
    outputs=[main_output],
    nodes=[call_add_two_node],
    name="MainGraph",
    opset_imports={"": 18, CUSTOM_DOMAIN: 1},  # Import both standard and custom domains
)

model = ir.Model(
    main_graph,
    ir_version=10,
    functions=[add_one_func, add_two_func],  # Provide the function definitions
)


# === Step 5: Save the Model ===
print(f"Step 5: Saving the model to '{output_path}'...")
ir.save(model, output_path)

print("\n✅ Model with nested functions saved successfully.")
