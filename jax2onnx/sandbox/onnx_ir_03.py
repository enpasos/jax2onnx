import onnx_ir as ir
import numpy as np
import os

# --- Setup ---
output_dir = "tmp"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, "if_model.onnx")


# === Step 1: Define Inputs for the Main Graph ===
# The main graph will take a boolean condition and a data tensor.
print("Step 1: Defining main graph inputs...")
cond_in = ir.Value(
    name="condition", shape=ir.Shape([1]), type=ir.TensorType(ir.DataType.BOOL)
)
x_in = ir.Value(
    name="X", shape=ir.Shape(("B", 4)), type=ir.TensorType(ir.DataType.FLOAT)
)


# === Step 2: Define the "Then" Branch (Subgraph) ===
# This graph will be executed if the condition is true.
# It takes 'X' as an input from the outer scope.
print("Step 2: Defining the 'then' branch subgraph...")

# --- 'Then' Branch Body ---
# It adds a constant to the input tensor.
then_const = ir.Value(
    name="then_add_val",
    const_value=ir.tensor(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)),
    type=ir.TensorType(ir.DataType.FLOAT),
    shape=ir.Shape([4]),
)
then_output = ir.Value(
    name="then_out", shape=ir.Shape(("B", 4)), type=ir.TensorType(ir.DataType.FLOAT)
)
then_node = ir.node(
    op_type="Add",
    inputs=[x_in, then_const],  # Note: x_in is from the outer scope
    outputs=[then_output],
)

# --- 'Then' Graph Definition ---
# The subgraph has no explicit inputs; it captures them from the parent scope.
then_graph = ir.Graph(
    inputs=[],
    outputs=[then_output],
    nodes=[then_node],
    initializers=[then_const],
    name="then_branch_graph",
)


# === Step 3: Define the "Else" Branch (Subgraph) ===
# This graph will be executed if the condition is false.
print("Step 3: Defining the 'else' branch subgraph...")

# --- 'Else' Branch Body ---
# It multiplies the input tensor by a constant.
else_const = ir.Value(
    name="else_mul_val",
    const_value=ir.tensor(np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)),
    type=ir.TensorType(ir.DataType.FLOAT),
    shape=ir.Shape([4]),
)
else_output = ir.Value(
    name="else_out", shape=ir.Shape(("B", 4)), type=ir.TensorType(ir.DataType.FLOAT)
)
else_node = ir.node(
    op_type="Mul",
    inputs=[x_in, else_const],  # Note: x_in is from the outer scope
    outputs=[else_output],
)

# --- 'Else' Graph Definition ---
else_graph = ir.Graph(
    inputs=[],
    outputs=[else_output],
    nodes=[else_node],
    initializers=[else_const],
    name="else_branch_graph",
)


# === Step 4: Define the Main Graph with the If Node ===
print("Step 4: Defining the main graph with the 'If' node...")

# --- Main Graph Output ---
# The output of the 'If' node must have the same shape/type as the subgraph outputs.
main_output = ir.Value(
    name="Y", shape=ir.Shape(("B", 4)), type=ir.TensorType(ir.DataType.FLOAT)
)

# --- Main Graph Node ---
# The 'If' node takes the condition as input and the two branches as attributes.
if_node = ir.node(
    op_type="If",
    inputs=[cond_in],
    outputs=[main_output],
    attributes={"then_branch": then_graph, "else_branch": else_graph},
)


# === Step 5: Construct the Final Model ===
print("Step 5: Building the final Model object...")
main_graph = ir.Graph(
    inputs=[cond_in, x_in],
    outputs=[main_output],
    nodes=[if_node],
    name="MainIfGraph",
    opset_imports={"": 18},
)

model = ir.Model(main_graph, ir_version=10)


# === Step 6: Save the Model ===
print(f"Step 6: Saving the model to '{output_path}'...")
ir.save(model, output_path)

print("\n✅ Model with 'If' node saved successfully.")
