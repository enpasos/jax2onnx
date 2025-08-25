import os

import onnx
from onnx import TensorProto, helper

# Define a unique domain for our custom functions
CUSTOM_DOMAIN = "my.custom.domain.embedded"
OPSET_VERSION = 14  # Choose an appropriate opset version


# === Helper Function to Create Constant Node for Float Scalar ===
def make_float_scalar_constant_node(name, output_name, value):
    """Creates an ONNX Constant node for a scalar float value."""
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[output_name],
        name=name,
        value=helper.make_tensor(
            name=output_name + "_value",  # Tensor name inside attribute
            data_type=TensorProto.FLOAT,
            dims=[],  # Scalar
            vals=[float(value)],
        ),
    )


# === 1. Define FunctionProto 1: ScaleAndAdd_A ===
# Function: Y_out = (X_in * 2.0) + 1.0
# Constants 2.0 and 1.0 are embedded using Constant nodes

func_A_input_names = ["X_in_A"]
func_A_output_names = ["Y_out_A"]

# Define Constant nodes *inside* the function for the scalar values
func_A_node_const_scale = make_float_scalar_constant_node(
    name="FuncA_Const_Scale", output_name="scale_A_local_val", value=2.0
)
func_A_node_const_bias = make_float_scalar_constant_node(
    name="FuncA_Const_Bias", output_name="bias_A_local_val", value=1.0
)

# Define computation nodes using the output names of the Constant nodes
func_A_node_mul = helper.make_node(
    "Mul",
    inputs=["X_in_A", "scale_A_local_val"],  # Use Constant node output
    outputs=["mul_result_A"],
    name="FuncA_Node_Mul",
)
func_A_node_add = helper.make_node(
    "Add",
    inputs=["mul_result_A", "bias_A_local_val"],  # Use Constant node output
    outputs=["Y_out_A"],
    name="FuncA_Node_Add",
)

# Create the FunctionProto for ScaleAndAdd_A
# Include the Constant nodes in the 'nodes' list
scale_and_add_func_A = helper.make_function(
    domain=CUSTOM_DOMAIN,
    fname="ScaleAndAdd_A",
    inputs=func_A_input_names,
    outputs=func_A_output_names,
    nodes=[
        func_A_node_const_scale,  # Constant node comes first
        func_A_node_const_bias,  # Constant node comes first
        func_A_node_mul,
        func_A_node_add,
    ],
    # NO 'initializer' argument here
    opset_imports=[helper.make_opsetid("", OPSET_VERSION)],
)

# === 2. Define FunctionProto 2: ScaleAndAdd_B ===
# Function: Y_out = (X_in * 3.0) + 5.0
# Constants 3.0 and 5.0 are embedded using Constant nodes

func_B_input_names = ["X_in_B"]
func_B_output_names = ["Y_out_B"]

# Define Constant nodes *inside* this function
func_B_node_const_scale = make_float_scalar_constant_node(
    name="FuncB_Const_Scale",
    output_name="scale_B_local_val",
    value=3.0,  # Different value
)
func_B_node_const_bias = make_float_scalar_constant_node(
    name="FuncB_Const_Bias",
    output_name="bias_B_local_val",
    value=5.0,  # Different value
)

# Define computation nodes using the output names of the Constant nodes
func_B_node_mul = helper.make_node(
    "Mul",
    inputs=["X_in_B", "scale_B_local_val"],  # Use Constant node output
    outputs=["mul_result_B"],
    name="FuncB_Node_Mul",
)
func_B_node_add = helper.make_node(
    "Add",
    inputs=["mul_result_B", "bias_B_local_val"],  # Use Constant node output
    outputs=["Y_out_B"],
    name="FuncB_Node_Add",
)

# Create the FunctionProto for ScaleAndAdd_B
scale_and_add_func_B = helper.make_function(
    domain=CUSTOM_DOMAIN,
    fname="ScaleAndAdd_B",
    inputs=func_B_input_names,
    outputs=func_B_output_names,
    nodes=[
        func_B_node_const_scale,
        func_B_node_const_bias,
        func_B_node_mul,
        func_B_node_add,
    ],
    # NO 'initializer' argument here
    opset_imports=[helper.make_opsetid("", OPSET_VERSION)],
)


# === 3. Define the Main GraphProto ===
# (This part remains the same as the previous attempt)

graph_input_1 = helper.make_tensor_value_info("Input_1", TensorProto.FLOAT, None)
graph_input_2 = helper.make_tensor_value_info("Input_2", TensorProto.FLOAT, None)
graph_output_1 = helper.make_tensor_value_info("Output_1", TensorProto.FLOAT, None)
graph_output_2 = helper.make_tensor_value_info("Output_2", TensorProto.FLOAT, None)

graph_node_1 = helper.make_node(
    op_type="ScaleAndAdd_A",
    inputs=["Input_1"],
    outputs=["Output_1"],
    name="FunctionCall_A",
    domain=CUSTOM_DOMAIN,
)
graph_node_2 = helper.make_node(
    op_type="ScaleAndAdd_B",
    inputs=["Input_2"],
    outputs=["Output_2"],
    name="FunctionCall_B",
    domain=CUSTOM_DOMAIN,
)

main_graph = helper.make_graph(
    nodes=[graph_node_1, graph_node_2],
    name="MainGraph_EmbeddedConstants",
    inputs=[graph_input_1, graph_input_2],
    outputs=[graph_output_1, graph_output_2],
    initializer=[],  # No initializers needed here for scale/bias
)

# === 4. Assemble and Save the ModelProto ===
# (This part remains the same)

opset_imports = [
    helper.make_opsetid("", OPSET_VERSION),
    helper.make_opsetid(CUSTOM_DOMAIN, 1),
]

model = helper.make_model(
    main_graph,
    functions=[
        scale_and_add_func_A,
        scale_and_add_func_B,
    ],  # Include both function defs
    opset_imports=opset_imports,
    producer_name="onnx-embedded-constants-example-v2",
)

try:
    onnx.checker.check_model(model)
    print("ONNX model check passed.")
except onnx.checker.ValidationError as e:
    print(f"ONNX model check failed: {e}")

output_dir = "./onnx_output"
os.makedirs(output_dir, exist_ok=True)
# Use a different filename to avoid confusion
output_path = os.path.join(output_dir, "embedded_constants_corrected_example.onnx")

onnx.save(model, output_path)

print(f"✅ Corrected ONNX model saved to: {output_path}")
print("This version uses Constant nodes inside the functions to embed parameters.")
