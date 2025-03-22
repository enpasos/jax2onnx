import onnx
import os
from onnx import helper, TensorProto

# === Define the LinearRegression function in the 'custom' domain ===

# Inputs and outputs of the function
lr_inputs = ["X", "A"]
lr_outputs = ["Y"]

# Attribute: bias
bias_attr = "bias"

# Nodes inside the function
const_node = helper.make_node(
    "Constant", inputs=[], outputs=["B"], domain="", ref_attr_name="bias"
)

matmul_node = helper.make_node("MatMul", inputs=["X", "A"], outputs=["XA"], domain="")

add_node = helper.make_node("Add", inputs=["XA", "B"], outputs=["Y"], domain="")

linear_regression_function = onnx.helper.make_function(
    domain="custom",
    fname="LinearRegression_001",
    inputs=lr_inputs,
    outputs=lr_outputs,
    nodes=[const_node, matmul_node, add_node],
    opset_imports=[
        helper.make_opsetid("", 14),
        helper.make_opsetid("custom", 1),
    ],
    attributes=[bias_attr],
)

# === Define the main graph ===

# Input tensors
X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [None, None])

# Output tensor
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None])

# Attribute for LinearRegression (bias tensor)
bias_tensor = helper.make_tensor(
    name="former_B", data_type=TensorProto.FLOAT, dims=[1], vals=[0.67]
)

lr_node = helper.make_node(
    "LinearRegression_001",
    inputs=["X", "A"],
    outputs=["Y1"],
    domain="custom",
    bias=bias_tensor,  # bind tensor attribute
)

abs_node = helper.make_node("Abs", inputs=["Y1"], outputs=["Y"])

graph = helper.make_graph(
    nodes=[lr_node, abs_node], name="example", inputs=[X, A], outputs=[Y]
)

# === Assemble the model ===
model = helper.make_model(
    graph,
    opset_imports=[helper.make_opsetid("", 14), helper.make_opsetid("custom", 1)],
    functions=[linear_regression_function],
    ir_version=9,
)


# Save the model to file
dir = "./docs/onnx/sandbox"
path = os.path.join(dir, "custom_onnx_function_example.onnx")
# create directory if it doesn't exist


if not os.path.exists(os.path.dirname(path)):
    os.makedirs(os.path.dirname(path))
onnx.save(model, path)
