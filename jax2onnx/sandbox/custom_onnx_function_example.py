import onnx
import os
from onnx import helper, TensorProto
from jax2onnx import to_onnx
import jax.numpy as jnp
from flax import nnx
from onnx import GraphProto

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


class MLPBlock(nnx.Module):
    """MLP block for Transformer layers."""

    def __init__(self, num_hiddens, mlp_dim, dropout_rate=0.1, *, rngs: nnx.Rngs):
        self.layers = [
            nnx.Linear(num_hiddens, mlp_dim, rngs=rngs),
            lambda x: nnx.gelu(x, approximate=False),
            nnx.Dropout(rate=dropout_rate, rngs=rngs),
            nnx.Linear(mlp_dim, num_hiddens, rngs=rngs),
            nnx.Dropout(rate=dropout_rate, rngs=rngs),
        ]

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        for layer in self.layers:
            if isinstance(layer, nnx.Dropout):
                x = layer(x, deterministic=deterministic)
            else:
                x = layer(x)
        return x


my_callable = MLPBlock(num_hiddens=256, mlp_dim=512, dropout_rate=0.1, rngs=nnx.Rngs(0))


onnx_model = to_onnx(my_callable, [("B", 30)])


# Step 1: Extract the graph from the exported model
mlp_graph: GraphProto = onnx_model.graph

# Step 2: Create a function from the graph
mlp_function = helper.make_function(
    domain="custom",
    fname="MLPBlock_001",
    inputs=[i.name for i in mlp_graph.input],
    outputs=[o.name for o in mlp_graph.output],
    nodes=mlp_graph.node,
    opset_imports=onnx_model.opset_import,
    attributes=[],  # No attributes, unless you define Dropout rates etc. as attributes
)

# Step 3: Create a new graph that calls the function
input_tensor = helper.make_tensor_value_info("Input", TensorProto.FLOAT, [1, 30])
output_tensor = helper.make_tensor_value_info("Output", TensorProto.FLOAT, [1, 30])

mlp_node = helper.make_node(
    "MLPBlock_001",
    inputs=["Input"],
    outputs=["Output"],
    domain="custom",
)

top_graph = helper.make_graph(
    nodes=[mlp_node],
    name="MLPBlockGraph",
    inputs=[input_tensor],
    outputs=[output_tensor],
)

# Step 4: Assemble the top-level model
top_model = helper.make_model(
    top_graph,
    opset_imports=onnx_model.opset_import,
    functions=[mlp_function],
    ir_version=9,
    producer_name="jax2onnx + your-brain",
)

# Save the model
onnx.save(top_model, "./docs/onnx/sandbox/onnx_with_mlp_function.onnx")
print("Function-based ONNX model saved.")
