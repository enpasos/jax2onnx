import onnx
import onnx.helper as helper
from onnx import TensorProto

# === Main Graph Inputs ===
cond_input = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
x_input = helper.make_tensor_value_info("x", TensorProto.INT32, [])
y_output = helper.make_tensor_value_info("y", TensorProto.INT32, [])

# === THEN branch: input x, output x + 1 ===
then_input = helper.make_tensor_value_info("x", TensorProto.INT32, [])
then_output = helper.make_tensor_value_info("y", TensorProto.INT32, [])

const_one_then = helper.make_node(
    "Constant",
    [],
    ["const1"],
    value=helper.make_tensor("value", TensorProto.INT32, [], [1]),
)
add_node = helper.make_node("Add", ["x", "const1"], ["y"])

then_graph = helper.make_graph(
    nodes=[const_one_then, add_node],
    name="then_body",
    inputs=[then_input],
    outputs=[then_output],
)

# === ELSE branch: input x, output x - 1 ===
else_input = helper.make_tensor_value_info("x", TensorProto.INT32, [])
else_output = helper.make_tensor_value_info("y", TensorProto.INT32, [])

const_one_else = helper.make_node(
    "Constant",
    [],
    ["const1"],
    value=helper.make_tensor("value", TensorProto.INT32, [], [1]),
)
sub_node = helper.make_node("Sub", ["x", "const1"], ["y"])

else_graph = helper.make_graph(
    nodes=[const_one_else, sub_node],
    name="else_body",
    inputs=[else_input],
    outputs=[else_output],
)

# === If Node ===
if_node = helper.make_node(
    "If", inputs=["cond"], outputs=["y"], then_branch=then_graph, else_branch=else_graph
)

# === Main Graph ===
main_graph = helper.make_graph(
    nodes=[if_node],
    name="IfExplicitBranchInputs",
    inputs=[cond_input, x_input],
    outputs=[y_output],
)

# === Model ===
model = helper.make_model(
    main_graph,
    opset_imports=[helper.make_opsetid("", 23)],
    producer_name="onnx-if-with-branch-inputs",
)


onnx.save_model(model, "docs/onnx/primitives/lax/cond_scalar_2.onnx")
