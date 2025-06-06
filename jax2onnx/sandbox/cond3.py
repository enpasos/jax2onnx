import onnx
import onnx.helper as helper
from onnx import TensorProto

# === Main Graph Output ===
cond_output = helper.make_tensor_value_info("y", TensorProto.INT32, [])

# === Constant cond = True ===
cond_node = helper.make_node(
    "Constant",
    inputs=[],
    outputs=["cond"],
    value=helper.make_tensor("value", TensorProto.BOOL, [], [True]),
)

# === Constant x = 3 ===
const_x_node = helper.make_node(
    "Constant",
    inputs=[],
    outputs=["x"],
    value=helper.make_tensor("value", TensorProto.INT32, [], [3]),
)

# === THEN branch: input x, output x + 1 ===
then_input = helper.make_tensor_value_info("x", TensorProto.INT32, [])
then_output = helper.make_tensor_value_info("y", TensorProto.INT32, [])

const1_then = helper.make_node(
    "Constant",
    [],
    ["const1"],
    value=helper.make_tensor("value", TensorProto.INT32, [], [1]),
)
add_node = helper.make_node("Add", ["x", "const1"], ["y"])

then_graph = helper.make_graph(
    nodes=[const1_then, add_node],
    name="then_body",
    inputs=[then_input],
    outputs=[then_output],
)

# === ELSE branch: input x, output x - 1 ===
else_input = helper.make_tensor_value_info("x", TensorProto.INT32, [])
else_output = helper.make_tensor_value_info("y", TensorProto.INT32, [])

const1_else = helper.make_node(
    "Constant",
    [],
    ["const1"],
    value=helper.make_tensor("value", TensorProto.INT32, [], [1]),
)
sub_node = helper.make_node("Sub", ["x", "const1"], ["y"])

else_graph = helper.make_graph(
    nodes=[const1_else, sub_node],
    name="else_body",
    inputs=[else_input],
    outputs=[else_output],
)

# === If Node (x is implicitly captured in subgraphs) ===
if_node = helper.make_node(
    "If", inputs=["cond"], outputs=["y"], then_branch=then_graph, else_branch=else_graph
)

# === Main Graph ===
main_graph = helper.make_graph(
    nodes=[cond_node, const_x_node, if_node],
    name="IfWithConstantXOuterScope",
    inputs=[],
    outputs=[cond_output],
)

# === Model ===
model = helper.make_model(
    main_graph,
    opset_imports=[helper.make_opsetid("", 23)],
    producer_name="onnx-if-outer-constant-x",
)

onnx.save_model(model, "docs/onnx/primitives/lax/cond_scalar_3.onnx")
print("Saved: docs/onnx/primitives/lax/cond_scalar_3.onnx")
