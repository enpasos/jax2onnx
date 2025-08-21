import onnx
from onnx import TensorProto, TensorShapeProto, TypeProto, ValueInfoProto, helper

# Define constants for clarity
OPSET_VERSION = 14  # Use a reasonably recent opset
FUNCTION_DOMAIN = "local.functions.example"  # Custom domain for our function
MODEL_NAME = "function_with_intermediate_shape"
MODEL_PATH = f"docs/onnx/sandbox/{MODEL_NAME}.onnx"

# Define Tensor shapes and types
# Using a symbolic dimension 'N' and a fixed dimension 10
shape_N_10 = ["N", 10]
input_type = TensorProto.FLOAT


# Helper to create ValueInfoProto with symbolic dimensions
def make_tensor_value_info_symbolic(
    name: str, elem_type: int, shape: list
) -> ValueInfoProto:
    """Creates ValueInfoProto allowing symbolic dimensions (strings)."""
    value_info = ValueInfoProto()
    value_info.name = name
    tensor_type = TypeProto.Tensor()
    tensor_type.elem_type = elem_type
    tensor_shape = TensorShapeProto()
    for dim in shape:
        dim_proto = TensorShapeProto.Dimension()
        if isinstance(dim, str):
            dim_proto.dim_param = dim  # Symbolic dimension
        elif isinstance(dim, int):
            dim_proto.dim_value = dim  # Fixed dimension
        else:
            # Add basic handling for None or other unexpected types if needed
            dim_proto.dim_param = "?"  # Or raise an error
        tensor_shape.dim.append(dim_proto)
    tensor_type.shape.CopyFrom(tensor_shape)
    value_info.type.tensor_type.CopyFrom(tensor_type)
    return value_info


print("Creating ONNX model components...")

# --- 1. Define ValueInfo for all relevant tensors ---
# We define these upfront for clarity, including the intermediate one.
vi_func_in = make_tensor_value_info_symbolic("func_in", input_type, shape_N_10)
vi_intermediate = make_tensor_value_info_symbolic(
    "intermediate_tensor", input_type, shape_N_10
)
vi_func_out = make_tensor_value_info_symbolic("func_out", input_type, shape_N_10)

# --- 2. Define the Nodes inside the Function ---
node1_add = helper.make_node(
    op_type="Add",
    inputs=[vi_func_in.name, vi_func_in.name],  # Add input to itself for simplicity
    outputs=[vi_intermediate.name],
    name="function_node_add",
)

node2_relu = helper.make_node(
    op_type="Relu",
    inputs=[vi_intermediate.name],
    outputs=[vi_func_out.name],
    name="function_node_relu",
)

# --- 3. Define the Function Proto ---
# Define the opset used *within* the function body
func_opset_import = [helper.make_opsetid("", OPSET_VERSION)]

my_function_proto = helper.make_function(
    domain=FUNCTION_DOMAIN,  # Custom domain
    fname="MySimpleFunction",  # Function name
    inputs=[vi_func_in.name],  # Function input names
    outputs=[vi_func_out.name],  # Function output names
    nodes=[node1_add, node2_relu],  # Nodes implementing the function
    opset_imports=func_opset_import,  # Opset for function's internal nodes
    doc_string="A simple function with Add and Relu.",
)
print(f"FunctionProto '{my_function_proto.name}' created.")

# --- 4. Define the Main Graph that calls the Function ---
# Define ValueInfo for the main graph's external inputs/outputs
vi_graph_in = make_tensor_value_info_symbolic("graph_in", input_type, shape_N_10)
vi_graph_out = make_tensor_value_info_symbolic("graph_out", input_type, shape_N_10)

# Define the node that calls the function
# Note: Its op_type matches the function name (fname)
#       Its domain matches the function domain
node_function_call = helper.make_node(
    op_type="MySimpleFunction",  # Must match function fname
    inputs=[vi_graph_in.name],
    outputs=[vi_graph_out.name],
    name="main_graph_function_call_node",
    domain=FUNCTION_DOMAIN,  # Must match function domain
)
print(f"Main graph node calling '{node_function_call.op_type}' created.")

# --- Crucial Step: Collect ALL ValueInfo protos ---
# Include graph I/O AND the function's I/O and intermediate tensors
# This makes the shapes explicitly available in the graph's value_info list.
all_value_info = [
    vi_graph_in,
    vi_graph_out,
    vi_func_in,  # Explicitly add function input info
    vi_intermediate,  # Explicitly add intermediate info
    vi_func_out,  # Explicitly add function output info
]
print("Collected ValueInfo for graph (including function internals).")


# Define the opsets used by the main graph (including the custom function domain)
graph_opset_import = [
    helper.make_opsetid("", OPSET_VERSION),  # Standard ONNX opset
    helper.make_opsetid(FUNCTION_DOMAIN, 1),  # Custom function domain opset
]

# Create the main graph proto
main_graph = helper.make_graph(
    nodes=[node_function_call],  # Node(s) in the main graph
    name="main_graph_calling_function",
    inputs=[vi_graph_in],  # Graph Input ValueInfo
    outputs=[vi_graph_out],  # Graph Output ValueInfo
    value_info=all_value_info,  # <--- Explicitly list all known tensor infos here
    # initializers=None               # No initializers needed for this example
)
print(f"GraphProto '{main_graph.name}' created.")


# --- 5. Define the Model Proto ---
# Model also needs opset imports
model_opset_import = graph_opset_import  # Can be the same as graph in this case

model = helper.make_model(
    graph=main_graph,
    functions=[my_function_proto],  # Include the function definition
    producer_name="ONNX_Example_Script",
    opset_imports=model_opset_import,
)
print(f"ModelProto '{MODEL_NAME}' created.")


# --- 6. Validate and Save ---
print("Checking model validity...")
try:
    onnx.checker.check_model(model)
    print("Model check passed.")

    # Optional: Run shape inference (though not strictly needed here as we added info manually)
    # try:
    #     model = onnx.shape_inference.infer_shapes(model)
    #     print("Shape inference ran (optional step).")
    # except Exception as e:
    #     print(f"Optional shape inference failed: {e}")

    print(f"Saving model to: {MODEL_PATH}")
    onnx.save(model, MODEL_PATH)
    print("Model saved successfully.")
    print(f"\n--> You can now open '{MODEL_PATH}' in Netron.")

except onnx.checker.ValidationError as e:
    print(f"Model validation failed: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback

    traceback.print_exc()
