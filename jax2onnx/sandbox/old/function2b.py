import os  # Import os for path manipulation

import onnx
from onnx import TensorProto, TensorShapeProto, TypeProto, ValueInfoProto, helper

# Define constants for clarity
OPSET_VERSION = 14  # Use a reasonably recent opset
FUNCTION_DOMAIN = "local.functions.example"  # Custom domain for our function
MODEL_NAME = "function_with_internal_vi"  # Changed name slightly
# Ensure the sandbox directory exists or adjust the path as needed
output_dir = "docs/onnx/sandbox/"
os.makedirs(output_dir, exist_ok=True)  # Create dir if it doesn't exist
MODEL_PATH = os.path.join(output_dir, f"{MODEL_NAME}.onnx")


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
# ValueInfo for the function's scope (input, intermediate, output)
vi_func_in = make_tensor_value_info_symbolic("func_in", input_type, shape_N_10)
vi_intermediate = make_tensor_value_info_symbolic(
    "intermediate_tensor", input_type, shape_N_10
)
vi_func_out = make_tensor_value_info_symbolic("func_out", input_type, shape_N_10)

# ValueInfo for the main graph's external inputs/outputs
vi_graph_in = make_tensor_value_info_symbolic("graph_in", input_type, shape_N_10)
vi_graph_out = make_tensor_value_info_symbolic("graph_out", input_type, shape_N_10)

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

# *** MODIFICATION START ***
# Collect ValueInfo specific to the function's scope
function_value_info = [vi_func_in, vi_intermediate, vi_func_out]
print("Collected ValueInfo for FunctionProto.")

my_function_proto = helper.make_function(
    domain=FUNCTION_DOMAIN,  # Custom domain
    fname="MySimpleFunction",  # Function name
    inputs=[vi_func_in.name],  # Function input names
    outputs=[vi_func_out.name],  # Function output names
    nodes=[node1_add, node2_relu],  # Nodes implementing the function
    opset_imports=func_opset_import,  # Opset for function's internal nodes
    value_info=function_value_info,  # <-- Pass function's ValueInfo list here
    doc_string="A simple function with Add and Relu, VI inside FunctionProto.",
)
# *** MODIFICATION END ***
print(f"FunctionProto '{my_function_proto.name}' created with internal ValueInfo.")

# --- 4. Define the Main Graph that calls the Function ---
# Define the node that calls the function
node_function_call = helper.make_node(
    op_type="MySimpleFunction",  # Must match function fname
    inputs=[vi_graph_in.name],
    outputs=[vi_graph_out.name],
    name="main_graph_function_call_node",
    domain=FUNCTION_DOMAIN,  # Must match function domain
)
print(f"Main graph node calling '{node_function_call.op_type}' created.")

# *** MODIFICATION START ***
# Collect ValueInfo relevant ONLY to the main graph scope
# (Graph inputs, outputs, and any intermediates specific to the main graph, if any)
graph_value_info = [vi_graph_in, vi_graph_out]
print("Collected ValueInfo for GraphProto (graph I/O only).")
# *** MODIFICATION END ***


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
    value_info=graph_value_info,  # <-- Pass GRAPH's ValueInfo list here
    # initializers=None               # No initializers needed for this example
)
print(f"GraphProto '{main_graph.name}' created.")


# --- 5. Define the Model Proto ---
# Model also needs opset imports
model_opset_import = graph_opset_import  # Can be the same as graph in this case

model = helper.make_model(
    graph=main_graph,
    functions=[my_function_proto],  # Include the function definition
    producer_name="ONNX_Example_Script_V2",
    opset_imports=model_opset_import,
)
print(f"ModelProto '{MODEL_NAME}' created.")


# --- 6. Validate and Save ---
print("Checking model validity...")
try:
    onnx.checker.check_model(model)
    print("Model check passed.")

    # Optional: Run shape inference. It *should* ideally pick up the info
    # we added inside the function now, but its main role is adding info
    # that might be missing.
    try:
        # Strict mode might fail if symbolic dims aren't fully handled by infer.
        inferred_model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
        print("Shape inference ran (optional step).")
        # Decide whether to save the inferred model or the original one
        # Saving the original ensures we only have the manually added info.
        # model_to_save = inferred_model
        model_to_save = model  # Save the one with only manually added VI
    except Exception as e:
        print(f"Optional shape inference failed: {e}. Saving original model.")
        model_to_save = model

    print(f"Saving model to: {MODEL_PATH}")
    onnx.save(model_to_save, MODEL_PATH)
    print("Model saved successfully.")
    print(f"\n--> You can now open '{MODEL_PATH}' in Netron.")
    print("--> Check if shapes now appear on connections INSIDE the function view.")

except onnx.checker.ValidationError as e:
    print(f"Model validation failed: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback

    traceback.print_exc()
