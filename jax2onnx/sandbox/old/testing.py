# file: jax2onnx/sandbox/testing.py

import onnx

# model = onnx.load("docs/onnx/examples/onnx_functions/000_one_function_outer.onnx")
model = onnx.load("docs/onnx/examples/onnx_functions/011_vit_conv_embedding.onnx")
# model = onnx.load("docs/onnx/examples/onnx_functions/006_one_function_outer.onnx") #


def print_single_shape_info(vi, prefix=""):  # Added prefix for indentation
    shape = [
        # Handle both dim_value and dim_param for flexibility
        (
            dim.dim_value
            if (dim.HasField("dim_value"))
            else dim.dim_param if (dim.HasField("dim_param")) else "?"
        )
        for dim in vi.type.tensor_type.shape.dim
    ]
    # Added dtype for more info
    dtype = onnx.TensorProto.DataType.Name(vi.type.tensor_type.elem_type)
    print(f"{prefix}{vi.name}: {dtype}{shape}")


def print_single_initializer(ini, prefix=""):  # Added prefix for indentation
    dims = list(ini.dims)
    dtype = onnx.TensorProto.DataType.Name(ini.data_type)
    print(f"{prefix}{ini.name}: {dtype}{dims}")


def print_shape_info(model):
    print("=== MAIN GRAPH INPUT INFO ===")
    for vi in list(model.graph.input):
        print_single_shape_info(vi, prefix="  ")

    print("\n=== MAIN GRAPH INITIALIZER ===")
    if not model.graph.initializer:
        print("  None")
    for ini in list(model.graph.initializer):
        print_single_initializer(ini, prefix="  ")

    print("\n=== MAIN GRAPH VALUE INFO ===")
    if not model.graph.value_info:
        print("  None")
    for vi in list(model.graph.value_info):
        print_single_shape_info(vi, prefix="  ")

    # --- MINIMAL ADDITION FOR FUNCTION INFO ---
    print("\n\n=== LIBRARY FUNCTIONS ===")
    if not model.functions:
        print("  No functions found in the model.")
    else:
        print(f"  Found {len(model.functions)} function(s).")

    for func in model.functions:
        print(f"\n--- Function: {func.name} (Domain: {func.domain}) ---")
        print(f"  Function Inputs: {[i for i in func.input]}")
        print(f"  Function Outputs: {[o for o in func.output]}")
        # Check if function has value_info attribute and print it
        if hasattr(func, "value_info") and func.value_info:
            print("\n  Function Value Info:")
            for vi in func.value_info:
                print_single_shape_info(vi, prefix="    ")  # Use existing helper
        else:
            print("\n  Function Value Info: Not available or empty.")
        # Optionally print nodes for context
        print("\n  Function Nodes:")
        if not func.node:
            print("    None")
        for node in func.node:
            print(
                f"    Node Name: {node.name}, Op Type: {node.op_type}, Inputs: {list(node.input)}, Outputs: {list(node.output)}"
            )
    # --- END OF MINIMAL ADDITION ---

    # print("=== OUTPUT INFO ===") # Keep commented as in original
    # for vi in list(model.graph.output):
    #     print_single_shape_info(vi)


print("--- Model Info Before Shape Inference ---")
print_shape_info(model)

print("\n\n######### RUNNING SHAPE INFERENCE #########\n")
# Note: Shape inference might populate value_info within functions
try:
    inferred_model = onnx.shape_inference.infer_shapes(model)
    print("Shape inference successful.")
    print_shape_info(inferred_model)
except Exception as e:
    print(f"Shape inference failed: {e}")
