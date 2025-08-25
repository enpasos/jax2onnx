import onnx
from onnx import TensorProto, shape_inference


# Helper to extract shape and dtype from ValueInfoProto
def extract_value_info_shape_dtype(
    value_info: onnx.ValueInfoProto,
) -> tuple[list[str], str]:
    """Extracts shape and dtype from a ValueInfoProto."""
    try:
        t = value_info.type.tensor_type
        if t.elem_type == TensorProto.UNDEFINED:
            dtype = "UNDEFINED"
        else:
            dtype = TensorProto.DataType.Name(t.elem_type)

        shape = []
        if t.HasField("shape"):
            shape = [
                (
                    str(dim.dim_value)
                    if dim.HasField("dim_value")
                    else dim.dim_param if dim.HasField("dim_param") else "?"
                )
                for dim in t.shape.dim
            ]
        else:
            shape = ["?"]  # Unknown shape
        return shape, dtype
    except Exception:
        # Fallback for types other than TensorProto if needed, or if errors occur
        return ["ERROR"], "ERROR"


# Helper to extract shape and dtype from Initializer (TensorProto)
def extract_initializer_shape_dtype(
    initializer: onnx.TensorProto,
) -> tuple[list[str], str]:
    """Extracts shape and dtype from a TensorProto (Initializer)."""
    try:
        dtype = TensorProto.DataType.Name(initializer.data_type)
        # Shape is directly available in dims
        shape = [str(dim) for dim in initializer.dims]
        return shape, dtype
    except Exception:
        return ["ERROR"], "ERROR"


# --- Enhanced Collection Function ---
def collect_all_tensor_details(
    model: onnx.ModelProto,
) -> dict[str, tuple[list[str], str]]:
    """
    Collects shape and dtype for all known tensors in the model (after shape inference).
    Includes graph inputs, outputs, initializers, and value_info.
    """
    tensor_details = {}
    graph = model.graph

    # 1. Graph Inputs
    for inp in graph.input:
        if inp.name not in tensor_details:  # Avoid overwriting if also in value_info
            shape, dtype = extract_value_info_shape_dtype(inp)
            tensor_details[inp.name] = (shape, dtype)

    # 2. Graph Outputs
    for outp in graph.output:
        if outp.name not in tensor_details:  # Avoid overwriting if also in value_info
            shape, dtype = extract_value_info_shape_dtype(outp)
            tensor_details[outp.name] = (shape, dtype)

    # 3. Initializers (Constants)
    for init in graph.initializer:
        # Initializers provide concrete shape/type
        shape, dtype = extract_initializer_shape_dtype(init)
        tensor_details[init.name] = (shape, dtype)

    # 4. Value Info (often populated by shape inference for intermediate tensors)
    for vi in graph.value_info:
        # This might overwrite info from inputs/outputs if shape inference provided more detail
        shape, dtype = extract_value_info_shape_dtype(vi)
        tensor_details[vi.name] = (shape, dtype)

    # Note: We don't explicitly look inside model.functions here because
    # shape_inference should populate the main graph's value_info
    # with details about tensors used within function calls.

    return tensor_details


# --- Enhanced Printing Function ---
def print_model_details(model: onnx.ModelProto, indent_str: str = "  "):
    """
    Prints detailed information about the main graph and all functions,
    including inputs, outputs, initializers, and intermediate tensors.
    """
    print("Collecting tensor details after shape inference...")
    # Run shape inference *first* to get maximum information
    try:
        # Adding data propagation can sometimes provide more type info,
        # especially if there are constants defined inside functions.
        # It requires the model's weights, so it might fail if they aren't
        # in the same directory or if external data is used.
        inferred_model = shape_inference.infer_shapes(
            model, check_type=True, strict_mode=True, data_prop=True
        )
        print("Shape inference successful.")
    except Exception as e:
        print(f"Shape inference with data propagation failed: {e}")
        print("Falling back to basic shape inference.")
        try:
            inferred_model = shape_inference.infer_shapes(model)
            print("Basic shape inference successful.")
        except Exception as e2:
            print(f"Basic shape inference also failed: {e2}")
            print("Proceeding with potentially incomplete information.")
            inferred_model = model  # Use original model if inference fails

    all_tensor_details = collect_all_tensor_details(inferred_model)
    graph = inferred_model.graph  # Use the graph from the *inferred* model

    print(f"\n=== Main Graph: {graph.name or 'N/A'} ===")

    # --- Main Graph Details ---
    graph_input_names = {inp.name for inp in graph.input}
    graph_output_names = {outp.name for outp in graph.output}
    graph_initializer_names = {init.name for init in graph.initializer}

    print(f"\n{indent_str}Inputs:")
    for name in sorted(list(graph_input_names)):
        shape, dtype = all_tensor_details.get(name, (["?"], "UNKNOWN"))
        print(f"{indent_str*2}{name:<40s} shape={shape}, dtype={dtype}")

    print(f"\n{indent_str}Initializers (Constants):")
    for name in sorted(list(graph_initializer_names)):
        shape, dtype = all_tensor_details.get(name, (["?"], "UNKNOWN"))
        # Initializers are technically inputs but often treated separately
        print(f"{indent_str*2}{name:<40s} shape={shape}, dtype={dtype}")

    print(f"\n{indent_str}Outputs:")
    for name in sorted(list(graph_output_names)):
        shape, dtype = all_tensor_details.get(name, (["?"], "UNKNOWN"))
        print(f"{indent_str*2}{name:<40s} shape={shape}, dtype={dtype}")

    print(f"\n{indent_str}Intermediate Tensors (from value_info):")
    intermediate_printed = False
    for name, (shape, dtype) in sorted(all_tensor_details.items()):
        if (
            name not in graph_input_names
            and name not in graph_output_names
            and name not in graph_initializer_names
        ):
            print(f"{indent_str*2}{name:<40s} shape={shape}, dtype={dtype}")
            intermediate_printed = True
    if not intermediate_printed:
        print(f"{indent_str*2}(None explicitly listed in value_info)")

    # --- Function Details ---
    if not model.functions:
        print("\nNo functions defined in this model.")
        return

    print("\n=== Functions ===")
    for func in model.functions:
        print(
            f"\n{indent_str}Function: {func.name} (domain: {func.domain or 'ai.onnx.functions'})"
        )

        func_input_names = set(func.input)
        func_output_names = set(func.output)

        print(f"{indent_str*2}Inputs:")
        if not func_input_names:
            print(f"{indent_str*3}(None)")
        for name in sorted(list(func_input_names)):
            shape, dtype = all_tensor_details.get(name, (["?"], "UNKNOWN"))
            print(f"{indent_str*3}{name:<40s} shape={shape}, dtype={dtype}")

        print(f"{indent_str*2}Outputs:")
        if not func_output_names:
            print(f"{indent_str*3}(None)")
        for name in sorted(list(func_output_names)):
            shape, dtype = all_tensor_details.get(name, (["?"], "UNKNOWN"))
            print(f"{indent_str*3}{name:<40s} shape={shape}, dtype={dtype}")

        print(f"{indent_str*2}Intermediate Tensors (produced by internal nodes):")
        # Find tensors produced by nodes *inside* the function
        internal_node_outputs = set()
        for node in func.node:
            internal_node_outputs.update(node.output)

        # Intermediates are those produced internally but *not* listed as function outputs
        intermediate_names = internal_node_outputs - func_output_names

        if not intermediate_names:
            print(f"{indent_str*3}(None or all internal outputs are function outputs)")
        else:
            for name in sorted(list(intermediate_names)):
                # These might not always have entries in all_tensor_details if shape
                # inference couldn't determine them, especially for complex functions.
                shape, dtype = all_tensor_details.get(name, (["?"], "UNKNOWN"))
                print(
                    f"{indent_str*3}{name:<40s} shape={shape}, dtype={dtype} {'(Info maybe incomplete)' if shape == ['?'] else ''}"
                )


# --- Main Execution ---
# Replace with the actual path to your ONNX model
# onnx_model_path = "docs/onnx/examples/onnx_functions/012_vit_conv_embedding.onnx"
# onnx_model_path = "docs/onnx/examples/onnx_functions/000_one_function_outer.onnx"
onnx_model_path = "docs/onnx/sandbox/function_with_intermediate_shape.onnx"


try:
    # Load the model
    print(f"Loading model: {onnx_model_path}")
    model = onnx.load(onnx_model_path)
    print("Model loaded successfully.")

    # Check model validity before proceeding
    print("Checking model format...")
    onnx.checker.check_model(model)
    print("Model format check passed.")

    # Print the detailed structure
    print_model_details(model)

except FileNotFoundError:
    print(f"Error: Model file not found at {onnx_model_path}")
    print("Please ensure the path is correct.")
except onnx.checker.ValidationError as e:
    print(f"Error: Model validation failed: {e}")
    print("The model might be invalid or corrupted.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
