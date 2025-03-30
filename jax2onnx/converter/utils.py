# file: jax2onnx/converter/utils.py

import numpy as np
from onnx import TensorProto  # Added helper import
import jax.numpy as jnp

# Added missing typing imports from previous steps
from typing import TYPE_CHECKING, Callable
from jax.extend.core import Literal

# Assuming these are correctly defined in your project:
from jax2onnx.converter.onnx_builder import (
    OnnxBuilder,
    CUSTOM_DOMAIN,
)  # Import CUSTOM_DOMAIN

# Conditionally import Jaxpr2OnnxConverter for type hinting only
if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


# Ensure these helpers are defined or imported correctly in utils.py
def _tensorproto_dtype_to_numpy(onnx_dtype: int) -> np.dtype:
    """
    Converts ONNX TensorProto data types to NumPy data types.
    """
    dtype_map = {
        TensorProto.FLOAT: np.float32,
        TensorProto.DOUBLE: np.float64,
        TensorProto.INT32: np.int32,
        TensorProto.INT64: np.int64,
        TensorProto.BOOL: np.bool_,
        TensorProto.INT8: np.int8,
        TensorProto.UINT8: np.uint8,
    }
    np_dtype = dtype_map.get(onnx_dtype)
    if np_dtype is None:
        print(
            f"Warning: Unsupported ONNX dtype {onnx_dtype} encountered in _tensorproto_dtype_to_numpy. Defaulting to np.float32."
        )
        return np.float32
    return np_dtype


def _propagate_nested_functions(parent_builder: OnnxBuilder, sub_builder: OnnxBuilder):
    """
    Propagates nested ONNX functions from a sub-builder to a parent builder.
    Ensures nested functions are added only once.
    """
    # This implementation seems correct based on stable state
    for nested_func_name, nested_func_proto in sub_builder.functions.items():
        if nested_func_name not in parent_builder.functions:
            parent_builder.functions[nested_func_name] = nested_func_proto
            print(f"🚀 Propagated nested ONNX function: {nested_func_name}")


def function_handler(
    name: str, converter: "Jaxpr2OnnxConverter", eqn, orig_fn: Callable, params
):
    """
    Handles nested JAX functions by creating a nested ONNX function and propagating it to the parent builder.
    Uses unique instance names for functions.
    """
    if orig_fn is None:
        raise RuntimeError(f"Original function for {name} not recorded.")

    print(f"\n🚀 Encountered function primitive: {name}")

    # === Minimal Step 3 Change 1: Generate UNIQUE function name ===
    # 'name' here is the original function name (e.g., "TransformerBlock009")
    unique_func_name = converter.builder.get_unique_instance_name(name)
    print(f"   -> Generating unique ONNX name: {unique_func_name}")

    # --- Logic below is from the stable version, adapted to use unique_func_name ---
    try:
        # Get input shapes and non-literal var names
        input_avals = [
            var.aval for var in eqn.invars
        ]  # Get abstract values for tracing args
        non_literal_invars = [v for v in eqn.invars if not isinstance(v, Literal)]
        # Get input names from the parent converter context
        input_names = [converter.get_name(v) for v in non_literal_invars]
        # Prepare example args for tracing based on abstract values
        example_args = [jnp.ones(aval.shape, dtype=aval.dtype) for aval in input_avals]
    except AttributeError as e:
        print(
            f"Error accessing attributes in function_handler for {name}. eqn: {eqn}, Error: {e}"
        )
        raise e
    except Exception as e:  # Broader catch for unexpected issues
        print(f"Unexpected error processing inputs for {name}. eqn: {eqn}, Error: {e}")
        raise e

    parent_builder: OnnxBuilder = converter.builder

    # We only trace/define the function if this unique instance hasn't been seen
    if unique_func_name not in parent_builder.functions:
        print(f"   -> Tracing function body for: {unique_func_name}")
        # Create sub-builder. Crucially, it should use the *parent's* initializer list
        # and the *parent's* name counter for its internal nodes, as per the stable version logic.
        # Using a new GlobalNameCounter() here caused issues before.
        sub_builder = OnnxBuilder(
            parent_builder.name_counter,  # Use parent's counter for sub-graph node names
            parent_builder.opset,
            unique_func_name + "_graph",  # Give subgraph a unique name
            initializers=parent_builder.initializers,  # Share initializers
        )
        # Create sub-converter (without passing parent_converter, as we reverted that logic)
        sub_converter = converter.__class__(sub_builder)

        # Trace the function
        sub_converter.trace_jaxpr(
            orig_fn,
            example_args,
            preserve_graph=True,  # Keep graph state within sub_builder
        )

        # Identify parameters (constants used) - using the stable version's logic
        initializers_in_shared_list = {
            init.name: init for init in parent_builder.initializers
        }
        all_constants_used_in_subgraph = set()
        # Scan nodes created in the sub-builder
        for node in sub_builder.nodes:
            for inp_name in node.input:
                if inp_name in initializers_in_shared_list:
                    all_constants_used_in_subgraph.add(inp_name)

        # Also need to check constants used in nested functions called by this one
        processed_funcs = set()
        queue = list(sub_builder.nodes)
        while queue:
            node = queue.pop(0)
            # Check if it calls a known custom function (use original name 'name' for lookup?)
            # No, the builder should have the unique name if propagated correctly. Check sub_builder.functions
            if node.domain == CUSTOM_DOMAIN and node.op_type in sub_builder.functions:
                nested_func_name = node.op_type
                if nested_func_name not in processed_funcs:
                    processed_funcs.add(nested_func_name)
                    sub_builder.functions[nested_func_name]
                    # How to get parameters of nested function? Assume they are handled recursively.
                    # Just add its nodes to the scan? Risky.
                    # Let's rely on the direct scan for now, assuming propagation works.

        final_param_input_names = sorted(list(all_constants_used_in_subgraph))
        print(f"   -> Identified parameters (constants): {final_param_input_names}")

        # === Minimal Step 3 Change 2: Use unique_func_name for add_function ===
        parent_builder.add_function(
            unique_func_name, sub_builder, final_param_input_names
        )

        _propagate_nested_functions(parent_builder, sub_builder)
        print(f"✅ Finished tracing function body: {unique_func_name}")

    else:
        # Function already defined, need to retrieve/determine parameters for the call
        print(f"   -> Function definition for {unique_func_name} already exists.")
        # Retrieve parameters from the stored FunctionProto
        func_proto = parent_builder.functions[unique_func_name]
        num_expected_graph_inputs = len(input_names)
        if len(func_proto.input) >= num_expected_graph_inputs:
            final_param_input_names = func_proto.input[num_expected_graph_inputs:]
            print(
                f"   -> Retrieved parameters (constants) from proto: {final_param_input_names}"
            )
        else:
            print(
                f"Warning: Could not determine parameters for existing function {unique_func_name}. Input count mismatch."
            )
            final_param_input_names = []  # Fallback

    # === Minimal Step 3 Change 3: Use unique_func_name for add_function_call_node ===
    call_inputs = input_names + final_param_input_names
    node_output_names = [converter.get_var_name(v) for v in eqn.outvars]

    parent_builder.add_function_call_node(
        unique_func_name,  # Use the unique name for the call
        call_inputs,
        node_output_names,
    )
    print(f"   -> Added call node for: {unique_func_name}")

    # Original stable version had print here, keep it if desired
    # print(f"✅ Finished tracing function: {name}\n") # Log original name for clarity?
