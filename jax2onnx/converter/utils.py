# file: jax2onnx/converter/utils.py

import numpy as np
from onnx import TensorProto
import jax.numpy as jnp

# Ensure all needed types are imported
from typing import TYPE_CHECKING, Callable

# === Fix: Import core explicitly ===
import jax.core
from jax.extend.core import Literal

# Assuming these are correctly defined in your project:
from jax2onnx.converter.onnx_builder import OnnxBuilder

# Conditionally import Jaxpr2OnnxConverter for type hinting only
if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


# Helper functions (_tensorproto_dtype_to_numpy, _propagate_nested_functions) remain the same as previous stable version


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

    # Generate UNIQUE function name
    unique_func_name = converter.builder.get_unique_instance_name(name)
    print(f"   -> Generating unique ONNX name: {unique_func_name}")

    try:
        # Identify non-literal variables and their names in parent context
        non_literal_invars = [v for v in eqn.invars if not isinstance(v, Literal)]
        input_names = [
            converter.get_name(v) for v in non_literal_invars
        ]  # Names for call node

        # === Fix: Correctly prepare example_args for tracing ===
        example_args = []
        for var in eqn.invars:
            # Ensure using correct Var/Literal types (adjust imports if needed)
            if isinstance(var, jax.core.Var):  # Use jax.core based on converter.py
                # Create placeholder for variable inputs based on aval
                if var.aval.shape == ():
                    example_args.append(jnp.zeros((), dtype=var.aval.dtype))
                else:
                    example_args.append(jnp.ones(var.aval.shape, dtype=var.aval.dtype))

            elif isinstance(var, Literal):  # Use jax.extend.core.Literal
                # Use the actual value for literal inputs
                example_args.append(var.val)
            else:
                raise TypeError(f"Unexpected type in eqn.invars: {type(var)}")

    except AttributeError as e:
        print(
            f"Error accessing attributes in function_handler for {name}. eqn: {eqn}, Error: {e}"
        )
        raise e
    except Exception as e:
        print(
            f"Unexpected error processing inputs/example_args for {name}. eqn: {eqn}, Error: {e}"
        )
        raise e

    parent_builder: OnnxBuilder = converter.builder

    # Trace/define the function only if this unique instance hasn't been seen
    if unique_func_name not in parent_builder.functions:
        print(f"   -> Tracing function body for: {unique_func_name}")
        # Create sub-builder (use parent's counter and initializers as per original stable logic)
        sub_builder = OnnxBuilder(
            parent_builder.name_counter,  # Pass the counter from parent
            parent_builder.opset,
            unique_func_name + "_graph",
            initializers=parent_builder.initializers,  # Pass initializers reference
        )
        # Use the same converter class, passing the new sub-builder
        sub_converter = converter.__class__(sub_builder)

        # Trace the function using the correctly prepared example_args
        try:
            sub_converter.trace_jaxpr(
                orig_fn,
                example_args,  # Pass corrected args
                preserve_graph=True,  # Preserve sub-builder state
            )
        except Exception as e:
            print(
                f"Error during sub_converter.trace_jaxpr for {unique_func_name}. Error: {e}"
            )
            print(f"   -> Example Args causing error: {example_args}")
            raise e

        # Identify parameters (constants used) - original stable logic
        initializers_in_shared_list = {
            init.name: init for init in parent_builder.initializers
        }
        all_constants_used_in_subgraph = set()
        for node in sub_builder.nodes:
            for inp_name in node.input:
                if inp_name in initializers_in_shared_list:
                    all_constants_used_in_subgraph.add(inp_name)

        final_param_input_names = sorted(list(all_constants_used_in_subgraph))
        print(f"   -> Identified parameters (constants): {final_param_input_names}")

        # Add function definition using unique name
        parent_builder.add_function(
            unique_func_name, sub_builder, final_param_input_names
        )

        _propagate_nested_functions(parent_builder, sub_builder)
        print(f"✅ Finished tracing function body: {unique_func_name}")

    else:
        # Function already defined, retrieve parameters - original stable logic
        print(f"   -> Function definition for {unique_func_name} already exists.")
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

    # Add function CALL node using unique name
    call_inputs = input_names + final_param_input_names
    node_output_names = [converter.get_var_name(v) for v in eqn.outvars]

    parent_builder.add_function_call_node(
        unique_func_name,
        call_inputs,
        node_output_names,
    )
    print(f"   -> Added call node for: {unique_func_name}")
