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

    from jax2onnx.plugin_system import (
        get_qualified_name,
    )  # Local import to avoid circularity

    impl_key = get_qualified_name(orig_fn)
    print(f"\n🚀 Encountered function primitive: {impl_key}")

    instance_base_name = name.split(".")[-1]

    if instance_base_name == "MultiHeadAttention":
        print("⚠️ MultiHeadAttention ...")

    if instance_base_name == "TransformerBlock":
        print("⚠️ TransformerBlock ...")

    unique_node_name = converter.builder.get_unique_instance_name(instance_base_name)
    print(f"   -> Generating unique ONNX node name: {unique_node_name}")

    try:
        input_names = [
            converter.get_name(v) for v in eqn.invars if not isinstance(v, Literal)
        ]
        example_args = []
        for var in eqn.invars:
            if isinstance(var, jax.core.Var):
                aval = var.aval
                example_args.append(
                    jnp.ones(aval.shape, dtype=aval.dtype)
                    if aval.shape
                    else jnp.zeros((), dtype=aval.dtype)
                )
            elif isinstance(var, Literal):
                example_args.append(var.val)
            else:
                raise TypeError(f"Unexpected input var type: {type(var)}")
    except Exception as e:
        print(f"❌ Failed to prepare inputs for {impl_key}: {e}")
        raise

    parent_builder = converter.builder

    # if impl_key not in parent_builder.function_name_cache:
    # unique_func_name = parent_builder.get_unique_name(name + "_def")
    unique_func_name = unique_node_name
    # parent_builder.function_name_cache[impl_key] = unique_func_name
    print(f"   -> Tracing function body for: {unique_func_name}")

    sub_builder = OnnxBuilder(
        parent_builder.name_counter,
        parent_builder.name_generator,
        parent_builder.opset,
        unique_func_name + "_graph",
        initializers=parent_builder.initializers,
    )
    sub_converter = converter.__class__(sub_builder)

    try:
        sub_converter.trace_jaxpr(orig_fn, example_args, preserve_graph=True)
    except Exception as e:
        print(f"❌ Failed to trace {impl_key}: {e}")
        raise

    initializer_names = {i.name for i in parent_builder.initializers}
    used_constants = {
        inp
        for node in sub_builder.nodes
        for inp in node.input
        if inp in initializer_names
    }
    param_inputs = sorted(used_constants)
    print(f"   -> Identified parameters (constants): {param_inputs}")

    internal_name = parent_builder.add_function(
        name=unique_func_name,
        sub_builder=sub_builder,
        param_input_names=param_inputs,
        user_display_name=name,
        # allow_duplicates=True,
    )
    parent_builder.functions[impl_key] = parent_builder.functions[internal_name]

    _propagate_nested_functions(parent_builder, sub_builder)
    print(f"✅ Finished tracing function body: {unique_func_name}")
    # else:
    #     internal_name = parent_builder.function_name_cache[impl_key]
    #     print(f"   -> Function definition for {internal_name} already exists.")
    #     func_proto = parent_builder.functions.get(internal_name)
    #     num_inputs = len(input_names)
    #     if func_proto and len(func_proto.input) >= num_inputs:
    #         param_inputs = func_proto.input[num_inputs:]
    #         print(f"   -> Retrieved parameters: {param_inputs}")
    #     else:
    #         print(f"⚠️ Warning: Cannot extract parameters for {impl_key}")
    #         param_inputs = []

    call_inputs = input_names + param_inputs
    output_names = [converter.get_var_name(v) for v in eqn.outvars]

    parent_builder.add_function_call_node(
        function_name=name,
        input_names=call_inputs,
        output_names=output_names,
        node_name=unique_node_name,
        user_display_name=name,
    )

    print(f"   -> Added call node for: {internal_name}")
