# file: jax2onnx/converter/utils.py

import numpy as np
from onnx import (
    TensorProto,
)  # Ensure necessary onnx imports
import jax.numpy as jnp
from typing import (
    TYPE_CHECKING,
)  # Ensure necessary types are imported
from jax.extend.core import Literal

# Assuming these are correctly defined in your project:
from jax2onnx.converter.onnx_builder import OnnxBuilder

# Conditionally import Jaxpr2OnnxConverter for type hinting only
if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


# Ensure these helpers are defined or imported correctly in utils.py
def _tensorproto_dtype_to_numpy(onnx_dtype):
    """
    Converts ONNX TensorProto data types to NumPy data types.
    """
    return {
        TensorProto.FLOAT: np.float32,
        TensorProto.DOUBLE: np.float64,
        TensorProto.INT32: np.int32,
        TensorProto.INT64: np.int64,
        TensorProto.BOOL: np.bool_,
    }.get(onnx_dtype, np.float32)


def _propagate_nested_functions(parent_builder, sub_builder):
    """
    Propagates nested ONNX functions from a sub-builder to a parent builder.
    """
    for nested_func_name, nested_func_proto in sub_builder.functions.items():
        if nested_func_name not in parent_builder.functions:
            parent_builder.functions[nested_func_name] = nested_func_proto
            print(f"🚀 Propagated nested ONNX function: {nested_func_name}")


def function_handler(name, converter: "Jaxpr2OnnxConverter", eqn, orig_fn, params):
    """
    Handles nested JAX functions by creating a nested ONNX function and propagating it to the parent builder.
    """
    if orig_fn is None:
        raise RuntimeError(f"Original function for {name} not recorded.")

    print(f"\n🚀 Tracing function: {name}")
    try:
        input_shapes = [var.aval.shape for var in eqn.invars]
        non_literal_invars = [v for v in eqn.invars if not isinstance(v, Literal)]
        input_names = [converter.get_name(v) for v in non_literal_invars]
        example_args = [
            jnp.ones(shape, dtype=var.aval.dtype)
            for shape, var in zip(input_shapes, eqn.invars)
        ]
    except AttributeError as e:
        print(
            f"Error accessing eqn.invars in function_handler for {name}. eqn type: {type(eqn)}"
        )
        raise e

    parent_builder: OnnxBuilder = converter.builder
    sub_builder = OnnxBuilder(
        parent_builder.name_counter,
        parent_builder.opset,
        parent_builder.model_name,
        initializers=parent_builder.initializers,
    )
    sub_converter = converter.__class__(sub_builder)

    sub_converter.trace_jaxpr(
        orig_fn,
        example_args,
        preserve_graph=True,
    )

    # Detect constants used in subgraph
    initializers_in_shared_list = {
        init.name: init for init in parent_builder.initializers
    }
    all_constants_used_in_subgraph = set()
    for node in sub_builder.nodes:
        for inp_name in node.input:
            if inp_name in initializers_in_shared_list:
                all_constants_used_in_subgraph.add(inp_name)
    # These are the names needed as parameters for the function definition and call
    final_param_input_names = sorted(list(all_constants_used_in_subgraph))

    parent_builder.add_function(name, sub_builder, final_param_input_names)
    _propagate_nested_functions(parent_builder, sub_builder)

    parent_builder.add_function_call_node(
        name,
        input_names + final_param_input_names,
        [converter.get_var_name(v) for v in eqn.outvars],
    )

    print(f"✅ Finished tracing function: {name}\n")
