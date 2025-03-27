import numpy as np
from onnx import TensorProto
import jax.numpy as jnp

from jax2onnx.converter.onnx_builder import OnnxBuilder


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


def function_handler(name, converter, eqn, orig_fn, params):
    """
    Handles nested JAX functions by creating a nested ONNX function and propagating it to the parent builder.
    """
    if orig_fn is None:
        raise RuntimeError(f"Original function for {name} not recorded.")

    print(f"\n🚀 Tracing function: {name}")
    input_shapes = [var.aval.shape for var in eqn.invars]
    input_names = [converter.get_name(v) for v in eqn.invars]
    example_args = [
        jnp.ones(shape, dtype=var.aval.dtype)
        for shape, var in zip(input_shapes, eqn.invars)
    ]

    parent_builder = converter.builder
    builder = OnnxBuilder(
        parent_builder.name_counter,
        parent_builder.opset,
        parent_builder.model_name,
    )
    sub_converter = converter.__class__(builder)

    sub_converter.trace_jaxpr(
        orig_fn,
        example_args,
        preserve_graph=True,
    )

    # Transfer initializers as function parameters
    function_initializers = sub_converter.builder.initializers
    param_input_names = [init.name for init in function_initializers]

    existing_input_names = {vi.name for vi in parent_builder.inputs}
    for init in function_initializers:
        if init.name not in existing_input_names:
            parent_builder.add_input(
                init.name,
                list(init.dims),
                _tensorproto_dtype_to_numpy(init.data_type),
            )
            parent_builder.initializers.append(init)
        else:
            print(f"⚠️ Constant '{init.name}' already exists in parent builder inputs.")

    parent_builder.add_function(name, sub_converter.builder, param_input_names)

    # Explicitly propagate nested ONNX functions upward
    _propagate_nested_functions(parent_builder, sub_converter.builder)

    parent_builder.name_counter = sub_converter.builder.name_counter

    parent_builder.add_function_call_node(
        name,
        input_names + param_input_names,
        [converter.get_var_name(v) for v in eqn.outvars],
    )

    print(f"✅ Finished tracing function: {name}\n")
