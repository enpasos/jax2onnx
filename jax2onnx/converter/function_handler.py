# file: jax2onnx/converter/function_handler.py

import jax.numpy as jnp
import numpy as np
from onnx import TensorProto


def function_handler(name, converter, eqn, orig_fn, params):

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
    sub_converter = converter.__class__(name_counter=parent_builder.name_counter)

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

    parent_builder.add_function(name, sub_converter.builder, param_input_names)

    # ✅ Explicitly propagate nested ONNX functions upward
    for (
        nested_func_name,
        nested_func_proto,
    ) in sub_converter.builder.functions.items():
        if nested_func_name not in parent_builder.functions:
            parent_builder.functions[nested_func_name] = nested_func_proto
            print(f"🚀 Propagated nested ONNX function: {nested_func_name}")

    parent_builder.name_counter = sub_converter.builder.name_counter

    parent_builder.add_function_call_node(
        name,
        input_names + param_input_names,
        [converter.get_var_name(v) for v in eqn.outvars],
    )

    print(f"✅ Finished tracing function: {name}\n")


def _tensorproto_dtype_to_numpy(onnx_dtype):
    return {
        TensorProto.FLOAT: np.float32,
        TensorProto.DOUBLE: np.float64,
        TensorProto.INT32: np.int32,
        TensorProto.INT64: np.int64,
        TensorProto.BOOL: np.bool_,
    }.get(onnx_dtype, np.float32)
