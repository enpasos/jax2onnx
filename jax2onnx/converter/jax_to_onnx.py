import onnx
import jax.numpy as jnp
from typing import Any
from jax2onnx.converter.onnx_builder import OnnxBuilder
from jax2onnx.converter.converter import Jaxpr2OnnxConverter
from jax2onnx.converter.optimize_onnx_graph import improve_onnx_model
from jax2onnx.converter.utils import (
    _tensorproto_dtype_to_numpy,
    _propagate_nested_functions,
)


def prepare_example_args(input_shapes, default_batch_size=2):
    """
    Prepares example arguments for tracing by replacing dynamic batch dimensions ('B') with a default value.
    """
    if any("B" in shape for shape in input_shapes):
        print("Dynamic batch dimensions detected.")

    def replace_B(s):
        return [default_batch_size if d == "B" else d for d in s]

    return [jnp.zeros(replace_B(s)) for s in input_shapes]


def to_onnx(
    fn: Any,
    input_shapes: Any,
    model_name: str = "jax_model",
    opset: int = 21,
) -> onnx.ModelProto:
    """
    Converts a JAX function into an ONNX model.
    """
    example_args = prepare_example_args(input_shapes)

    builder = OnnxBuilder(name_counter=0, opset=opset)
    converter = Jaxpr2OnnxConverter(builder)

    converter.trace_jaxpr(fn, example_args)

    builder.adjust_dynamic_batch_dimensions(input_shapes)
    builder.filter_unused_initializers()

    model = builder.create_onnx_model(model_name)

    return improve_onnx_model(model)


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
