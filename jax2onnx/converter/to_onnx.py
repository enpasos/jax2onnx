# file: jax2onnx/converter/simple_handler_conversion.py
import onnx
from typing import Any
import jax.numpy as jnp
from onnx import helper
from jax2onnx.converter.converter import Jaxpr2OnnxConverter
from jax2onnx.converter.onnx_builder import OnnxBuilder
from jax2onnx.converter.optimize_onnx_graph import improve_onnx_model
from jax2onnx.plugin_system import (
    ONNX_FUNCTION_PLUGIN_REGISTRY,
)


def prepare_example_args(input_shapes, default_batch_size=2):
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
    from jax2onnx.plugin_system import PLUGIN_REGISTRY, PrimitiveLeafPlugin

    example_args = prepare_example_args(input_shapes)

    builder = OnnxBuilder(name_counter=0, opset=opset)
    converter = Jaxpr2OnnxConverter(builder)

    converter.trace_jaxpr(fn, example_args)

    builder.adjust_dynamic_batch_dimensions(input_shapes)

    # Filter unused initializers using the new method
    builder.filter_unused_initializers()

    # Create the ONNX model using the new method
    model = builder.create_onnx_model(model_name)

    return improve_onnx_model(model)
