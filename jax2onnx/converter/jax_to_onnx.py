import onnx
import jax.numpy as jnp
from typing import Any
from jax2onnx.converter.name_generator import GlobalNameCounter
from jax2onnx.converter.onnx_builder import OnnxBuilder
from jax2onnx.converter.converter import Jaxpr2OnnxConverter
from jax2onnx.converter.optimize_onnx_graph import improve_onnx_model


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

    counter = GlobalNameCounter()
    builder = OnnxBuilder(counter, opset=opset)
    # builder = OnnxBuilder(opset=opset)
    converter = Jaxpr2OnnxConverter(builder)

    converter.trace_jaxpr(fn, example_args)

    builder.adjust_dynamic_batch_dimensions(input_shapes)
    builder.filter_unused_initializers()

    model = builder.create_onnx_model(model_name)

    model = improve_onnx_model(model)

    onnx.checker.check_model(model)

    return model
