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


def prepare_example_args(input_shapes):
    if any("B" in shape for shape in input_shapes):
        print(
            "Dynamic batch dimensions detected. Setting include_intermediate_shapes=False"
        )

    def replace_B(s):
        return [2 if d == "B" else d for d in s]

    return [jnp.zeros(replace_B(s)) for s in input_shapes]


def to_onnx(
    fn: Any,
    input_shapes: Any,
    model_name: str = "jax_model",
    include_intermediate_shapes: bool = False,
    opset: int = 21,
) -> onnx.ModelProto:
    from jax2onnx.plugin_system import PLUGIN_REGISTRY, PrimitiveLeafPlugin

    # concrete example inputs

    if any("B" in shape for shape in input_shapes):
        include_intermediate_shapes = False

    example_args = prepare_example_args(input_shapes)

    builder = OnnxBuilder(name_counter=0, opset=opset)
    converter = Jaxpr2OnnxConverter(builder)

    converter.trace_jaxpr(fn, example_args)

    # Adjust dynamic batch dimensions using the new method
    builder.adjust_dynamic_batch_dimensions(input_shapes)

    value_info = converter.builder.value_info if include_intermediate_shapes else []
    used_inputs = {i for node in converter.builder.nodes for i in node.input}
    converter.builder.initializers = [
        init for init in converter.builder.initializers if init.name in used_inputs
    ]

    graph = helper.make_graph(
        nodes=converter.builder.nodes,
        name=model_name,
        inputs=converter.builder.inputs,
        outputs=converter.builder.outputs,
        initializer=converter.builder.initializers,
        value_info=value_info,
    )

    # explicitly add functions (FunctionProto) to model
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", opset),
            helper.make_opsetid("custom", 1),  # <-- ✅ explicitly import custom domain
        ],
        functions=list(converter.builder.functions.values()),
    )

    return improve_onnx_model(model)
