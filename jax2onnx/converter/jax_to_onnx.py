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

    analyze_constants(model)

    return model


def analyze_constants(model: onnx.ModelProto):
    print("\n🔍 Constant Analysis Report (Verbose)")

    graph = model.graph
    graph_inputs = {inp.name for inp in graph.input}
    initializers = {init.name for init in graph.initializer}
    const_nodes = {
        node.output[0]: node for node in graph.node if node.op_type == "Constant"
    }
    function_names = {f.name for f in model.functions}

    print("\n📦 Top-Level Inputs:")
    for inp in graph.input:
        print(f"  - {inp.name}")

    print("\n🧊 Initializers (Style 2):")
    for init in graph.initializer:
        print(f"  - {init.name}")

    print("\n🧱 Constant Nodes in Main Graph (Style 2):")
    for name in const_nodes:
        print(f"  - {name}")

    print("\n🧩 Function Call Inputs:")
    for node in graph.node:
        if node.op_type in function_names:
            print(f"\n▶ Function Call: {node.op_type}")
            for inp in node.input:
                if inp in initializers:
                    style = "Style 2 (initializer reused)"
                elif inp in graph_inputs:
                    style = "Style 1 (passed in as input)"
                elif inp in const_nodes:
                    style = "Style 2 (constant node)"
                else:
                    style = "Unknown/Intermediate"
                print(f"  - {inp} → {style}")

    print("\n🔗 Constant Usage Map:")
    for node in graph.node:
        for inp in node.input:
            if inp.startswith("const_") or inp.startswith("var_"):
                print(f"  - {inp} used in {node.op_type}")
