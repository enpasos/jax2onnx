# file: jax2onnx/converter/jax_to_onnx.py
import onnx
import jax.numpy as jnp
from typing import Any, Optional  # <-- 🔧 FIXED: now allowed

# === Use original name generator import ===
from jax2onnx.converter.name_generator import UniqueNameGenerator

# ==========================================
from jax2onnx.converter.onnx_builder import OnnxBuilder
from jax2onnx.converter.converter import Jaxpr2OnnxConverter
from jax2onnx.converter.optimize_onnx_graph import improve_onnx_model


def prepare_example_args(input_shapes, default_batch_size=2):
    """
    Prepares example arguments for tracing by replacing dynamic batch dimensions ('B') with a default value.
    """
    dynamic_dim_found = False
    processed_shapes = []
    for shape in input_shapes:
        # Ensure shape is iterable
        current_shape = shape if isinstance(shape, (list, tuple)) else (shape,)
        new_shape = []
        for dim in current_shape:
            if dim == "B":
                new_shape.append(default_batch_size)
                dynamic_dim_found = True
            else:
                new_shape.append(dim)
        processed_shapes.append(tuple(new_shape))

    if dynamic_dim_found:
        print("Dynamic batch dimensions detected.")

    return [jnp.zeros(s, dtype=jnp.float32) for s in processed_shapes]  # Assume float32


def to_onnx(
    fn: Any,
    input_shapes: Any,
    model_name: str = "jax_model",
    opset: int = 21,
    input_params: Optional[dict] = None,  # <-- 🔧 FIXED: now allowed
) -> onnx.ModelProto:
    """
    Converts a JAX function into an ONNX model.
    """
    from jax2onnx.converter.jax_to_onnx import prepare_example_args

    example_args = prepare_example_args(input_shapes)

    unique_name_generator = UniqueNameGenerator()
    builder = OnnxBuilder(unique_name_generator, opset=opset)
    converter = Jaxpr2OnnxConverter(builder)

    converter.trace_jaxpr(fn, example_args)
    builder.adjust_dynamic_batch_dimensions(input_shapes)
    builder.filter_unused_initializers()

    model = builder.create_onnx_model(model_name)
    model = improve_onnx_model(model)

    return model


def analyze_constants(model: onnx.ModelProto):
    # Original debugging function
    print("\n🔍 Constant Analysis Report (Verbose)")
    graph = model.graph
    graph_inputs = {inp.name for inp in graph.input}
    initializers = {init.name for init in graph.initializer}
    const_nodes = {
        node.output[0]: node for node in graph.node if node.op_type == "Constant"
    }
    function_names = {f.name for f in model.functions}
    print("\n📦 Top-Level Inputs:")
    [print(f"  - {inp.name}") for inp in graph.input]
    print("\n🧊 Initializers (Style 2):")
    [print(f"  - {init.name}") for init in graph.initializer]
    print("\n🧱 Constant Nodes in Main Graph (Style 2):")
    [print(f"  - {name}") for name in const_nodes]
    print("\n🧩 Function Call Inputs:")
    for node in graph.node:
        if node.op_type in function_names:
            print(f"\n▶ Function Call: {node.op_type}")
            for inp in node.input:
                style = "Unknown/Intermediate"
                if inp in initializers:
                    style = "Style 2 (initializer reused)"
                elif inp in graph_inputs:
                    style = "Style 1 (passed in as input)"
                elif inp in const_nodes:
                    style = "Style 2 (constant node)"
                print(f"  - {inp} → {style}")
    print("\n🔗 Constant Usage Map:")
    for node in graph.node:
        for inp in node.input:
            if inp.startswith("const_") or inp.startswith("var_"):
                print(f"  - {inp} used in {node.op_type}")
