# file: jax2onnx/converter/jax_to_onnx.py
from typing import Any  # <-- 🔧 FIXED: now allowed

import jax.numpy as jnp
import onnx

from jax2onnx.converter.converter import Jaxpr2OnnxConverter

# === Use original name generator import ===
from jax2onnx.converter.name_generator import UniqueNameGenerator

# ==========================================
from jax2onnx.converter.onnx_builder import OnnxBuilder
from jax2onnx.converter.optimize_onnx_graph import improve_onnx_model


def prepare_example_args(input_shapes, default_batch_size=2):
    """
    Prepares example arguments for tracing by replacing dynamic batch dimensions ('B') with a default value.

    Args:
        input_shapes: List of input shapes, where 'B' represents a dynamic batch dimension.
        default_batch_size: Default value to replace 'B' with.

    Returns:
        List of NumPy arrays with the specified shapes, filled with zeros.
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
    input_params: dict | None = None,
) -> onnx.ModelProto:
    """
    Converts a JAX function into an ONNX model.

    Args:
        fn: JAX function to convert.
        input_shapes: Shapes of the inputs to the function.
        model_name: Name of the ONNX model.
        opset: ONNX opset version to use.
        input_params: Additional parameters for inference (optional). These will be
                     converted to ONNX model inputs rather than baked into the model.

    Returns:
        An ONNX ModelProto object representing the converted model.
    """
    from jax2onnx.converter.jax_to_onnx import prepare_example_args

    # Generate concrete example arguments based on provided shapes
    example_args = prepare_example_args(input_shapes)

    unique_name_generator = UniqueNameGenerator()
    builder = OnnxBuilder(unique_name_generator, opset=opset)
    converter = Jaxpr2OnnxConverter(builder)

    # Store the parameters that should be exposed as inputs in the ONNX model
    converter.call_params = input_params or {}

    # Trace the function to capture its structure
    converter.trace_jaxpr(fn, example_args)

    # Now add additional ONNX inputs for each parameter in input_params
    if input_params:
        for param_name, param_value in input_params.items():
            # Create appropriate scalar inputs for each parameter type
            if isinstance(param_value, bool):
                # Add a boolean input parameter to the ONNX model
                builder.add_scalar_input(param_name, dtype=onnx.TensorProto.BOOL)
                print(f"Added parameter '{param_name}' as BOOL input to the ONNX model")
            elif isinstance(param_value, int):
                builder.add_scalar_input(param_name, dtype=onnx.TensorProto.INT64)
                print(
                    f"Added parameter '{param_name}' as INT64 input to the ONNX model"
                )
            elif isinstance(param_value, float):
                builder.add_scalar_input(param_name, dtype=onnx.TensorProto.FLOAT)
                print(
                    f"Added parameter '{param_name}' as FLOAT input to the ONNX model"
                )
            else:
                print(
                    f"Warning: Parameter {param_name} of type {type(param_value)} not supported as ONNX input"
                )

    # Continue with the normal conversion process
    builder.adjust_dynamic_batch_dimensions(input_shapes)
    builder.filter_unused_initializers()

    model = builder.create_onnx_model(model_name)
    model = improve_onnx_model(model)

    return model


def analyze_constants(model: onnx.ModelProto):
    """
    Analyzes constants in an ONNX model and prints a detailed report.

    Args:
        model: The ONNX model to analyze.
    """
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
