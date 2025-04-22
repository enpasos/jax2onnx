"""
Conversion API Module

This module provides the core API functions for converting JAX functions and models to ONNX format.
It serves as the implementation layer for the public API exposed through user_interface.py.
"""

from typing import Any, Dict

import onnx

import logging

from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter
from jax2onnx.converter.name_generator import UniqueNameGenerator
from jax2onnx.converter.onnx_builder import OnnxBuilder
from jax2onnx.converter.optimize_onnx_graph import improve_onnx_model
import jax.export as export
import jax.numpy as jnp
from jax import ShapeDtypeStruct


def prepare_example_args(input_shapes, default_batch_size=2):
    """
    Prepares example arguments for tracing, supporting symbolic dimensions.

    Args:
        input_shapes: List of input shapes, where symbolic dims are any non-int (e.g., str).
        default_batch_size: Default value to use for symbolic dims if not using shape polymorphism.

    Returns:
        List of example arguments for tracing (ShapeDtypeStruct if symbolic, else zero arrays).
    """

    # 1. Collect all symbolic dimension tokens
    symbolic_dims = {
        d
        for s in input_shapes
        for d in (s if isinstance(s, (tuple, list)) else (s,))
        if not isinstance(d, int)
    }

    if symbolic_dims:
        if export.symbolic_shape is None:
            raise RuntimeError(
                "symbolic_shape not found in jax. Please upgrade your JAX version."
            )
        # 2. Create symbolic variables for each token
        dim_vars = export.symbolic_shape(",".join(sorted(symbolic_dims)))
        token_to_var = dict(zip(sorted(symbolic_dims), dim_vars))
        # 3. Build ShapeDtypeStruct for each input
        example_args = []
        for spec in input_shapes:
            shape = tuple(
                token_to_var.get(d, d)
                for d in (spec if isinstance(spec, (tuple, list)) else (spec,))
            )
            example_args.append(ShapeDtypeStruct(shape, jnp.float32))
        return example_args
    else:
        # Fallback: all static dims, use zero arrays as before
        processed_shapes = []
        for shape in input_shapes:
            current_shape = shape if isinstance(shape, (list, tuple)) else (shape,)
            new_shape = []
            for dim in current_shape:
                if not isinstance(dim, int):
                    new_shape.append(default_batch_size)
                else:
                    new_shape.append(dim)
            processed_shapes.append(tuple(new_shape))
        return [jnp.zeros(s, dtype=jnp.float32) for s in processed_shapes]


def to_onnx(
    fn: Any,
    input_shapes: Any,
    input_params: Dict[str, Any] | None = None,
    model_name: str = "jax_model",
    opset: int = 21,
) -> onnx.ModelProto:
    """
    Converts a JAX function into an ONNX model.

    This is the core implementation of the conversion process. It traces the JAX function,
    captures its computational graph as JAXPR, and then converts it to the ONNX format.

    Args:
        fn: JAX function to convert.
        input_shapes: Shapes of the inputs to the function.
        input_params: Additional parameters for inference (optional). These will be
                     converted to ONNX model inputs rather than baked into the model.
        model_name: Name of the ONNX model.
        opset: ONNX opset version to use.

    Returns:
        An ONNX ModelProto object representing the converted model.
    """
    # Generate concrete example arguments based on provided shapes
    example_args = prepare_example_args(input_shapes)

    unique_name_generator = UniqueNameGenerator()
    builder = OnnxBuilder(unique_name_generator, opset=opset)
    converter = Jaxpr2OnnxConverter(builder)

    # Store the parameters that should be exposed as inputs in the ONNX model
    converter.call_params = input_params or {}

    # Trace the function to capture its structure
    # Pass the input_params explicitly to the trace_jaxpr function
    # This ensures parameters affect the JAX call graph during tracing
    converter.trace_jaxpr(fn, example_args, params=input_params)

    # Continue with the normal conversion process
    builder.adjust_dynamic_batch_dimensions(input_shapes)
    builder.filter_unused_initializers()

    model = builder.create_onnx_model(model_name)
    model = improve_onnx_model(model)

    return model


def analyze_constants(model: onnx.ModelProto):
    """
    Analyzes constants in an ONNX model and prints a detailed report.

    This function is useful for debugging and understanding how constants are
    represented and used within the ONNX graph.

    Args:
        model: The ONNX model to analyze.
    """
    logging.info("\n🔍 Constant Analysis Report (Verbose)")
    graph = model.graph
    graph_inputs = {inp.name for inp in graph.input}
    initializers = {init.name for init in graph.initializer}
    const_nodes = {
        node.output[0]: node for node in graph.node if node.op_type == "Constant"
    }
    function_names = {f.name for f in model.functions}
    logging.info("\n📦 Top-Level Inputs:")
    for inp in graph.input:
        logging.info(f"  - {inp.name}")
    logging.info("\n🧊 Initializers (Style 2):")
    for init in graph.initializer:
        logging.info(f"  - {init.name}")
    logging.info("\n🧱 Constant Nodes in Main Graph (Style 2):")
    for name in const_nodes:
        logging.info(f"  - {name}")
    logging.info("\n🧩 Function Call Inputs:")
    for node in graph.node:
        if node.op_type in function_names:
            logging.info(f"\n▶ Function Call: {node.op_type}")
            for inp in node.input:
                style = "Unknown/Intermediate"
                if inp in initializers:
                    style = "Style 2 (initializer reused)"
                elif inp in graph_inputs:
                    style = "Style 1 (passed in as input)"
                elif inp in const_nodes:
                    style = "Style 2 (constant node)"
                logging.info(f"  - {inp} → {style}")
    logging.info("\n🔗 Constant Usage Map:")
    for node in graph.node:
        for inp in node.input:
            if inp.startswith("const_") or inp.startswith("var_"):
                logging.info(f"  - {inp} used in {node.op_type}")
