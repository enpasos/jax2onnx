# file: jax2onnx/converter/conversion_api.py

"""
Conversion API Module

This module provides the core API functions for converting JAX functions and models to ONNX format.
It serves as the implementation layer for the public API exposed through user_interface.py.
"""

from typing import Any, Dict, Sequence, Tuple, Union

import onnx

import logging

from jax2onnx.converter.dynamic_utils import _create_symbolic_input_avals
from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter
from jax2onnx.converter.name_generator import UniqueNameGenerator
from jax2onnx.converter.onnx_builder import OnnxBuilder
from jax2onnx.converter.optimize_onnx_graph import improve_onnx_model
import jax.export as export
import jax.numpy as jnp
from jax import ShapeDtypeStruct

logger = logging.getLogger("jax2onnx.converter.conversion_api")


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

        var_to_symbol_map = {}
        for token, dim_obj in token_to_var.items():
            var_to_symbol_map[dim_obj] = token  # exact object
            var_to_symbol_map[id(dim_obj)] = token  # same object id
            var_to_symbol_map[str(dim_obj)] = token  # same repr / str

        return example_args, var_to_symbol_map
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
        return [jnp.zeros(s, dtype=jnp.float32) for s in processed_shapes], {}


def to_onnx(
    fn: Any,
    # Change signature: 'inputs' likely holds [(shape, dtype), ...] structure
    inputs: Sequence[
        Tuple[Sequence[Union[int, str]], Any]
    ],  # Or adjust based on actual input structure
    input_params: Dict[str, Any] | None = None,
    model_name: str = "jax_model",
    opset: int = 21,
    # ... other parameters ...
) -> onnx.ModelProto:
    """
    Converts a JAX function into an ONNX model. (Docstring potentially updated)
    """
    logger.info(f"Starting JAX to ONNX conversion for '{model_name}'")
    logger.debug(f"Received inputs spec: {inputs}")
    logger.debug(
        f"Received input_params: {input_params.keys() if input_params else 'None'}"
    )

    # --- Step 1: Prepare Abstract Inputs with Symbolic Dimensions ---
    # Assuming 'inputs' is the validated list like [(shape_tuple, dtype), ...]
    # If 'inputs' has a different structure, adjust the call accordingly.
    symbolic_avals, var_to_symbol_map = _create_symbolic_input_avals(inputs)

    # --- Setup Converter and Builder ---
    unique_name_generator = UniqueNameGenerator()
    # Pass the reverse map to the builder/converter for later use in ONNX mapping
    builder = OnnxBuilder(
        unique_name_generator,
        opset=opset,
        converter=None,
        var_to_symbol_name_map=var_to_symbol_map,  # Pass the map here
    )
    converter = Jaxpr2OnnxConverter(builder)
    builder.converter = converter

    converter.call_params = input_params or {}

    # --- Step 2: Trace the function using Symbolic Avals ---
    # Note: converter.trace_jaxpr needs to be modified next to *accept*
    # symbolic_avals directly instead of example_args / input_shapes.
    logger.info("Initiating JAX tracing with symbolic abstract values...")
    converter.trace_jaxpr(
        fn, symbolic_avals, params=input_params
    )  # Pass symbolic_avals
    logger.info("JAX tracing finished.")

    # --- Step 3: Build and Optimize ONNX model ---
    logger.info("Building ONNX model...")
    builder.filter_unused_initializers()
    model = builder.create_onnx_model(model_name)
    logger.info("Optimizing ONNX model...")
    model = improve_onnx_model(model)
    logger.info("ONNX model conversion complete.")

    return model


def analyze_constants(model: onnx.ModelProto):
    """
    Analyzes constants in an ONNX model and prints a detailed report.

    This function is useful for debugging and understanding how constants are
    represented and used within the ONNX graph.

    Args:
        model: The ONNX model to analyze.
    """
    logger.info("\n🔍 Constant Analysis Report (Verbose)")
    graph = model.graph
    graph_inputs = {inp.name for inp in graph.input}
    initializers = {init.name for init in graph.initializer}
    const_nodes = {
        node.output[0]: node for node in graph.node if node.op_type == "Constant"
    }
    function_names = {f.name for f in model.functions}
    logger.info("\n📦 Top-Level Inputs:")
    for inp in graph.input:
        logger.info(f"  - {inp.name}")
    logger.info("\n🧊 Initializers (Style 2):")
    for init in graph.initializer:
        logger.info(f"  - {init.name}")
    logger.info("\n🧱 Constant Nodes in Main Graph (Style 2):")
    for name in const_nodes:
        logger.info(f"  - {name}")
    logger.info("\n🧩 Function Call Inputs:")
    for node in graph.node:
        if node.op_type in function_names:
            logger.info(f"\n▶ Function Call: {node.op_type}")
            for inp in node.input:
                style = "Unknown/Intermediate"
                if inp in initializers:
                    style = "Style 2 (initializer reused)"
                elif inp in graph_inputs:
                    style = "Style 1 (passed in as input)"
                elif inp in const_nodes:
                    style = "Style 2 (constant node)"
                logger.info(f"  - {inp} → {style}")
    logger.info("\n🔗 Constant Usage Map:")
    for node in graph.node:
        for inp in node.input:
            if inp.startswith("const_") or inp.startswith("var_"):
                logger.info(f"  - {inp} used in {node.op_type}")
