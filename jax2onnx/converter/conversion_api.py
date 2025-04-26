# In file: jax2onnx/converter/conversion_api.py

# Keep existing imports
from typing import Any, Dict, Sequence, Tuple, Union, List  # Add List
import onnx
import logging
from jax2onnx.converter.dynamic_utils import (
    _create_symbolic_input_avals,
)  # Import the helper
from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter
from jax2onnx.converter.name_generator import UniqueNameGenerator
from jax2onnx.converter.onnx_builder import OnnxBuilder
from jax2onnx.converter.optimize_onnx_graph import improve_onnx_model
import jax
import jax.export as export  # Keep this import
import jax.numpy as jnp
from jax import ShapeDtypeStruct, core  # Keep core

logger = logging.getLogger("jax2onnx.converter.conversion_api")


# Remove or comment out the old prepare_example_args if no longer needed
# def prepare_example_args(...): ...


def to_onnx(
    fn: Any,
    # Assume 'inputs' is passed as a list/sequence of shape tuples
    inputs: Sequence[Sequence[Union[int, str]]],
    input_params: Dict[str, Any] | None = None,
    model_name: str = "jax_model",
    opset: int = 21,
    default_dtype: Any = jnp.float32,  # Default dtype if not specified otherwise
    # ... other parameters ...
) -> onnx.ModelProto:
    """
    Converts a JAX function into an ONNX model.
    Handles symbolic dimensions specified as strings in input shapes.
    """
    logger.info(f"Starting JAX to ONNX conversion for '{model_name}'")
    logger.debug(f"Received raw inputs (shapes): {inputs}")
    logger.debug(
        f"Received input_params: {input_params.keys() if input_params else 'None'}"
    )
    logger.debug(f"Using default dtype: {default_dtype}")

    # --- Step 0: Format input_specs ---
    # Create the list of (shape, dtype) tuples needed by the helper
    # This assumes 'inputs' is a list of shapes and uses default_dtype.
    # Future enhancement: Allow user to pass [(shape, dtype), ...] directly.
    try:
        input_specs: List[Tuple[Sequence[Union[int, str]], Any]] = []
        for shape_spec in inputs:
            # Ensure shape_spec is a tuple/list before processing
            if not isinstance(shape_spec, (tuple, list)):
                # Handle scalar shapes like (B,) potentially passed as just "B"
                shape_spec = (shape_spec,)
            # Pair the processed shape tuple with the default dtype
            input_specs.append((tuple(shape_spec), default_dtype))
    except Exception as e:
        logger.error(
            f"Failed to format input shapes/dtypes. Input: {inputs}. Error: {e}",
            exc_info=True,
        )
        raise TypeError(
            "Input shapes must be sequences (tuples/lists) of int or str."
        ) from e

    logger.debug(f"Formatted input_specs: {input_specs}")

    # --- Step 1: Prepare Abstract Inputs with Symbolic Dimensions ---
    # Call the helper function (defined in dynamic_utils.py)
    symbolic_avals, var_to_symbol_map = _create_symbolic_input_avals(input_specs)

    # --- Setup Converter and Builder ---
    unique_name_generator = UniqueNameGenerator()
    # Pass the reverse map (symbol obj -> name str) to the builder
    builder = OnnxBuilder(
        unique_name_generator,
        opset=opset,
        converter=None,
        var_to_symbol_name_map=var_to_symbol_map,  # Pass map for later ONNX mapping
    )
    converter = Jaxpr2OnnxConverter(builder)
    builder.converter = converter

    converter.call_params = input_params or {}

    # --- Step 2: Trace the function using Symbolic Avals ---
    # Reminder: converter.trace_jaxpr needs modification next to accept symbolic_avals
    logger.info("Initiating JAX tracing with symbolic abstract values...")
    # *** NEXT STEP: Modify converter.trace_jaxpr to accept symbolic_avals ***
    converter.trace_jaxpr(fn, symbolic_avals, params=input_params)
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
