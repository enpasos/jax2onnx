# file: jax2onnx/converter/conversion_api.py

# Add Tuple, Union to imports if not already present
from typing import (
    Any,
    Dict,
    Optional,
    Sequence,
    Union,
)  # Added Tuple, List
import onnx
import logging
import numpy as np
from onnx import helper, mapping
from jax2onnx.converter.dynamic_utils import (
    _create_symbolic_input_avals,
)  # Import the helper
from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter
from jax2onnx.converter.name_generator import UniqueNameGenerator
from jax2onnx.converter.onnx_builder import OnnxBuilder
from jax2onnx.converter.optimize_onnx_graph import improve_onnx_model
from jax import config as jax_config  # NEW
import jax.numpy as jnp

logger = logging.getLogger("jax2onnx.converter.conversion_api")


# Remove or comment out the old prepare_example_args if no longer needed
# def prepare_example_args(...): ...


# ------------------------------------------------------------------
# Promote items passed via *input_params* to proper graph inputs
# ------------------------------------------------------------------
def _elem_type_from_numpy(arr: np.ndarray) -> int:
    return mapping.NP_TYPE_TO_TENSOR_TYPE[arr.dtype]


def _promote_params_to_inputs(model: onnx.ModelProto, params: Dict[str, Any] | None):
    if not params:
        return

    for name, value in params.items():
        # ① drop initializer (if any)
        kept = [init for init in model.graph.initializer if init.name != name]
        model.graph.ClearField("initializer")
        model.graph.initializer.extend(kept)

        # ② drop stale value_info (INT32 in our case)
        kept = [vi for vi in model.graph.value_info if vi.name != name]
        model.graph.ClearField("value_info")
        model.graph.value_info.extend(kept)

        # ③ add graph input once
        if any(inp.name == name for inp in model.graph.input):
            continue
        dtype = _elem_type_from_numpy(np.asarray(value))
        vi = helper.make_tensor_value_info(name, dtype, [])  # scalar
        model.graph.input.append(vi)


# -----------------------------------------------------------------------------
# drop duplicate initializers for parameters promoted to real graph inputs
# -----------------------------------------------------------------------------
def _strip_param_initializers(model, input_params):
    if not input_params:
        return
    param_names = set(input_params)
    keep = [init for init in model.graph.initializer if init.name not in param_names]
    del model.graph.initializer[:]  # in‑place update
    model.graph.initializer.extend(keep)


def to_onnx(
    fn: Any,
    # Assume 'inputs' is passed as a list/sequence of shape tuples
    inputs: Sequence[Sequence[Union[int, str]]],
    input_params: Dict[str, Any] | None = None,
    model_name: str = "jax_model",
    opset: int = 21,
    *,
    enable_double_precision: bool = False,
    default_dtype: Any | None = None,
    record_primitive_calls_file: Optional[str] = None,
    # ... other parameters ...
) -> onnx.ModelProto:
    """
    Converts a JAX function into an ONNX model.
    Handles symbolic dimensions specified as strings in input shapes.

    Parameters:
    -----------
    fn : Any
        The JAX function to convert to ONNX.
    inputs : Sequence[Sequence[Union[int, str]]]
        Shapes of the input tensors. String values represent symbolic dimensions.
    input_params : Dict[str, Any] | None, optional
        Additional parameters to be passed to the function, by default None
    model_name : str, optional
        Name of the ONNX model, by default "jax_model"
    opset : int, optional
        ONNX opset version to target, by default 21
    enable_double_precision : bool, optional
        If **True**, the converter keeps every tensor in double
        precision (`tensor(double)`).  If **False** (default) the
        graph is down-cast to single precision.
    default_dtype : Any | None, optional
        Default data type for inputs if not specified, by default None

    Returns:
    --------
    onnx.ModelProto
        The converted ONNX model
    """
    logger.info(f"Starting JAX to ONNX conversion for '{model_name}'")
    logger.debug(f"Received raw inputs (shapes): {inputs}")
    logger.debug(
        f"Received input_params: {input_params.keys() if input_params else 'None'}"
    )

    # ------------------------------------------------------------------
    # 1) Decide the working dtype, 2) flip JAX's global x64 switch
    #    before we trace the function.  Needs to happen **before**
    #    any array creation inside this call.
    # ------------------------------------------------------------------
    if enable_double_precision:
        jax_config.update("jax_enable_x64", True)
        # If enable_double_precision is True, working_dtype MUST be float64,
        # unless default_dtype is explicitly something else (which might be an edge case to clarify or restrict)
        working_dtype = (
            jnp.float64
        )  # Prioritize float64 if enable_double_precision is true
        if default_dtype is not None and default_dtype != jnp.float64:
            logger.warning(
                f"enable_double_precision is True, but default_dtype is {default_dtype}. Using jnp.float64."
            )
    else:
        jax_config.update("jax_enable_x64", False)
        working_dtype = jnp.float32 if default_dtype is None else default_dtype

    logger.debug(
        f"🔧 enable_double_precision = {enable_double_precision} → working dtype = {working_dtype}"
    )

    # --- Step 0: Format input_specs ---
    # build symbolic input avals — accept shapes, Array-like, or ShapeDtypeStruct
    from jax import ShapeDtypeStruct

    normalized_specs = []
    for spec in inputs:
        if isinstance(spec, ShapeDtypeStruct):
            # already has shape & dtype
            normalized_specs.append((spec.shape, spec.dtype))
        elif hasattr(spec, "shape") and hasattr(spec, "dtype"):
            # real JAX/NumPy array
            normalized_specs.append((tuple(spec.shape), spec.dtype))
        elif isinstance(spec, (tuple, list)):
            # plain shape tuple/list → assume working_dtype
            normalized_specs.append((tuple(spec), working_dtype))
        else:
            raise TypeError(
                f"Unsupported inputs element: {type(spec)}. "
                "Must be shape tuple, Array, or ShapeDtypeStruct."
            )

    logger.debug(f"Normalized input_specs: {normalized_specs}")

    # --- Step 1: Prepare Abstract Inputs with Symbolic Dimensions ---
    # (Assumes this part is now correct)
    symbolic_avals, var_to_symbol_map = _create_symbolic_input_avals(normalized_specs)

    # --- Setup Converter and Builder ---
    unique_name_generator = UniqueNameGenerator()

    # Initialize OnnxBuilder with the enable_double_precision flag
    builder = OnnxBuilder(
        unique_name_generator,
        opset=opset,
        converter=None,  # Will be set later
        enable_double_precision=enable_double_precision,  # Pass the flag
    )

    # Set the map as an attribute *after* initialization
    builder.var_to_symbol_map = var_to_symbol_map
    logger.debug(f"Set builder.var_to_symbol_map: {builder.var_to_symbol_map}")

    # Initialize Converter and link back (no change here)
    converter = Jaxpr2OnnxConverter(
        builder,
        record_primitive_calls_file=record_primitive_calls_file,
        function_context_for_recording=getattr(
            fn, "__name__", model_name
        ),  # carry top-level fn name or use model_name as fallback
    )
    builder.converter = converter

    converter.call_params = input_params or {}

    # --- Step 2: Trace the function using Symbolic Avals ---
    # Reminder: converter.trace_jaxpr needs modification next
    logger.info("Initiating JAX tracing with symbolic abstract values...")
    # *** NEXT STEP: Modify converter.trace_jaxpr to accept symbolic_avals ***
    converter.trace_jaxpr(fn, symbolic_avals, params=input_params)
    logger.info("JAX tracing finished.")

    # --- Step 3: Build and Optimize ONNX model ---
    logger.info("Building ONNX model...")
    builder.filter_unused_initializers()
    model = builder.create_onnx_model(model_name)

    # Replace with the new function for properly handling parameter inputs
    _promote_params_to_inputs(
        model, input_params
    )  # ← new instead of _strip_param_initializers

    logger.info("Optimizing ONNX model...")
    model = improve_onnx_model(model)

    # Ensure compatibility with the ONNX Runtime version being used for testing.
    # If the 'onnx' library (used for model creation) defaults to an IR version
    # higher than what the testing 'onnxruntime' supports (e.g., IRv11 vs max IRv10),
    # explicitly set the model's IR version to a compatible one.
    # For onnxruntime 1.18.0, the max supported IR version is 10.
    # Opset 21 (often used by jax2onnx) should correspond to IR version 10
    # according to ONNX specifications (see onnx.helper.VERSION_TABLE).
    # However, the `onnx` library might default to a newer IR version based on its own release.

    # Target IR version for compatibility with onnxruntime that supports up to IR version 10
    target_ir_version = 10
    if model.ir_version > target_ir_version:
        logger.info(
            f"Current model IR version is {model.ir_version}. "
            f"Setting IR version to {target_ir_version} for compatibility "
            f"with an ONNX Runtime that supports up to IR version {target_ir_version}."
        )
        model.ir_version = target_ir_version

    logger.info("ONNX model conversion complete.")
    logger.debug(onnx.helper.printable_graph(model.graph))

    # if primitive-call recording was enabled, flush the log to disk
    if record_primitive_calls_file and hasattr(converter, "recorded_calls_log"):
        from jax2onnx.utils.debug import save_primitive_calls_log

        getattr(fn, "__name__", model_name)
        # honor exactly the path the user passed in
        save_primitive_calls_log(
            converter.recorded_calls_log,
            record_primitive_calls_file,
        )

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
