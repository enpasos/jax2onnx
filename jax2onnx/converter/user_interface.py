# file: jax2onnx/converter/user_interface.py

from collections import defaultdict
from typing import Any, Callable

import jax.numpy as jnp
import jax
import numpy as np
import onnx
import onnxruntime as ort

from jax2onnx.converter.jax_to_onnx import to_onnx as core_to_onnx


def to_onnx(
    fn: Any,
    input_shapes: Any,
    model_name: str = "jax_model",
    opset: int = 21,
    *,
    kwargs: dict[str, Any] | None = None,
) -> onnx.ModelProto:

    # Extract input_params from kwargs if present
    input_params = None
    if kwargs is not None and "input_params" in kwargs:
        input_params = kwargs.get("input_params")

    return core_to_onnx(
        fn=fn,
        input_shapes=input_shapes,
        model_name=model_name,
        opset=opset,
        input_params=input_params,
    )


def save_onnx(
    fn: Any,
    input_shapes: Any,
    output_path: str = "model.onnx",
    model_name: str = "jax_model",
    opset: int = 21,
):
    onnx_model = to_onnx(fn, input_shapes, model_name=model_name, opset=opset)
    onnx.save_model(onnx_model, output_path)


def allclose(callable, onnx_model_path, *xs, jax_kwargs=None, rtol=1e-3, atol=1e-5):
    """
    Checks if JAX and ONNX Runtime outputs are close.

    Args:
        callable: JAX function to test
        onnx_model_path: Path to the ONNX model
        *xs: Tensor arguments to pass to both JAX and ONNX
        jax_kwargs: Optional keyword arguments to pass to the JAX function
        rtol: Relative tolerance for comparison (default: 1e-3)
        atol: Absolute tolerance for comparison (default: 1e-5)

    Returns:
        Tuple of (is_match: bool, message: str)
    """
    if jax_kwargs is None:
        jax_kwargs = {}

    # Load ONNX model and create inference session
    session = ort.InferenceSession(onnx_model_path)

    # Extract actual input names from model
    input_names = [inp.name for inp in session.get_inputs()]

    # Get input shapes to help identify scalar parameters
    input_shapes = [tuple(inp.shape) for inp in session.get_inputs()]

    # Scalars will have shape () - these are likely to be parameters
    scalar_inputs = [
        name for name, shape in zip(input_names, input_shapes) if shape == ()
    ]

    # If we have more inputs than tensor arguments, assume the extras are parameters
    has_param_inputs = len(input_names) > len(xs)

    # Determine how many inputs are tensors (should match xs)
    tensor_input_count = len(xs)

    # Get the tensor input names (the first n names)
    tensor_input_names = input_names[:tensor_input_count]

    # The rest are parameter inputs
    param_input_names = input_names[tensor_input_count:]

    # Print debug info to understand what's happening
    print(f"ONNX model inputs: {input_names}")
    print(f"Tensor inputs: {tensor_input_names}")
    print(f"Parameter inputs: {param_input_names}")
    print(f"JAX kwargs: {jax_kwargs}")

    # Prepare ONNX input dictionary for tensor inputs
    p = {name: np.array(x) for name, x in zip(tensor_input_names, xs, strict=False)}

    # Get more detailed type information from ONNX model inputs
    input_details = [(inp.name, inp.type, inp.shape) for inp in session.get_inputs()]
    print(f"Detailed input info: {input_details}")

    # Handle deterministic parameter and other parameters more carefully
    for param_name in param_input_names:
        if param_name == "deterministic":
            # Special handling for deterministic parameter
            det_value = jax_kwargs.get(
                "deterministic", True
            )  # Default to True if not specified

            # Use bool_ type for deterministic parameter - ONNX expects boolean type
            p[param_name] = np.array(det_value, dtype=np.bool_)

            # Also pass it in jax_kwargs so the JAX model receives the same parameter
            jax_kwargs[param_name] = det_value
            print(f"Added deterministic={det_value} as bool_ value and to jax_kwargs")

        elif param_name in jax_kwargs:
            # General handling for other parameters
            param_value = jax_kwargs[param_name]

            # Convert the parameter to the appropriate numpy array based on its type
            if isinstance(param_value, bool):
                try:
                    p[param_name] = np.array(
                        int(param_value), dtype=np.int64
                    )  # Use int64 for booleans
                except (TypeError, ValueError):
                    p[param_name] = np.array(
                        param_value, dtype=np.bool_
                    )  # Fallback to bool_
            elif isinstance(param_value, int):
                p[param_name] = np.array(param_value, dtype=np.int64)
            elif isinstance(param_value, float):
                p[param_name] = np.array(param_value, dtype=np.float32)
            else:
                print(
                    f"Warning: Parameter {param_name} has unsupported type {type(param_value)}"
                )
        else:
            # Parameter not found in jax_kwargs, provide a reasonable default
            print(
                f"Warning: Parameter {param_name} not provided in jax_kwargs, using default value"
            )
            # For boolean parameters, default to True
            if param_name == "deterministic":
                p[param_name] = np.array(True, dtype=np.bool_)
            # For other parameters, we could add more default handling if needed

    # Run ONNX model with both tensor and parameter inputs
    onnx_output = session.run(None, p)

    # Call JAX function directly with tensor args and keyword args
    jax_output = callable(*xs, **jax_kwargs)

    if not isinstance(jax_output, list):
        jax_output = [jax_output]
    if not isinstance(onnx_output, list):
        onnx_output = [onnx_output]

    isOk = all(
        np.allclose(o, j, rtol=rtol, atol=atol)
        for o, j in zip(onnx_output, jax_output, strict=False)
    )

    return (
        isOk,
        (
            "ONNX and JAX outputs match :-)"
            if isOk
            else "ONNX and JAX outputs do not match :-("
        ),
    )


class ModelExportContext:
    """
    Holds model-specific state for naming and caching.
    """

    def __init__(self, model_id: str | None = None):
        self.model_id: str = model_id or "default_model"
        self.function_cache: dict[str, Any] = {}
        self.instance_counters: dict[str, int] = defaultdict(int)

    def next_function_name(self, base_name: str) -> str:
        """
        Generates a unique ONNX function name scoped to this model.
        E.g., TransformerBlock_1, TransformerBlock_2, ...
        """
        self.instance_counters[base_name] += 1
        return f"{base_name}_{self.instance_counters[base_name]}"


# # === NEW V2 Function: to_onnx_v2 ===
# def to_onnx_v2(
#     fn: Callable,  # Can be original or lambda wrapper
#     example_args: (
#         list[jax.Array] | tuple[jax.Array, ...]
#     ),  # Accepts concrete tensor args
#     model_name: str = "jax_model",
#     opset: int = 21,
#     verbose: bool = False,  # Added verbose based on previous exploration
#     **kwargs,  # Pass other kwargs to core_to_onnx
# ) -> onnx.ModelProto:
#     """
#     Converts a JAX callable (potentially wrapped) into an ONNX model. (V2 - Corrected Approach)

#     Args:
#         fn: The JAX function (or lambda wrapper) to convert.
#         example_args: Concrete JAX arrays for tracing (only tensor inputs).
#         model_name: Name for the ONNX model.
#         opset: ONNX opset version.
#         verbose: Enable verbose logging.
#         **kwargs: Additional options passed to the core converter.
#     """
#     print(f"\n[V2] Starting ONNX conversion for '{model_name}' (opset {opset})...")
#     # Calls core_to_onnx correctly with positional fn and example_args
#     # Assumes core_to_onnx expects fn, example_args positionally,
#     # then keyword args like model_name, default_opset (or opset).
#     # Adjust keyword names like 'default_opset' vs 'opset' if needed based on core_to_onnx definition.
#     try:
#         # Try passing opset as default_opset first, as seen in converter logic
#         result = core_to_onnx(
#             fn,  # Positional arg 1 (potentially wrapped fn)
#             example_args,  # Positional arg 2 (concrete tensor args)
#             # Keyword arguments for the rest
#             model_name=model_name,
#             default_opset=opset,  # Assuming core expects default_opset
#             verbose=verbose,
#             **kwargs,  # Pass through other kwargs
#         )
#         print(f"[V2] Conversion successful for '{model_name}'.")
#         return result
#     except TypeError as e:
#         # Fallback if default_opset caused TypeError, try 'opset'
#         if "default_opset" in str(e):
#             print(
#                 "[V2 Warning] 'default_opset' unexpected, trying 'opset' keyword for core_to_onnx."
#             )
#             result = core_to_onnx(
#                 fn,
#                 example_args,  # Positional
#                 model_name=model_name,
#                 opset=opset,  # Use 'opset' keyword
#                 verbose=verbose,
#                 **kwargs,
#             )
#             print(f"[V2] Conversion successful for '{model_name}' (using 'opset').")
#             return result
#         else:
#             print(f"[V2 Error] Conversion failed for '{model_name}': {e}")
#             raise  # Re-raise original error if not related to opset keyword
#     except Exception as e:
#         print(f"[V2 Error] Conversion failed for '{model_name}': {e}")
#         raise


# # === NEW V2 Function: allclose_v2 ===
# def allclose_v2(
#     jax_callable: Callable,
#     onnx_model_path: str,
#     *xs: jax.Array,  # Tensor arguments only
#     rtol=1e-03,  # Using baseline tolerances
#     atol=1e-05,  # Using baseline tolerances
#     jax_kwargs=None,  # Accepts static kwargs for JAX call
# ):
#     """Checks if JAX and ONNX Runtime outputs are close. (V2 - Corrected Approach)"""
#     if jax_kwargs is None:
#         jax_kwargs = {}

#     print(f"\n[V2] Running allclose for {onnx_model_path}...")
#     # --- JAX Execution using jax_kwargs ---
#     try:
#         print(
#             f"  [V2] Calling JAX function with {len(xs)} tensor args and kwargs: {list(jax_kwargs.keys())}"
#         )
#         jax_results = jax_callable(*xs, **jax_kwargs)  # Use kwargs here
#         print("  [V2] JAX function call successful.")
#     except Exception as e:
#         print(f"[Error] V2 allclose: JAX function call failed: {e}")
#         raise

#     if not isinstance(jax_results, (list, tuple)):
#         jax_results = (jax_results,)
#     jax_results_np = [np.array(res) for res in jax_results]

#     # --- ONNX Execution ---
#     try:
#         session = ort.InferenceSession(onnx_model_path)
#         print("  [V2] ONNX model loaded.")
#     except Exception as e:
#         print(
#             f"[Error] V2 allclose: Failed to load ONNX model '{onnx_model_path}': {e}"
#         )
#         raise

#     input_names = [inp.name for inp in session.get_inputs()]
#     output_names = [out.name for out in session.get_outputs()]
#     print(f"  [V2] ONNX expected inputs: {input_names} ({len(input_names)})")
#     print(f"  [V2] Provided tensor args for ONNX: {len(xs)}")

#     if len(input_names) != len(xs):
#         # This error indicates the V2 conversion still produced wrong number of inputs
#         raise ValueError(
#             f"[V2 Error] ONNX model '{onnx_model_path}' expects {len(input_names)} inputs "
#             f"({input_names}), but got {len(xs)} tensor arguments for comparison."
#         )

#     onnx_inputs = {name: np.array(arg) for name, arg in zip(input_names, xs)}

#     try:
#         onnx_results = session.run(output_names, onnx_inputs)
#         print("  [V2] ONNX inference successful.")
#     except Exception as e:
#         print(f"[Error] V2 allclose: ONNX Runtime inference failed: {e}")
#         raise

#     if not isinstance(onnx_results, (list, tuple)):
#         onnx_results = (onnx_results,)

#     # --- Comparison ---
#     if len(jax_results_np) != len(onnx_results):
#         raise ValueError(
#             f"[V2 Error] Output count mismatch: JAX {len(jax_results_np)}, ONNX {len(onnx_results)}"
#         )

#     is_ok = True
#     max_diff = 0.0
#     for i, (jax_res_np, onnx_res) in enumerate(zip(jax_results_np, onnx_results)):
#         diff = np.max(np.abs(jax_res_np - onnx_res))
#         max_diff = max(max_diff, diff)
#         if not np.allclose(jax_res_np, onnx_res, rtol=rtol, atol=atol):
#             print(f"  [V2 FAIL] Output {i} mismatch!")
#             is_ok = False
#             # Don't break, report max diff across all outputs

#     print(f"  [V2] Comparison complete. Max difference: {max_diff:.6g}")
#     # Match the return type of the original allclose (tuple)
#     return (is_ok, "[V2] Match :-)" if is_ok else "[V2] Mismatch :-(")
