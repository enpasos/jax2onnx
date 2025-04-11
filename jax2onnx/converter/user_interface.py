# file: jax2onnx/converter/user_interface.py

from collections import defaultdict
from typing import Any

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
    kwargs = kwargs or {}
    input_params = kwargs.get("input_params", {})
    print(f"Found input_params in kwargs: {input_params}")

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
    [name for name, shape in zip(input_names, input_shapes) if shape == ()]

    # If we have more inputs than tensor arguments, assume the extras are parameters
    len(input_names) > len(xs)

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

    # Create a mapping of parameter name to expected ONNX type
    onnx_type_map = {}
    for inp in session.get_inputs():
        if inp.name in param_input_names:
            onnx_type = inp.type
            # Extract numpy dtype from ONNX type string (e.g. "tensor(float)" -> np.float32)
            if "float" in onnx_type:
                if "float64" in onnx_type:
                    onnx_type_map[inp.name] = np.float64
                else:
                    onnx_type_map[inp.name] = np.float32
            elif "int" in onnx_type:
                if "int64" in onnx_type:
                    onnx_type_map[inp.name] = np.int64
                else:
                    onnx_type_map[inp.name] = np.int32
            elif "bool" in onnx_type:
                onnx_type_map[inp.name] = np.bool_
            else:
                # Default to float32 if we can't determine the type
                onnx_type_map[inp.name] = np.float32
                print(
                    f"Warning: Unknown ONNX type {onnx_type} for parameter {inp.name}, using float32"
                )

    print(f"ONNX parameter types: {onnx_type_map}")

    # Handle deterministic parameter and other parameters more carefully
    for param_name in param_input_names:
        if param_name == "deterministic":
            # Special handling for deterministic parameter
            det_value = jax_kwargs.get(
                "deterministic", True
            )  # Default to True if not specified

            # Use the dtype expected by ONNX for this parameter
            if param_name in onnx_type_map:
                expected_dtype = onnx_type_map[param_name]
                p[param_name] = np.array(det_value, dtype=expected_dtype)
                print(f"Added deterministic={det_value} with dtype={expected_dtype}")
            else:
                # Fallback to bool_ if not in the type map
                p[param_name] = np.array(det_value, dtype=np.bool_)
                print(f"Added deterministic={det_value} with dtype=bool_")

            # Also ensure it's in jax_kwargs
            jax_kwargs[param_name] = det_value

        elif param_name in jax_kwargs:
            # General handling for other parameters
            param_value = jax_kwargs[param_name]

            # Use the dtype expected by ONNX for this parameter, if available
            if param_name in onnx_type_map:
                expected_dtype = onnx_type_map[param_name]
                try:
                    p[param_name] = np.array(param_value, dtype=expected_dtype)
                    print(
                        f"Added {param_name}={param_value} with dtype={expected_dtype}"
                    )
                except (TypeError, ValueError) as e:
                    print(
                        f"Warning: Failed to convert {param_name}={param_value} to {expected_dtype}: {e}"
                    )
                    # Fall back to intelligent guessing based on the value type
                    if isinstance(param_value, bool):
                        p[param_name] = np.array(
                            int(param_value),
                            dtype=(
                                np.int64 if "int" in str(expected_dtype) else np.bool_
                            ),
                        )
                    elif isinstance(param_value, (int, float)):
                        p[param_name] = np.array(
                            param_value, dtype=type(param_value).__name__
                        )
                    else:
                        print(
                            f"Warning: Unsupported parameter type for {param_name}: {type(param_value)}"
                        )
            else:
                # Fall back to intelligent guessing based on the value type
                if isinstance(param_value, bool):
                    # Booleans in ONNX are often represented as int64
                    p[param_name] = np.array(int(param_value), dtype=np.int64)
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

            # For boolean parameters like deterministic, default to True
            if param_name == "deterministic":
                if param_name in onnx_type_map:
                    expected_dtype = onnx_type_map[param_name]
                    p[param_name] = np.array(True, dtype=expected_dtype)
                else:
                    p[param_name] = np.array(True, dtype=np.bool_)
            # For other parameters, we might need more sophisticated defaults
            elif param_name in onnx_type_map:
                # Set a reasonable default based on the expected type
                dtype = onnx_type_map[param_name]
                if np.issubdtype(dtype, np.integer):
                    p[param_name] = np.array(0, dtype=dtype)
                elif np.issubdtype(dtype, np.floating):
                    p[param_name] = np.array(0.0, dtype=dtype)
                elif np.issubdtype(dtype, np.bool_):
                    p[param_name] = np.array(False, dtype=dtype)
                else:
                    print(
                        f"Warning: Cannot determine default for parameter {param_name} with type {dtype}"
                    )
            # If we don't have type information, we can't provide a reasonable default

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
