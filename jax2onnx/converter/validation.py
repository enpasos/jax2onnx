"""
Model Validation Module

This module provides utilities for validating ONNX models converted from JAX functions
by comparing their outputs when given identical inputs. It helps ensure that the
conversion process preserves the behavior of the original JAX functions.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import onnxruntime as ort
import logging


def allclose(
    fn,
    onnx_model_path,
    *xs,
    rtol=1e-3,
    atol=1e-5,
    **jax_kwargs,
) -> Tuple[bool, str]:
    """
    Checks if JAX and ONNX Runtime outputs are numerically close.

    This function runs both the original JAX function and the converted ONNX model
    with identical inputs, then compares their outputs using numpy's allclose.
    It handles various parameter types and provides detailed diagnostics.

    Args:
        fn: JAX function to test
        onnx_model_path: Path to the ONNX model
        *xs: Value inputs (not shapes) to pass to both JAX and ONNX
        rtol: Relative tolerance for comparison (default: 1e-3)
        atol: Absolute tolerance for comparison (default: 1e-5)
        **jax_kwargs: Optional keyword arguments to pass to the JAX function

    Returns:
        Tuple of (is_match: bool, message: str)
    """

    # Accept either value inputs or shape tuples/lists
    def is_shape(x):
        return isinstance(x, (tuple, list)) and all(
            isinstance(dim, (int, str)) for dim in x
        )

    # If all inputs are shapes, generate random values
    if all(is_shape(x) for x in xs):
        xs = tuple(
            np.random.rand(*[d if isinstance(d, int) else 2 for d in shape]).astype(
                np.float32
            )
            for shape in xs
        )

    # Load ONNX model and create inference session
    session = ort.InferenceSession(onnx_model_path)

    # Extract actual input names from model
    input_names = [inp.name for inp in session.get_inputs()]

    # Get input shapes to help identify scalar parameters
    [tuple(inp.shape) for inp in session.get_inputs()]

    # Determine how many inputs are tensors (should match xs)
    tensor_input_count = len(xs)

    # Get the tensor input names (the first n names)
    tensor_input_names = input_names[:tensor_input_count]

    # The rest are parameter inputs
    param_input_names = input_names[tensor_input_count:]

    # Prepare ONNX input dictionary for tensor inputs
    onnx_inputs = {
        name: np.array(x) for name, x in zip(tensor_input_names, xs, strict=False)
    }

    # Create a mapping of parameter name to expected ONNX type
    onnx_type_map = _create_onnx_type_map(session, param_input_names)

    # Handle parameters (deterministic and others)
    _add_parameters_to_inputs(onnx_inputs, param_input_names, jax_kwargs, onnx_type_map)

    # Run ONNX model with both tensor and parameter inputs
    onnx_output = session.run(None, onnx_inputs)

    # compute JAX output, making sure inputs are JAX arrays
    import jax.numpy as jnp

    # Wrap any array-like inputs into JAX arrays so methods like .at work
    jax_inputs = [jnp.asarray(x) for x in xs]
    jax_output = fn(*jax_inputs, **jax_kwargs)

    # --- START OF PATCH ---
    # Ensure outputs are in a flat list format for comparison
    if isinstance(jax_output, tuple):
        jax_output_list = list(jax_output)
    elif not isinstance(jax_output, list):
        jax_output_list = [jax_output]
    else:
        jax_output_list = jax_output

    if not isinstance(onnx_output, list):
        onnx_output_list = [onnx_output]
    else:
        onnx_output_list = onnx_output
    # --- END OF PATCH ---

    # Compare all outputs
    all_match = True
    detailed_messages = []

    for i, (o, j) in enumerate(zip(onnx_output_list, jax_output_list, strict=False)):
        # Convert outputs to numpy arrays if they aren't already
        o_np = np.array(o)
        j_np = np.array(j)

        # Check if shapes match
        if o_np.shape != j_np.shape:
            all_match = False
            detailed_messages.append(
                f"Output {i}: Shape mismatch - ONNX: {o_np.shape}, JAX: {j_np.shape}"
            )
            continue

        # Check if values are close
        if np.allclose(o_np, j_np, rtol=rtol, atol=atol, equal_nan=True):
            detailed_messages.append(
                f"Output {i}: Values match within tolerance (rtol={rtol}, atol={atol})"
            )
        else:
            all_match = False
            # Calculate statistics for the differences
            abs_diff = np.abs(o_np - j_np)
            max_diff = np.max(abs_diff)
            mean_diff = np.mean(abs_diff)
            median_diff = np.median(abs_diff)

            # Find the location of the maximum difference
            max_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)

            detailed_messages.append(
                f"Output {i}: Values differ beyond tolerance. "
                f"Max diff: {max_diff:.6e} at {max_idx}, "
                f"Mean diff: {mean_diff:.6e}, Median diff: {median_diff:.6e}"
            )

    # Create the final message
    if all_match:
        message = "All outputs match within tolerance:\n" + "\n".join(detailed_messages)
    else:
        message = "Outputs do not match:\n" + "\n".join(detailed_messages)

    logging.info(f"Comparison result: {message}")
    logging.info(f"JAX output: {jax_output}")
    logging.info(f"ONNX output: {onnx_output}")
    logging.info(f"ONNX inputs: {onnx_inputs}")
    logging.info(f"JAX inputs: {jax_kwargs}")
    logging.info(f"ONNX type map: {onnx_type_map}")
    logging.info(f"Tensor input names: {tensor_input_names}")

    return all_match, message


def _create_onnx_type_map(
    session: ort.InferenceSession, param_input_names: List[str]
) -> Dict[str, np.dtype]:
    """
    Creates a mapping from parameter names to their expected numpy data types based on ONNX model inputs.

    Args:
        session: The ONNX runtime inference session
        param_input_names: List of parameter input names

    Returns:
        Dictionary mapping parameter names to numpy dtypes
    """
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
                logging.warning(
                    f"Warning: Unknown ONNX type {onnx_type} for parameter {inp.name}, using float32"
                )

    logging.debug(f"ONNX parameter types: {onnx_type_map}")
    return onnx_type_map


def _add_parameters_to_inputs(
    onnx_inputs: Dict[str, np.ndarray],
    param_input_names: List[str],
    jax_kwargs: Dict[str, Any],
    onnx_type_map: Dict[str, np.dtype],
):
    """
    Adds parameter values to the ONNX inputs dictionary.

    Args:
        onnx_inputs: Dictionary of ONNX inputs to modify
        param_input_names: List of parameter input names
        jax_kwargs: Dictionary of keyword arguments passed to the JAX function
        onnx_type_map: Mapping from parameter names to expected numpy dtypes
    """
    # Handle deterministic parameter and other parameters
    for param_name in param_input_names:
        if param_name == "deterministic":
            # Special handling for deterministic parameter
            det_value = jax_kwargs.get(
                "deterministic", True
            )  # Default to True if not specified

            # Use the dtype expected by ONNX for this parameter
            if param_name in onnx_type_map:
                expected_dtype = onnx_type_map[param_name]
                onnx_inputs[param_name] = np.array(det_value, dtype=expected_dtype)
            else:
                # Fallback to bool_ if not in the type map
                onnx_inputs[param_name] = np.array(det_value, dtype=np.bool_)

            # Also ensure it's in jax_kwargs
            jax_kwargs["deterministic"] = det_value

        elif param_name in jax_kwargs:
            # General handling for other parameters
            param_value = jax_kwargs[param_name]
            _add_parameter_value(onnx_inputs, param_name, param_value, onnx_type_map)
        else:
            # Parameter not found in jax_kwargs, provide a reasonable default
            logging.info(
                f"Warning: Parameter {param_name} not provided in jax_kwargs, using default value"
            )
            _add_default_parameter(onnx_inputs, param_name, onnx_type_map)


def _add_parameter_value(
    onnx_inputs: Dict[str, np.ndarray],
    param_name: str,
    param_value: Any,
    onnx_type_map: Dict[str, np.dtype],
):
    """
    Adds a parameter value to the ONNX inputs dictionary with appropriate type conversion.

    Args:
        onnx_inputs: Dictionary of ONNX inputs to modify
        param_name: Name of the parameter
        param_value: Value of the parameter
        onnx_type_map: Mapping from parameter names to expected numpy dtypes
    """
    # Use the dtype expected by ONNX for this parameter, if available
    if param_name in onnx_type_map:
        expected_dtype = onnx_type_map[param_name]
        try:
            onnx_inputs[param_name] = np.array(param_value, dtype=expected_dtype)
        except (TypeError, ValueError) as e:
            logging.info(
                f"Warning: Failed to convert {param_name}={param_value} to {expected_dtype}: {e}"
            )
            # Fall back to intelligent guessing based on the value type
            if isinstance(param_value, bool):
                onnx_inputs[param_name] = np.array(
                    int(param_value),
                    dtype=(np.int64 if "int" in str(expected_dtype) else np.bool_),
                )
            elif isinstance(param_value, (int, float)):
                onnx_inputs[param_name] = np.array(
                    param_value, dtype=type(param_value).__name__
                )
            else:
                logging.info(
                    f"Warning: Unsupported parameter type for {param_name}: {type(param_value)}"
                )
    else:
        # Fall back to intelligent guessing based on the value type
        if isinstance(param_value, bool):
            # Booleans in ONNX are often represented as int64
            onnx_inputs[param_name] = np.array(int(param_value), dtype=np.int64)
        elif isinstance(param_value, int):
            onnx_inputs[param_name] = np.array(param_value, dtype=np.int64)
        elif isinstance(param_value, float):
            onnx_inputs[param_name] = np.array(param_value, dtype=np.float32)
        else:
            logging.info(
                f"Warning: Parameter {param_name} has unsupported type {type(param_value)}"
            )


def _add_default_parameter(
    onnx_inputs: Dict[str, np.ndarray],
    param_name: str,
    onnx_type_map: Dict[str, np.dtype],
):
    """
    Adds a default parameter value to the ONNX inputs dictionary.

    Args:
        onnx_inputs: Dictionary of ONNX inputs to modify
        param_name: Name of the parameter
        onnx_type_map: Mapping from parameter names to expected numpy dtypes
    """
    # For boolean parameters like deterministic, default to True
    if param_name == "deterministic":
        if param_name in onnx_type_map:
            expected_dtype = onnx_type_map[param_name]
            onnx_inputs[param_name] = np.array(True, dtype=expected_dtype)
        else:
            onnx_inputs[param_name] = np.array(True, dtype=np.bool_)
    # For other parameters, we might need more sophisticated defaults
    elif param_name in onnx_type_map:
        # Set a reasonable default based on the expected type
        dtype = onnx_type_map[param_name]
        if np.issubdtype(dtype, np.integer):
            onnx_inputs[param_name] = np.array(0, dtype=dtype)
        elif np.issubdtype(dtype, np.floating):
            onnx_inputs[param_name] = np.array(0.0, dtype=dtype)
        elif np.issubdtype(dtype, np.bool_):
            onnx_inputs[param_name] = np.array(False, dtype=dtype)
        else:
            logging.info(
                f"Warning: Cannot determine default for parameter {param_name} with type {dtype}"
            )
    # If we don't have type information, we can't provide a reasonable default
    else:
        logging.info(
            f"Warning: No type information for parameter {param_name}, skipping default"
        )
