# file: jax2onnx/converter/user_interface.py

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import argparse
import logging

import onnx
from jax import config, core
from jax2onnx.converter.conversion_api import to_onnx as to_onnx_impl
from jax2onnx.converter.validation import allclose as allclose_impl
from jax2onnx.plugin_system import onnx_function as onnx_function_impl

config.update("jax_dynamic_shapes", True)

# NEW -----------------------------------------------------------------
_FLOAT64_HELP = (
    "Export the entire ONNX graph in double precision (tensor(double)). "
    "If omitted, tensors are exported in single precision (tensor(float))."
)


def to_onnx(
    fn: Callable,
    inputs: List[Any],
    input_params: Optional[Dict[str, Any]] = None,  # Made Optional more explicit
    model_name: str = "jax_model",
    opset: int = 21,
    *,  # All arguments after this must be keyword-only
    enable_double_precision: bool = False,
    record_primitive_calls_file: Optional[str] = None,
) -> onnx.ModelProto:
    """
    Converts a JAX function or model into an ONNX model.

    This function serves as the main entry point for converting JAX/Flax models to ONNX format.
    It supports dynamic shapes and additional runtime parameters.

    Args:
        fn: The JAX function or Flax module to convert.
        inputs: Either shapes (List[Tuple|List]) or actual input values (List[np.ndarray]) to the function.
        input_params: Optional parameters that should be exposed as inputs in the ONNX model
                     rather than baked into the model. Useful for runtime parameters like
                     'deterministic' flags.
        model_name: Name to give the ONNX model. Defaults to "jax_model".
        opset: ONNX opset version to target. Defaults to 21.
        enable_double_precision: If True, export tensors as tensor(double). Defaults to False (use tensor(float)).
        record_primitive_calls_file: Optional path to a file. If provided,
            details of each JAX primitive encountered during conversion will be
            recorded to this file. This log can be used by developers to manually
            create new test cases. Defaults to None (disabled).

    Returns:
        An ONNX ModelProto object representing the converted model.

    Example:
        >>> import jax.numpy as jnp
        >>> from flax import nnx
        >>> from jax2onnx import to_onnx
        >>>
        >>> model = MyFlaxModel(...)
        >>> onnx_model = to_onnx(model, inputs=[('B', 32, 32, 3)])
        >>> import onnx
        >>> onnx.save(onnx_model, "model.onnx")
    """

    logging.info(
        f"Converting JAX function to ONNX model with parameters: "
        f"model_name={model_name}, opset={opset}, input_shapes={inputs}, "
        f"input_params={input_params}, enable_double_precision={enable_double_precision}, "
        f"record_primitive_calls_file={record_primitive_calls_file}"
    )

    # Determine the nature of the 'inputs' argument to prepare for to_onnx_impl
    processed_inputs_for_impl: list

    if not inputs:  # Handle empty inputs list
        processed_inputs_for_impl = []
    else:
        # Check if all elements are already ShapeDtypeStructs (or compatible ShapedArray)
        if all(isinstance(x, core.ShapedArray) for x in inputs):
            # Case 1: Inputs are already ShapeDtypeStructs (e.g., from t_generator with input_values)
            # Preserve them as they contain both shape and dtype.
            processed_inputs_for_impl = list(inputs)
        else:
            # Case 2: Inputs might be shape tuples or actual JAX/NumPy arrays.

            # Define helper to check for shape tuples
            def is_shape_tuple(item):
                return isinstance(item, (tuple, list)) and all(
                    isinstance(dim, (int, str)) for dim in item
                )

            if all(is_shape_tuple(x) for x in inputs):
                # All inputs are shape tuples.
                # to_onnx_impl will create ShapedArrays using a default dtype.
                processed_inputs_for_impl = list(inputs)
            else:
                # Assume inputs are actual JAX arrays or NumPy arrays.
                # Convert them to ShapeDtypeStructs.
                try:
                    import jax  # Ensure jax is imported for jax.ShapeDtypeStruct

                    processed_inputs_for_impl = [
                        jax.ShapeDtypeStruct(x.shape, x.dtype) for x in inputs
                    ]
                except AttributeError as e:
                    # This might happen if the list is mixed and some items don't have .shape/.dtype
                    # or if an item is not a ShapeDtypeStruct, not a shape tuple, and not an array.
                    raise ValueError(
                        "Invalid 'inputs' argument. Expected a list of JAX/NumPy arrays, "
                        "jax.ShapeDtypeStruct objects, or shape tuples. "
                        f"Got an element of type {type(inputs[0]) if inputs else 'Unknown'} in the list. Error: {e}"
                    )

    return to_onnx_impl(
        fn=fn,
        inputs=processed_inputs_for_impl,
        input_params=input_params,
        model_name=model_name,
        opset=opset,
        enable_double_precision=enable_double_precision,
        record_primitive_calls_file=record_primitive_calls_file,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jax2onnx",
        description="Convert a JAX function to an ONNX model.",
    )
    p.add_argument("module", help="Python module containing the JAX function")
    p.add_argument("fn", help="Name of the JAX function inside the module")
    p.add_argument("--out", help="Output .onnx file", default="model.onnx")
    p.add_argument("--opset", type=int, default=21, help="ONNX opset version")
    # … other existing args …

    # ──────────────── NEW FLAG ────────────────
    p.add_argument(
        "--float64",
        dest="enable_double_precision",
        action="store_true",
        default=False,
        help=_FLOAT64_HELP,
    )
    # ──────────────────────────────────────────

    # Add new argument for primitive call recording
    p.add_argument(
        "--record-primitives",
        dest="record_primitive_calls_file",
        help="File path to record JAX primitive calls during conversion",
        default=None,
    )

    return p


def run_command_line():
    args = build_arg_parser().parse_args()

    # Import the module and get the function
    import importlib
    import sys

    sys.path.append(".")
    try:
        module = importlib.import_module(args.module)
        function = getattr(module, args.fn)
    except (ImportError, AttributeError) as e:
        logging.error(f"Error loading function: {e}")
        sys.exit(1)

    # Parse input shapes or use reasonable defaults
    input_specs = []  # Default to empty list if not specified
    if hasattr(args, "input_shapes") and args.input_shapes:
        try:
            # Parse input shapes from command line
            # This is a placeholder - actual implementation would depend on how shapes are specified
            input_specs = eval(args.input_shapes)
        except Exception as e:
            logging.error(f"Error parsing input shapes: {e}")
            sys.exit(1)

    to_onnx(
        function,
        inputs=input_specs,
        model_name=args.fn,
        opset=args.opset,
        enable_double_precision=args.enable_double_precision,
        record_primitive_calls_file=args.record_primitive_calls_file,
    )

    # ...existing code...


def convert(
    *,
    enable_double_precision: bool = False,
    record_primitive_calls_file: Optional[str] = None,
    **kwargs,
):
    """
    Python API thin-wrapper around :pyfunc:`jax2onnx.to_onnx`.

    Parameters
    ----------
    enable_double_precision : bool, optional
        If *True*, export tensors as ``tensor(double)``.  Defaults to *False*.
    record_primitive_calls_file : str, optional
        Path to a file to record JAX primitive calls during conversion. Defaults to None.
    """
    return to_onnx(
        enable_double_precision=enable_double_precision,
        record_primitive_calls_file=record_primitive_calls_file,
        **kwargs,
    )


def onnx_function(target: Union[Callable, type]) -> Union[Callable, type]:
    """
    Decorator to mark a function or class as an ONNX function.

    This decorator is used to indicate that a function or class should be converted to
    an ONNX function node when included in a model. It allows the function to be traced
    and exported as a reusable component with its own namespace in the ONNX graph.

    Args:
        target: The target function or class to decorate.

    Returns:
        The decorated function or class with ONNX function capabilities.

    Example:
        >>> from jax2onnx import onnx_function
        >>> from flax import nnx
        >>>
        >>> @onnx_function
        >>> class MLPBlock(nnx.Module):
        >>>     def __init__(self, features, rngs):
        >>>         self.dense = nnx.Linear(features, rngs=rngs)
        >>>         self.activation = nnx.relu
        >>>
        >>>     def __call__(self, x):
        >>>         return self.activation(self.dense(x))
    """

    return onnx_function_impl(target)


def allclose(
    fn: Callable,
    onnx_model_path: str,
    inputs: List[Any],
    input_params: Optional[Dict[str, Any]] = None,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> Tuple[bool, str]:
    """
    Checks if JAX and ONNX Runtime outputs produce numerically similar results.

    This function is useful for validating that a converted ONNX model behaves
    similarly to the original JAX model within specified tolerance thresholds.

    Args:
        fn: JAX function or model to compare against the ONNX model
        onnx_model_path: Path to the saved ONNX model file
        inputs: Either input tensors (List[np.ndarray]) or shapes (List[Tuple|List]) to pass to both the JAX function and ONNX model
        rtol: Relative tolerance for numerical comparison (default: 1e-3)
        atol: Absolute tolerance for numerical comparison (default: 1e-5)
        input_params: Optional keyword arguments to pass to the JAX function only

    Returns:
        Tuple containing (is_match: bool, message: str) where:
          - is_match: True if outputs are numerically close, False otherwise
          - message: Descriptive message with comparison details

    Example:
        >>> import numpy as np
        >>> from jax2onnx import to_onnx, allclose
        >>> # First convert a model
        >>> onnx_model = to_onnx(my_jax_fn, inputs=[(3, 224, 224)])
        >>> import onnx
        >>> onnx.save(onnx_model, "my_model.onnx")
        >>> # Then validate the outputs match
        >>> test_input = np.random.rand(3, 224, 224).astype(np.float32)
        >>> is_close, message = allclose(my_jax_fn, "my_model.onnx", inputs=[test_input], deterministic=True)
        >>> print(f"Models match: {is_close}")
        >>> print(message)
    """
    import numpy as np

    logging.info(
        f"Comparing JAX and ONNX outputs with parameters: "
        f"onnx_model_path={onnx_model_path}, inputs={inputs}, "
        f"input_params={input_params}, rtol={rtol}, atol={atol}"
    )

    def is_shape(x):
        return isinstance(x, (tuple, list)) and all(
            isinstance(dim, (int, str)) for dim in x
        )

    # If all inputs are shapes, generate random values for them
    if all(is_shape(x) for x in inputs):
        xs = tuple(
            np.random.rand(*[d if isinstance(d, int) else 2 for d in shape]).astype(
                np.float32
            )
            for shape in inputs
        )
    else:
        xs = tuple(inputs)

    if input_params is None:
        return allclose_impl(fn, onnx_model_path, *xs, rtol=rtol, atol=atol)
    else:
        return allclose_impl(
            fn, onnx_model_path, *xs, rtol=rtol, atol=atol, **input_params
        )
