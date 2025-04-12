# file: jax2onnx/converter/user_interface.py

from typing import Any

import onnx

from jax2onnx.converter.conversion_api import to_onnx as to_onnx_impl
from jax2onnx.converter.validation import allclose as allclose_impl
from jax2onnx.plugin_system import onnx_function as onnx_function_impl


def to_onnx(
    fn: Any,
    input_shapes: Any,
    input_params: dict[str, Any] | None = None,
    model_name: str = "jax_model",
    opset: int = 21,
) -> onnx.ModelProto:
    """
    Converts a JAX function or model into an ONNX model.

    This function serves as the main entry point for converting JAX/Flax models to ONNX format.
    It supports dynamic shapes and additional runtime parameters.

    Args:
        fn: The JAX function or Flax module to convert.
        input_shapes: Shapes of the inputs to the function. Can include symbolic dimensions
                     like ('B', 28, 28, 1) where 'B' represents a dynamic batch dimension.
        input_params: Optional parameters that should be exposed as inputs in the ONNX model
                     rather than baked into the model. Useful for runtime parameters like
                     'deterministic' flags.
        model_name: Name to give the ONNX model. Defaults to "jax_model".
        opset: ONNX opset version to target. Defaults to 21.

    Returns:
        An ONNX ModelProto object representing the converted model.

    Example:
        >>> import jax.numpy as jnp
        >>> from flax import nnx
        >>> from jax2onnx import to_onnx
        >>>
        >>> model = MyFlaxModel(...)
        >>> onnx_model = to_onnx(model, [('B', 32, 32, 3)])
        >>> import onnx
        >>> onnx.save(onnx_model, "model.onnx")
    """

    return to_onnx_impl(
        fn=fn,
        input_shapes=input_shapes,
        input_params=input_params,
        model_name=model_name,
        opset=opset,
    )


def onnx_function(target):
    """
    Decorator to mark a function as an ONNX function.

    This decorator is used to indicate that a function should be converted to ONNX format.
    It allows the function to be traced and exported with additional metadata.

    Args:
        target: The target function or class to decorate.

    Returns:
        The decorated function or class.
    """
    return onnx_function_impl(target)


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
    # Delegate to the implementation in validation.py
    return allclose_impl(
        callable, onnx_model_path, *xs, jax_kwargs=jax_kwargs, rtol=rtol, atol=atol
    )
