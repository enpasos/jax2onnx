# file: jax2onnx/converter/user_interface.py

from typing import Any, Callable, Dict, List, Tuple, Union

import onnx

from jax2onnx.converter.conversion_api import to_onnx as to_onnx_impl
from jax2onnx.converter.validation import allclose as allclose_impl
from jax2onnx.plugin_system import onnx_function as onnx_function_impl


def to_onnx(
    fn: Callable,
    input_shapes: List[Union[Tuple, List]],
    input_params: Dict[str, Any] | None = None,
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
    *xs: Any,
    rtol: float = 1e-3,
    atol: float = 1e-5,
    **jax_kwargs,
) -> Tuple[bool, str]:
    """
    Checks if JAX and ONNX Runtime outputs produce numerically similar results.

    This function is useful for validating that a converted ONNX model behaves
    similarly to the original JAX model within specified tolerance thresholds.

    Args:
        fn: JAX function or model to compare against the ONNX model
        onnx_model_path: Path to the saved ONNX model file
        *xs: Input tensors to pass to both the JAX function and ONNX model
        rtol: Relative tolerance for numerical comparison (default: 1e-3)
        atol: Absolute tolerance for numerical comparison (default: 1e-5)
        **jax_kwargs: Optional keyword arguments to pass to the JAX function only

    Returns:
        Tuple containing (is_match: bool, message: str) where:
          - is_match: True if outputs are numerically close, False otherwise
          - message: Descriptive message with comparison details

    Example:
        >>> import numpy as np
        >>> from jax2onnx import to_onnx, allclose
        >>> # First convert a model
        >>> onnx_model = to_onnx(my_jax_fn, [(3, 224, 224)])
        >>> import onnx
        >>> onnx.save(onnx_model, "my_model.onnx")
        >>> # Then validate the outputs match
        >>> test_input = np.random.rand(3, 224, 224).astype(np.float32)
        >>> is_close, message = allclose(my_jax_fn, "my_model.onnx", test_input, deterministic=True)
        >>> print(f"Models match: {is_close}")
        >>> print(message)
    """
    # Delegate to the implementation in validation.py
    return allclose_impl(fn, onnx_model_path, *xs, rtol=rtol, atol=atol, **jax_kwargs)
