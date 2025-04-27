# file: jax2onnx/converter/patched_callable_wrapper.py

import functools
from typing import Callable, Any
from jax import Array, core
import logging
import jax.numpy as jnp  # Need jnp for default axis value potentially
from jax.extend.core import Primitive
import numpy as np

logger_wrapper = logging.getLogger("jax2onnx.PatchedCallableWrapper")


class PatchedCallableWrapper:
    """Wraps an original function to bind a primitive, passing the original."""

    def __init__(self, original_fn: Callable, primitive: Primitive):
        self._original_fn = original_fn
        self._primitive = primitive
        # Copy metadata for better introspection/debugging
        functools.update_wrapper(self, original_fn)

    def __call__(self, *args, **kwargs):
        """
        Called when the patched function is invoked.
        Binds the primitive, passing the original function and axis in params.
        """
        logger_wrapper.debug(
            f"Wrapper called for {self._original_fn.__name__} "
            f"-> binding primitive {self._primitive.name}"
        )

        # --- Argument Handling (Specific to jnp.concatenate) ---
        # jnp.concatenate expects (arrays, axis=0, ...)
        if not args:
            raise TypeError(
                f"Patched {self._original_fn.__name__} expects at least one argument (the arrays tuple)."
            )

        arrays_tuple = args[0]
        remaining_args = args[1:]  # Should be empty for jnp.concatenate

        # Ensure first arg is a sequence
        if not isinstance(arrays_tuple, (tuple, list)):
            # Allow single array input, treat as sequence of one
            if isinstance(
                arrays_tuple, (np.ndarray, Array, core.Tracer, core.ShapedArray)
            ):
                logger_wrapper.debug(
                    "Single array passed to concatenate, wrapping in tuple."
                )
                arrays_tuple = (arrays_tuple,)
            else:
                raise TypeError(
                    f"Expected first argument to {self._original_fn.__name__} to be a sequence of arrays or single array, got {type(arrays_tuple)}"
                )

        # Warn if extra positional args were passed (unexpected for concatenate)
        if remaining_args:
            logger_wrapper.warning(
                f"Patched {self._original_fn.__name__} received unexpected positional arguments: {remaining_args}"
            )

        # Extract axis, default to 0 if not present
        axis = kwargs.pop("axis", 0)

        # Combine remaining kwargs (if any) with the special params
        bind_params = {
            **kwargs,  # Pass any other user kwargs
            "_original_fn": self._original_fn,
            "axis": axis,
        }

        # Bind the primitive, passing arrays_tuple elements as *args
        # Important: Bind expects the *elements* of the sequence, not the sequence itself
        logger_wrapper.debug(
            f"Binding {self._primitive.name} with {len(arrays_tuple)} array args and params: {list(bind_params.keys())}"
        )

        # QUICKFIX inside PatchedCallableWrapper before primitive.bind
        if "axis" in bind_params:
            axis = bind_params["axis"]
            if isinstance(axis, core.Tracer):
                if hasattr(axis, "aval") and hasattr(axis.aval, "constant_value"):
                    constant_value = axis.aval.constant_value
                    if constant_value is None:
                        raise TypeError(
                            "Axis tracer has no constant value during bind."
                        )
                    bind_params["axis"] = int(constant_value)
                else:
                    raise TypeError(f"Axis is tracer and cannot be concretized: {axis}")

        return self._primitive.bind(*arrays_tuple, **bind_params)
