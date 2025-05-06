# file: jax2onnx/converter/patched_callable_wrapper.py

from typing import Callable
import logging
from jax import core
from jax.extend.core import Primitive
from jax._src.typing import Array
import numpy as np

logger_wrapper = logging.getLogger("jax2onnx.converter.patched_callable_wrapper")


class PatchedCallableWrapper:
    """
    A wrapper that replaces an original JAX function during tracing.

    It intercepts calls to the original function and instead binds a specified
    JAX primitive, passing necessary parameters like the original function itself
    and static arguments (e.g., 'axis') extracted from the call.
    """

    def __init__(self, original_fn: Callable, primitive: Primitive):
        """
        Initializes the wrapper.

        Args:
            original_fn: The original JAX function being patched (e.g., jnp.concatenate).
            primitive: The JAX primitive to bind instead (e.g., jnp.concat_p).
        """
        self._original_fn = original_fn
        self._primitive = primitive
        logger_wrapper.debug(
            f"Wrapper created for {original_fn.__name__} using primitive {primitive.name}"
        )

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
                f"Patched {self._original_fn.__name__} expects at least one argument (the arrays tuple/list)."
            )

        arrays_tuple = args[0]
        remaining_args = args[1:]  # Should be empty for jnp.concatenate

        # Ensure first arg is a sequence
        if not isinstance(arrays_tuple, (tuple, list)):
            # Allow single array input, treat as sequence of one
            # Check for actual arrays or tracers
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

        # === MODIFICATION START ===
        # Prepare bind_params *without* _original_fn, as abstract_eval gets it
        # from the class variable set during patching. Keep other user kwargs.
        bind_params = {
            **kwargs,  # Pass any other user kwargs
            "axis": axis,
        }
        # === MODIFICATION END ===

        # Bind the primitive, passing arrays_tuple elements as *args
        # Important: Bind expects the *elements* of the sequence, not the sequence itself
        logger_wrapper.debug(
            f"Binding {self._primitive.name} with {len(arrays_tuple)} array args and params: {list(bind_params.keys())}"
        )

        # QUICKFIX: Ensure axis is concrete int before bind if it's a tracer
        # This might be needed if axis itself is traced, although typically it's static.
        if "axis" in bind_params:
            axis_val = bind_params["axis"]
            if isinstance(axis_val, core.Tracer):
                if hasattr(axis_val, "aval") and hasattr(
                    axis_val.aval, "get_literal_value"
                ):
                    # Prefer get_literal_value for newer JAX versions
                    try:
                        constant_value = axis_val.aval.get_literal_value()
                        logger_wrapper.debug(
                            f"Concretized axis tracer to: {constant_value}"
                        )
                        bind_params["axis"] = int(constant_value)
                    except TypeError:  # Not concrete
                        raise TypeError(
                            f"Axis tracer cannot be concretized during bind: {axis_val}"
                        )
                elif hasattr(axis_val, "aval") and hasattr(
                    axis_val.aval, "constant_value"
                ):
                    # Fallback for older JAX versions?
                    constant_value = axis_val.aval.constant_value
                    if constant_value is None:
                        raise TypeError(
                            f"Axis tracer has no constant value during bind: {axis_val}"
                        )
                    logger_wrapper.debug(
                        f"Concretized axis tracer (legacy) to: {constant_value}"
                    )
                    bind_params["axis"] = int(constant_value)

                else:
                    raise TypeError(
                        f"Axis is tracer and cannot be concretized: {axis_val}"
                    )
            elif not isinstance(axis_val, (int, tuple)):
                # If it's already concrete but not int or tuple, try converting
                try:
                    bind_params["axis"] = int(axis_val)
                except (ValueError, TypeError):
                    raise TypeError(
                        f"Axis must be an integer, tuple, or concretizable tracer, got {type(axis_val)}"
                    )
            elif isinstance(axis_val, tuple):
                # Handle tuple of axes - no conversion needed for squeeze which accepts tuple axes
                logger_wrapper.debug(f"Using tuple axis value directly: {axis_val}")

        return self._primitive.bind(*arrays_tuple, **bind_params)
