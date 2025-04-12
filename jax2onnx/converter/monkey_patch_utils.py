"""
Monkey Patching Utilities for JAX2ONNX

This module provides utilities for temporarily monkey patching JAX functions and
classes to enable capture and conversion of JAX operations to ONNX format.
These utilities are primarily used during the tracing phase of JAX to ONNX conversion.
"""

import contextlib
import inspect
from typing import Any, Callable, Generator

from jax2onnx.plugin_system import (
    ONNX_FUNCTION_PLUGIN_REGISTRY,
    PLUGIN_REGISTRY,
    PrimitiveLeafPlugin,
)


@contextlib.contextmanager
def temporary_monkey_patches(
    allow_function_primitives: bool = False,
) -> Generator[None, None, None]:
    """
    Context manager that temporarily patches JAX functions and classes.

    This function applies patches from both the primitive leaf plugins and,
    if enabled, function primitive plugins. All patches are automatically
    reverted when the context is exited.

    Args:
        allow_function_primitives: If True, also patch function primitives
                                   from the ONNX function plugin registry.

    Yields:
        None: A context where the monkey patches are active.

    Example:
        with temporary_monkey_patches(allow_function_primitives=True):
            # JAX code executed here will use the patched functions
            result = my_jax_function(args)
    """
    with contextlib.ExitStack() as stack:
        # Patch leaf plugin primitives
        for key, plugin in PLUGIN_REGISTRY.items():
            if not isinstance(plugin, PrimitiveLeafPlugin) or not plugin.patch_info:
                continue
            target, attr, patch_func = plugin.get_patch_params()
            stack.enter_context(_temporary_patch(target, attr, patch_func))

        if allow_function_primitives:
            # Patch function primitives from the registry
            for qualname, plugin in ONNX_FUNCTION_PLUGIN_REGISTRY.items():
                primitive = plugin.primitive
                patch_fn = plugin.get_patch_fn(primitive)
                target = plugin.target

                if inspect.isclass(target):
                    # For classes: patch the __call__ method
                    stack.enter_context(_temporary_patch(target, "__call__", patch_fn))
                elif callable(target):
                    # For functions: patch the function in its module
                    module = inspect.getmodule(target)
                    func_name = target.__name__
                    if hasattr(module, func_name):
                        stack.enter_context(
                            _temporary_patch(module, func_name, patch_fn)
                        )
                else:
                    raise TypeError(f"Unsupported target type: {type(target)}")

        yield


@contextlib.contextmanager
def _temporary_patch(
    target: Any, attr: str, patch_func: Callable
) -> Generator[None, None, None]:
    """
    Internal helper that temporarily patches a single attribute.

    This context manager saves the original attribute value, replaces it with
    the patched version, and ensures the original is restored when the context exits.

    Args:
        target: The object to patch (class, module, etc.)
        attr: The attribute name to patch
        patch_func: The function that produces the patch or is the patch itself

    Yields:
        None: A context where the patch is active
    """
    # Save the original attribute
    original = getattr(target, attr)

    # Apply the patch - either directly or by calling with the original
    # We determine which approach to use by checking if patch_func accepts parameters
    patched = (
        patch_func(original)
        if inspect.signature(patch_func).parameters
        else patch_func()
    )
    setattr(target, attr, patched)

    try:
        yield
    finally:
        # Restore the original attribute when done
        setattr(target, attr, original)
