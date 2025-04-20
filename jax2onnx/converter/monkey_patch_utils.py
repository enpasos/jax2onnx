"""
Monkey Patching Utilities for JAX2ONNX

This module provides utilities for temporarily monkey patching JAX functions and
classes to enable capture and conversion of JAX operations to ONNX format.
These utilities are primarily used during the tracing phase of JAX to ONNX conversion.
"""

import contextlib
import inspect
from typing import Any, Callable, Generator, Mapping

from jax2onnx.plugin_system import (
    FunctionPlugin,
    ExamplePlugin,
    PrimitiveLeafPlugin,
    ONNX_FUNCTION_PLUGIN_REGISTRY,
    PLUGIN_REGISTRY,
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
        registries: list[
            Mapping[str, FunctionPlugin | ExamplePlugin | PrimitiveLeafPlugin]
        ] = [PLUGIN_REGISTRY]
        if allow_function_primitives:
            registries.append(ONNX_FUNCTION_PLUGIN_REGISTRY)
        for registry in registries:
            for plugin in registry.values():
                if not hasattr(plugin, "get_patch_params"):
                    continue
                try:
                    target, attr, patch_func = plugin.get_patch_params()
                except Exception:
                    continue
                stack.enter_context(_temporary_patch(target, attr, patch_func))
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
