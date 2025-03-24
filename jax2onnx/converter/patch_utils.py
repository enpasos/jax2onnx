# file: jax2onnx/converter/patch_utils.py

import contextlib
import inspect
from jax2onnx.plugin_system import PLUGIN_REGISTRY, PrimitivePlugin
from jax2onnx.converter.onnx_functions import ONNX_FUNCTION_PRIMITIVE_REGISTRY


@contextlib.contextmanager
def temporary_monkey_patches(allow_function_primitives=False):
    """
    Temporarily monkey-patch modules/classes to emit primitives.

    - Includes plugin patches from PLUGIN_REGISTRY
    - Optionally includes ONNX_FUNCTION_PRIMITIVE_REGISTRY if `allow_function_primitives` is True
    """
    with contextlib.ExitStack() as stack:
        # Patch plugin-based modules
        for key in PLUGIN_REGISTRY:
            plugin = PLUGIN_REGISTRY[key]
            if not isinstance(plugin, PrimitivePlugin) or not plugin.patch_info:
                continue
            patch_info = plugin.patch_info()
            target = patch_info["patch_targets"][0]
            patch_func = patch_info["patch_function"]
            attr = patch_info.get("target_attribute", "__call__")
            stack.enter_context(_temporary_patch(target, attr, patch_func))

        # Optionally patch ONNX-function-decorated classes
        if allow_function_primitives:
            for name, primitive in ONNX_FUNCTION_PRIMITIVE_REGISTRY.items():
                cls = _get_class_from_primitive(name)
                if cls is None:
                    continue

                original_call = cls.__call__

                def make_wrapped_call(orig, prim):
                    def wrapped(self, *args):
                        if not _is_tracing():
                            return orig(self, *args)
                        return prim.bind(*args)

                    return wrapped

                wrapped_call = make_wrapped_call(original_call, primitive)
                stack.enter_context(
                    _temporary_patch(cls, "__call__", lambda: wrapped_call)
                )

        yield


def _get_class_from_primitive(name):
    """
    Helper to find the class that owns a given primitive by name.
    Assumes the class has `_onnx_primitive.name` == name
    """
    for cls in ONNX_FUNCTION_PRIMITIVE_REGISTRY.values():
        if hasattr(cls, "_onnx_primitive") and cls._onnx_primitive.name == name:
            return cls
    return None


@contextlib.contextmanager
def _temporary_patch(target, attr, patch_func):
    """
    Temporarily override a method or attribute on a class or module.
    """
    original = getattr(target, attr)
    if inspect.signature(patch_func).parameters:
        patched = patch_func(original)
    else:
        patched = patch_func()
    setattr(target, attr, patched)
    try:
        yield
    finally:
        setattr(target, attr, original)


def _is_tracing():
    import jax

    return not jax.core.trace_state_clean()
