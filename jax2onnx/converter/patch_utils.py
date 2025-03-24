# file: jax2onnx/converter/patch_utils.py

import contextlib
import inspect

from jax2onnx.plugin_system import PLUGIN_REGISTRY, PrimitivePlugin
from jax2onnx.converter.onnx_functions import ONNX_FUNCTION_PRIMITIVE_REGISTRY


@contextlib.contextmanager
def temporary_monkey_patches():
    with contextlib.ExitStack() as stack:
        # 🔧 Patch plugin-defined primitives
        for key in PLUGIN_REGISTRY:
            plugin = PLUGIN_REGISTRY[key]
            if not isinstance(plugin, PrimitivePlugin) or not plugin.patch_info:
                continue
            patch_info = plugin.patch_info()

            target = patch_info["patch_targets"][0]
            patch_func = patch_info["patch_function"]
            attr = patch_info.get("target_attribute", "__call__")
            stack.enter_context(_temporary_patch(target, attr, patch_func))

        # 🧠 Patch @onnx_function-decorated classes
        for class_name, primitive in ONNX_FUNCTION_PRIMITIVE_REGISTRY.items():
            cls = _get_class_from_primitive_name(class_name)
            if cls is None:
                continue

            original_call = getattr(cls, "__call__")

            def patched_call(self, *args, _prim=primitive, _orig=original_call):
                import jax

                if not jax.core.trace_state_clean():
                    return _prim.bind(*args)
                return _orig(self, *args)

            stack.enter_context(_temporary_patch(cls, "__call__", lambda: patched_call))

        yield


@contextlib.contextmanager
def _temporary_patch(target, attr, patch_func):
    original = getattr(target, attr)

    # Check if the patch function expects an argument
    if inspect.signature(patch_func).parameters:
        patched = patch_func(original)
    else:
        patched = patch_func()  # Call without arguments if none are expected

    setattr(target, attr, patched)
    try:
        yield
    finally:
        setattr(target, attr, original)


def _get_class_from_primitive_name(name):
    """
    Utility to reverse lookup class from ONNX_FUNCTION_PRIMITIVE_REGISTRY key.
    Assumes decorator used the class name as the key.
    """
    from jax2onnx.converter.onnx_functions import ONNX_FUNCTION_REGISTRY

    return ONNX_FUNCTION_REGISTRY.get(name, None)
