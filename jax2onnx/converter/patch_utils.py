# file: jax2onnx/converter/patch_utils.py

import contextlib
import inspect

from jax2onnx.plugin_system import PLUGIN_REGISTRY, PrimitivePlugin
from jax2onnx.converter.onnx_functions import ONNX_FUNCTION_PRIMITIVE_REGISTRY


@contextlib.contextmanager
def temporary_monkey_patches(allow_function_primitives=False):
    """
    Temporarily patch all primitives:
    - Plugin primitives (from PLUGIN_REGISTRY)
    - Function primitives (from ONNX_FUNCTION_PRIMITIVE_REGISTRY, if enabled)
    """
    with contextlib.ExitStack() as stack:
        # Plugin primitives
        for key, plugin in PLUGIN_REGISTRY.items():
            if not isinstance(plugin, PrimitivePlugin) or not plugin.patch_info:
                continue
            patch_info = plugin.patch_info()
            target = patch_info["patch_targets"][0]
            patch_func = patch_info["patch_function"]
            attr = patch_info.get("target_attribute", "__call__")
            stack.enter_context(_temporary_patch(target, attr, patch_func))

        # ONNX function primitives
        if allow_function_primitives:
            for name, (primitive, cls) in ONNX_FUNCTION_PRIMITIVE_REGISTRY.items():

                def make_patch(prim):
                    def patch(original_call):
                        def wrapped(self, *args):
                            return prim.bind(*args)

                        return wrapped

                    return patch

                stack.enter_context(
                    _temporary_patch(cls, "__call__", make_patch(primitive))
                )

        yield


@contextlib.contextmanager
def _temporary_patch(target, attr, patch_func):
    original = getattr(target, attr)
    patched = (
        patch_func(original)
        if inspect.signature(patch_func).parameters
        else patch_func()
    )
    setattr(target, attr, patched)
    try:
        yield
    finally:
        setattr(target, attr, original)
