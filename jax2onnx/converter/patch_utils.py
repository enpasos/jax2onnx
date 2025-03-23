# file: jax2onnx/converter/patch_utils.py

import contextlib
import inspect

from jax2onnx.plugin_system import PLUGIN_REGISTRY, PrimitivePlugin


@contextlib.contextmanager
def temporary_monkey_patches():
    with contextlib.ExitStack() as stack:
        for key in PLUGIN_REGISTRY:
            plugin = PLUGIN_REGISTRY[key]
            if not isinstance(plugin, PrimitivePlugin) or not plugin.patch_info:
                continue
            patch_info = plugin.patch_info()

            target = patch_info["patch_targets"][0]
            patch_func = patch_info["patch_function"]
            attr = patch_info.get("target_attribute", "__call__")
            stack.enter_context(_temporary_patch(target, attr, patch_func))
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
