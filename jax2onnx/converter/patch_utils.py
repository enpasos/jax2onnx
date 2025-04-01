import contextlib
import inspect
from jax2onnx.plugin_system import (
    PLUGIN_REGISTRY,
    PrimitiveLeafPlugin,
    ONNX_FUNCTION_PLUGIN_REGISTRY,
)


@contextlib.contextmanager
def temporary_monkey_patches(allow_function_primitives=False):
    with contextlib.ExitStack() as stack:
        # Patch leaf plugin primitives
        for key, plugin in PLUGIN_REGISTRY.items():
            if not isinstance(plugin, PrimitiveLeafPlugin) or not plugin.patch_info:
                continue
            target, attr, patch_func = plugin.get_patch_params()
            stack.enter_context(_temporary_patch(target, attr, patch_func))

        if allow_function_primitives:
            for qualname, plugin in ONNX_FUNCTION_PLUGIN_REGISTRY.items():
                primitive = plugin.primitive
                patch_fn = plugin.get_patch_fn(primitive)
                target = plugin.target

                if inspect.isclass(target):
                    # Class-based: patch __call__
                    stack.enter_context(_temporary_patch(target, "__call__", patch_fn))
                elif callable(target):
                    # Function-based: patch the function reference in its module
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
