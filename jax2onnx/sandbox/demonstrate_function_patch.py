# file: jax2onnx/sandbox/demonstrate_function_patch.py

import contextlib
import inspect
from jax.core import Primitive

from jax2onnx.plugin_system import (
    ONNX_FUNCTION_REGISTRY,
)  # Import for type hinting and clarity

# Dummy registries and classes for demonstration

ONNX_FUNCTION_PRIMITIVE_REGISTRY: dict[str, Primitive] = {}


class PrimitivePlugin:  # Mock PrimitivePlugin for demonstration
    def __init__(self, patch_info):
        self.patch_info = lambda: patch_info
        self.patch_info_data = patch_info  # Store for easier access later

    def __call__(self, *args):  # To be able to call the patch_info method
        return self.patch_info()


PLUGIN_REGISTRY: dict[str, PrimitivePlugin] = {}


@contextlib.contextmanager
def _temporary_patch(target, attr, patch_func):
    """
    Temporarily patches an attribute of an object.

    Args:
        target: The object whose attribute will be patched.
        attr: The name of the attribute (string) to patch.
        patch_func:  A callable that either:
            - Takes the original attribute value as input and returns the patched value.
            - Takes no arguments and returns the patched value directly.
    """
    original = getattr(target, attr)  # Get the original attribute value.
    # Determine if patch_func expects the original value or not, and call it appropriately.
    patched = (
        patch_func(original)
        if inspect.signature(patch_func).parameters
        else patch_func()
    )
    setattr(target, attr, patched)  # Set the attribute to the patched value.
    try:
        yield  # This is where the code within the 'with' statement executes.
    finally:
        setattr(
            target, attr, original
        )  # Restore the original attribute value, even if errors occur.


@contextlib.contextmanager
def temporary_monkey_patches(allow_function_primitives=False):
    """
    Temporarily patch all primitives:
    - Plugin primitives (from PLUGIN_REGISTRY)
    - Function primitives (from ONNX_FUNCTION_PRIMITIVE_REGISTRY, if enabled)
    """
    with (
        contextlib.ExitStack() as stack
    ):  # Use ExitStack to manage multiple context managers (patches).
        # Plugin primitives
        for key, plugin in PLUGIN_REGISTRY.items():
            if not isinstance(plugin, PrimitivePlugin) or not plugin.patch_info:
                continue  # Skip if not a valid PrimitivePlugin or has no patch info.
            patch_info = plugin.patch_info()
            target = patch_info["patch_targets"][0]  # Get the target object to patch.
            patch_func = patch_info["patch_function"]  # Get the patching function.
            attr = patch_info.get(
                "target_attribute", "__call__"
            )  # Get the attribute to patch (default to "__call__").
            stack.enter_context(
                _temporary_patch(target, attr, patch_func)
            )  # Apply the patch using _temporary_patch.

        # ONNX function primitives
        if allow_function_primitives:
            # The structure is modified to keep track of the original callable.
            for name, primitive in ONNX_FUNCTION_PRIMITIVE_REGISTRY.items():

                def make_patch(prim, original_call):  # Add original_call parameter
                    def patch(
                        original,
                    ):  # Not used, but kept for consistency with _temporary_patch.
                        def wrapped(self, *args):
                            # The key change here is the registry update.
                            ONNX_FUNCTION_PRIMITIVE_REGISTRY[name] = (
                                prim,
                                original_call,
                            )
                            return prim.bind(*args)

                        return wrapped

                    return patch

                cls = ONNX_FUNCTION_REGISTRY[name]
                original_call = cls.__call__
                stack.enter_context(
                    _temporary_patch(
                        cls, "__call__", make_patch(primitive, original_call)
                    )  # Pass original_call
                )

        yield


# --- Example Usage & Demonstration ---
# Mock ONNX function and registration (similar to previous examples)


def onnx_function(cls):
    name = cls.__name__
    ONNX_FUNCTION_REGISTRY[name] = cls
    primitive = Primitive(name)
    ONNX_FUNCTION_PRIMITIVE_REGISTRY[name] = primitive
    # cls._onnx_primitive = primitive  # No longer strictly needed for reconstruction
    return cls


@onnx_function
class MyOnnxFunc:
    def __call__(self, x):
        print("Original __call__")
        return x * 2


# Mock plugin registration
class MyClass:
    def __call__(self, x):
        return x + 1


def my_patch_function(original_call):
    def patched_call(self, x):
        print("Patched __call__")
        return original_call(self, x) + 10

    return patched_call


my_plugin = PrimitivePlugin(
    {"patch_targets": [MyClass()], "patch_function": my_patch_function}
)
PLUGIN_REGISTRY["my_plugin"] = my_plugin

# --- Demonstrate Monkey Patching and Callable Preservation ---

instance = MyClass()
print("Original instance result", instance(2))  # > Original instance result 3

with temporary_monkey_patches(allow_function_primitives=True):
    instance = MyClass()
    print(
        "Inside monkey patch, result:", instance(2)
    )  # > Patched __call__ \n Inside monkey patch, result: 13
    func_instance = MyOnnxFunc()
    result_inside = func_instance(5)
    print(
        f"Inside monkey patch, MyOnnxFunc result: {result_inside}"
    )  # >Inside monkey patch, MyOnnxFunc result: [5]
    # Show that the callable is available from within the context
    print(
        "Inside monkey patch, callable:",
        ONNX_FUNCTION_PRIMITIVE_REGISTRY["MyOnnxFunc"][1],
    )

# Show that the callables are restored.
instance = MyClass()
print("Outside monkey patch, result:", instance(2))  # > Outside monkey patch, result: 3
func_instance = MyOnnxFunc()
result_outside = func_instance(5)
print(
    f"Outside monkey patch, MyOnnxFunc result: {result_outside}"
)  # >Outside monkey patch, MyOnnxFunc result: 10

# --- Show Callable Recovery AFTER Monkey Patching ---

# Reconstruct after the patch
_, reconstructed_callable = ONNX_FUNCTION_PRIMITIVE_REGISTRY.get(
    "MyOnnxFunc", (None, None)
)

if reconstructed_callable:
    # Because we have the class we can reinstantiate it
    reconstructed_instance = MyOnnxFunc()
    # We replace the call with the callable stored in the registry
    import types

    setattr(
        reconstructed_instance,
        "__call__",
        types.MethodType(reconstructed_callable, reconstructed_instance),
    )
    result = reconstructed_instance(5)
    print(
        f"Result with reconstructed callable: {result}"
    )  # > Original __call__ \n Result with reconstructed callable: 10

else:
    print("Could not reconstruct callable.")
