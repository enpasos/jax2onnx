# file: jax2onnx/plugin_system.py
import functools
import inspect
import pkgutil
import importlib
import os
from jax.extend.core import Primitive
from typing import Optional, Callable, Dict, Any, Tuple, Type, Union

from jax2onnx.converter.utils import function_handler

PLUGIN_REGISTRY: Dict[
    str, Union["FunctionPlugin", "ExamplePlugin", "PrimitiveLeafPlugin"]
] = {}

# Track ONNX-decorated modules and their plugins
ONNX_FUNCTION_REGISTRY: Dict[str, Any] = {}
ONNX_FUNCTION_PRIMITIVE_REGISTRY: Dict[str, Tuple[Primitive, Any]] = {}
ONNX_FUNCTION_PLUGIN_REGISTRY: Dict[str, "FunctionPlugin"] = {}


#####################################
# Primitive Plugin System
#####################################


class PrimitivePlugin:

    def get_patch_params(self):
        raise NotImplementedError("Subclasses should implement this method")

    def get_handler(self, converter: Any) -> Callable:
        raise NotImplementedError("Subclasses should implement this method")


class PrimitiveLeafPlugin(PrimitivePlugin):
    primitive: str
    metadata: Dict[str, Any]
    patch_info: Optional[Callable[[], Dict[str, Any]]] = None

    def get_patch_params(self):
        patch_info = self.patch_info()
        target = patch_info["patch_targets"][0]
        patch_func = patch_info["patch_function"]
        attr = patch_info.get("target_attribute", "__call__")
        return target, attr, patch_func

    def get_handler(self, converter: Any) -> Callable:
        return lambda converter, eqn, params: self.to_onnx(
            converter, eqn.invars, eqn.outvars, params
        )

    def to_onnx(
        self, converter: Any, node_inputs: Any, node_outputs: Any, params: Any
    ) -> None:
        raise NotImplementedError


class FunctionPlugin(PrimitivePlugin):
    def __init__(self, name: str, target: Any):
        self.name = name
        self.target = target
        self.primitive = Primitive(name)
        self.primitive.def_abstract_eval(self.abstract_eval_with_kwargs)
        self.primitive.def_impl(self.primitive_impl)
        self._orig_fn = None

    def abstract_eval_with_kwargs(self, *args, **kwargs):
        return args[0]

    def primitive_impl(self, *args, **kwargs):
        if self._orig_fn is None:
            raise ValueError("Original function not set for primitive!")
        return self._orig_fn(*args, **kwargs)

    def get_patch_fn(self, primitive):
        def patch(original_call):
            # if args2 or kwargs2:
            #     raise ValueError("No args or kwargs expected for this function plugin")
            sig = inspect.signature(original_call)
            params = list(sig.parameters.keys())

            @functools.wraps(original_call)
            def wrapped(*args, **kwargs):
                # Check if the original callable expects 'self'
                expects_self = params and params[0] == "self"

                if expects_self:
                    # It's a method - args[0] is 'self'
                    instance = args[0]
                    class_name = instance.__class__.__name__
                    if class_name in ONNX_FUNCTION_PLUGIN_REGISTRY:
                        ONNX_FUNCTION_PLUGIN_REGISTRY[class_name]._orig_fn = (
                            original_call.__get__(instance, type(instance))
                        )
                    # Do NOT forward 'self' to primitive!
                    return primitive.bind(*args[1:], **kwargs)
                else:
                    # Standalone function, no 'self'

                    ONNX_FUNCTION_PLUGIN_REGISTRY[self.name]._orig_fn = original_call
                    # self._orig_fn = original_call
                    return primitive.bind(*args, **kwargs)

            return wrapped

        return patch

    def get_patch_params(self):
        return (self.target, "__call__", self.get_patch_fn(self.primitive))

    # Add this implementation
    def get_handler(self, converter: Any) -> Callable:
        return lambda conv, eqn, params: self._function_handler(
            converter, conv, eqn, params
        )

    def _function_handler(self, plugin_converter, converter, eqn, params):
        # Implementation for how this function plugin is handled during conversion
        function_handler(self.name, converter, eqn, self._orig_fn, params)


########################################
# Decorators
########################################


def onnx_function(target):
    name = target.__name__
    primitive = Primitive(name)
    primitive.def_abstract_eval(lambda x: x)

    target._onnx_primitive = primitive

    ONNX_FUNCTION_REGISTRY[name] = target
    ONNX_FUNCTION_PRIMITIVE_REGISTRY[name] = (primitive, target)

    plugin = FunctionPlugin(name, target)
    ONNX_FUNCTION_PLUGIN_REGISTRY[name] = plugin

    # PLUGIN_REGISTRY[name] = plugin

    return target


# def onnx_function(target):
#     print(f"Registering ONNX function: {target.__name__}")
#     plugin = FunctionPlugin(name=target.__name__, target=target)
#     ONNX_FUNCTION_PLUGIN_REGISTRY[target.__name__] = plugin

#     if isinstance(target, type):
#         # For classes, patch __call__
#         return target
#     elif callable(target):
#         primitive = plugin.primitive

#         @functools.wraps(target)
#         def wrapped_function(*args, **kwargs):
#             return primitive.bind(*args, **kwargs)

#         primitive.def_impl(target)

#         # Set an abstract_eval if necessary
#         primitive.def_abstract_eval(lambda *args, **kwargs: args[0])

#         return wrapped_function
#     else:
#         raise TypeError("onnx_function decorator expects a class or function.")


class ExamplePlugin:
    metadata: Dict[str, Any]


def register_example(**metadata: Any) -> ExamplePlugin:
    instance = ExamplePlugin()
    instance.metadata = metadata
    component = metadata.get("component")
    if isinstance(component, str):
        PLUGIN_REGISTRY[component] = instance
    return instance


def register_primitive(
    **metadata: Any,
) -> Callable[[Type[PrimitiveLeafPlugin]], Type[PrimitiveLeafPlugin]]:
    primitive = metadata.get("jaxpr_primitive", "")

    def decorator(cls: Type[PrimitiveLeafPlugin]) -> Type[PrimitiveLeafPlugin]:
        if not issubclass(cls, PrimitiveLeafPlugin):
            raise TypeError("Plugin must subclass PrimitivePlugin")

        instance = cls()
        instance.primitive = primitive
        instance.metadata = metadata or {}

        if hasattr(cls, "patch_info"):
            instance.patch_info = getattr(cls, "patch_info")

        if isinstance(primitive, str):
            PLUGIN_REGISTRY[primitive] = instance
        return cls

    return decorator


_already_imported_plugins = False


def import_all_plugins() -> None:
    global _already_imported_plugins
    if _already_imported_plugins:
        return
    plugins_path = os.path.join(os.path.dirname(__file__), "plugins")
    for _, module_name, _ in pkgutil.walk_packages(
        [plugins_path], prefix="jax2onnx.plugins."
    ):
        importlib.import_module(module_name)
    _already_imported_plugins = True
