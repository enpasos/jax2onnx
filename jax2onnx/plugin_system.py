# file: jax2onnx/plugin_system.py
import functools
import inspect
import pkgutil
import importlib
import os
from jax.extend.core import Primitive
from typing import Optional, Callable, Dict, Any, Tuple, Type, Union
from abc import ABC, abstractmethod

from jax2onnx.converter.utils import function_handler

from jax2onnx.converter.name_generator import get_qualified_name

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


class PrimitivePlugin(ABC):

    @abstractmethod
    def get_patch_params(self):
        """Retrieve patch parameters for the plugin."""
        pass

    @abstractmethod
    def get_handler(self, converter: Any) -> Callable:
        """Retrieve the handler function for the plugin."""
        pass


class PrimitiveLeafPlugin(PrimitivePlugin):
    primitive: str
    metadata: Dict[str, Any]
    patch_info: Optional[Callable[[], Dict[str, Any]]] = None

    def get_patch_params(self):
        if not self.patch_info:
            raise ValueError("patch_info is not defined for this plugin.")
        patch_info = self.patch_info()
        target = patch_info["patch_targets"][0]
        patch_func = patch_info["patch_function"]
        attr = patch_info.get("target_attribute", "__call__")
        return target, attr, patch_func

    def get_handler(self, converter: Any) -> Callable:
        return lambda converter, eqn, params: self.to_onnx(
            converter, eqn.invars, eqn.outvars, params
        )

    @abstractmethod
    def to_onnx(
        self, converter: Any, node_inputs: Any, node_outputs: Any, params: Any
    ) -> None:
        """Convert the plugin to ONNX format."""
        pass


class FunctionPlugin(PrimitivePlugin):
    def __init__(self, name: str, target: Any):
        self.name = name
        self.target = target
        self.primitive = Primitive(name)
        self.primitive.def_abstract_eval(self.abstract_eval_with_kwargs)
        self.primitive.def_impl(self.primitive_impl)
        self._orig_fn = None

    def to_function_proto(self, context, builder, inputs, outputs):
        # Generate a unique name for this function instance
        function_name = context.next_function_name(self.target.__name__)

        # Start building the FunctionProto
        builder.start_function(function_name, inputs, outputs)

        # The actual conversion logic would go here...
        # e.g., trace self.target, emit intermediate nodes, etc.

        return builder.end_function()

    def abstract_eval_with_kwargs(self, *args, **kwargs):
        return args[0]

    def primitive_impl(self, *args, **kwargs):
        if self._orig_fn is None:
            raise ValueError("Original function not set for primitive!")
        return self._orig_fn(*args, **kwargs)

    def get_patch_fn(self, primitive):
        def patch(original_call):
            sig = inspect.signature(original_call)
            params = list(sig.parameters.keys())

            @functools.wraps(original_call)
            def wrapped(*args, **kwargs):
                expects_self = params and params[0] == "self"

                if expects_self:
                    instance = args[0]
                    qualname = get_qualified_name(instance.__class__)
                    if qualname in ONNX_FUNCTION_PLUGIN_REGISTRY:
                        plugin = ONNX_FUNCTION_PLUGIN_REGISTRY[qualname]
                        plugin._orig_fn = original_call.__get__(
                            instance, type(instance)
                        )
                    return primitive.bind(*args[1:], **kwargs)
                else:
                    # Non-class function
                    qualname = self.name  # self.name is already qualified
                    if qualname in ONNX_FUNCTION_PLUGIN_REGISTRY:
                        plugin = ONNX_FUNCTION_PLUGIN_REGISTRY[qualname]
                        plugin._orig_fn = original_call
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
        function_handler(self.name, converter, eqn, self._orig_fn, params)


########################################
# Decorators
########################################


def onnx_function(target):
    name = get_qualified_name(target)
    primitive = Primitive(name)
    primitive.def_abstract_eval(lambda x: x)

    target._onnx_primitive = primitive

    ONNX_FUNCTION_REGISTRY[name] = target
    ONNX_FUNCTION_PRIMITIVE_REGISTRY[name] = (primitive, target)

    plugin = FunctionPlugin(name, target)
    ONNX_FUNCTION_PLUGIN_REGISTRY[name] = plugin

    return target


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
