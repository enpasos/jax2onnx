# file: jax2onnx/plugin_system.py
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
        return lambda node_inputs, node_outputs, params: self.to_onnx(
            converter, node_inputs, node_outputs, params
        )

    def to_onnx(
        self, converter: Any, node_inputs: Any, node_outputs: Any, params: Any
    ) -> None:
        raise NotImplementedError


class FunctionPlugin(PrimitivePlugin):
    target: Any

    def __init__(self, name: str, target: Any):
        self.name = name
        self.target = target
        self.primitive = Primitive(name)
        self.primitive.def_abstract_eval(lambda *args: args[0])
        self._orig_fn = None
        self._nested_builder = None

    def get_handler(self, converter: Any) -> Callable:
        return lambda converter, eqn, params: self._function_handler(
            converter, eqn, params
        )

    def _function_handler(self, converter, eqn, params):
        function_handler(self.name, converter, eqn, self._orig_fn, params)

    def get_patch_fn(self, primitive):
        def patch(original_call):
            def wrapped(instance, *args):
                class_name = instance.__class__.__name__
                print(f"🧠 Calling ONNX-decorated function: {class_name}")
                if class_name in ONNX_FUNCTION_PLUGIN_REGISTRY:
                    # ✅ Capture the original callable directly bound to the instance
                    ONNX_FUNCTION_PLUGIN_REGISTRY[class_name]._orig_fn = (
                        original_call.__get__(instance, type(instance))
                    )
                return primitive.bind(*args)

            return wrapped

        return patch


########################################
# Decorators
########################################


def onnx_function(cls):
    name = cls.__name__
    primitive = Primitive(name)
    primitive.def_abstract_eval(lambda x: x)

    cls._onnx_primitive = primitive

    ONNX_FUNCTION_REGISTRY[name] = cls
    ONNX_FUNCTION_PRIMITIVE_REGISTRY[name] = (primitive, cls)

    plugin = FunctionPlugin(name, cls)
    ONNX_FUNCTION_PLUGIN_REGISTRY[name] = plugin

    # PLUGIN_REGISTRY[name] = plugin

    return cls


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
