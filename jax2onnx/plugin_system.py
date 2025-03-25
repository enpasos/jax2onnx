# file: jax2onnx/plugin_system.py

import pkgutil
import importlib
import os
from jax.extend.core import Primitive
from typing import Optional, Callable, Dict, Any, Tuple, Type, Union

PLUGIN_REGISTRY: Dict[str, Union["ExamplePlugin", "PrimitiveLeafPlugin"]] = {}

# Track ONNX-decorated modules and their plugins
ONNX_FUNCTION_REGISTRY: Dict[str, Any] = {}
ONNX_FUNCTION_PRIMITIVE_REGISTRY: Dict[str, Tuple[Primitive, Any]] = {}
ONNX_FUNCTION_PLUGIN_REGISTRY: Dict[str, "FunctionPlugin"] = {}


#####################################
# Primitive Plugin System
#####################################


class PrimitivePlugin:

    # for the monkey patch before making jaxpr
    # we need to provide the patch params
    def get_patch_params(self):
        raise NotImplementedError("Subclasses should implement this method")

    # during _process_eqn we need to get the handler
    # to do the conversion
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
        """Handles JAX to ONNX conversion; must be overridden."""
        raise NotImplementedError


class FunctionPlugin(PrimitivePlugin):
    """
    A plugin to handle the ONNX function conversion for decorated functions.
    """

    target: Any

    def __init__(self, name: str, target: Any):
        self.name = name
        self.target = target  # class or function

    def get_handler(self, converter: Any) -> Callable:
        """
        Returns the handler that processes this function.
        """
        return lambda converter, eqn, params: self._function_handler(
            converter, eqn, params
        )

    def _function_handler(self, converter, eqn, params):
        # we need to attach the function to the ONNX graph (at the parent node)
        # we need to initiate the next recursion

        # fn_name = eqn.primitive.name  # like TransformerBlock

        # we need to reconstruct the function fn and the example args
        fn = self.plugin_class  ## placeholder for the actual function
        example_args = [converter.get_name(v) for v in eqn.invars]  # placeholder

        converter.trace_jaxpr(fn, example_args)

    def get_patch_fn(self, original_call):
        self.original_call = original_call  # original call must be attached to  ?

        def patch(self, *args):
            return original_call(self, *args)

        return patch


########################################
# Decorators
########################################


def onnx_function(cls):
    """
    Decorator to mark a class as an ONNX function and register its handler plugin.
    """
    name = cls.__name__
    primitive = Primitive(name)

    primitive.def_abstract_eval(lambda x: x)  # Simple identity abstract_eval

    # Attach the primitive for introspection
    cls._onnx_primitive = primitive

    # Register the class and its primitive
    ONNX_FUNCTION_REGISTRY[name] = cls
    ONNX_FUNCTION_PRIMITIVE_REGISTRY[name] = (primitive, cls)

    # Register the plugin for this class
    plugin = FunctionPlugin(name, cls)
    ONNX_FUNCTION_PLUGIN_REGISTRY[name] = plugin

    return cls


class ExamplePlugin:
    metadata: Dict[str, Any]


def register_example(**metadata: Any) -> ExamplePlugin:
    """
    Decorator for registering an example plugin.
    The metadata must be a dictionary of attributes.
    """
    instance = ExamplePlugin()
    instance.metadata = metadata
    component = metadata.get("component")
    if isinstance(component, str):
        PLUGIN_REGISTRY[component] = instance
    return instance


def register_primitive(
    **metadata: Any,
) -> Callable[[Type[PrimitiveLeafPlugin]], Type[PrimitiveLeafPlugin]]:
    """
    Decorator to register a plugin with the given primitive and metadata.
    """
    primitive = metadata.get("jaxpr_primitive", "")

    def decorator(cls: Type[PrimitiveLeafPlugin]) -> Type[PrimitiveLeafPlugin]:
        if not issubclass(cls, PrimitiveLeafPlugin):
            raise TypeError("Plugin must subclass PrimitivePlugin")

        instance = cls()
        instance.primitive = primitive
        instance.metadata = metadata or {}

        # Register patch_info if defined in the class
        if hasattr(cls, "patch_info"):
            instance.patch_info = getattr(cls, "patch_info")

        if isinstance(primitive, str):
            PLUGIN_REGISTRY[primitive] = instance
        return cls

    return decorator


_already_imported_plugins = False


def import_all_plugins() -> None:
    """Imports all plugins dynamically from the 'plugins' directory."""
    global _already_imported_plugins
    if _already_imported_plugins:
        return  # Already imported plugins; no-op
    plugins_path = os.path.join(os.path.dirname(__file__), "plugins")
    for _, module_name, _ in pkgutil.walk_packages(
        [plugins_path], prefix="jax2onnx.plugins."
    ):
        importlib.import_module(module_name)
    _already_imported_plugins = True  # Mark as imported
