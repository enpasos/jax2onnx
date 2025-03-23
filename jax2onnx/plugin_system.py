# file: jax2onnx/plugin_system.py

import pkgutil
import importlib
import os
import inspect
from typing import Optional, Callable, Dict, Any, Type, Union

PLUGIN_REGISTRY: Dict[str, Union["ExamplePlugin", "PrimitivePlugin"]] = {}


class PrimitivePlugin:
    primitive: str
    metadata: Dict[str, Any]
    patch_info: Optional[Callable[[], Dict[str, Any]]] = None

    def get_handler(self, converter: Any) -> Callable:
        sig = inspect.signature(self.to_onnx)
        param_names = list(sig.parameters)

        # Remove first param if it looks like a method-bound instance (self/s/etc)
        if len(param_names) >= 1 and param_names[0] not in {
            "converter",
            "node_inputs",
            "eqn",
        }:
            param_names = param_names[1:]

        # Normalize by length and position, not exact parameter names
        if len(param_names) == 3:
            # Could be (converter, eqn, params) or (converter, node_inputs, node_outputs)
            # if param_names[0] == "converter":
            #     return lambda converter, eqn, params: self.to_onnx(
            #         converter, eqn, params
            #     )
            # else:
            return lambda converter, eqn, params: self.to_onnx(
                converter, eqn.invars, eqn.outvars, params
            )

        # elif len(param_names) == 2:
        # Old style: (node_inputs, node_outputs, params)
        # return lambda converter, eqn, params: self.to_onnx(
        #     eqn.invars, eqn.outvars, params
        # )

        # elif len(param_names) == 1 and param_names[0] == "eqn":
        #     return lambda converter, eqn, params: self.to_onnx(eqn)

        raise TypeError(
            f"Unsupported to_onnx() signature in {type(self).__name__}. Expected one of:\n"
            f"  (converter, eqn, params)\n"
            f"  (converter, node_inputs, node_outputs, params)\n"
            f"  (node_inputs, node_outputs, params)\n"
            f"  or (eqn)\n"
            f"Got: {param_names}"
        )

    def to_onnx(
        self, converter: Any, node_inputs: Any, node_outputs: Any, params: Any
    ) -> None:
        """Handles JAX to ONNX conversion; must be overridden."""
        raise NotImplementedError


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
) -> Callable[[Type[PrimitivePlugin]], Type[PrimitivePlugin]]:
    """
    Decorator to register a plugin with the given primitive and metadata.
    """
    primitive = metadata.get("jaxpr_primitive", "")

    def decorator(cls: Type[PrimitivePlugin]) -> Type[PrimitivePlugin]:
        if not issubclass(cls, PrimitivePlugin):
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
