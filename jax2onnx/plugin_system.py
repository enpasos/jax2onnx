# file: jax2onnx/plugin_system.py
import pkgutil
import importlib
import os
from typing import Optional, Callable, Dict, Any, TYPE_CHECKING

PLUGIN_REGISTRY = {}

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


class PrimitivePlugin:
    """Base class for ONNX conversion plugins."""

    primitive: str
    metadata: Dict[str, Any]
    patch_info: Optional[Callable] = None  # Method returning patch details

    def get_handler(self, converter):
        return lambda node_inputs, node_outputs, params: self.to_onnx(
            converter, node_inputs, node_outputs, params
        )

    def to_onnx(self, converter, node_inputs, node_outputs, params):
        """Handles JAX to ONNX conversion; must be overridden."""
        raise NotImplementedError


def register_plugin(**metadata: Optional[Dict[str, Any]]):
    """
    Decorator to register a plugin with the given primitive and metadata.
    """
    primitive = metadata.get("jaxpr_primitive")

    def decorator(cls):
        if not issubclass(cls, PrimitivePlugin):
            raise TypeError("Plugin must subclass PrimitivePlugin")

        instance = cls()
        instance.primitive = primitive
        instance.metadata = metadata or {}

        # Register patch_info if defined in the class
        if hasattr(cls, "patch_info"):
            instance.patch_info = getattr(cls, "patch_info")

        PLUGIN_REGISTRY[primitive] = instance
        return cls

    return decorator


_already_imported_plugins = False


def import_all_plugins():
    global _already_imported_plugins
    if _already_imported_plugins:
        return  # Already imported plugins; no-op
    plugins_path = os.path.join(os.path.dirname(__file__), "plugins")
    for _, module_name, _ in pkgutil.walk_packages(
        [plugins_path], prefix="jax2onnx.plugins."
    ):
        importlib.import_module(module_name)
    _already_imported_plugins = True  # Mark as imported
