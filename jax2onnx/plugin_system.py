# file: jax2onnx/plugin_system.py
import pkgutil
import importlib
import os
from typing import Optional, Callable, Dict, Any

PLUGIN_REGISTRY = {}


class PrimitivePlugin:
    """Base class for ONNX conversion plugins."""

    primitive: str
    metadata: Dict[str, Any]
    patch_info: Optional[Callable] = None  # Method returning patch details

    def abstract_eval(self, *args, **kwargs):
        """Handles shape inference; must be overridden."""
        raise NotImplementedError

    def to_onnx(self, converter, node_inputs, node_outputs, params):
        """Handles JAX to ONNX conversion; must be overridden."""
        raise NotImplementedError


def register_plugin(primitive: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Decorator to register a plugin with the given primitive and metadata.
    """

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
