# file: jax2onnx/plugin_system.py
import pkgutil
import importlib
import os

PLUGIN_REGISTRY = {}


def register_plugin(primitive=None, metadata=None, patch_fn=None):
    def wrapper(cls):
        instance = cls()
        instance.metadata = metadata or {}
        instance.patch_fn = patch_fn
        PLUGIN_REGISTRY[primitive] = instance
        return cls

    return wrapper


_already_imported_plugins = False


def import_all_plugins():
    global _already_imported_plugins
    if _already_imported_plugins:
        return  # No-op if we've already imported everything
    plugins_path = os.path.join(os.path.dirname(__file__), "plugins")
    for finder, name, ispkg in pkgutil.walk_packages(
        [plugins_path], prefix="jax2onnx.plugins."
    ):
        importlib.import_module(name)


class PrimitivePlugin:
    """Base class for all ONNX conversion plugins."""

    primitive = None

    def abstract_eval(self, *args, **kwargs):
        """Handles shape inference; to be overridden in subclasses."""
        raise NotImplementedError

    def to_onnx(self, node, graph, inputs, outputs):
        """Handles transformation from JAX to ONNX; to be overridden in subclasses."""
        raise NotImplementedError
