# jax2onnx/converter/conversion.py

from jax2onnx.plugin_system import PLUGIN_REGISTRY
from jax2onnx.legacy_plugins import LEGACY_PLUGIN_REGISTRY


def get_plugin(primitive):
    """Retrieve the plugin for a given JAX primitive."""
    if primitive in PLUGIN_REGISTRY:
        return PLUGIN_REGISTRY[primitive]
    elif primitive in LEGACY_PLUGIN_REGISTRY:
        return LEGACY_PLUGIN_REGISTRY[primitive]
    else:
        raise ValueError(f"No plugin found for primitive: {primitive}")
