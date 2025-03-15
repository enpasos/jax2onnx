# jax2onnx/converter/plugins/plugin_registry.py
import os
import importlib
from typing import Dict, Any

# The static registry is a mapping from a primitive name (str) to its module path (str)
plugin_registry: Dict[str, str]
if os.getenv("GENERATE_PLUGIN_REGISTRY") == "1":
    plugin_registry = {}
else:
    try:
        from .plugin_registry_static import plugin_registry
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "The static plugin registry file was not found. "
            "Please run `generate_registry_and_tests.py` before running the tests."
        )


def get_all_plugins() -> Dict[str, Any]:
    """
    Load and return a mapping from a plugin's primitive name to its plugin module.
    """
    plugins: Dict[str, Any] = {}
    for prim_name, mod_path in plugin_registry.items():
        try:
            module = importlib.import_module(mod_path)
            primitive = module.get_primitive()
            # Use the primitive's name (which should match prim_name) as the key.
            plugins[primitive.name] = module
        except Exception as e:
            print(
                f"Error loading plugin for primitive {prim_name} from {mod_path}: {e}"
            )
    return plugins


if __name__ == "__main__":
    plugins = get_all_plugins()
    for prim, plugin in plugins.items():
        print(f"Loaded plugin for primitive {prim} from module {plugin.__name__}")
