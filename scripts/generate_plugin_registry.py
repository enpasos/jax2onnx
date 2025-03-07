import pkgutil
import importlib
import os
from typing import Dict, Any

# Set the environment flag so that plugin_registry.py doesn't try to load a static file.
os.environ["GENERATE_PLUGIN_REGISTRY"] = "1"

# Update the root package to the new plugins location.
ROOT_PACKAGE = "jax2onnx.converter.plugins"


def generate_registry() -> Dict[Any, str]:
    """
    Scan the plugins package and return a mapping from a plugin's primitive
    name to the plugin's module path.
    """
    registry: Dict[Any, str] = {}
    package = importlib.import_module(ROOT_PACKAGE)
    for importer, modname, ispkg in pkgutil.walk_packages(
        package.__path__, prefix=package.__name__ + "."
    ):
        try:
            module = importlib.import_module(modname)
            # Check if the module provides the required interface.
            if all(
                hasattr(module, func)
                for func in ("get_primitive", "get_handler", "get_metadata")
            ):
                primitive = module.get_primitive()
                # Use the primitive's "name" attribute if available; otherwise, fallback to str(primitive)
                key = primitive.name if hasattr(primitive, "name") else str(primitive)
                registry[key] = modname
        except Exception as e:
            print(f"Error importing module {modname}: {e}")
    return registry


def write_registry_file(registry: Dict[Any, str], output_file: str) -> None:
    """
    Write the registry dictionary to a Python file as a variable.
    """
    with open(output_file, "w") as f:
        f.write("# This file is auto-generated. Do not modify manually.\n")
        f.write("plugin_registry = {\n")
        for prim_name, mod_path in registry.items():
            f.write(f"    '{prim_name}': '{mod_path}',\n")
        f.write("}\n")


if __name__ == "__main__":
    registry = generate_registry()
    # Compute the output directory:
    base_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "jax2onnx", "converter", "plugins"
        )
    )
    print("Computed base_dir:", base_dir)

    output_path = os.path.join(base_dir, "plugin_registry_static.py")
    print("Output path:", output_path)

    write_registry_file(registry, output_path)
    print(f"Plugin registry generated and written to {output_path}")
