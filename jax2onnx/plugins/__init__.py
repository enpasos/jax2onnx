import importlib
import pkgutil
import sys


# Dynamically import all submodules in the `jax2onnx.plugins` package
def _import_all_plugins():
    package_name = __name__  # 'jax2onnx.plugins'
    package = sys.modules[package_name]  # Get the module object

    for _, module_name, is_pkg in pkgutil.walk_packages(
        package.__path__, package_name + "."
    ):
        if not is_pkg:  # Avoid importing directories, only modules
            importlib.import_module(module_name)


# Execute auto-importing
_import_all_plugins()
