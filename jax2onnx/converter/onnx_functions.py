# file: jax2onnx/converter/onnx_functions.py

from jax.core import Primitive
from typing import Dict, Any, Tuple, Callable

# Track ONNX-decorated modules and their plugins
ONNX_FUNCTION_REGISTRY: Dict[str, Any] = {}
ONNX_FUNCTION_PRIMITIVE_REGISTRY: Dict[str, Tuple[Primitive, Any]] = {}
ONNX_FUNCTION_PLUGIN_REGISTRY: Dict[str, "FunctionPlugin"] = {}


class FunctionPlugin:
    """
    A plugin to handle the ONNX function conversion for decorated functions.
    """

    def __init__(self, name: str, plugin_class: Any):
        self.name = name
        self.plugin_class = plugin_class

    def get_handler(self) -> Callable:
        """
        Returns the handler that processes this function.
        """
        return lambda conv, eqn, params: self._function_handler(conv, eqn, params)

    def _function_handler(self, converter, eqn, params):
        """
        The handler that will convert the ONNX function during the ONNX conversion process.
        This handler adds the function to the ONNX graph and ensures recursion for child functions.
        """
        op_type = eqn.primitive.name
        input_names = [converter.get_name(v) for v in eqn.invars]
        output_names = [converter.get_var_name(v) for v in eqn.outvars]

        node = converter.builder.create_node(
            op_type,
            inputs=input_names,
            outputs=output_names,
            name=converter.builder.get_unique_name(f"call_{op_type}"),
            domain="jax2onnx.fn",
        )

        converter.builder.add_node(node)

        # Recursively handle decorated function calls inside this function
        for child_func_name in find_decorated_calls_in_jaxpr(
            eqn, ONNX_FUNCTION_REGISTRY
        ):
            handler = ONNX_FUNCTION_PLUGIN_REGISTRY[child_func_name].get_handler()
            handler(converter, eqn, params)


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


def find_decorated_calls_in_jaxpr(eqn, registry):
    """
    Detects calls to decorated functions within the JAXPR equation.
    Returns a list of function names that are decorated with @onnx_function.
    """
    found = []
    if eqn.primitive.name in registry:
        found.append(eqn.primitive.name)
    return found
