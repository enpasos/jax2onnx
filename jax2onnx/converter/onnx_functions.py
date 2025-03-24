# file: jax2onnx/converter/onnx_functions.py

from jax.core import Primitive
from typing import Dict, Any, Tuple

# Track ONNX-decorated modules and their primitives
ONNX_FUNCTION_REGISTRY: Dict[str, Any] = {}
ONNX_FUNCTION_PRIMITIVE_REGISTRY: Dict[str, Tuple[Primitive, Any]] = {}


def onnx_function(cls):
    """
    Decorator to mark a class as an ONNX function.

    Registers the class and its primitive under the same name.
    Does not modify __call__ or behavior.
    """
    name = cls.__name__
    primitive = Primitive(name)
    primitive.def_abstract_eval(lambda x: x)  # Simple identity abstract_eval

    cls._onnx_primitive = primitive  # Optional metadata for introspection

    ONNX_FUNCTION_REGISTRY[name] = cls
    ONNX_FUNCTION_PRIMITIVE_REGISTRY[name] = (primitive, cls)

    return cls


def custom_primitive_handler(converter, eqn, params):
    op_type = eqn.primitive.name
    input_names = [converter.get_name(v) for v in eqn.invars]
    output_names = [converter.get_var_name(v) for v in eqn.outvars]

    node = converter.builder.create_node(
        op_type,
        inputs=input_names,
        outputs=output_names,
        name=converter.builder.get_unique_name(f"call_{op_type}"),
        domain="jax2onnx.fn",  # Ensures linkage to ONNX FunctionProto
    )
    converter.builder.add_node(node)


def find_decorated_calls_in_jaxpr(jaxpr, registry):
    found = {}
    for eqn in jaxpr.eqns:
        if eqn.primitive.name in registry:
            found[eqn.primitive.name] = registry[eqn.primitive.name]
    return found
