import jax
from jax.core import Primitive
from typing import Dict, Any

# Registries to track decorated modules
ONNX_FUNCTION_REGISTRY: Dict[str, Any] = {}
ONNX_PRIMITIVE_REGISTRY: Dict[str, Primitive] = {}


def onnx_function(cls):
    """
    Decorator to mark a class as an ONNX function.
    This decorator registers the class and ensures
    that it is represented as a JAX primitive during tracing.
    """
    name = cls.__name__
    ONNX_FUNCTION_REGISTRY[name] = cls

    # Create a corresponding JAX primitive
    primitive = Primitive(name)
    ONNX_PRIMITIVE_REGISTRY[name] = primitive
    cls._onnx_primitive = primitive

    def abstract_eval(x):
        # Simplified abstract eval, returns input shape
        return x

    primitive.def_abstract_eval(abstract_eval)

    original_call = cls.__call__

    def wrapped_call(self, *args):
        # Emit primitive during JAX tracing
        if not jax.core.trace_state_clean():
            return primitive.bind(*args)
        return original_call(self, *args)

    cls.__call__ = wrapped_call
    return cls


def custom_primitive_handler(converter, eqn, params):
    """
    Handles custom primitives by emitting corresponding ONNX nodes.
    This function should convert JAX primitives emitted by decorated modules
    into ONNX nodes within the converted graph.
    """
    op_type = eqn.primitive.name
    input_names = [converter.get_name(v) for v in eqn.invars]
    output_names = [converter.get_var_name(v) for v in eqn.outvars]

    node = converter.builder.create_node(
        op_type,
        inputs=input_names,
        outputs=output_names,
        name=converter.builder.get_unique_name(f"call_{op_type}"),
        domain="",  # default domain
    )

    converter.builder.add_node(node)


def find_decorated_calls_in_jaxpr(jaxpr, registry):
    """
    Finds all primitives corresponding to decorated modules in the given JAXPR.
    """
    found = {}
    for eqn in jaxpr.eqns:
        if eqn.primitive.name in registry:
            found[eqn.primitive.name] = registry[eqn.primitive.name]
    return found
