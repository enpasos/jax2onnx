# file: jax2onnx/converter/onnx_functions.py

from jax.core import Primitive
from typing import Dict, Any

# ✅ Registry of decorated classes
ONNX_FUNCTION_REGISTRY: Dict[str, Any] = {}

# ✅ Registry of corresponding JAX primitives (created per class)
ONNX_FUNCTION_PRIMITIVE_REGISTRY: Dict[str, Primitive] = {}


def onnx_function(cls):
    """
    Decorator to register a class as an ONNX function.

    This decorator does NOT override any behavior. It simply:
    - Registers the class name to the ONNX function registry
    - Creates and registers a corresponding JAX primitive
    - Stores the primitive on the class for future use by monkey-patching

    Example:
        @onnx_function
        class MyBlock(nnx.Module):
            ...
    """
    name = cls.__name__

    # ✅ Register the class
    ONNX_FUNCTION_REGISTRY[name] = cls

    # ✅ Create and register a JAX primitive (but don't bind it)
    primitive = Primitive(name)
    primitive.multiple_results = False  # optional, depending on your use case

    # Simplified abstract_eval for tracing
    def abstract_eval(x):
        return x

    primitive.def_abstract_eval(abstract_eval)

    # ✅ Store primitive on class for later monkey-patching
    ONNX_FUNCTION_PRIMITIVE_REGISTRY[name] = primitive
    cls._onnx_primitive = primitive

    return cls


def custom_primitive_handler(converter, eqn, params):
    """
    Default ONNX handler for decorated ONNX function calls.

    This is invoked when a traced function call (e.g., MyBlock(x)) has been replaced
    by a primitive (e.g., primitive.bind(x)) during tracing.
    """
    op_type = eqn.primitive.name
    input_names = [converter.get_name(v) for v in eqn.invars]
    output_names = [converter.get_var_name(v) for v in eqn.outvars]

    node = converter.builder.create_node(
        op_type=op_type,
        inputs=input_names,
        outputs=output_names,
        name=converter.builder.get_unique_name(f"call_{op_type}"),
        domain="",  # could be custom domain if desired
    )
    converter.builder.add_node(node)


def find_decorated_calls_in_jaxpr(jaxpr, registry):
    """
    Find all occurrences of decorated ONNX functions (by primitive name) in the given JAXPR.

    This is used to determine which classes still need to be traced and converted.
    """
    found = {}
    for eqn in jaxpr.eqns:
        if eqn.primitive.name in registry:
            found[eqn.primitive.name] = registry[eqn.primitive.name]
    return found
