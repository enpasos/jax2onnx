# jax2onnx/onnx_functions.py

import jax
import jax.numpy as jnp
import onnx
from onnx import helper
from jax.core import Primitive
from typing import Dict, Any, List, Callable

ONNX_FUNCTION_REGISTRY: Dict[str, Any] = {}
ONNX_FUNCTION_MODELS: Dict[str, onnx.ModelProto] = {}
ONNX_PRIMITIVE_REGISTRY: Dict[str, Primitive] = {}


def onnx_function(cls):
    name = cls.__name__
    ONNX_FUNCTION_REGISTRY[name] = cls

    primitive = Primitive(name)
    cls._onnx_primitive = primitive
    ONNX_PRIMITIVE_REGISTRY[name] = primitive  # ✅ Register it here

    def abstract_eval(*args):
        return args[0]  # Or more sophisticated shape inference

    primitive.def_abstract_eval(abstract_eval)

    original_call = cls.__call__

    def wrapped_call(self, *args):
        if not jax.core.trace_state_clean():
            return primitive.bind(*args)
        return original_call(self, *args)

    cls.__call__ = wrapped_call
    return cls


def custom_primitive_handler(converter, eqn, params):
    op_type = eqn.primitive.name
    input_names = [converter.get_name(v) for v in eqn.invars]
    output_names = [converter.get_var_name(v) for v in eqn.outvars]
    node = helper.make_node(
        op_type=op_type,
        inputs=input_names,
        outputs=output_names,
        name=converter.get_unique_name(f"call_{op_type}"),
        domain="",
    )
    converter.add_node(node)


def trace_to_jaxpr(fn: Callable, input_shapes: List[Any]):
    example_args = [jnp.zeros([2 if d == "B" else d for d in s]) for s in input_shapes]
    closed_jaxpr = jax.make_jaxpr(fn)(*example_args)
    print(closed_jaxpr)
    return closed_jaxpr.jaxpr


def find_decorated_calls_in_jaxpr(jaxpr, registry):
    found = {}
    for eqn in jaxpr.eqns:
        if eqn.primitive.name in registry:
            found[eqn.primitive.name] = registry[eqn.primitive.name]
    return found


def enhanced_to_onnx(top_module, input_shapes):
    from jax2onnx.converter.converter import to_onnx

    pending_registry = ONNX_FUNCTION_REGISTRY.copy()

    while pending_registry:
        jaxpr = trace_to_jaxpr(top_module, input_shapes)
        decorated_calls = find_decorated_calls_in_jaxpr(jaxpr, pending_registry)

        if not decorated_calls:
            break

        for func_name, cls in decorated_calls.items():
            instance = cls()

            # Remove the current function from registry temporarily
            del ONNX_FUNCTION_REGISTRY[func_name]
            sub_model = to_onnx(
                fn=instance, input_shapes=input_shapes, model_name=func_name
            )
            ONNX_FUNCTION_MODELS[func_name] = sub_model
            # Re-register it to avoid issues downstream
            ONNX_FUNCTION_REGISTRY[func_name] = cls
            del pending_registry[func_name]

    final_model = to_onnx(
        fn=top_module, input_shapes=input_shapes, model_name="top_model"
    )

    for name, model in ONNX_FUNCTION_MODELS.items():
        g = model.graph
        f = helper.make_function(
            name=name,
            inputs=g.input,
            outputs=g.output,
            nodes=g.node,
            opset_imports=model.opset_import,
            domain="",
        )
        final_model.functions.append(f)

    print("Defined ONNX functions in final_model:")
    for f in final_model.functions:
        print(" - Function:", f.name)

    return final_model
