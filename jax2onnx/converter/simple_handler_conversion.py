import jax
import onnx
from typing import Any
import jax.numpy as jnp
from onnx import helper
from jax2onnx.converter.converter import Jaxpr2OnnxConverter
from jax2onnx.converter.patch_utils import temporary_monkey_patches
from jax2onnx.converter.optimize_onnx_graph import improve_onnx_model
from jax2onnx.converter.onnx_functions import (
    ONNX_FUNCTION_REGISTRY,
    find_decorated_calls_in_jaxpr,
)


def trace_to_jaxpr(fn, input_shapes):
    example_args = [jnp.zeros([2 if d == "B" else d for d in s]) for s in input_shapes]
    closed_jaxpr = jax.make_jaxpr(fn)(*example_args)
    print(closed_jaxpr)
    return closed_jaxpr.jaxpr


def hierarchical_to_onnx(fn, input_shapes, model_name="jax_model", opset=21):
    pending_registry = ONNX_FUNCTION_REGISTRY.copy()
    extracted_models = {}

    while pending_registry:
        jaxpr = trace_to_jaxpr(fn, input_shapes)
        decorated_calls = find_decorated_calls_in_jaxpr(jaxpr, pending_registry)

        if not decorated_calls:
            break

        for func_name, cls in decorated_calls.items():
            instance = cls()

            # Temporarily unregister to isolate nested calls
            del ONNX_FUNCTION_REGISTRY[func_name]

            sub_model = to_onnx(
                instance, input_shapes, model_name=func_name, opset=opset
            )
            extracted_models[func_name] = sub_model

            # Re-register after extraction
            ONNX_FUNCTION_REGISTRY[func_name] = cls
            del pending_registry[func_name]

    final_model = to_onnx(fn, input_shapes, model_name=model_name, opset=opset)

    # Inject extracted ONNX functions into final ONNX model
    for name, model in extracted_models.items():
        graph = model.graph
        onnx_func = onnx.FunctionProto()
        onnx_func.name = name
        onnx_func.domain = ""  # default domain
        onnx_func.input.extend([i.name for i in graph.input])
        onnx_func.output.extend([o.name for o in graph.output])
        onnx_func.node.extend(graph.node)

        final_model.functions.append(onnx_func)

    return final_model


def to_onnx(
    fn: Any,
    input_shapes: Any,
    model_name: str = "jax_model",
    include_intermediate_shapes: bool = False,
    opset: int = 21,
) -> onnx.ModelProto:

    from jax2onnx.converter.onnx_functions import (
        ONNX_PRIMITIVE_REGISTRY,
        custom_primitive_handler,
    )
    from jax2onnx.plugin_system import PLUGIN_REGISTRY, PrimitivePlugin

    # Handle symbolic dimensions
    if any("B" in shape for shape in input_shapes):
        include_intermediate_shapes = False
        print(
            "Dynamic batch dimensions detected. Setting include_intermediate_shapes=False"
        )

    # Generate example inputs
    def replace_B(s):
        return [2 if d == "B" else d for d in s]

    example_args = [jnp.zeros(replace_B(s)) for s in input_shapes]

    # Dry-run to patch layers
    with temporary_monkey_patches():
        jax.make_jaxpr(fn)(*example_args)

    # Create converter
    converter = Jaxpr2OnnxConverter()

    # Register handlers from plugin system
    for name, plugin in PLUGIN_REGISTRY.items():
        if isinstance(plugin, PrimitivePlugin):
            converter.primitive_handlers[name] = plugin.get_handler(converter)

    # Register handlers for ONNX primitives/functions
    for name, primitive in ONNX_PRIMITIVE_REGISTRY.items():
        converter.primitive_handlers[name] = (
            lambda conv, eqn, params, n=name: custom_primitive_handler(
                conv, eqn, params
            )
        )

    # Trace and convert
    converter.trace_jaxpr(fn, example_args)

    # Annotate input shapes
    for tensor, input_shape in zip(converter.builder.inputs, input_shapes):
        tensor_shape = tensor.type.tensor_type.shape.dim
        for idx, dim in enumerate(input_shape):
            if dim == "B":
                tensor_shape[idx].dim_param = "B"

    # Annotate output shapes
    batch_dims = {
        idx for shape in input_shapes for idx, dim in enumerate(shape) if dim == "B"
    }
    for tensor in converter.builder.outputs:
        tensor_shape = tensor.type.tensor_type.shape.dim
        for idx in batch_dims:
            if idx < len(tensor_shape):
                tensor_shape[idx].dim_param = "B"

    # Clean up value_info if not needed
    value_info = converter.builder.value_info if include_intermediate_shapes else []

    # Prune unused initializers
    used_inputs = {i for node in converter.builder.nodes for i in node.input}
    converter.builder.initializers = [
        init for init in converter.builder.initializers if init.name in used_inputs
    ]

    # Build graph
    graph = helper.make_graph(
        nodes=converter.builder.nodes,
        name=model_name,
        inputs=converter.builder.inputs,
        outputs=converter.builder.outputs,
        initializer=converter.builder.initializers,
        value_info=value_info,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model = improve_onnx_model(model)
    return model


def _validate_input_shapes(input_shapes):
    for shape in input_shapes:
        assert isinstance(shape, tuple), "Each input shape must be a tuple"


def _shape_with_example_batch(shape, example_batch=2):
    return tuple(example_batch if d == "B" else d for d in shape)
