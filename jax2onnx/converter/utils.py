# file: jax2onnx/converter/utils.py

import numpy as np
from onnx import TensorProto, helper
import jax.numpy as jnp

from jax.extend.core import Literal, Var

# Ensure all needed types are imported
from typing import TYPE_CHECKING, Callable

# Assuming these are correctly defined in your project:
from jax2onnx.converter.dtype_utils import numpy_dtype_to_tensorproto
from jax2onnx.converter.onnx_builder import OnnxBuilder


if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def _propagate_nested_functions(parent_builder: OnnxBuilder, sub_builder: OnnxBuilder):
    """
    Propagates nested ONNX functions from a sub-builder to a parent builder.
    Ensures nested functions are added only once.
    """
    for nested_func_name, nested_func_proto in sub_builder.functions.items():
        if nested_func_name not in parent_builder.functions:
            parent_builder.functions[nested_func_name] = nested_func_proto
            print(f"Propagated nested ONNX function: {nested_func_name}")


def function_handler(
    name: str, converter: "Jaxpr2OnnxConverter", eqn, orig_fn: Callable, params
):
    if orig_fn is None:
        raise RuntimeError(f"Original function for {name} not recorded.")

    from jax2onnx.plugin_system import get_qualified_name

    impl_key = get_qualified_name(orig_fn)
    print(f"Encountered function primitive: {impl_key}")

    instance_base_name = name.split(".")[-1]
    unique_node_name = converter.builder.get_unique_instance_name(instance_base_name)
    print(f"Generating unique ONNX node name: {unique_node_name}")

    parent_builder = converter.builder
    input_names = []
    example_args = []

    for var in eqn.invars:
        if isinstance(var, Var):
            aval = var.aval
            name = converter.get_name(var)
            input_names.append(name)
            example_args.append(
                jnp.ones(aval.shape, dtype=aval.dtype)
                if aval.shape
                else jnp.zeros((), dtype=aval.dtype)
            )
            parent_builder.register_value_info_metadata(
                name, tuple(aval.shape), aval.dtype
            )
        elif isinstance(var, Literal):
            example_args.append(var.val)
        else:
            raise TypeError(f"Unexpected input var type: {type(var)}")

    print(f"Tracing function body for: {unique_node_name}")
    sub_builder = OnnxBuilder(
        parent_builder.name_generator,
        parent_builder.opset,
        unique_node_name + "_graph",
        initializers=parent_builder.initializers,
    )
    sub_converter = converter.__class__(sub_builder)
    sub_converter.trace_jaxpr(orig_fn, example_args, preserve_graph=True)

    initializer_names = {i.name for i in parent_builder.initializers}
    used_constants = {
        inp
        for node in sub_builder.nodes
        for inp in node.input
        if inp in initializer_names
    }
    param_inputs = sorted(used_constants)
    print(f"Identified parameters (constants): {param_inputs}")

    sub_output_names = [vi.name for vi in sub_builder.outputs]

    for name in sub_output_names:
        if name not in sub_builder.value_info_metadata:
            print(f"[WARNING] Subgraph output '{name}' is missing value_info metadata.")

    internal_name = parent_builder.add_function(
        name=unique_node_name,
        sub_builder=sub_builder,
        param_input_names=param_inputs,
    )

    parent_builder.merge_value_info_metadata_from(sub_builder)

    for i, var in enumerate(eqn.outvars):
        parent_name = converter.get_var_name(var)
        sub_name = sub_converter.get_name(var)

        print(
            f"[DEBUG] Mapping subgraph output '{sub_name}' → parent output '{parent_name}'"
        )

        if parent_name in parent_builder.value_info:
            continue

        shape_dtype = sub_builder.value_info_metadata.get(sub_name)
        origin = None

        if not shape_dtype:
            print(f"[WARN] Missing metadata for '{sub_name}'")
            if hasattr(var, "aval"):
                shape = tuple(var.aval.shape)
                dtype = numpy_dtype_to_tensorproto(var.aval.dtype)
                shape_dtype = (shape, dtype)
                origin = "recovered"
                print(f"[RECOVER] Inferred metadata from var.aval for '{parent_name}'")

                sub_builder.register_value_info_metadata(sub_name, shape, dtype, origin)
                if all(vi.name != sub_name for vi in sub_builder.value_info):
                    sub_builder.add_value_info(sub_name, shape, dtype)
            elif i < len(sub_output_names):
                sub_name = sub_output_names[i]
                shape_dtype = sub_builder.value_info_metadata.get(sub_name)
                origin = "fallback"
                print(
                    f"[FALLBACK] Using positional subgraph output '{sub_name}' for '{parent_name}'"
                )

        if shape_dtype:
            shape, dtype = shape_dtype
            if sub_name in getattr(sub_builder, "value_info_origin", {}):
                origin = sub_builder.value_info_origin[sub_name]
                print(
                    f"[TRACE] Output '{parent_name}' registered from subgraph with origin: {origin}"
                )
            parent_builder.register_value_info_metadata(
                parent_name, shape, dtype, origin
            )
            parent_builder.add_value_info(parent_name, shape, dtype)
        else:
            print(
                f"[❌] Could not determine shape/type metadata for output '{parent_name}'"
            )
            raise RuntimeError(
                f"Missing shape/type metadata for output '{parent_name}'"
            )

    print("[DEBUG] Final parent value_info entries:")
    for vi in parent_builder.value_info:
        print(
            f"  - {vi.name}: shape={list(vi.type.tensor_type.shape.dim)}, dtype={vi.type.tensor_type.elem_type}"
        )

    print("[DEBUG] Current registered outputs:")
    for vi in parent_builder.outputs:
        print(f"  - {vi.name}")

    _propagate_nested_functions(parent_builder, sub_builder)
    print(f"Finished tracing function body: {unique_node_name}")

    call_inputs = input_names + param_inputs
    output_names = [converter.get_var_name(v) for v in eqn.outvars]

    parent_builder.add_function_call_node(
        function_name=unique_node_name,
        input_names=call_inputs,
        output_names=output_names,
        node_name=unique_node_name,
        user_display_name=name,
    )

    print(f"Added call node for: {internal_name}")
