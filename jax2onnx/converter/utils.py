# file: jax2onnx/converter/utils.py

import jax.numpy as jnp
from jax.core import ShapedArray
from jax.extend.core import Literal


# Ensure all needed types are imported
from typing import TYPE_CHECKING, Callable

# Assuming these are correctly defined in your project:
from jax2onnx.converter.dtype_utils import (
    tensorproto_dtype_to_numpy,
)
from jax2onnx.converter.name_generator import get_qualified_name
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

    impl_key = get_qualified_name(orig_fn)
    print(f"Encountered function primitive: {impl_key}")

    instance_base_name = name.split(".")[-1]
    unique_node_name = converter.builder.get_unique_instance_name(instance_base_name)
    print(f"Generating unique ONNX node name: {unique_node_name}")

    parent_builder = converter.builder
    input_names = []
    example_args = []

    for var in eqn.invars:
        if hasattr(var, "aval"):
            aval = var.aval
            var_name = converter.get_name(var)
            input_names.append(var_name)

            # Create example tensor
            example_args.append(
                jnp.ones(aval.shape, dtype=aval.dtype)
                if aval.shape
                else jnp.zeros((), dtype=aval.dtype)
            )

            # 🛡️ Protect against overwriting existing value info with conflicting shape/dtype
            if var_name in parent_builder.value_info_metadata:
                old_shape, old_dtype = parent_builder.value_info_metadata[var_name]
                new_shape, new_dtype = tuple(aval.shape), aval.dtype
                if old_shape != new_shape or old_dtype != new_dtype:
                    print(
                        f"[❌ OverwriteError] Refusing to overwrite '{var_name}' "
                        f"(old shape={old_shape}, dtype={old_dtype}) with "
                        f"(new shape={new_shape}, dtype={new_dtype})"
                    )
                    continue

            parent_builder.register_value_info_metadata(
                var_name, tuple(aval.shape), aval.dtype
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

    sub_output_names = [vi.name for vi in sub_builder.outputs]
    print(f"[⚠️ DEBUG] Subgraph output names: {sub_output_names}")
    print("[⚠️ DEBUG] Mapping subgraph outputs to top-level ONNX outputs:")

    parent_builder.add_function(
        name=unique_node_name,
        sub_builder=sub_builder,
        param_input_names=param_inputs,
    )

    parent_builder.merge_value_info_metadata_from(sub_builder)

    call_outputs = []

    for i, sub_name in enumerate(sub_output_names):
        var = eqn.outvars[i]
        # sub_name = sub_output_names[i]
        shape_dtype = sub_builder.value_info_metadata.get(sub_name)

        if shape_dtype is None:
            raise RuntimeError(
                f"[❌] Missing metadata for subgraph output '{sub_name}'."
            )

        shape, dtype = shape_dtype

        # here the original shape is wrong
        # it was set to the shape of the input (intentionally in the primitive)
        var.aval = ShapedArray(shape, tensorproto_dtype_to_numpy(dtype))

        # ✅ Generate fresh output name to avoid conflict
        parent_output_name = parent_builder.get_unique_name("var")

        # can I change the type of var

        converter.var_to_name[var] = parent_output_name
        converter.name_to_var[parent_output_name] = var

        call_outputs.append(parent_output_name)

        parent_builder.register_value_info_metadata(parent_output_name, shape, dtype)
        parent_builder.add_value_info(parent_output_name, shape, dtype)

    _propagate_nested_functions(parent_builder, sub_builder)

    call_inputs = input_names + param_inputs

    parent_builder.add_function_call_node(
        function_name=unique_node_name,
        input_names=call_inputs,
        output_names=call_outputs,
        node_name=unique_node_name,
        user_display_name=name,
    )

    print(f"✅ Added call node for: {unique_node_name}")
