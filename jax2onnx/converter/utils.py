from collections.abc import Callable
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax.core import ShapedArray
from jax.extend.core import Literal

from jax2onnx.converter.dtype_utils import (
    numpy_dtype_to_tensorproto,
    tensorproto_dtype_to_numpy,
)
from jax2onnx.converter.name_generator import get_qualified_name
from jax2onnx.converter.onnx_builder import OnnxBuilder

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def _propagate_nested_functions(parent_builder: OnnxBuilder, sub_builder: OnnxBuilder):
    for name, func in sub_builder.functions.items():
        if name not in parent_builder.functions:
            parent_builder.functions[name] = func


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
    outer_input_vars_avals = []

    for var in eqn.invars:
        if hasattr(var, "aval"):
            aval = var.aval
            var_name = converter.get_name(var)
            input_names.append(var_name)
            outer_input_vars_avals.append((var, aval))
            example_args.append(
                jnp.ones(aval.shape, dtype=aval.dtype)
                if aval.shape
                else jnp.zeros((), dtype=aval.dtype)
            )
            # 🛡️ Avoid overwriting existing value info
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

    # ✅ Store params in sub_converter
    sub_converter.params = params

    sub_converter.trace_jaxpr(orig_fn, example_args, preserve_graph=True, params=params)

    internal_input_vars = sub_converter.jaxpr.invars
    if len(internal_input_vars) != len(outer_input_vars_avals):
        raise RuntimeError(
            f"Mismatch between outer function inputs ({len(outer_input_vars_avals)}) "
            f"and traced internal inputs ({len(internal_input_vars)}) for {name}."
        )

    for internal_var, (outer_var, outer_aval) in zip(
        internal_input_vars, outer_input_vars_avals, strict=False
    ):
        internal_name = sub_converter.get_name(internal_var)
        shape = tuple(outer_aval.shape)
        onnx_dtype_enum = numpy_dtype_to_tensorproto(outer_aval.dtype)
        sub_builder.register_value_info_metadata(
            internal_name, shape, onnx_dtype_enum, origin="function_input"
        )
        sub_builder.add_value_info(internal_name, shape, onnx_dtype_enum)

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
        shape_dtype = sub_builder.value_info_metadata[sub_name]
        if shape_dtype is None:
            raise RuntimeError(
                f"[❌] Missing metadata for subgraph output '{sub_name}'."
            )
        shape, dtype = shape_dtype
        var.aval = ShapedArray(shape, tensorproto_dtype_to_numpy(dtype))
        parent_output_name = parent_builder.get_unique_name("var")
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
