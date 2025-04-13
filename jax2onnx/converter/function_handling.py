from collections.abc import Callable
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax.core import ShapedArray
from jax.extend.core import Literal
from onnx import helper

# Use built-in ONNX helper functions instead
# helper.np_dtype_to_tensor_dtype replaces numpy_dtype_to_tensorproto
# helper.tensor_dtype_to_np_dtype replaces tensorproto_dtype_to_numpy

from jax2onnx.converter.name_generator import get_qualified_name
from jax2onnx.converter.onnx_builder import OnnxBuilder

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


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

    # Process regular inputs from the equation
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

    # Add function parameters to the function's inputs
    # This ensures parameters like deterministic are passed to function nodes
    extra_param_inputs = []

    # First check if we have function-specific parameters from the function_handler's params argument
    if params:
        for param_name, param_value in params.items():
            # For boolean parameters like 'deterministic', first check if it already exists as a standard input
            if isinstance(param_value, bool):
                # Look for an existing boolean parameter in the equation's input variables
                standard_var_found = False
                standard_var_name = None

                # Keep track of which boolean inputs we've found
                bool_input_indices = []
                for i, var in enumerate(eqn.invars):
                    if hasattr(var, "aval") and var.aval.dtype == jnp.bool_:
                        bool_input_indices.append(i)

                # If there's exactly one boolean input, we can confidently map it to the boolean param
                if len(bool_input_indices) == 1:
                    bool_idx = bool_input_indices[0]
                    bool_var = eqn.invars[bool_idx]
                    standard_var_name = converter.get_name(bool_var)
                    standard_var_found = True
                    print(
                        f"[INFO] Found exactly one boolean input '{standard_var_name}', mapping to parameter '{param_name}'"
                    )
                # If there are multiple boolean inputs, we need to be careful about which one maps to this param
                elif len(bool_input_indices) > 1:
                    # We'd need more info to disambiguate, for now just use the first one
                    # This could be improved in the future with more context
                    bool_idx = bool_input_indices[0]
                    bool_var = eqn.invars[bool_idx]
                    standard_var_name = converter.get_name(bool_var)
                    standard_var_found = True
                    print(
                        f"[INFO] Found multiple boolean inputs, using '{standard_var_name}' for parameter '{param_name}'"
                    )

                if standard_var_found and standard_var_name:
                    # Add the standard input name to the function's input list if not already there
                    if standard_var_name not in input_names:
                        input_names.append(standard_var_name)

                    # Record that we found a standard variable for this parameter
                    extra_param_inputs.append((param_name, standard_var_name))
                    print(
                        f"[INFO] Using standard boolean input '{standard_var_name}' for parameter '{param_name}'"
                    )
                    continue

        for i, param in enumerate(params.items()):
            param_name, param_value = param

            input_names.append(param_name)
            extra_param_inputs.append((param_name, param_value))

            # For example args, add a reasonable default value based on type
            if isinstance(param_value, bool):
                example_args.append(param_value)
            elif isinstance(param_value, int):
                example_args.append(param_value)
            elif isinstance(param_value, float):
                example_args.append(param_value)
            else:
                print(
                    f"[WARN] Unsupported parameter type for {param_name}: {type(param_value)}"
                )

    print(f"Tracing function body for: {unique_node_name}")
    sub_builder = OnnxBuilder(
        parent_builder.name_generator,
        parent_builder.opset,
        unique_node_name + "_graph",
        initializers=parent_builder.initializers,
    )
    sub_converter = converter.__class__(sub_builder)

    # Pass all parameters to the subconverter
    sub_converter.params = params

    # Also pass call_params from parent to sub_converter
    if hasattr(converter, "call_params"):
        sub_converter.call_params = converter.call_params

    # Extract any parameters from the equation that should be propagated
    # This ensures parameters are properly passed through nested function calls
    if eqn.params:
        # If we have equation parameters, extract them to params dictionary
        if params is None:
            params = {}

        for param_key, param_value in eqn.params.items():
            # Only propagate parameters that aren't already in params
            if param_key not in params:
                params[param_key] = param_value
                print(
                    f"[INFO] Propagating parameter '{param_key}' from equation params"
                )

    trace_kwargs = {"preserve_graph": True}

    # Don't duplicate parameters between trace_kwargs and example_args
    # This prevents the "got multiple values for argument" error
    param_keys_to_exclude = []
    if params is not None:
        trace_kwargs["params"] = params
        param_keys_to_exclude = list(params.keys())
        print(
            f"[INFO] Will exclude these parameters from example_args: {param_keys_to_exclude}"
        )

    # Remove any example_args that correspond to parameters already in trace_kwargs
    if example_args and param_keys_to_exclude:
        # Boolean params (like deterministic) are often at the end of example_args
        if (
            isinstance(example_args[-1], bool)
            and "deterministic" in param_keys_to_exclude
        ):
            print(
                "[INFO] Removing duplicated 'deterministic' parameter from example_args"
            )
            example_args = example_args[:-1]

        # Remove None values that correspond to parameters being passed in kwargs
        # This avoids duplicate parameters like 'mask'
        for i, param_name in enumerate(param_keys_to_exclude):
            if param_name in [
                "mask",
                "dropout_rng",
                "dtype",
                "precision",
                "module",
            ] and i < len(example_args):
                if example_args[i] is None:
                    print(
                        f"[INFO] Removing duplicated '{param_name}' parameter from example_args"
                    )
                    example_args = example_args[:i] + example_args[i + 1 :]

    sub_converter.trace_jaxpr(orig_fn, example_args, **trace_kwargs)

    internal_input_vars = sub_converter.jaxpr.invars

    # MODIFIED: Account for extra parameter inputs when checking match
    expected_inputs = len(outer_input_vars_avals) + len(extra_param_inputs)
    if len(internal_input_vars) != expected_inputs:
        print(
            f"[WARNING] Mismatch in input count! Expected {expected_inputs}, got {len(internal_input_vars)}"
        )
        print(f"  - Regular inputs: {len(outer_input_vars_avals)}")
        print(f"  - Extra param inputs: {len(extra_param_inputs)}")
        # Continue anyway - we'll skip the mismatched inputs

    # Process the regular inputs first
    for internal_var, (outer_var, outer_aval) in zip(
        internal_input_vars[: len(outer_input_vars_avals)],
        outer_input_vars_avals,
        strict=False,
    ):
        internal_name = sub_converter.get_name(internal_var)
        shape = tuple(outer_aval.shape)
        onnx_dtype_enum = helper.np_dtype_to_tensor_dtype(outer_aval.dtype)
        sub_builder.register_value_info_metadata(
            internal_name, shape, onnx_dtype_enum, origin="function_input"
        )
        sub_builder.add_value_info(internal_name, shape, onnx_dtype_enum)

    # NEW CODE: Now process any extra parameter inputs
    remaining_internal_vars = internal_input_vars[len(outer_input_vars_avals) :]
    for internal_var, (param_name, param_value) in zip(
        remaining_internal_vars, extra_param_inputs, strict=False
    ):
        internal_name = sub_converter.get_name(internal_var)

        # Get shape and dtype from the parameter
        if isinstance(param_value, bool):
            shape = ()
            onnx_dtype_enum = 9  # TensorProto.BOOL
        elif isinstance(param_value, int):
            shape = ()
            onnx_dtype_enum = 7  # TensorProto.INT64
        elif isinstance(param_value, float):
            shape = ()
            onnx_dtype_enum = 1  # TensorProto.FLOAT
        else:
            print(f"[WARN] Unsupported parameter type, skipping: {type(param_value)}")
            continue

        print(
            f"[INFO] Registering param input in function: {internal_name} <- {param_name}"
        )

        # Register metadata for the internal parameter name
        sub_builder.register_value_info_metadata(
            internal_name, shape, onnx_dtype_enum, origin="function_param_input"
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

        if sub_name not in sub_builder.value_info_metadata:
            sub_var = sub_converter.name_to_var.get(sub_name)
            if sub_var and hasattr(sub_var, "aval"):
                aval = sub_var.aval
                shape = tuple(aval.shape)
                dtype = helper.np_dtype_to_tensor_dtype(aval.dtype)
                sub_builder.register_value_info_metadata(
                    sub_name, shape, dtype, origin="function_output"
                )
                sub_builder.add_value_info(sub_name, shape, dtype)

        shape_dtype = sub_builder.value_info_metadata.get(sub_name)
        if shape_dtype is None:
            raise RuntimeError(
                f"[❌] Missing metadata for subgraph output '{sub_name}'."
            )
        shape, dtype = shape_dtype
        var.aval = ShapedArray(shape, helper.tensor_dtype_to_np_dtype(dtype))
        parent_output_name = parent_builder.get_unique_name("var")
        converter.var_to_name[var] = parent_output_name
        converter.name_to_var[parent_output_name] = var
        call_outputs.append(parent_output_name)
        parent_builder.register_value_info_metadata(parent_output_name, shape, dtype)
        parent_builder.add_value_info(parent_output_name, shape, dtype)

    parent_builder._propagate_nested_functions(sub_builder)

    # Ensure we include all parameter inputs in the final call inputs
    # This combines our regular inputs with weight parameters and scalar parameters like 'deterministic'
    call_inputs = input_names + param_inputs

    parent_builder.add_function_call_node(
        function_name=unique_node_name,
        input_names=call_inputs,
        output_names=call_outputs,
        node_name=unique_node_name,
        user_display_name=name,
    )

    print(f"✅ Added call node for: {unique_node_name}")
