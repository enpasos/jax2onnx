# file: jax2onnx/converter/function_handling.py

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
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


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

    # Check for static parameter context from the parent converter
    static_params = getattr(converter, "static_params", {})

    # First check if we have function-specific parameters from the function_handler's params argument
    if params:
        # Track parameters that need special handling (like boolean flags)
        scalar_params_to_process = {}

        # First pass: identify parameters that map to existing inputs
        for param_name, param_value in params.items():
            # Special handling for tracers - try to resolve from static context
            is_tracer = str(type(param_value)).find("DynamicJaxprTracer") >= 0
            if is_tracer:
                print(
                    f"[WARN] Parameter '{param_name}' is a tracer: {type(param_value)}"
                )
                # Check if we have a static value in the converter context
                if param_name in static_params:
                    print(
                        f"[INFO] Using static value {static_params[param_name]} for tracer parameter '{param_name}'"
                    )
                    param_value = static_params[param_name]
                else:
                    print(
                        f"[WARN] No static value found for tracer parameter '{param_name}', defaulting to True"
                    )
                    if param_name in ["deterministic", "training", "is_training"]:
                        param_value = True  # Default to deterministic=True

            # Handle scalar parameters (like deterministic=True)
            if isinstance(param_value, (bool, int, float)) or is_tracer:
                scalar_params_to_process[param_name] = param_value

                # For boolean parameters, try to map to existing boolean inputs
                if isinstance(param_value, bool) or (
                    param_name in ["deterministic", "training"]
                ):
                    # Keep track of which boolean inputs we've found
                    bool_input_indices = []
                    for i, var in enumerate(eqn.invars):
                        if hasattr(var, "aval") and var.aval.dtype == jnp.bool_:
                            bool_input_indices.append(i)

                    # If there's exactly one boolean input, we can map it confidently
                    if len(bool_input_indices) == 1:
                        bool_idx = bool_input_indices[0]
                        bool_var = eqn.invars[bool_idx]
                        standard_var_name = converter.get_name(bool_var)
                        if standard_var_name not in input_names:
                            input_names.append(standard_var_name)

                        # Record that we found a standard variable for this parameter
                        extra_param_inputs.append((param_name, standard_var_name))
                        print(
                            f"[INFO] Using standard boolean input '{standard_var_name}' for parameter '{param_name}'"
                        )
                        # Remove from params to process as we've handled it
                        scalar_params_to_process.pop(param_name, None)

        # Second pass: create constants for remaining scalar parameters
        import onnx

        for param_name, param_value in scalar_params_to_process.items():
            # Check if this parameter was already processed above
            if any(name == param_name for name, _ in extra_param_inputs):
                continue

            # For well-known control parameters, always create a constant
            if param_name in ["deterministic", "training", "is_training"]:
                # Ensure boolean value
                if isinstance(param_value, bool):
                    bool_value = param_value
                else:
                    # Default for control parameters
                    bool_value = True
                    print(
                        f"[INFO] Converting non-boolean value for '{param_name}' to boolean True"
                    )

                # Create a unique constant tensor name for this parameter
                const_name = f"{param_name}_const__{parent_builder.get_unique_name('')}"

                # Create the tensor with the boolean value
                const_tensor = onnx.helper.make_tensor(
                    name=const_name,
                    data_type=onnx.TensorProto.BOOL,
                    dims=(),
                    vals=[int(bool_value)],  # Convert bool to int for ONNX
                )

                # Add to initializers
                parent_builder.initializers.append(const_tensor)
                print(
                    f"[INFO] Created constant tensor '{const_name}' for parameter '{param_name}' with value {bool_value}"
                )

                # Record this constant for the function definition
                input_names.append(const_name)
                extra_param_inputs.append((param_name, const_name))

                # Add to example args
                example_args.append(bool_value)
            elif isinstance(param_value, bool):
                # Handle general boolean parameters
                const_name = f"{param_name}_const__{parent_builder.get_unique_name('')}"
                const_tensor = onnx.helper.make_tensor(
                    name=const_name,
                    data_type=onnx.TensorProto.BOOL,
                    dims=(),
                    vals=[int(param_value)],
                )
                parent_builder.initializers.append(const_tensor)
                print(
                    f"[INFO] Created constant tensor '{const_name}' for boolean parameter '{param_name}'"
                )
                input_names.append(const_name)
                extra_param_inputs.append((param_name, const_name))
                example_args.append(param_value)
            elif isinstance(param_value, int):
                # Handle integer parameters
                const_name = f"{param_name}_const__{parent_builder.get_unique_name('')}"
                const_tensor = onnx.helper.make_tensor(
                    name=const_name,
                    data_type=onnx.TensorProto.INT64,
                    dims=(),
                    vals=[param_value],
                )
                parent_builder.initializers.append(const_tensor)
                print(
                    f"[INFO] Created constant tensor '{const_name}' for integer parameter '{param_name}'"
                )
                input_names.append(const_name)
                extra_param_inputs.append((param_name, const_name))
                example_args.append(param_value)
            elif isinstance(param_value, float):
                # Handle float parameters
                const_name = f"{param_name}_const__{parent_builder.get_unique_name('')}"
                const_tensor = onnx.helper.make_tensor(
                    name=const_name,
                    data_type=onnx.TensorProto.FLOAT,
                    dims=(),
                    vals=[param_value],
                )
                parent_builder.initializers.append(const_tensor)
                print(
                    f"[INFO] Created constant tensor '{const_name}' for float parameter '{param_name}'"
                )
                input_names.append(const_name)
                extra_param_inputs.append((param_name, const_name))
                example_args.append(param_value)

                # Create the constant tensor with the parameter value
                const_tensor = onnx.helper.make_tensor(
                    name=const_name,
                    data_type=onnx.TensorProto.INT64,
                    dims=(),
                    vals=[param_value],
                )

                # Add the constant to initializers
                parent_builder.initializers.append(const_tensor)

                print(
                    f"[INFO] Created constant tensor '{const_name}' for parameter '{param_name}' with value {param_value}"
                )

                # Use the constant tensor's name instead of the parameter name
                input_names.append(const_name)
                extra_param_inputs.append((param_name, const_name))
            elif isinstance(param_value, float):
                # Create a constant tensor for float parameters
                import onnx

                # Generate a unique constant name for this parameter
                const_name = f"{param_name}_const_{parent_builder.get_unique_name('')}"

                # Create the constant tensor with the parameter value
                const_tensor = onnx.helper.make_tensor(
                    name=const_name,
                    data_type=onnx.TensorProto.FLOAT,
                    dims=(),
                    vals=[param_value],
                )

                # Add the constant to initializers
                parent_builder.initializers.append(const_tensor)

                print(
                    f"[INFO] Created constant tensor '{const_name}' for parameter '{param_name}' with value {param_value}"
                )

                # Use the constant tensor's name instead of the parameter name
                input_names.append(const_name)
                extra_param_inputs.append((param_name, const_name))
            else:
                # Fall back to original behavior for other parameter types
                input_names.append(param_name)
                extra_param_inputs.append((param_name, param_value))
                print(
                    f"[WARN] Unsupported parameter type for {param_name}: {type(param_value)}"
                )

            # For example args, add the parameter value
            example_args.append(param_value)

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

    # Get initializer names before processing parameter inputs
    initializer_names = {i.name for i in parent_builder.initializers}

    # Process any extra parameter inputs with improved name preservation
    remaining_internal_vars = internal_input_vars[len(outer_input_vars_avals) :]
    for internal_var, (param_name, param_value) in zip(
        remaining_internal_vars, extra_param_inputs, strict=False
    ):
        # Explicitly force descriptive parameter names in the sub-function
        # This ensures deterministic_const__* names appear in the function definition
        if isinstance(param_value, str) and (
            param_value in initializer_names
            or param_value.startswith(f"{param_name}_const")
        ):
            # For constant parameters, use the exact constant name from parent scope
            internal_name = param_value

            # Crucial: First remove existing name mappings from sub_converter if they exist
            if internal_var in sub_converter.var_to_name:
                old_name = sub_converter.var_to_name[internal_var]
                print(
                    f"[INFO] Replacing generic name '{old_name}' with constant name '{internal_name}' for parameter '{param_name}'"
                )
                # Remove the old name from name_to_var mapping if it exists
                if old_name in sub_converter.name_to_var:
                    del sub_converter.name_to_var[old_name]

            # Apply our preferred name directly
            sub_converter.var_to_name[internal_var] = internal_name
            sub_converter.name_to_var[internal_name] = internal_var
            print(
                f"[INFO] Using constant name '{internal_name}' for parameter '{param_name}'"
            )
        else:
            # For non-constant parameters, create a descriptive name
            internal_name = f"{param_name}_{sub_builder.get_unique_name('')}"

            # Again, first remove existing name if it exists
            if internal_var in sub_converter.var_to_name:
                old_name = sub_converter.var_to_name[internal_var]
                print(
                    f"[INFO] Replacing generic name '{old_name}' with descriptive name '{internal_name}' for parameter '{param_name}'"
                )
                if old_name in sub_converter.name_to_var:
                    del sub_converter.name_to_var[old_name]

            # Apply our preferred name
            sub_converter.var_to_name[internal_var] = internal_name
            sub_converter.name_to_var[internal_name] = internal_var

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
