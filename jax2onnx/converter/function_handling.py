# file: jax2onnx/converter/function_handling.py


from __future__ import annotations

from collections.abc import Callable
import inspect
import logging
import re
from typing import TYPE_CHECKING
import numpy as np  # Add missing numpy import
from typing import Any  # type: ignore

import jax.numpy as jnp
from jax.core import ShapedArray
from jax.extend.core import Literal
from onnx import helper
from onnx import TensorProto  # noqa: F401 – used by helpers
import onnx

from jax2onnx.converter.name_generator import get_qualified_name
from jax2onnx.converter.onnx_builder import OnnxBuilder

logger = logging.getLogger("jax2onnx.converter.function_handling")

if TYPE_CHECKING:  # circular‑import guard
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


def create_scalar_constant_tensor(param_name, param_value, dtype_enum, parent_builder):
    # Use the parameter name directly for better clarity and to pass the test
    const_name = param_name

    # Check if we already have this constant
    for initializer in parent_builder.initializers:
        if initializer.name == const_name:
            logger.debug(
                f"Using existing constant tensor '{const_name}' for parameter '{param_name}'"
            )
            return const_name

    const_tensor = onnx.helper.make_tensor(
        name=const_name,
        data_type=dtype_enum,
        dims=(),
        vals=[int(param_value) if isinstance(param_value, bool) else param_value],
    )
    parent_builder.initializers.append(const_tensor)
    logger.debug(
        f"Created constant tensor '{const_name}' for parameter '{param_name}' with value {param_value}"
    )
    return const_name


def prepare_function_names(converter, orig_fn, name):
    impl_key = get_qualified_name(orig_fn)
    logger.debug(f"Encountered function primitive: {impl_key}")

    unique_node_name = converter.builder.get_unique_instance_name(name.split(".")[-1])
    logger.debug(f"Generating unique ONNX node name: {unique_node_name}")

    parent_builder = converter.builder
    return impl_key, unique_node_name, parent_builder


def check_parameters(
    name: str, converter: "Jaxpr2OnnxConverter", eqn, orig_fn: Callable, params
):
    if orig_fn is None:
        raise RuntimeError(f"Original function for {name} not recorded.")


def resolve_function_inputs(converter, eqn, parent_builder):
    input_names, example_args, outer_input_vars_avals = [], [], []
    for var in eqn.invars:
        if hasattr(var, "aval"):
            aval, var_name = var.aval, converter.get_name(var)
            input_names.append(var_name)
            outer_input_vars_avals.append((var, aval))
            example_args.append(create_example_arg(aval))
            register_input_metadata(parent_builder, var_name, aval)
        elif isinstance(var, Literal):
            example_args.append(var.val)
        else:
            raise TypeError(f"Unexpected input var type: {type(var)}")
    return input_names, example_args, outer_input_vars_avals


def create_example_arg(aval):
    """Create an example argument for function tracing.

    Handles both concrete shapes and shapes with symbolic dimensions.
    """
    if not aval.shape:
        return jnp.zeros((), dtype=aval.dtype)

    # Check if the shape contains any symbolic dimensions
    has_symbolic_dim = any(not isinstance(dim, int) for dim in aval.shape)

    if has_symbolic_dim:
        # Create a placeholder ShapedArray instead of concrete array
        # This avoids trying to materialize arrays with symbolic dimensions
        return aval
    else:
        # For concrete shapes, create actual array
        return jnp.ones(aval.shape, dtype=aval.dtype)


def register_input_metadata(builder, var_name, aval):
    shape, dtype = tuple(aval.shape), aval.dtype
    builder.register_value_info_metadata(var_name, shape, dtype)


def register_constant_parameter(
    const_name, param_name, param_value, input_names, extra_param_inputs, example_args
):
    input_names.append(const_name)
    extra_param_inputs.append((param_name, const_name))
    example_args.append(param_value)


def process_scalar_parameters(
    scalar_params_to_process,
    converter,
    eqn,
    parent_builder,
    input_names,
    extra_param_inputs,
    example_args,
    special_param_handling=None,  # New argument for explicit parameter handling
):
    """
    Process scalar parameters for function conversion.
    special_param_handling: Optional[dict], maps param_name to one of: 'input', 'constant', 'static'.
    """
    special_param_handling = special_param_handling or {}
    for param_name, param_value in scalar_params_to_process.items():
        if any(name == param_name for name, _ in extra_param_inputs):
            continue

        handling_mode = special_param_handling.get(param_name, None)

        # If the parameter is a tracer (not static), expose as ONNX input and skip all constant logic
        is_tracer = str(type(param_value)).find("DynamicJaxprTracer") >= 0
        if is_tracer:
            logger.debug(
                f"Exposing tracer parameter '{param_name}' as ONNX input (no constant created)."
            )
            if param_name not in input_names:
                input_names.append(param_name)
            extra_param_inputs.append((param_name, param_name))
            # Do NOT create a constant or add to example_args
            continue

        if handling_mode == "input":
            # Treat as graph input
            if param_name in converter.name_to_var:
                var_name = param_name
                logger.debug(
                    f"Using existing graph input '{var_name}' for parameter '{param_name}' (explicit input mode)"
                )
                if var_name not in input_names:
                    input_names.append(var_name)
                extra_param_inputs.append((param_name, var_name))
                continue
        elif handling_mode == "constant":
            # Treat as constant
            # use INT32 for plain Python ints to match JAX's default
            dtype_enum = (
                onnx.TensorProto.BOOL
                if isinstance(param_value, bool)
                else (
                    onnx.TensorProto.INT32
                    if isinstance(param_value, int)
                    else onnx.TensorProto.FLOAT
                )
            )
            const_name = create_scalar_constant_tensor(
                param_name, param_value, dtype_enum, parent_builder
            )
            register_constant_parameter(
                const_name,
                param_name,
                param_value,
                input_names,
                extra_param_inputs,
                example_args,
            )
            continue
        elif handling_mode == "static":
            # Do not add as input or constant, just skip
            logger.debug(
                f"Parameter '{param_name}' marked as static, skipping input/constant registration."
            )
            continue

        # Fallback: legacy behavior for backward compatibility
        # Remove all hardcoded param_name checks
        if isinstance(param_value, (bool, int, float)):
            dtype_enum = (
                onnx.TensorProto.BOOL
                if isinstance(param_value, bool)
                else (
                    onnx.TensorProto.INT32
                    if isinstance(param_value, int)
                    else onnx.TensorProto.FLOAT
                )
            )
            const_name = create_scalar_constant_tensor(
                param_name, param_value, dtype_enum, parent_builder
            )
            register_constant_parameter(
                const_name,
                param_name,
                param_value,
                input_names,
                extra_param_inputs,
                example_args,
            )
        else:
            input_names.append(param_name)
            extra_param_inputs.append((param_name, param_value))
            logger.warning(
                f"Unsupported parameter type for {param_name}: {type(param_value)}"
            )
            example_args.append(param_value)


def classify_scalar_params(
    params, static_params, eqn, converter, input_names, extra_param_inputs, orig_fn=None
):
    """
    Classify scalar parameters for ONNX function export.

    Required (no default) parameters are always exposed as ONNX inputs.
    Optional parameters (with a default) are exported as ONNX constants if possible.
    Uses Python's inspect to determine which parameters are required.
    Tracer values are treated as dynamic inputs unless a static value is allowed.
    """
    scalar_params_to_process = {}
    # Determine which parameters are required (no default) using inspect
    required_params = set()
    if orig_fn is not None:
        sig = inspect.signature(orig_fn)
        for name, param in sig.parameters.items():
            if param.default is inspect.Parameter.empty and param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                required_params.add(name)
    for param_name, param_value in params.items():
        is_tracer = str(type(param_value)).find("DynamicJaxprTracer") >= 0
        # Only use static value if parameter is NOT required (i.e., has a default)
        if is_tracer:
            if param_name in static_params and (
                orig_fn is None or param_name not in required_params
            ):
                # Optional parameter: use static value
                param_value = static_params[param_name]
            # else: required parameter, expose as ONNX input
        if isinstance(param_value, (bool, int, float)) or is_tracer:
            scalar_params_to_process[param_name] = param_value

            # Remove legacy logic for deterministic/training guessing
            # Always expose as input if it's a tracer and not static
            # (No more bool_input_indices guessing)
    return scalar_params_to_process


def handle_function_parameters(
    params, converter, eqn, parent_builder, input_names, example_args, orig_fn=None
):
    extra_param_inputs = []
    if not params:
        return extra_param_inputs

    static_params = getattr(converter, "static_params", {})
    scalar_params_to_process = classify_scalar_params(
        params,
        static_params,
        eqn,
        converter,
        input_names,
        extra_param_inputs,
        orig_fn=orig_fn,
    )
    process_scalar_parameters(
        scalar_params_to_process,
        converter,
        eqn,
        parent_builder,
        input_names,
        extra_param_inputs,
        example_args,
    )
    return extra_param_inputs


def prepare_trace_kwargs_and_example_args(
    params: dict[str, Any] | None,
    example_args: list[Any],
    orig_fn: Callable,
) -> tuple[dict[str, Any] | None, list[Any]]:
    """
    Strip duplicate/keyword-only parameters **and** trim superfluous positionals.

    Ensures we never forward more positional arguments than the wrapped function
    explicitly accepts (unless it defines a *\*args* catch‑all).
    """
    import inspect

    sig = inspect.signature(orig_fn)

    # Build list of names to exclude from example_args
    excluded_param_names: list[str] = []
    if params is not None:
        # 1) keyword-only or **kwargs parameters always passed via params
        excluded_param_names.extend(
            [
                name
                for name, param in sig.parameters.items()
                if param.kind
                in (
                    inspect.Parameter.KEYWORD_ONLY,
                    inspect.Parameter.VAR_KEYWORD,
                )
                and name in params
            ]
        )
        # 2) any param in both params and signature
        excluded_param_names.extend(name for name in params if name in sig.parameters)
        # de-duplicate
        seen: set[str] = set()
        unique_names: list[str] = []
        for x in excluded_param_names:
            if x not in seen:
                unique_names.append(x)
                seen.add(x)
        excluded_param_names = unique_names

    # Remove trailing duplicates from example_args
    for _ in range(len(example_args)):
        if not excluded_param_names:
            break
        last = example_args[-1]
        if (
            isinstance(last, (int, float, bool))
            and params is not None
            and any(params.get(name) is last for name in excluded_param_names)
        ):
            example_args.pop()
            continue
        break

    # Trim surplus positional args if no *args in signature
    has_varargs = any(
        p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
    )
    if not has_varargs:
        # Count explicit positional parameters
        positional_count = sum(
            1
            for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        )
        if len(example_args) > positional_count:
            logger.debug(
                "Truncating example_args from %d → %d to match positional arity of '%s'",
                len(example_args),
                positional_count,
                orig_fn.__name__,
            )
            del example_args[positional_count:]

    # Always preserve_graph; params passed via kwargs
    trace_kwargs: dict[str, Any] = {"preserve_graph": True}
    if params is not None:
        trace_kwargs["params"] = params

    return trace_kwargs, example_args


def propagate_eqn_parameters(eqn, params):
    if eqn.params:
        if params is None:
            params = {}
        for param_key, param_value in eqn.params.items():
            if param_key not in params:
                params[param_key] = param_value
                logger.debug(
                    f"Propagating parameter '{param_key}' from equation params"
                )
    return params


def setup_sub_converter(converter, eqn, params, unique_node_name, parent_builder):
    """Set up a sub-converter for handling function bodies.

    This function creates a child ONNX builder and converter, ensuring symbolic
    dimensions are properly propagated from parent to child.
    """
    logger.debug(
        "⇢ enter @onnx_function  (parent symbols: %s)",
        (
            parent_builder.var_to_symbol_map
            if hasattr(parent_builder, "var_to_symbol_map")
            else {}
        ),
    )

    # ------------------------------------------------------------------ #
    # 1.  Create a *child* builder, but seed it with the symbol aliases   #
    #     already discovered for the outer graph so that the original     #
    #     symbolic names ('B', etc.) survive inside the Function.         #
    # ------------------------------------------------------------------ #
    sub_builder = OnnxBuilder(
        parent_builder.name_generator,
        parent_builder.opset,
        unique_node_name + "_graph",
        initializers=parent_builder.initializers,
        converter=converter,  # Pass converter reference
    )

    # ─────────────────────────  DEBUG / bookkeeping  ────────────────
    if not hasattr(sub_builder, "var_to_symbol_map"):
        sub_builder.var_to_symbol_map = {}

    if hasattr(parent_builder, "var_to_symbol_map"):
        sub_builder.var_to_symbol_map.update(parent_builder.var_to_symbol_map)

    for k, v in list(sub_builder.var_to_symbol_map.items()):
        sub_builder.var_to_symbol_map.setdefault(str(k), v)
        sub_builder.var_to_symbol_map.setdefault(v, v)
    logger.debug("   inherited symbols → %s", sub_builder.var_to_symbol_map)
    # ────────────────────────────────────────────────────────────────

    # Propagate other symbolic dimension related maps
    if hasattr(parent_builder, "symbolic_shapes") and hasattr(
        sub_builder, "symbolic_shapes"
    ):
        sub_builder.symbolic_shapes.update(parent_builder.symbolic_shapes)

    # Optional: keep a link so the Function can fall back to parent map for new aliases
    sub_builder.parent = parent_builder

    # ------------------------------------------------------------------ #
    # 2.  Create the converter for the Function and propagate symbolic    #
    #     dimension mapping information                                   #
    # ------------------------------------------------------------------ #

    # ──► Create the *Function*-local JaxprConverter and hand over
    #     its private symbol tables
    sub_converter = converter.__class__(sub_builder)

    # share Var → 'B' table
    if hasattr(converter, "_dimvar_to_name"):
        # Start with a copy of parent's dimension mapping
        sub_converter._dimvar_to_name = dict(converter._dimvar_to_name)

    # share the canonical tuple of abstracted axes
    if hasattr(converter, "symbolic_axes"):
        sub_converter.symbolic_axes = converter.symbolic_axes

    # make sure to enable shape polymorphism in the sub-converter
    if hasattr(converter, "use_abstracted_axes"):
        sub_converter.use_abstracted_axes = converter.use_abstracted_axes

    # (optional) allow the child converter to fall back to the parent
    sub_converter.parent = converter

    # Propagate equation parameters
    params = propagate_eqn_parameters(eqn, params)
    sub_converter.params = params

    # Propagate other parameters if available
    if hasattr(converter, "call_params"):
        sub_converter.call_params = converter.call_params

    return sub_converter, sub_builder, params


def register_function_inputs(
    sub_converter, sub_builder, internal_input_vars, outer_input_vars_avals
):
    for internal_var, (outer_var, outer_aval) in zip(
        internal_input_vars[: len(outer_input_vars_avals)],
        outer_input_vars_avals,
        strict=False,
    ):
        internal_name = sub_converter.get_name(internal_var)
        shape = tuple(outer_aval.shape)
        dtype = helper.np_dtype_to_tensor_dtype(outer_aval.dtype)
        sub_builder.register_value_info_metadata(
            internal_name, shape, dtype, origin="function_input"
        )
        sub_builder.add_value_info(internal_name, shape, dtype)


def rename_and_register_param_inputs(
    sub_converter, sub_builder, remaining_internal_vars, extra_param_inputs
):
    """
    Ensure that parameter inputs in the ONNX function graph use descriptive names
    (matching the original function parameter names) instead of generic variable names.
    This improves readability and correctness, and is especially important for
    required call parameters that must be exposed as ONNX inputs.
    Any user-supplied parameter must be a graph input (variable), never a constant/initializer.
    """

    for internal_var, (param_name, param_value) in zip(
        remaining_internal_vars, extra_param_inputs, strict=False
    ):
        # Use the actual parameter name as the internal name for better readability
        internal_name = param_name

        # Handle the mapping in the converter
        if internal_var in sub_converter.var_to_name:
            old_name = sub_converter.var_to_name[internal_var]
            logger.debug(
                f"Replacing generic name '{old_name}' with descriptive name '{internal_name}' for parameter '{param_name}'"
            )

            # Clean up old mappings
            if old_name in sub_converter.name_to_var:
                del sub_converter.name_to_var[old_name]

            # Update mappings for all references to this variable in the graph
            for node in sub_builder.nodes:
                for i, input_name in enumerate(node.input):
                    if input_name == old_name:
                        node.input[i] = internal_name
                        logger.debug(
                            f"Updated node input from '{old_name}' to '{internal_name}'"
                        )

        # Update the converter mappings
        sub_converter.var_to_name[internal_var] = internal_name
        sub_converter.name_to_var[internal_name] = internal_var

        # Register the value info with proper metadata as a graph input (never a constant/initializer)
        # Try to infer shape and dtype from param_value if possible, otherwise default to scalar float32
        try:
            if hasattr(param_value, "shape") and hasattr(param_value, "dtype"):
                shape = tuple(param_value.shape)
                dtype = helper.np_dtype_to_tensor_dtype(param_value.dtype)
            elif isinstance(param_value, (bool, int, float)):
                shape = ()

                if isinstance(param_value, bool):
                    dtype = onnx.TensorProto.BOOL
                elif isinstance(param_value, (int, np.integer)):
                    dtype = onnx.TensorProto.INT32
                else:
                    dtype = onnx.TensorProto.FLOAT
            else:
                shape = ()
                dtype = onnx.TensorProto.FLOAT
        except Exception:
            shape = ()
            dtype = onnx.TensorProto.FLOAT

        sub_builder.register_value_info_metadata(
            internal_name, shape, dtype, origin="function_param_input"
        )
        sub_builder.add_value_info(internal_name, shape, dtype)


def collect_used_param_inputs(sub_builder, parent_builder):
    initializer_names = {i.name for i in parent_builder.initializers}
    used_constants = {
        inp
        for node in sub_builder.nodes
        for inp in node.input
        if inp in initializer_names
    }
    return sorted(used_constants)


def create_function_call(
    unique_node_name,
    input_names,
    param_inputs,
    call_outputs,
    parent_builder,
    display_name,
):
    """Create a function call node that invokes the ONNX function in the parent graph.

    Args:
        unique_node_name: Unique name for the function
        input_names: List of standard input tensor names
        param_inputs: List of parameter input names (constants, weights, etc.)
        call_outputs: List of output tensor names
        parent_builder: The parent ONNX graph builder
        display_name: Human-readable name for debugging
    """
    # Ensure we include all parameter inputs in the final call inputs without duplicates
    # This combines our regular inputs with weight parameters and scalar parameters like 'deterministic'

    # Use a dictionary to track seen inputs
    seen_inputs = {}
    call_inputs = []

    # First add standard input tensor names
    for name in input_names:
        if name not in seen_inputs:
            call_inputs.append(name)
            seen_inputs[name] = True

    # Then add parameter input names, avoiding duplicates
    for name in param_inputs:
        if name not in seen_inputs:
            call_inputs.append(name)
            seen_inputs[name] = True
        else:
            logger.info(
                f"Skipping duplicate input '{name}' in function call to {unique_node_name}"
            )

    parent_builder.add_function_call_node(
        function_name=unique_node_name,
        input_names=call_inputs,
        output_names=call_outputs,
        node_name=unique_node_name,
        user_display_name=display_name,
    )

    logger.debug(f"✅ Added call node for: {unique_node_name}")


def trace_function_body(
    converter,
    orig_fn,
    params,
    example_args,
    unique_node_name,
    parent_builder,
    outer_input_vars_avals,
    extra_param_inputs,
    eqn,
):
    """Trace the function body and set up the function's inputs.

    This function performs the JAX function tracing and registers all inputs
    including regular inputs and parameter inputs.

    Args:
        converter: The parent converter instance
        orig_fn: Original JAX function to trace
        params: Function parameters
        example_args: Example arguments for tracing
        unique_node_name: Unique name for the function
        parent_builder: The parent ONNX graph builder
        outer_input_vars_avals: Input variable information from parent scope
        extra_param_inputs: Extra parameter inputs information
        eqn: Equation from JAX's JaxprTrace

    Returns:
        A tuple of (sub_converter, sub_builder, internal_input_vars)
    """

    sub_converter, sub_builder, params = setup_sub_converter(
        converter, eqn, params, unique_node_name, parent_builder
    )

    # Propagate equation parameters (fix for dual param dicts)
    if isinstance(params, dict) and "onnx_params" in params and "jax_params" in params:
        params_to_use = params["jax_params"]
    else:
        params_to_use = params
    params_to_use = propagate_eqn_parameters(eqn, params_to_use)
    sub_converter.params = params_to_use
    params = params_to_use

    trace_kwargs, example_args = prepare_trace_kwargs_and_example_args(
        params, example_args, orig_fn
    )
    sub_converter.trace_jaxpr(orig_fn, example_args, **trace_kwargs)

    internal_input_vars = sub_converter.jaxpr.invars
    register_function_inputs(
        sub_converter, sub_builder, internal_input_vars, outer_input_vars_avals
    )

    remaining_internal_vars = internal_input_vars[len(outer_input_vars_avals) :]
    rename_and_register_param_inputs(
        sub_converter, sub_builder, remaining_internal_vars, extra_param_inputs
    )

    return sub_converter, sub_builder, internal_input_vars


def map_and_register_outputs(
    unique_node_name, sub_builder, parent_builder, sub_converter, converter, eqn
):
    sub_output_names = [vi.name for vi in sub_builder.outputs]
    logger.debug(f"[⚠️ DEBUG] Subgraph output names: {sub_output_names}")
    logger.debug("[⚠️ DEBUG] Mapping subgraph outputs to top-level ONNX outputs:")

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

        # Get the shape and dtype from metadata
        shape, dtype = shape_dtype

        # Check for symbolic dimensions in shape
        # Skip ShapedArray creation if there are symbolic dimensions - just preserve var.aval
        if all(isinstance(dim, (int, float)) for dim in shape):
            # Only create ShapedArray for concrete shapes
            var.aval = ShapedArray(shape, helper.tensor_dtype_to_np_dtype(dtype))
        else:
            # For symbolic shapes, preserve dimensions but update the dtype if needed
            if hasattr(var, "aval") and var.aval is not None:
                # If we already have an aval with the right shape, just update dtype if needed
                if var.aval.dtype != helper.tensor_dtype_to_np_dtype(dtype):
                    var.aval = var.aval.update(
                        dtype=helper.tensor_dtype_to_np_dtype(dtype)
                    )
            else:
                # If we don't have an aval, we need to construct one that preserves symbolic dims
                # This is a fallback that might not handle all cases
                logger.warning(
                    f"Creating placeholder aval for var with symbolic dimensions: {shape}"
                )
                # Use the first dimension from the original var if available
                if (
                    hasattr(var, "aval")
                    and hasattr(var.aval, "shape")
                    and var.aval.shape
                ):
                    placeholder_shape = var.aval.shape
                else:
                    # Create a default shape - this is not ideal but better than failing
                    placeholder_shape = (2,) * len(shape)
                var.aval = ShapedArray(
                    placeholder_shape, helper.tensor_dtype_to_np_dtype(dtype)
                )

        parent_output_name = parent_builder.get_unique_name("var")
        converter.var_to_name[var] = parent_output_name
        converter.name_to_var[parent_output_name] = var
        call_outputs.append(parent_output_name)

        # Important: Register shape metadata in parent builder
        # We need to preserve the original shape with symbolic dimensions here
        parent_builder.register_value_info_metadata(parent_output_name, shape, dtype)
        # keep the converter’s symbolic table in sync
        if hasattr(converter, "symbolic_shapes"):
            converter.symbolic_shapes[parent_output_name] = shape
        parent_builder.add_value_info(parent_output_name, shape, dtype)

    return call_outputs


# --- little helper -----------------------------------------------------------


# ──────────────────────────────────────────────────────────────────────────────
# utilities
# ──────────────────────────────────────────────────────────────────────────────


def _base_name(name: str) -> str:
    """Return `name` without the trailing ``_<digits>`` suffix."""
    return re.sub(r"_\d+$", "", name)


def function_handler(
    name: str,
    converter: "Jaxpr2OnnxConverter",
    eqn,
    orig_fn: Callable,
    onnx_params,
    jax_params=None,
    params=None,
):
    """
    Convert a primitive produced by ``@onnx_function``.

    Compared with the previous implementation this version
    **(a)** inlines trivial pass‑through wrappers and
    **(b)** registers every non‑trivial function as a proper
    `FunctionProto`, guaranteeing ORT can resolve the
    op‑type at load time.
    """

    # ------------------------------------------------------------------
    # 1) Boiler‑plate: checks, name generation, input preparation
    # ------------------------------------------------------------------
    if orig_fn is None:
        raise RuntimeError(f"Original function for {name} not recorded.")

    impl_key, unique_node_name, parent_builder = prepare_function_names(
        converter, orig_fn, name
    )

    input_names, example_args, outer_input_avals = resolve_function_inputs(
        converter, eqn, parent_builder
    )

    extra_param_inputs = handle_function_parameters(
        params if params is not None else onnx_params,
        converter,
        eqn,
        parent_builder,
        input_names,
        example_args,
        orig_fn=orig_fn,
    )

    logger.debug(f"Tracing function body for: {unique_node_name}")

    # Use jax_params for tracing if provided, else fallback to onnx_params
    trace_params = jax_params if jax_params is not None else onnx_params
    sub_converter, sub_builder, _ = trace_function_body(
        converter,
        orig_fn,
        trace_params,
        example_args,
        unique_node_name,
        parent_builder,
        outer_input_avals,
        extra_param_inputs,
        eqn,
    )

    # ------------------------------------------------------------------
    # 2)  INLINE  trivial wrappers  (the body contains exactly one node
    #     that merely calls the *real* function)
    # ------------------------------------------------------------------
    inner_nodes = list(sub_builder.nodes)
    if len(inner_nodes) == 1 and _base_name(inner_nodes[0].op_type) == _base_name(
        unique_node_name
    ):
        logger.debug(f"Inlining trivial wrapper '{unique_node_name}' (calls itself).")

        parent_builder._propagate_nested_functions(sub_builder)
        parent_builder.merge_value_info_metadata_from(sub_builder)

        call_outputs = map_and_register_outputs(
            inner_nodes[0].op_type,
            sub_builder,
            parent_builder,
            sub_converter,
            converter,
            eqn,
        )

        param_inputs = collect_used_param_inputs(sub_builder, parent_builder)

        create_function_call(
            inner_nodes[0].op_type,
            input_names,
            param_inputs,
            call_outputs,
            parent_builder,
            name,
        )
        return  # wrapper collapsed – done!

    # ------------------------------------------------------------------
    # 3)  NORMAL path – keep the wrapper as a reusable ONNX function
    # ------------------------------------------------------------------
    param_inputs = collect_used_param_inputs(sub_builder, parent_builder)

    parent_builder.add_function(
        name=unique_node_name,
        sub_builder=sub_builder,
        param_input_names=param_inputs,
        sub_converter=sub_converter,
    )

    parent_builder.merge_value_info_metadata_from(sub_builder)

    call_outputs = map_and_register_outputs(
        unique_node_name,
        sub_builder,
        parent_builder,
        sub_converter,
        converter,
        eqn,
    )

    parent_builder._propagate_nested_functions(sub_builder)

    create_function_call(
        unique_node_name,
        input_names,
        param_inputs,
        call_outputs,
        parent_builder,
        name,
    )
