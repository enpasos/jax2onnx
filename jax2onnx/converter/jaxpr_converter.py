# file: jax2onnx/converter/jaxpr_converter.py

"""
JAXPR to ONNX Converter Module

This module contains the core functionality for converting JAX's JAXPR representation
to ONNX format. It provides the main Jaxpr2OnnxConverter class which traverses the JAXPR
representation of a JAX function and converts it to equivalent ONNX operations.
"""

from typing import Any, Dict

import jax
import jax.random
import jax.numpy as jnp
import numpy as np
import onnx
from jax.extend import core as extend_core
from onnx import helper

# Using ONNX's built-in mapping instead of custom dtype_utils
from jax2onnx.converter.onnx_builder import OnnxBuilder
from jax2onnx.converter.monkey_patch_utils import temporary_monkey_patches
from jax2onnx.plugin_system import (
    ONNX_FUNCTION_PLUGIN_REGISTRY,
    PLUGIN_REGISTRY,
    PrimitiveLeafPlugin,
    import_all_plugins,
)


class Jaxpr2OnnxConverter:
    """
    Converts JAX's JAXPR representation to ONNX format, enabling interoperability
    between JAX and ONNX-based tools.

    This class handles the core conversion logic from JAX's internal representation
    to the ONNX graph format. It traverses the JAXPR computation graph and
    generates equivalent ONNX operations.
    """

    def __init__(self, builder: OnnxBuilder):
        # Initialize the converter with an ONNX builder instance.
        self.builder = builder

        self.params: Dict[str, Any] = {}  # Parameters for tracing
        self.call_params: Dict[str, Any] = {}  # Parameters that should be ONNX inputs

        # Mapping between variables and their names in the ONNX graph.
        self.var_to_name: dict[Any, str] = {}
        self.name_to_var: dict[str, Any] = {}

        # Handlers for JAX primitives.
        self.primitive_handlers: dict[str, Any] = {}

        # Environment to track variable shapes.
        self.shape_env: dict[str, tuple[int, ...]] = {}

        # Mapping for constants in the ONNX graph.
        self.name_to_const: dict[str, Any] = {}

        # Import and register all plugins.
        import_all_plugins()
        self._register_primitive_handlers()

    def new_var(self, dtype: np.dtype, shape: tuple[int, ...]):
        """Create a new JAX variable with the given dtype and shape."""
        return jax.core.Var(
            self.builder.get_unique_name(""), jax.core.ShapedArray(shape, dtype)
        )

    def add_node(self, node):
        """Add an ONNX node to the builder."""
        self.builder.add_node(node)

    def get_unique_name(self, prefix="node"):
        """Get a unique name for an ONNX node or variable."""
        return self.builder.get_unique_name(prefix)

    def get_var_name(self, var):
        """Get or create a unique name for a JAX variable."""
        if var not in self.var_to_name:
            name = self.get_unique_name("var")
            self.var_to_name[var] = name
            self.name_to_var[name] = var
        return self.var_to_name[var]

    def get_constant_name(self, val):
        """Get or create a name for a constant value in the ONNX graph."""
        return self.builder.get_constant_name(val)

    def _ensure_onnx_dtype(self, dtype):
        """
        Ensure the dtype is a valid ONNX TensorProto data type (integer).

        Args:
            dtype: The data type to convert (numpy.dtype, Python type, or ONNX enum)

        Returns:
            An integer representing an ONNX TensorProto data type
        """
        from onnx import TensorProto

        # If it's already an int, assume it's a valid ONNX enum
        if isinstance(dtype, int):
            return dtype

        # Handle JAX array types (add this check)
        if hasattr(dtype, "__module__") and dtype.__module__.startswith("jax"):
            # For JAX arrays used in shape operations, ensure they're INT64
            if str(dtype).find("int") >= 0:
                return TensorProto.INT64
            elif str(dtype).find("float") >= 0:
                return TensorProto.FLOAT
            elif str(dtype).find("bool") >= 0:
                return TensorProto.BOOL

        # Handle numpy dtypes
        if hasattr(dtype, "type"):
            # Map common numpy types to ONNX TensorProto types
            numpy_to_onnx_dtype = {
                np.float32: TensorProto.FLOAT,
                np.float64: TensorProto.DOUBLE,
                np.int32: TensorProto.INT32,
                np.int64: TensorProto.INT64,
                np.bool_: TensorProto.BOOL,
                np.uint8: TensorProto.UINT8,
                np.int8: TensorProto.INT8,
                np.uint16: TensorProto.UINT16,
                np.int16: TensorProto.INT16,
                np.uint32: TensorProto.UINT32,
                np.uint64: TensorProto.UINT64,
                np.float16: TensorProto.FLOAT16,
                np.complex64: TensorProto.COMPLEX64,
                np.complex128: TensorProto.COMPLEX128,
            }

            for np_type, onnx_type in numpy_to_onnx_dtype.items():
                if dtype.type == np_type:
                    return onnx_type

        # Handle numpy dtype objects
        if hasattr(dtype, "name"):
            if dtype.name == "float32":
                return TensorProto.FLOAT
            elif dtype.name == "float64":
                return TensorProto.DOUBLE
            elif dtype.name == "int32":
                return TensorProto.INT32
            elif dtype.name == "int64":
                return TensorProto.INT64
            elif dtype.name == "bool":
                return TensorProto.BOOL
            elif dtype.name == "uint8":
                return TensorProto.UINT8
            elif dtype.name == "int8":
                return TensorProto.INT8
            elif dtype.name == "uint16":
                return TensorProto.UINT16
            elif dtype.name == "int16":
                return TensorProto.INT16
            elif dtype.name == "uint32":
                return TensorProto.UINT32
            elif dtype.name == "uint64":
                return TensorProto.UINT64
            elif dtype.name == "float16":
                return TensorProto.FLOAT16
            elif dtype.name == "complex64":
                return TensorProto.COMPLEX64
            elif dtype.name == "complex128":
                return TensorProto.COMPLEX128

        # Handle string type names
        if isinstance(dtype, str):
            if dtype == "float32":
                return TensorProto.FLOAT
            elif dtype == "float64":
                return TensorProto.DOUBLE
            elif dtype == "int32":
                return TensorProto.INT32
            elif dtype == "int64":
                return TensorProto.INT64
            elif dtype == "bool":
                return TensorProto.BOOL

        # Try ONNX's helper (might raise TypeError for some inputs)
        try:
            return helper.np_dtype_to_tensor_dtype(dtype)
        except (TypeError, ValueError):
            # Default to FLOAT if all else fails
            print(
                f"[WARN] Could not convert dtype {dtype} to ONNX dtype, defaulting to FLOAT"
            )
            return TensorProto.FLOAT

    def register_shape(self, name, shape, dtype):
        """
        Central method for registering shape information consistently.
        This consolidates shape tracking across different methods.

        Args:
            name: The name of the tensor
            shape: The shape of the tensor
            dtype: The data type of the tensor

        Returns:
            The registered tensor name
        """
        # Convert dtype to ONNX TensorProto enum if needed
        onnx_dtype = self._ensure_onnx_dtype(dtype)

        # Register with the builder
        self.builder.register_value_info_metadata(name, shape, onnx_dtype)

        # Store locally for quick access
        self.shape_env[name] = shape

        return name

    def add_input(self, var, shape, dtype=np.float32):
        """Add an input variable to the ONNX graph and store its shape."""
        name = self.get_var_name(var)
        self.builder.add_input(name, shape, dtype)
        self.register_shape(name, shape, dtype)
        return name

    def add_output(self, var, shape, dtype=np.float32):
        """Add an output variable to the ONNX graph and store its shape."""
        name = self.get_var_name(var)
        self.builder.add_output(name, shape, dtype)
        self.register_shape(name, shape, dtype)
        return name

    def add_shape_info(self, name, shape, dtype=np.float32):
        """Add shape information for a variable in the ONNX graph."""
        self.builder.add_value_info(name, shape, dtype)
        self.register_shape(name, shape, dtype)
        return name

    def get_name(self, var):
        """Get the ONNX name for a JAX variable or literal."""
        if isinstance(var, jax._src.core.Var):
            return self.get_var_name(var)
        elif isinstance(var, extend_core.Literal):
            return self.get_constant_name(var)
        else:
            raise NotImplementedError("not yet implemented")

    def _create_and_save_model(
        self, model_name: str, output_path: str | None = None
    ) -> onnx.ModelProto:
        """
        Create and optionally save an ONNX model with the given name.
        This centralizes model creation and saving logic to avoid duplication.

        Args:
            model_name: Name for the ONNX model
            output_path: Optional path to save the model. If None, model is only created, not saved.

        Returns:
            The created ONNX ModelProto object
        """
        # Clean up unused initializers
        used_initializers = {i for node in self.builder.nodes for i in node.input}
        self.builder.initializers = [
            init for init in self.builder.initializers if init.name in used_initializers
        ]

        # Create graph and model
        graph = self.builder.create_graph(model_name)
        onnx_model = self.builder.create_model(graph)

        # Save model if path is provided
        if output_path:
            onnx.save_model(onnx_model, output_path)
            print(f"[INFO] Model saved to {output_path}")

        return onnx_model

    def trace_jaxpr(self, fn, example_args, preserve_graph=False, params=None):
        """
        Trace a JAX function to generate its JAXPR representation and convert it to ONNX.

        Args:
            fn: JAX function to trace
            example_args: Example arguments to trace the function with
            preserve_graph: Whether to preserve the existing graph
            params: Additional parameters for the function
        """
        print(f"trace_jaxpr ... preserve_graph= {preserve_graph}")
        if not preserve_graph:
            self.builder.reset()
            self.var_to_name.clear()
            self.name_to_const.clear()
            self.shape_env.clear()

        # Check if any parameters might be duplicated in example_args and params
        modified_args = list(example_args)

        # Extract static parameter values for function definitions
        static_params = {}
        if params:
            # Copy parameters that should be considered static for ONNX function definitions
            for param_name, param_value in params.items():
                # For common control parameters like deterministic, training, etc.
                # Store their concrete values or defaults for proper static embedding in ONNX functions
                if param_name in ["deterministic", "training", "is_training"]:
                    if isinstance(param_value, bool):
                        # Store concrete boolean value
                        static_params[param_name] = param_value
                    else:
                        # For tracers or other types, use sensible defaults based on common patterns
                        is_tracer = (
                            str(type(param_value)).find("DynamicJaxprTracer") >= 0
                        )
                        if is_tracer:
                            print(
                                f"[INFO] Resolving tracer for static parameter '{param_name}' to default value (True)"
                            )
                            static_params[param_name] = True

        # Store static parameters in the converter instance for use during nested function handling
        self.static_params = static_params

        # Handle potential duplicate parameters that are passed both in example_args and params
        if params and len(modified_args) >= 2:
            # Check if the last arg is a potential duplicate parameter
            last_arg = modified_args[-1]
            is_tracer = str(type(last_arg)).find("DynamicJaxprTracer") >= 0

            # Check for static parameters that might be duplicated
            for param_name in static_params.keys():
                if param_name in params and (
                    isinstance(last_arg, bool)
                    or is_tracer
                    or (
                        isinstance(last_arg, (int, float))
                        and not hasattr(last_arg, "shape")
                    )
                ):
                    print(
                        f"[INFO] Removing potential duplicate '{param_name}' parameter from example_args"
                    )
                    modified_args = modified_args[:-1]
                    break
            else:
                # If no static parameter matches, try signature inspection
                import inspect

                try:
                    sig = inspect.signature(fn)
                    param_names = list(sig.parameters.keys())

                    # Check if any parameter in params might be duplicated in example_args
                    for param_name, param_value in params.items():
                        if param_name in param_names and param_name != param_names[0]:
                            # This is likely a duplicate parameter (passed as both positional and keyword)
                            print(
                                f"[INFO] Removing potential duplicate '{param_name}' parameter from example_args"
                            )
                            modified_args = modified_args[:-1]
                            break
                except (ValueError, TypeError):
                    # If we can't inspect the signature, use a heuristic based on common patterns
                    # Check if the last arg might be a scalar or tracer (common for flag parameters)
                    last_arg_is_scalar = (
                        isinstance(last_arg, (bool, int, float))  # Basic scalar types
                        or is_tracer  # JAX tracers
                        or (not hasattr(last_arg, "shape"))  # No shape attribute
                        or (
                            hasattr(last_arg, "shape")
                            and (last_arg.shape == () or len(last_arg.shape) == 0)
                        )  # Empty shape
                    )

                    if last_arg_is_scalar and len(params) > 0:
                        print(
                            "[INFO] Removing potential duplicate parameter from example_args"
                        )
                        modified_args = modified_args[:-1]

        # Simply trace the function with all parameters
        with temporary_monkey_patches(allow_function_primitives=True):
            if params is None:
                closed_jaxpr = jax.make_jaxpr(fn)(*modified_args)
            else:
                closed_jaxpr = jax.make_jaxpr(fn)(*modified_args, **params)

        print(closed_jaxpr)

        self.jaxpr = closed_jaxpr.jaxpr
        self.output_vars = self.jaxpr.outvars
        jaxpr, consts = self.jaxpr, closed_jaxpr.consts

        self._process_jaxpr(jaxpr, consts)

        for var in jaxpr.outvars:
            name = self.get_var_name(var)
            if name in self.builder.value_info:
                continue

            if hasattr(var, "aval"):
                shape = tuple(var.aval.shape)
                dtype = helper.np_dtype_to_tensor_dtype(var.aval.dtype)
                self.builder.register_value_info_metadata(name, shape, dtype)
                self.builder.add_value_info(name, shape, dtype)
            else:
                raise RuntimeError(
                    f"[MissingShape] Cannot infer shape for output var {name}"
                )

    def convert(
        self, fn, example_args, output_path="model.onnx", model_name="jax_model"
    ):
        """
        Convert a JAX function to ONNX.

        Args:
            fn: JAX function to convert
            example_args: Example input arguments to trace the function
            output_path: Path to save the ONNX model
            model_name: Name for the ONNX model

        Returns:
            Path to the saved ONNX model
        """
        # Trace the JAX function to generate JAXPR representation
        self.trace_jaxpr(fn, example_args)

        # Create and save the ONNX model using our centralized method
        self._create_and_save_model(model_name, output_path)

        return output_path

    def add_initializer(
        self, name, vals, data_type=helper.TensorProto.INT64, dims=None
    ):
        """
        Add a tensor initializer to the model.

        Args:
            name: The name of the initializer
            vals: The values to initialize with
            data_type: The data type of the tensor (default: INT64)
            dims: The dimensions of the tensor (default: [len(vals)])

        Returns:
            The name of the created initializer
        """
        if dims is None:
            dims = [len(vals)]

        tensor = helper.make_tensor(
            name=name,
            data_type=data_type,
            dims=dims,
            vals=vals,
        )
        self.builder.initializers.append(tensor)
        return name

    # Handler methods have been moved to their respective plugin files
    # in the jax2onnx/plugins/ directory

    def _process_jaxpr(self, jaxpr, consts):
        """
        Process a JAXPR and convert it to ONNX nodes.

        Args:
            jaxpr: The JAX program representation to convert
            consts: Constants used in the JAXPR
        """
        # Add input variables to the ONNX graph, skipping any that are already added
        # (such as parameters added via add_scalar_input)
        for var in jaxpr.invars:
            # here we need to call a function that returns the name of the variable
            # match_call_param_by_type_and_order may use call_params
            # call_params should be stored by name and type (not value)
            var_name = self.match_call_param_by_type_and_order(var)
            if var_name is None:
                var_name = self.get_var_name(var)
            # Check if this input is already in the builder's inputs
            # This avoids duplicate inputs for parameters that were added as scalar inputs
            if not any(
                input_info.name == var_name for input_info in self.builder.inputs
            ):
                self.add_input(var, var.aval.shape, var.aval.dtype)

        # Add constants to the ONNX graph.
        for i, const in enumerate(consts):
            const_name = self.get_constant_name(const)
            const_var = jaxpr.constvars[i]
            self.var_to_name[const_var] = const_name
            self.name_to_var[const_name] = const_var
            self.name_to_const[const_name] = const

        # Process equations in the JAXPR.
        for eqn in jaxpr.eqns:
            self._process_eqn(eqn)

        # Add output variables to the ONNX graph.
        for var in jaxpr.outvars:
            name = self.get_var_name(var)
            shape = None
            dtype = None

            metadata = self.builder.get_value_info_metadata_with_origin(name)
            if metadata:
                shape, dtype_enum, _ = metadata
                try:
                    dtype = helper.tensor_dtype_to_np_dtype(dtype_enum)
                except Exception:
                    print(
                        f"[WARN] Could not convert dtype enum {dtype_enum} for {name}, fallback to var.aval"
                    )
                    shape = var.aval.shape
                    dtype = var.aval.dtype
            else:
                print(
                    f"[WARN] No metadata found for output var '{name}', using fallback."
                )
                shape = var.aval.shape
                dtype = var.aval.dtype

            self.add_output(var, shape, dtype)

    def _process_eqn(self, eqn):
        """
        Process a single JAXPR equation.

        Args:
            eqn: The JAXPR equation to process
        """
        if not hasattr(eqn, "primitive"):
            raise NotImplementedError(f"Non-primitive equation: {eqn}")

        primitive = eqn.primitive
        name = primitive.name

        is_function_handler = name in ONNX_FUNCTION_PLUGIN_REGISTRY.keys()

        handler = self.primitive_handlers.get(name)
        if handler is None:
            raise NotImplementedError(f"Primitive {name} not implemented")

        handler(self, eqn, eqn.params)

        if not is_function_handler:
            for outvar in eqn.outvars:
                output_name = self.get_name(outvar)
                if hasattr(outvar, "aval"):
                    self.add_shape_info(
                        output_name, outvar.aval.shape, outvar.aval.dtype
                    )
                else:
                    print(
                        f"[WARN] Cannot add shape info for {output_name}, missing .aval."
                    )

    def match_call_param_by_type_and_order(self, var):
        """
        Match a variable to a parameter in call_params based on type and order.

        Args:
            var: The variable to match

        Returns:
            The name of the matched parameter, or None if no match is found
        """
        if not self.call_params or not hasattr(var, "aval"):
            return None

        # Check if this variable matches any parameter by type and shape
        var_dtype = var.aval.dtype
        var_shape = tuple(var.aval.shape)

        # Special handling for boolean parameters like 'deterministic'
        if var_dtype == jnp.bool_ and var_shape == ():
            # Look for boolean parameters in call_params
            for param_name, param_value in self.call_params.items():
                if isinstance(param_value, bool):
                    # Skip parameters that have already been matched
                    param_key = f"{param_name}"
                    if param_key in self.var_to_name.values():
                        continue

                    print(
                        f"[INFO] Matching boolean variable to parameter '{param_name}'"
                    )
                    # Store this mapping
                    self.var_to_name[var] = param_name
                    self.name_to_var[param_name] = var
                    return param_name

        # Track position to maintain matching by order for non-boolean parameters
        matched_params = []

        for param_name, param_value in self.call_params.items():
            # Skip parameters that have already been matched
            param_key = f"{param_name}"
            if param_key in self.var_to_name.values():
                continue

            # Check if parameter type and shape match the variable
            if hasattr(param_value, "dtype") and hasattr(param_value, "shape"):
                param_dtype = param_value.dtype
                param_shape = tuple(param_value.shape)

                if param_dtype == var_dtype and param_shape == var_shape:
                    matched_params.append((param_name, param_value))

        # If we found matches, use the first one
        if matched_params:
            param_name, _ = matched_params[0]
            # Store this mapping
            self.var_to_name[var] = param_name
            self.name_to_var[param_name] = var
            return param_name

        return None

    # Required method that was called by the random handler methods but wasn't defined
    def _create_identity_node(self, node_inputs, node_outputs, prefix):
        """
        Create an Identity node to handle simple pass-through operations.

        Args:
            node_inputs: Input variables
            node_outputs: Output variables
            prefix: Prefix for the node name

        Returns:
            The created ONNX node
        """
        input_name = self.get_name(node_inputs[0])
        output_name = self.get_var_name(node_outputs[0])

        node = helper.make_node(
            "Identity",
            inputs=[input_name],
            outputs=[output_name],
            name=self.get_unique_name(f"{prefix}:identity"),
        )
        self.builder.add_node(node)
        return node

    def _register_primitive_handlers(self):
        """
        Register all primitive handlers from both plugin registries.
        This consolidates the plugin registration logic in one place.
        """
        # Register handlers from the main plugin registry
        for key, plugin in PLUGIN_REGISTRY.items():
            if isinstance(plugin, PrimitiveLeafPlugin):
                self.primitive_handlers[key] = plugin.get_handler(self)

        # Register handlers from the ONNX function plugin registry
        for plugin in ONNX_FUNCTION_PLUGIN_REGISTRY.values():
            primitive = plugin.primitive
            self.primitive_handlers[primitive.name] = plugin.get_handler(self)

        if self.primitive_handlers:
            print(
                f"[INFO] Registered {len(self.primitive_handlers)} primitive handlers"
            )
