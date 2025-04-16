# file: jax2onnx/converter/jaxpr_converter.py

"""
JAXPR to ONNX Converter Module

This module contains the core functionality for converting JAX's JAXPR representation
to ONNX format. It provides the main Jaxpr2OnnxConverter class which traverses the JAXPR
representation of a JAX function and converts it to equivalent ONNX operations.
"""

from typing import Any, Dict
import logging

import jax
import jax.random
import jax.numpy as jnp
import numpy as np
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
        self.logger = logging.getLogger("jax2onnx.converter.jaxpr_converter")
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

    def new_var(self, dtype: np.dtype, shape: tuple[int, ...]) -> extend_core.Var:
        """Create a new JAX variable with the given dtype and shape."""
        return extend_core.Var(
            self.builder.get_unique_name(""), extend_core.ShapedArray(shape, dtype)
        )

    def add_node(self, node: Any) -> None:
        """Add an ONNX node to the builder."""
        self.builder.add_node(node)

    def get_unique_name(self, prefix: str = "node") -> str:
        """Get a unique name for an ONNX node or variable."""
        return self.builder.get_unique_name(prefix)

    def get_var_name(self, var: Any) -> str:
        """Get or create a unique name for a JAX variable."""
        if var not in self.var_to_name:
            name = self.get_unique_name("var")
            self.var_to_name[var] = name
            self.name_to_var[name] = var
        return self.var_to_name[var]

    def get_constant_name(self, val: Any) -> str:
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

        # Centralized mapping for numpy and string dtypes
        dtype_map = {
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
            "float32": TensorProto.FLOAT,
            "float64": TensorProto.DOUBLE,
            "int32": TensorProto.INT32,
            "int64": TensorProto.INT64,
            "bool": TensorProto.BOOL,
            "uint8": TensorProto.UINT8,
            "int8": TensorProto.INT8,
            "uint16": TensorProto.UINT16,
            "int16": TensorProto.INT16,
            "uint32": TensorProto.UINT32,
            "uint64": TensorProto.UINT64,
            "float16": TensorProto.FLOAT16,
            "complex64": TensorProto.COMPLEX64,
            "complex128": TensorProto.COMPLEX128,
        }

        # If it's already an int, assume it's a valid ONNX enum
        if isinstance(dtype, int):
            return dtype

        # Handle JAX array types
        if hasattr(dtype, "__module__") and dtype.__module__.startswith("jax"):
            if "int" in str(dtype):
                return TensorProto.INT64
            elif "float" in str(dtype):
                return TensorProto.FLOAT
            elif "bool" in str(dtype):
                return TensorProto.BOOL

        # Handle numpy dtypes and string names
        if hasattr(dtype, "type") and dtype.type in dtype_map:
            return dtype_map[dtype.type]
        if hasattr(dtype, "name") and dtype.name in dtype_map:
            return dtype_map[dtype.name]
        if isinstance(dtype, str) and dtype in dtype_map:
            return dtype_map[dtype]

        # Try ONNX's helper (might raise TypeError for some inputs)
        try:
            return helper.np_dtype_to_tensor_dtype(dtype)
        except (TypeError, ValueError):
            self.logger.debug(
                "Could not convert dtype %s to ONNX dtype, defaulting to FLOAT", dtype
            )
            return TensorProto.FLOAT

    def register_shape(self, name: str, shape: tuple[int, ...], dtype: Any) -> str:
        """Register shape and dtype information for a tensor."""
        # Convert dtype to ONNX TensorProto enum if needed
        onnx_dtype = self._ensure_onnx_dtype(dtype)

        # Register with the builder
        self.builder.register_value_info_metadata(name, shape, onnx_dtype)

        # Store locally for quick access
        self.shape_env[name] = shape

        return name

    def add_input(
        self, var: Any, shape: tuple[int, ...], dtype: Any = np.float32
    ) -> str:
        """Add an input variable to the ONNX graph and store its shape."""
        name = self.get_var_name(var)
        self.builder.add_input(name, shape, dtype)
        self.register_shape(name, shape, dtype)
        return name

    def add_output(
        self, var: Any, shape: tuple[int, ...], dtype: Any = np.float32
    ) -> str:
        """Add an output variable to the ONNX graph and store its shape."""
        name = self.get_var_name(var)
        self.builder.add_output(name, shape, dtype)
        self.register_shape(name, shape, dtype)
        return name

    def add_shape_info(
        self, name: str, shape: tuple[int, ...], dtype: Any = np.float32
    ) -> str:
        """Add shape information for a variable in the ONNX graph."""
        self.builder.add_value_info(name, shape, dtype)
        self.register_shape(name, shape, dtype)
        return name

    def get_name(self, var: Any) -> str:
        """Get the ONNX name for a JAX variable or literal."""
        if isinstance(var, jax._src.core.Var):
            return self.get_var_name(var)
        elif isinstance(var, extend_core.Literal):
            return self.get_constant_name(var)
        else:
            raise NotImplementedError("not yet implemented")

    def trace_jaxpr(
        self,
        fn: Any,
        example_args: list[Any],
        preserve_graph: bool = False,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Trace a JAX function to generate its JAXPR representation and convert it to ONNX."""

        self.logger.debug("trace_jaxpr ... preserve_graph= %s", preserve_graph)
        if not preserve_graph:
            self.builder.reset()
            self.var_to_name.clear()
            self.name_to_const.clear()
            self.shape_env.clear()

        # Check if any parameters might be duplicated in example_args and params
        modified_args = list(example_args)

        # Handle potential duplicate parameters that are passed both in example_args and params
        if params and len(modified_args) >= 2:
            # Check if the last arg is a potential duplicate parameter
            last_arg = modified_args[-1]
            is_tracer = str(type(last_arg)).find("DynamicJaxprTracer") >= 0

            # Check for static parameters that might be duplicated
            for param_name in params.keys():
                if param_name in params and (
                    isinstance(last_arg, bool)
                    or is_tracer
                    or (
                        isinstance(last_arg, (int, float))
                        and not hasattr(last_arg, "shape")
                    )
                ):
                    self.logger.debug(
                        "Removing potential duplicate '%s' parameter from example_args",
                        param_name,
                    )
                    modified_args = modified_args[:-1]
                    break

        # Simply trace the function with all parameters
        with temporary_monkey_patches(allow_function_primitives=True):
            if params is None:
                closed_jaxpr = jax.make_jaxpr(fn)(*modified_args)
            else:
                closed_jaxpr = jax.make_jaxpr(fn)(*modified_args, **params)

        self.logger.debug(closed_jaxpr)

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

    def add_initializer(
        self,
        name: str,
        vals: Any,
        data_type: int = helper.TensorProto.INT64,
        dims: list[int] | None = None,
    ) -> str:
        """Add a tensor initializer to the model."""

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

    def _process_jaxpr(self, jaxpr: Any, consts: list[Any]) -> None:
        """Process a JAXPR and convert it to ONNX nodes."""

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
            shape: tuple[int, ...]
            dtype: Any

            metadata = self.builder.get_value_info_metadata_with_origin(name)
            if metadata:
                shape, dtype_enum, _ = metadata
                try:
                    dtype = helper.tensor_dtype_to_np_dtype(dtype_enum)
                except Exception:
                    self.logger.debug(
                        "Could not convert dtype enum %s for %s, fallback to var.aval",
                        dtype_enum,
                        name,
                    )
                    shape = tuple(var.aval.shape)
                    dtype = var.aval.dtype
            else:
                self.logger.warning(
                    "No metadata found for output var '%s', using fallback.", name
                )
                shape = tuple(var.aval.shape)
                dtype = var.aval.dtype

            self.add_output(var, shape, dtype)

    def _process_eqn(self, eqn: Any) -> None:
        """Process a single JAXPR equation."""

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
                    self.logger.warning(
                        "Cannot add shape info for %s, missing .aval.", output_name
                    )

    def match_call_param_by_type_and_order(self, var: Any) -> str | None:
        """Match a variable to a parameter in call_params based on type and order."""

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

                    self.logger.debug(
                        "Matching boolean variable to parameter '%s'", param_name
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

    def _create_identity_node(
        self, node_inputs: list[Any], node_outputs: list[Any], prefix: str
    ) -> Any:
        """Create an Identity node to handle simple pass-through operations."""

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

    def _register_primitive_handlers(self) -> None:
        """Register all primitive handlers from both plugin registries."""
        # Register handlers from the main plugin registry
        for key, plugin in PLUGIN_REGISTRY.items():
            if isinstance(plugin, PrimitiveLeafPlugin):
                self.primitive_handlers[key] = plugin.get_handler(self)

        # Register handlers from the ONNX function plugin registry
        for plugin in ONNX_FUNCTION_PLUGIN_REGISTRY.values():
            primitive = plugin.primitive
            self.primitive_handlers[primitive.name] = plugin.get_handler(self)

        if self.primitive_handlers:
            self.logger.debug(
                "Registered %d primitive handlers", len(self.primitive_handlers)
            )
