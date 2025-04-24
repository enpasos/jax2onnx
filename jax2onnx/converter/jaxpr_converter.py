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
from jax2onnx.converter.onnx_builder import _symbol_name, _canonical_symbol
from jax2onnx.utils.debug import sdebug


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
        self.builder = builder
        setattr(
            self.builder, "converter", self
        )  # Ensure builder has back-reference to this converter
        # Initialize the converter with an ONNX builder instance.
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

        self.symbolic_shapes = {}

        self.dim_to_symbol = lambda d: _canonical_symbol(self.builder, d)

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
        """Register shape and dtype information for a tensor, preserving symbolic dims."""
        # Convert dtype to ONNX TensorProto enum if needed
        onnx_dtype = self._ensure_onnx_dtype(dtype)

        # If the shape comes from a ShapeDtypeStruct or similar, preserve symbolic tokens
        # Try to recover symbolic names if present (e.g., from .symbol attribute or original spec)
        symbolic_shape = tuple(d.symbol if hasattr(d, "symbol") else d for d in shape)

        # Register with the builder
        self.builder.register_value_info_metadata(name, symbolic_shape, onnx_dtype)

        # Store locally for quick access
        self.shape_env[name] = symbolic_shape

        return name

    def add_input(
        self, var: Any, shape: tuple[int, ...], dtype: Any = np.float32
    ) -> str:
        """Add an input variable to the ONNX graph and store its shape."""
        name = self.get_var_name(var)
        self.builder.add_input(name, shape, dtype)
        sym_shape = tuple(self.dim_to_symbol(d) for d in var.aval.shape)
        self.register_shape(name, shape, dtype)
        self.symbolic_shapes[name] = sym_shape  # Store symbolic shape
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
        sym_shape = tuple(self.dim_to_symbol(d) for d in shape)

        self.register_shape(name, shape, dtype)
        self.symbolic_shapes[name] = sym_shape  # Store symbolic shape
        return name

    def get_name(self, var: Any) -> str:
        """Get the ONNX name for a JAX variable or literal."""
        if isinstance(var, jax._src.core.Var):
            return self.get_var_name(var)
        elif isinstance(var, extend_core.Literal):
            return self.get_constant_name(var)
        else:
            raise NotImplementedError("not yet implemented")

    def _extract_symbolic_axes(self, example_args):
        # Returns a tuple of all symbolic dimension tokens in the example args (if any)
        symbolic_axes = set()
        for arg in example_args:
            if hasattr(arg, "shape"):
                for d in arg.shape:
                    if not isinstance(d, int):
                        symbolic_axes.add(d)
        # JAX expects a tuple, not a set, for abstracted_axes
        return tuple(symbolic_axes) if symbolic_axes else None

    def dim_to_symbol(self, d):
        """Translate JAX shape-dimension `d` into a stable symbolic string
        or a concrete int."""
        # 0) plain integer → just return it
        if isinstance(d, int):
            return d

        # 1) exact identity hit
        if d in getattr(self, "_dimvar_to_name", {}):
            return self._dimvar_to_name[d]

        # 2) hit by (count, dtype) — survives Var cloning
        key = (getattr(d, "count", None), str(getattr(d, "aval", "")))
        if key in getattr(self, "_dimvar_to_name_by_count", {}):
            return self._dimvar_to_name_by_count[key]

        # 3) modern JAX: DimExpr carries .symbol
        if hasattr(d, "symbol") and d.symbol is not None:
            return str(d.symbol)

        # 4) fall back to old helper
        from jax2onnx.converter.onnx_builder import _symbol_name

        _logger = logging.getLogger("jax2onnx.converter.jaxpr_converter")
        _logger.debug("  - FALLBACK to _symbol_name: %s ⚠️", d)

        # try to reuse an existing symbol (same position in the arg-shape)
        sym = None
        if (
            hasattr(self.builder, "current_arg_axes")
            and self.builder.current_arg_axes is not None
        ):
            # current_arg_axes e.g. (None, 'B', None)
            idx = getattr(self.builder, "current_axis_index", 0)  # maintained by caller
            if idx < len(self.builder.current_arg_axes):
                sym = self.builder.current_arg_axes[idx]  # 'B' or None

        if not sym:  # still nothing? invent one
            if hasattr(self.builder, "_unique_symbol"):
                sym = self.builder._unique_symbol()  # e.g. '__sym0'
            else:
                sym = f"__sym{id(d) % 1000}"  # fallback if _unique_symbol doesn't exist

        # register every alias so the object can be found again later
        if not hasattr(self.builder, "var_to_symbol_map"):
            self.builder.var_to_symbol_map = {}

        self.builder.var_to_symbol_map[d] = sym
        self.builder.var_to_symbol_map[id(d)] = sym
        self.builder.var_to_symbol_map[str(d)] = sym

        logger.debug("[dim_to_symbol] %s (%s)  →  %s", d, type(d).__name__, sym)
        return sym  # <— now make_value_info gets "B" (or '__sym0')

        # Step	Description	Dynamic Dim Handling
        # -	User provides symbolic dimensions ("B")	User-level symbolic dimension
        # -	Map symbolic dimensions to concrete ints	Temporary numeric placeholders
        # -	Create concrete zero-arrays for JAX tracer	Concrete numeric array
        # -	Trace with abstracted_axes	JAX records symbolic shapes (DimVar)
        # -	Extract symbolic shapes post-tracing	Explicit symbolic shapes recorded
        # -	Export symbolic shapes into ONNX	ONNX dynamic shape (dim_param)

    ###############################################################################
    # NOTE: this *replaces* the old trace_jaxpr implementation
    ###############################################################################
    def trace_jaxpr(
        self,
        fn: Any,
        example_args: list[Any],
        preserve_graph: bool = False,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Trace a JAX function to JAXPR and convert it to ONNX, while preserving
        user-facing symbolic dimension names (e.g. the batch dim "B") even inside
        nested @onnx_function scopes."""
        import jax
        import jax.numpy as jnp
        from onnx import helper
        from jax2onnx.converter.onnx_builder import _symbol_name

        # ──────────────────────────────────────────────────────────────────────
        # 0) fresh / nested call housekeeping
        # ──────────────────────────────────────────────────────────────────────
        self.logger.debug("trace_jaxpr … preserve_graph=%s", preserve_graph)
        if not preserve_graph:  # top-level call
            self.builder.reset()
            self.var_to_name.clear()
            self.name_to_const.clear()
            self.shape_env.clear()
            self._dimvar_to_name = {}  # start from scratch
            self.builder.var_to_symbol_map.clear()
        else:  # nested call (e.g. inside a
            # keep the maps we inherited from the parent converter/builder
            self._dimvar_to_name = dict(getattr(self, "_dimvar_to_name", {}))

        # ──────────────────────────────────────────────────────────────────────
        # 1) trim duplicate params from example_args
        # ──────────────────────────────────────────────────────────────────────
        modified_args = list(example_args)
        if params and len(modified_args) >= 2:
            last = modified_args[-1]
            is_tracer = "DynamicJaxprTracer" in str(type(last))
            for pname in params:
                if (
                    isinstance(last, bool)
                    or is_tracer
                    or (isinstance(last, (int, float)) and not hasattr(last, "shape"))
                ):
                    self.logger.debug("Removing duplicate param '%s'", pname)
                    modified_args = modified_args[:-1]
                    break
        self.logger.debug("modified_args: %s", modified_args)

        # ──────────────────────────────────────────────────────────────────────
        # 2) build abstracted_axes tuple
        # ──────────────────────────────────────────────────────────────────────
        abstracted_axes = tuple(
            (
                tuple(
                    (
                        _symbol_name(self.builder, dim)
                        if not isinstance(dim, int)
                        else None
                    )
                    for dim in arg.shape
                )
                if hasattr(arg, "shape")
                else None
            )
            for arg in example_args
        )
        use_abstracted_axes = any(
            any(dim is not None for dim in axes)
            for axes in abstracted_axes
            if axes is not None
        )
        self.logger.debug(
            "use_abstracted_axes=%s  symbolic_axes=%s",
            use_abstracted_axes,
            abstracted_axes,
        )

        # ──────────────────────────────────────────────────────────────────────
        # 3) pick concrete placeholder sizes for symbolic dims ─ only for
        #    the *concrete* arrays we feed into jax.make_jaxpr
        # ──────────────────────────────────────────────────────────────────────
        symbolic_dim_to_ints: dict[Any, set[int]] = {}
        for arg in modified_args:
            if hasattr(arg, "shape"):
                for d in arg.shape:
                    if not isinstance(d, int):
                        symbolic_dim_to_ints.setdefault(d, set())
        for arg in modified_args:
            if hasattr(arg, "shape"):
                for d in arg.shape:
                    if not isinstance(d, int):
                        for other in modified_args:
                            if hasattr(other, "shape"):
                                for od in other.shape:
                                    if od == d and isinstance(od, int):
                                        symbolic_dim_to_ints[d].add(od)
        symbolic_dim_map = {
            d: max(vals) if vals else 2 for d, vals in symbolic_dim_to_ints.items()
        }
        self.logger.debug("symbolic_dim_map=%s", symbolic_dim_map)

        tracing_args: list[Any] = []
        for arg in modified_args:
            if hasattr(arg, "shape") and hasattr(arg, "dtype"):
                static = tuple(symbolic_dim_map.get(d, d) for d in arg.shape)
                tracing_args.append(jnp.zeros(static, dtype=arg.dtype))
            else:
                tracing_args.append(arg)
        self.logger.debug("tracing_args=%s", tracing_args)

        # ──────────────────────────────────────────────────────────────────────
        # 4) call jax.make_jaxpr
        # ──────────────────────────────────────────────────────────────────────
        with temporary_monkey_patches(allow_function_primitives=True):
            mk = (
                jax.make_jaxpr(fn, abstracted_axes=abstracted_axes)
                if use_abstracted_axes
                else jax.make_jaxpr(fn)
            )
            closed = mk(*tracing_args, **(params or {}))

        self.logger.debug(closed)
        self.jaxpr = closed.jaxpr
        self.output_vars = self.jaxpr.outvars

        # ──────────────────────────────────────────────────────────────────────
        # 5) extend the dim-var ↔ user-symbol table  (***key change***)
        # ──────────────────────────────────────────────────────────────────────
        offset = len(self.jaxpr.invars) - len(example_args)
        new_map: dict[Any, str] = {}

        for i, orig_arg in enumerate(example_args):
            invar = self.jaxpr.invars[i + offset]
            if not hasattr(orig_arg, "shape"):
                continue

            for orig_dim, traced_dim in zip(orig_arg.shape, invar.aval.shape):
                if isinstance(orig_dim, int):
                    continue

                canonical = _symbol_name(self.builder, orig_dim)
                # record both by object id and by object itself (defensive)
                new_map[id(traced_dim)] = canonical
                new_map[traced_dim] = canonical

        # merge – never overwrite inherited data
        self._dimvar_to_name.update(new_map)
        self.builder.var_to_symbol_map.update(new_map)

        # expose helper dicts for builder / plugins
        self._dimvar_to_name_by_str = {
            str(k): v for k, v in self._dimvar_to_name.items()
        }
        self.builder.dimvar_to_name = self._dimvar_to_name
        self.builder.dimvar_to_name_by_str = self._dimvar_to_name_by_str

        # add "robust" key (count, aval) → symbol
        self._dimvar_to_name_by_count = {
            (d.count, str(getattr(d, "aval", ""))): name
            for d, name in self._dimvar_to_name.items()
            if hasattr(d, "count")
        }

        # also let onnx_builder’s global helper resolve these
        from jax2onnx.converter import onnx_builder as _ob

        _ob.DIMVAR_STR2SYMBOL.update(self._dimvar_to_name_by_str)

        # ──────────────────────────────────────────────────────────────────────
        # 6) remember symbolic shapes for *every* variable so downstream
        #    plugins can query them.
        # ──────────────────────────────────────────────────────────────────────
        self.symbolic_shapes = {}
        for var in (*self.jaxpr.invars, *self.jaxpr.constvars, *self.jaxpr.outvars):
            if hasattr(var, "aval") and hasattr(var.aval, "shape"):
                name = self.get_var_name(var)
                sym_shape = tuple(self.dim_to_symbol(d) for d in var.aval.shape)
                self.symbolic_shapes[name] = sym_shape

        # ──────────────────────────────────────────────────────────────────────
        # 7) convert the JAXPR to an ONNX graph
        # ──────────────────────────────────────────────────────────────────────
        self._process_jaxpr(self.jaxpr, closed.consts)

        # ──────────────────────────────────────────────────────────────────────
        # 8) register graph outputs with their symbolic shapes
        # ──────────────────────────────────────────────────────────────────────
        for var in self.jaxpr.outvars:
            name = self.get_var_name(var)
            if name in self.builder.value_info:
                continue
            sym_shape = tuple(self.dim_to_symbol(d) for d in var.aval.shape)
            dtype = helper.np_dtype_to_tensor_dtype(var.aval.dtype)
            self.builder.register_value_info_metadata(name, sym_shape, dtype)
            self.builder.add_value_info(name, sym_shape, dtype)

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

        # needed for dim-name fallback when a DimVar has no entry in
        # self._dimvar_to_name (avoids NameError)
        from jax2onnx.converter.onnx_builder import _symbol_name

        # Add input variables to the ONNX graph, skipping any that are already added
        # (such as parameters added via add_scalar_input)
        for var in jaxpr.invars:
            # Skip scalar symbolic tokens
            if (
                hasattr(var, "aval")
                and getattr(var.aval, "dtype", None) == np.int32
                and var.aval.shape == ()
            ):
                continue

            var_name = self.match_call_param_by_type_and_order(
                var
            ) or self.get_var_name(var)

            # Convert each dim using the DimVar→name map if available
            shape = tuple(
                (
                    self._dimvar_to_name.get(
                        d, getattr(d, "symbol", None) or _symbol_name(self.builder, d)
                    )
                    if not isinstance(d, int)
                    else d
                )
                for d in var.aval.shape
            )

            if not any(inp.name == var_name for inp in self.builder.inputs):
                self.add_input(var, shape, var.aval.dtype)

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
