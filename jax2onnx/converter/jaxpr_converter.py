# file: jax2onnx/converter/jaxpr_converter.py

"""
JAXPR to ONNX Converter Module

This module contains the core functionality for converting JAX's JAXPR representation
to ONNX format. It provides the main Jaxpr2OnnxConverter class which traverses the JAXPR
representation of a JAX function and converts it to equivalent ONNX operations.
"""

from typing import Any, Dict, List
import logging
import jax
import jax.random
import jax.numpy as jnp
import numpy as np
from jax.extend import core as extend_core
from onnx import helper
from jax2onnx.converter.onnx_builder import OnnxBuilder
from jax2onnx.converter.monkey_patch_utils import temporary_monkey_patches
from jax2onnx.plugin_system import (
    ONNX_FUNCTION_PLUGIN_REGISTRY,
    PLUGIN_REGISTRY,
    PrimitiveLeafPlugin,
    import_all_plugins,
)

# Keep _symbol_name, _canonical_symbol if used elsewhere, maybe remove from here if only trace_jaxpr used them
from jax2onnx.converter.onnx_builder import _symbol_name, _canonical_symbol
from jax2onnx.utils.debug import sdebug
from jax import ShapeDtypeStruct  # Import ShapeDtypeStruct


logger = logging.getLogger("jax2onnx.converter.jaxpr_converter")


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
        setattr(self.builder, "converter", self)
        self.params: Dict[str, Any] = {}
        self.call_params: Dict[str, Any] = {}
        self.var_to_name: dict[Any, str] = {}
        self.name_to_var: dict[str, Any] = {}
        self.primitive_handlers: dict[str, Any] = {}
        self.shape_env: dict[str, tuple[int, ...]] = {}
        self.name_to_const: dict[str, Any] = {}
        import_all_plugins()
        self._register_primitive_handlers()
        self.symbolic_shapes = {}
        # Initialize DimVar -> Name mappings needed for dim_to_symbol
        self._dimvar_to_name = getattr(builder, "var_to_symbol_map", {})
        self._dimvar_to_name_by_str = {
            str(k): v for k, v in self._dimvar_to_name.items()
        }
        # Note: dim_to_symbol might need access to the builder's map directly later
        # self.dim_to_symbol = lambda d: _canonical_symbol(self.builder, d) # Maybe pass builder?

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
    # --- Helper for safe dimension-to-symbol conversion (Keep from response #33) ---
    def _dim_to_symbol_safe(self, d):
        if isinstance(d, int):
            return d
        # Use the builder's map which should be up-to-date
        # Try object, then str representation
        resolved = self.builder.var_to_symbol_map.get(d)
        if resolved is None:
            resolved = self.builder.var_to_symbol_map.get(str(d))
        if resolved is None:
            # Fallback for unknown symbolic objects (might be internal JAX exprs)
            logger.warning(
                f"Cannot resolve symbolic dim {d} (type: {type(d)}) to name. Using str()."
            )
            resolved = str(d)  # Use string representation as fallback name
        return resolved

    # --- MODIFIED trace_jaxpr Method ---
    def trace_jaxpr(
        self,
        fn: Any,
        # Change signature: Now expects the list of symbolic ShapeDtypeStruct avals
        symbolic_avals: List[ShapeDtypeStruct],  # Changed name and type hint
        preserve_graph: bool = False,
        params: dict[str, Any] | None = None,
    ) -> None:
        """
        Trace a JAX function to JAXPR using pre-computed symbolic abstract values
        and convert it to ONNX.
        """
        self.logger.debug(
            f"trace_jaxpr called with {len(symbolic_avals)} symbolic avals. preserve_graph={preserve_graph}"
        )
        if not preserve_graph:
            # Reset state for top-level call
            # NOTE: Ensure builder.reset() doesn't clear var_to_symbol_map or re-assign it after reset
            # Maybe builder reset needs adjustment, or we fetch the map *after* reset?
            # Let's assume builder keeps the map or we re-set it from conversion_api if needed.
            # self.builder.reset() # Defer reset or handle map persistence carefully
            self.var_to_name.clear()
            self.name_to_const.clear()
            self.shape_env.clear()  # This seems safe to clear
            # Fetch map from builder, assuming it was set in conversion_api
            self._dimvar_to_name = getattr(self.builder, "var_to_symbol_map", {})
            self._dimvar_to_name_by_str = {
                str(k): v for k, v in self._dimvar_to_name.items()
            }
            self.symbolic_shapes.clear()
        else:
            # For nested calls, inherit maps - ensure builder map is current context
            self._dimvar_to_name = getattr(self.builder, "var_to_symbol_map", {})
            self._dimvar_to_name_by_str = {
                str(k): v for k, v in self._dimvar_to_name.items()
            }
            # Keep existing self.symbolic_shapes for nested scope? This needs thought.

        # --- Step 1: Use received symbolic_avals directly ---
        # No need to create concrete tracing_args based on symbolic_dim_map
        tracing_args = symbolic_avals  # Use the symbolic avals directly
        self.logger.debug(f"Using tracing_args (symbolic avals): {tracing_args}")

        # --- Step 2: Call jax.make_jaxpr ---
        # We *should not* need abstracted_axes if symbolic shapes are explicit in avals
        # JAX handles polymorphism based on the symbolic objects in the input avals.
        with temporary_monkey_patches(allow_function_primitives=True):
            # Remove abstracted_axes argument if it was previously used
            mk = jax.make_jaxpr(fn)
            try:
                # Pass symbolic avals directly as arguments
                closed = mk(*tracing_args, **(params or {}))
            except Exception as e:
                self.logger.error(
                    f"jax.make_jaxpr failed with symbolic avals. Error: {e}",
                    exc_info=True,
                )
                self.logger.error(f"Function: {fn}")
                self.logger.error(f"Tracing Args (Symbolic Avals): {tracing_args}")
                self.logger.error(f"Params: {params}")
                raise

        self.logger.debug(f"Jaxpr generated: {closed}")
        self.jaxpr = closed.jaxpr
        self.output_vars = getattr(self.jaxpr, "outvars", [])  # Access outvars safely

        # --- Step 3: Post-trace Processing (Update internal state) ---
        # Ensure the builder has the necessary map for subsequent operations
        # It should have been set in conversion_api.py
        self.builder.var_to_symbol_map = self._dimvar_to_name

        # Store symbolic shapes for *all* vars seen in the final jaxpr
        # using the safe conversion back to string names
        self.symbolic_shapes = {}
        all_vars = (
            getattr(self.jaxpr, "invars", [])
            + getattr(self.jaxpr, "constvars", [])
            + getattr(self.jaxpr, "outvars", [])
        )
        for var in all_vars:
            if var is not None and hasattr(var, "aval") and hasattr(var.aval, "shape"):
                try:
                    name = self.get_name(var)  # Handles Vars and Literals
                    # Convert potentially symbolic shape back to tuple with string names
                    sym_shape = tuple(
                        self._dim_to_symbol_safe(d) for d in var.aval.shape
                    )
                    self.symbolic_shapes[name] = sym_shape
                    self.logger.debug(
                        f"Stored symbolic shape for {name} ('{type(var)}'): {sym_shape}"
                    )
                except Exception as e:
                    # Log details about the variable causing issues
                    var_repr = repr(var)
                    aval_repr = repr(getattr(var, "aval", None))
                    self.logger.warning(
                        f"Could not process/store symbolic shape for var {var_repr} with aval {aval_repr}. Error: {e}",
                        exc_info=False,
                    )  # Keep log concise

        # --- Step 4: Convert JAXPR to ONNX Graph ---
        self.logger.info("Processing generated jaxpr...")
        # Pass the map explicitly or ensure _process_jaxpr uses self.builder.var_to_symbol_map
        self._process_jaxpr(self.jaxpr, closed.consts)
        self.logger.info("Jaxpr processing complete.")

    def _process_jaxpr(self, jaxpr: Any, consts: list[Any]) -> None:
        # ... (Add constants logic - needs self.get_constant_name) ...
        for i, const in enumerate(consts):
            const_name = self.get_constant_name(
                const
            )  # Builder handles registration now
            const_var = jaxpr.constvars[i]
            self.var_to_name[const_var] = const_name
            self.name_to_var[const_name] = const_var
            # self.name_to_const[const_name] = const # Maybe builder stores this?

        # Add input variables with symbolic shapes
        for var in jaxpr.invars:
            if var is None:
                continue
            var_name = self.get_var_name(var)
            if not hasattr(var, "aval") or not hasattr(var.aval, "shape"):
                self.logger.warning(
                    f"Input var {var_name} missing aval/shape. Skipping add_input."
                )
                continue
            shape = tuple(self._dim_to_symbol_safe(d) for d in var.aval.shape)
            dtype = var.aval.dtype
            if not any(inp.name == var_name for inp in self.builder.inputs):
                self.add_input(var, shape, dtype)  # Passes symbolic shape tuple

        # Process equations
        for eqn in jaxpr.eqns:
            self._process_eqn(
                eqn
            )  # Ensure this handles symbolic shapes in output avals

        # Add outputs with symbolic shapes
        for var in jaxpr.outvars:
            if var is None:
                continue
            name = self.get_var_name(var)
            if not hasattr(var, "aval") or not hasattr(var.aval, "shape"):
                self.logger.warning(
                    f"Output var {name} missing aval/shape. Skipping add_output."
                )
                continue
            shape = tuple(self._dim_to_symbol_safe(d) for d in var.aval.shape)
            dtype = var.aval.dtype
            self.add_output(var, shape, dtype)  # Pass symbolic shape tuple

    # --- Ensure add_input/add_output/add_shape_info pass symbolic tuples ---
    # These might be simplified if the builder handles most logic now
    def add_input(self, var: Any, shape: tuple, dtype: Any = np.float32) -> str:
        name = self.get_var_name(var)
        self.builder.add_input(
            name, shape, dtype
        )  # Pass potentially symbolic shape tuple
        return name

    def add_output(self, var: Any, shape: tuple, dtype: Any = np.float32) -> str:
        name = self.get_var_name(var)
        self.builder.add_output(
            name, shape, dtype
        )  # Pass potentially symbolic shape tuple
        return name

    def add_shape_info(self, name: str, shape: tuple, dtype: Any = np.float32) -> str:
        # Note: shape passed here might already be symbolic strings from _dim_to_symbol_safe
        self.builder.add_value_info(name, shape, dtype)
        return name

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
