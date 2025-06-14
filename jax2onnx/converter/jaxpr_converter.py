"""
JAXPR to ONNX Converter Module

This module contains the core functionality for converting JAX's JAXPR representation
to ONNX format. It provides the main Jaxpr2OnnxConverter class which traverses the JAXPR
representation of a JAX function and converts it to equivalent ONNX operations.
"""

from typing import Any, Dict, List, Optional, Sequence, Union
import logging
import jax
import jax.random
import numpy as np
import jax.core as core
from jax.extend import core as extend_core
from jax.extend.core import Var, Literal, ClosedJaxpr
from onnx import helper
from jax2onnx.converter.onnx_builder import OnnxBuilder
from jax2onnx.converter.monkey_patch_utils import temporary_monkey_patches
from jax2onnx.utils.debug import RecordedPrimitiveCallLog  # Ensure correct import
from jax2onnx.plugin_system import (
    ONNX_FUNCTION_PLUGIN_REGISTRY,
    PLUGIN_REGISTRY,
    PrimitiveLeafPlugin,
    import_all_plugins,
)
from jax._src.export.shape_poly import _DimExpr
from jax import ShapeDtypeStruct

logger = logging.getLogger("jax2onnx.converter.jaxpr_converter")


class Jaxpr2OnnxConverter:
    """
    Converts JAX's JAXPR representation to ONNX format, enabling interoperability
    between JAX and ONNX-based tools.

    This class handles the core conversion logic from JAX's internal representation
    to the ONNX graph format. It traverses the JAXPR computation graph and
    generates equivalent ONNX operations.
    """

    # Map symbolic dimensions to their origin tensor names and axes

    # mapping from ONNX tensor name → its symbolic shape (a tuple of ints or dim-names)
    symbolic_shapes: dict[str, tuple[Union[int, str], ...]]

    def __init__(
        self,
        builder: OnnxBuilder,
        record_primitive_calls_file: Optional[str] = None,
        function_context_for_recording: Optional[str] = None,
    ):
        self.logger = logging.getLogger("jax2onnx.converter.jaxpr_converter")
        self.builder = builder
        self.record_primitive_calls_file = record_primitive_calls_file
        self.function_context_for_recording = function_context_for_recording

        if self.record_primitive_calls_file:
            self.recorded_calls_log: List[RecordedPrimitiveCallLog] = []
            self.primitive_call_counter: int = 0

        setattr(self.builder, "converter", self)
        self.params: Dict[str, Any] = {}
        self.call_params: Dict[str, Any] = {}
        self.var_to_name: dict[Any, str] = {}
        self.name_to_var: dict[str, Any] = {}
        self.primitive_handlers: dict[str, Any] = {}
        self.shape_env: dict[str, tuple[int, ...]] = {}
        self.name_to_const: dict[str, Any] = {}
        # ------------------------------------------------------------------
        # Mapping: symbolic dimension expression -> (tensor_name, axis)
        # ------------------------------------------------------------------
        self.symbolic_dim_to_origin: dict[_DimExpr, tuple[str, int]] = {}
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

    def new_var(self, dtype: np.dtype, shape: tuple[int, ...]) -> Var:
        """Create a new JAX variable with the given dtype and shape."""
        return Var(
            self.builder.get_unique_name(""), extend_core.ShapedArray(shape, dtype)
        )

    def add_node(self, node: Any) -> None:
        """Add an ONNX node to the builder."""
        self.builder.add_node(node)

    def _emit_result(self, jax_outvar, wanted_sym, src_sym):
        """
        Connect a JAX output variable to its ONNX representation.

        If the caller asked for a specific output symbol name that's different
        from the source symbol, insert an Identity node to create the alias.

        Args:
            jax_outvar: The JAX output variable
            wanted_sym: The desired output symbol name
            src_sym: The source symbol name currently producing the value
        """
        # If the caller asked for a different symbol, materialise an Identity
        if wanted_sym != src_sym:
            id_name = self.builder.name_generator.get("alias")
            self.builder.add_node(
                helper.make_node("Identity", [src_sym], [wanted_sym], name=id_name)
            )
            self.builder.add_value_info(
                wanted_sym, jax_outvar.aval.shape, jax_outvar.aval.dtype
            )
            self.var_to_name[jax_outvar] = wanted_sym
        else:
            self.var_to_name[jax_outvar] = src_sym

    def _import_var_as_input(
        self,
        var: Var,
        sym: str,
    ):
        """Make *sym* a formal graph input **iff** it is not
             already created inside the builder**.

        That is the case when *sym* is produced by a node that
        was added to ``self.builder`` **before** this converter
        started processing the JAXPR – for example the ``Cast``
        that makes ``iter64 → iter32`` in a Loop body.  Turning
        such an internal tensor into another graph input would
        (a) pollute the interface and (b) create duplicate-name
        errors at ONNX runtime.
        """

        # 1. Fast bail-out if another node already writes *sym*
        if any(sym in n.output for n in self.builder.nodes):
            # We only need shape-information (if not yet there)
            aval = var.aval
            self.builder.add_value_info(sym, aval.shape, aval.dtype)
            return

        # 2. Really add a new formal input
        aval = var.aval
        self.builder.add_input(sym, aval.shape, aval.dtype)

    def get_unique_name(self, prefix: str = "node") -> str:
        """Get a unique name for an ONNX node or variable."""
        return self.builder.get_unique_name(prefix)

    def get_var_name(self, var: Any) -> str:
        """Get or create a unique name for a JAX variable."""

        # ────────────────────────────────────────────────────────────────
        # Plain Python int / float (e.g., literal `3`)
        # ────────────────────────────────────────────────────────────────
        from numbers import Number

        if isinstance(var, Number) and not hasattr(var, "aval"):
            value = np.array(var, dtype=np.int32)
            const_name = self.get_constant_name(value)
            wanted_dtype = self._ensure_onnx_dtype(value.dtype)

            # Cast to ensure dtype matches int32 exactly
            cast_name = self.get_unique_name("lit_cast_to_i32")
            self.builder.add_node(
                helper.make_node(
                    "Cast",
                    inputs=[const_name],
                    outputs=[cast_name],
                    to=int(wanted_dtype),
                    name=cast_name,
                )
            )
            self.add_shape_info(cast_name, (), value.dtype)
            return cast_name

        # ────────────────────────────────────────────────────────────────
        # Handle Literal (JAX constant-folded literals)
        # ────────────────────────────────────────────────────────────────
        if isinstance(var, Literal):
            value = np.asarray(var.val)
            if np.issubdtype(value.dtype, np.integer):
                value = value.astype(np.int32)
            const_name = self.get_constant_name(value)
            self.add_shape_info(const_name, value.shape, value.dtype)
            return const_name

        # ────────────────────────────────────────────────────────────────
        # Every other normal Var
        # ────────────────────────────────────────────────────────────────
        if var not in self.var_to_name:
            name = self.get_unique_name("var")
            self.set_var_name(var, name)
        return self.var_to_name[var]

    def set_var_name(self, var: Any, name: str) -> None:
        """Set a custom name for a JAX variable."""
        self.var_to_name[var] = name
        self.name_to_var[name] = var

    def change_var_name(self, var: Any, name: str) -> None:
        """Change the name of a JAX variable."""
        if var in self.var_to_name:
            old_name = self.var_to_name[var]
            del self.name_to_var[old_name]
            self.var_to_name[var] = name
            self.name_to_var[name] = var
            self.builder.change_var_name(old_name, name)
        else:
            raise ValueError(f"Variable {var} not found in var_to_name mapping.")

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
            np.dtype("float32"): TensorProto.FLOAT,
            np.float64: TensorProto.DOUBLE,
            np.dtype("float64"): TensorProto.DOUBLE,
            np.int32: TensorProto.INT32,
            np.dtype("int32"): TensorProto.INT32,
            np.int64: TensorProto.INT64,
            np.dtype("int64"): TensorProto.INT64,
            np.bool_: TensorProto.BOOL,
            np.dtype("bool"): TensorProto.BOOL,
            bool: TensorProto.BOOL,
            "int64": TensorProto.INT64,
            "bool": TensorProto.BOOL,
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

    def _handle_pjit(self, eqn, params):
        """
        Processes a pjit call by creating a new, isolated converter context
        to prevent name collisions, then inlining the resulting nodes.
        """
        parent_converter = self
        self.logger.debug(f"Creating isolated context for pjit eqn: {eqn.primitive}")

        # ① Fetch the closed jaxpr from the parameters
        closed = params.get("call_jaxpr") or params.get("jaxpr")
        if isinstance(closed, ClosedJaxpr):
            inner_jaxpr = closed.jaxpr
            consts = closed.consts
        else:
            inner_jaxpr = closed
            consts = params.get("consts", ())

        # ② Create a new temporary converter to process the subgraph in isolation.
        #    It inherits the parent's builder to add nodes to the same graph,
        #    but it will have its own separate variable-to-name mapping.
        sub_converter = Jaxpr2OnnxConverter(
            parent_converter.builder,
            parent_converter.record_primitive_calls_file,
            parent_converter.function_context_for_recording,
        )

        # ③ Map the inputs for the subgraph. The inputs to the pjit call (eqn.invars)
        #    already have names in the parent converter. We tell the sub-converter
        #    to use these same names for the subgraph's input variables (inner_jaxpr.invars).
        for outer_invar, inner_invar in zip(eqn.invars, inner_jaxpr.invars):
            outer_name = parent_converter.get_name(outer_invar)
            sub_converter.set_var_name(inner_invar, outer_name)

        # ④ Process the subgraph using the new converter. This will generate all the
        #    necessary ONNX nodes, but any new variable names will be created in the
        #    isolated sub_converter and won't clash with the parent.
        sub_converter._process_jaxpr(inner_jaxpr, consts)

        # ── remove any outputs that the subgraph mistakenly added ────────────
        inner_output_names = {sub_converter.get_name(v) for v in inner_jaxpr.outvars}
        self.builder.outputs = [
            o for o in self.builder.outputs if o.name not in inner_output_names
        ]

        # ⑤ Wire the outputs. The output variables of the subgraph (inner_jaxpr.outvars)
        #    now have unique names within the subgraph's context. We need to alias the
        #    pjit's output variables (eqn.outvars) to these names in the parent converter.
        for outer_outvar, inner_outvar in zip(eqn.outvars, inner_jaxpr.outvars):
            inner_name = sub_converter.get_name(inner_outvar)
            parent_converter.set_var_name(outer_outvar, inner_name)

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

    def get_name(self, var: Any) -> str:
        """Get the ONNX name for a JAX variable, tracer, or literal."""
        if isinstance(var, Var):
            return self.get_var_name(var)
        if isinstance(var, core.Tracer):
            # Tracers are stand-ins for Vars during tracing. They are hashable
            # and used as keys in the var_to_name map.
            return self.get_var_name(var)
        if isinstance(var, extend_core.Literal):
            return self.get_constant_name(var)
        raise NotImplementedError(f"get_name not yet implemented for type: {type(var)}")

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
        # Always hand ShapeDtypeStructs to jax.make_jaxpr
        tracing_args = [
            (
                jax.ShapeDtypeStruct(a.shape, a.dtype)  # ✔ keeps _DimExpr symbols
                if hasattr(a, "shape")
                and hasattr(a, "dtype")  # Check for ShapedArray attributes
                else a
            )  # already a ShapeDtypeStruct
            for a in symbolic_avals
        ]
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

        # --------------------------------------------------------------
        # Record symbolic dimensions present in the graph inputs
        # --------------------------------------------------------------
        for input_var, input_spec in zip(self.jaxpr.invars, symbolic_avals):
            tensor_name = self.get_name(input_var)
            for axis, dim in enumerate(input_spec.shape):
                if isinstance(dim, _DimExpr):
                    self.symbolic_dim_to_origin[dim] = (tensor_name, axis)

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

        # Write the full primitive calls log if recording was enabled
        if (
            self.record_primitive_calls_file
            and hasattr(self, "recorded_calls_log")
            and self.recorded_calls_log
        ):
            try:
                self.logger.info(
                    f"Writing {len(self.recorded_calls_log)} primitive call records to {self.record_primitive_calls_file}"
                )
                from jax2onnx.utils.debug import save_primitive_calls_log

                save_primitive_calls_log(
                    self.recorded_calls_log, self.record_primitive_calls_file
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to save primitive calls log: {e}", exc_info=True
                )

    def _process_jaxpr(self, jaxpr: Any, consts: list[Any]) -> None:
        # Process equations
        for i, const in enumerate(consts):
            # register initializer-name → value
            const_name = self.get_constant_name(const)
            const_var = jaxpr.constvars[i]
            self.var_to_name[const_var] = const_name
            self.name_to_var[const_name] = const_var
            self.name_to_const[const_name] = const
            # also register the "logical var name" (in case it differs)
            var_name = self.get_var_name(const_var)
            if var_name != const_name:
                self.name_to_const[var_name] = const

        # Add input variables with symbolic shapes
        for var in jaxpr.invars:
            if var is None:
                continue
            var_name = self.get_var_name(var)
            if not hasattr(var, "aval") or not hasattr(var.aval, "shape"):
                continue
            shape = tuple(self._dim_to_symbol_safe(d) for d in var.aval.shape)
            dtype = var.aval.dtype
            if not any(inp.name == var_name for inp in self.builder.inputs):
                self.add_input(var, shape, dtype)
        for eqn in jaxpr.eqns:
            self._process_eqn(eqn)

        # Explicitly handle outputs
        for var in jaxpr.outvars:
            if var is None:
                continue
            name = self.get_var_name(var)

            if name in self.name_to_const:
                # Explicitly wrap initializer in Identity for valid ONNX output
                identity_output_name = self.get_unique_name("const_output")
                dtype = np.asarray(self.name_to_const[name]).dtype
                shape = ()

                identity_node = helper.make_node(
                    "Identity",
                    inputs=[name],
                    outputs=[identity_output_name],
                    name=self.get_unique_name("identity_const_out"),
                )
                self.builder.add_node(identity_node)

                # Set the output explicitly to the Identity node's output
                self.builder.add_output(identity_output_name, shape, dtype)
                self.add_shape_info(identity_output_name, shape, dtype)
            elif hasattr(var, "aval") and hasattr(var.aval, "shape"):
                # if the plugin already emitted this output, skip it
                if any(o.name == name for o in self.builder.outputs):
                    continue
                shape = tuple(self._dim_to_symbol_safe(d) for d in var.aval.shape)
                dtype = var.aval.dtype
                self.add_output(var, shape, dtype)
            else:
                if not any(o.name == name for o in self.builder.outputs):
                    self.builder.add_output(name, (), np.int32)

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
                if key == "lax.remat2":
                    self.primitive_handlers["remat2"] = plugin.get_handler(self)
                else:
                    self.primitive_handlers[key] = plugin.get_handler(self)

        # Register handlers from the ONNX function plugin registry
        for plugin in ONNX_FUNCTION_PLUGIN_REGISTRY.values():
            primitive = plugin.primitive

            self.primitive_handlers[primitive.name] = plugin.get_handler(self)

        # built‑in call‑style primitive that we inline
        self.primitive_handlers["pjit"] = lambda conv, eqn, params: conv._handle_pjit(
            eqn, params
        )

        if self.primitive_handlers:
            self.logger.debug(
                "Registered %d primitive handlers", len(self.primitive_handlers)
            )

    def log_primitive_call(
        self,
        eqn: extend_core.JaxprEqn,  # Updated to use jax.extend.core.JaxprEqn
        plugin_hint: str | None,
        current_fn_context_name: str | None = None,
    ):
        if not self.record_primitive_calls_file:
            return

        prim_name = str(eqn.primitive.name)

        # Cleaned params and their string representation (assuming this logic exists)
        params_cleaned = {
            k: v for k, v in eqn.params.items() if k != "sharding"
        }  # Example cleaning
        params_repr_str_list = []
        if params_cleaned:
            for k, v_param in params_cleaned.items():
                params_repr_str_list.append(
                    f"  - {k}: {self._format_param_for_log(v_param)}"
                )
        params_repr = (
            "\n".join(params_repr_str_list) if params_repr_str_list else "  (none)"
        )

        inputs_aval_log = []
        inputs_jax_vars_log = []
        inputs_onnx_names_log = []
        for var in eqn.invars:
            inputs_jax_vars_log.append(str(var))
            if isinstance(var, Literal):
                # Represent shape and dtype for literals, ONNX name is more complex (could be constant node)
                inputs_aval_log.append(
                    (
                        tuple(var.aval.shape),
                        str(var.aval.dtype),
                        f"Literal<val={var.val}>",
                    )
                )
                # For literals, an ONNX name might be the name of a Constant node if created,
                # or just indicate it's a literal value.
                # If the builder has a way to get/make names for literals that become constants:
                try:
                    # This assumes get_name can handle literals by finding/creating a constant node name
                    inputs_onnx_names_log.append(self.get_name(var))
                except Exception:
                    inputs_onnx_names_log.append("<Literal Value>")
            else:
                inputs_aval_log.append(
                    (tuple(var.aval.shape), str(var.aval.dtype), type(var).__name__)
                )
                try:
                    inputs_onnx_names_log.append(self.get_name(var))
                except Exception:
                    inputs_onnx_names_log.append("<ONNX name not found/assigned>")

        outputs_aval_log = []
        outputs_jax_vars_log = []
        outputs_onnx_names_log = []
        for var in eqn.outvars:
            outputs_jax_vars_log.append(str(var))
            outputs_aval_log.append(
                (tuple(var.aval.shape), str(var.aval.dtype), type(var).__name__)
            )
            try:
                # self.get_var_name is generally used to get/create a name for any var,
                # including outputs that are about to be generated.
                outputs_onnx_names_log.append(self.get_var_name(var))
            except Exception:
                outputs_onnx_names_log.append("<ONNX name TBD>")

        self.primitive_call_counter += (
            1  # Increment before creating log, if ID is 1-based from call
        )
        log_entry = RecordedPrimitiveCallLog(
            sequence_id=self.primitive_call_counter,
            primitive_name=prim_name,
            plugin_file_hint=plugin_hint,
            params=params_cleaned,
            params_repr=params_repr,
            inputs_aval=inputs_aval_log,
            outputs_aval=outputs_aval_log,
            conversion_context_fn_name=current_fn_context_name
            or self.function_context_for_recording,
            # New fields
            inputs_jax_vars=inputs_jax_vars_log,
            inputs_onnx_names=inputs_onnx_names_log,
            outputs_jax_vars=outputs_jax_vars_log,
            outputs_onnx_names=outputs_onnx_names_log,
        )
        self.recorded_calls_log.append(log_entry)

    def _format_param_for_log(self, param_val: Any) -> str:
        # Helper to format params nicely, especially large arrays or complex objects
        if isinstance(param_val, (np.ndarray, jax.Array)):
            if param_val.size > 10:  # Example threshold
                return f"Array<shape={param_val.shape}, dtype={param_val.dtype}> (values hidden)"
            return str(param_val)  # Or a more concise repr
        # Add more types as needed (e.g., jax.ShapedArray if params can be that)
        return repr(param_val)

    def _process_eqn(self, eqn: Any) -> None:
        """Process a single JAXPR equation by dispatching to the appropriate plugin handler."""

        if not hasattr(eqn, "primitive"):
            # Should not happen for standard jaxprs, maybe handle call primitives?
            self.logger.warning(
                f"Equation type without 'primitive' attribute encountered: {type(eqn)}. Skipping: {eqn}"
            )
            return

        primitive = eqn.primitive
        name = primitive.name

        # ------------------------------------------------------------------------------
        # -- record this primitive call if enabled -------------------------------------
        # ------------------------------------------------------------------------------
        if self.record_primitive_calls_file:
            # Get plugin hint
            plugin_hint = self._get_plugin_file_hint(primitive)

            # Use new logging method
            self.log_primitive_call(
                eqn=eqn,
                plugin_hint=plugin_hint,
                current_fn_context_name=self.function_context_for_recording,
            )

            # If we've reached a threshold or it's a less common primitive,
            # write the log to file
            if self.primitive_call_counter % 100 == 0 or name not in {
                "add",
                "mul",
                "reshape",
            }:
                try:
                    from jax2onnx.utils.debug import save_primitive_calls_log

                    save_primitive_calls_log(
                        self.recorded_calls_log, self.record_primitive_calls_file
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to save primitive calls log: {e}")
        # -- end recording hook -------------------------------------------------------

        # Check if it's handled by function plugins first (if applicable)
        is_function_handler = (
            name in ONNX_FUNCTION_PLUGIN_REGISTRY
        )  # Use the actual registry name

        handler = self.primitive_handlers.get(name)
        if handler is None:
            raise NotImplementedError(
                f"No ONNX handler registered for JAX primitive: '{name}'"
            )

        self.logger.debug(f"Processing eqn for primitive: {name}")
        try:
            # Call the handler (typically a method on the plugin instance or a lambda)
            # The handler expects: self (converter), eqn object, params dict
            handler(self, eqn, eqn.params)  # Pass self, eqn, params
        except Exception as e:
            self.logger.error(
                f"Error processing primitive '{name}' with handler {handler}. Eqn: {eqn}",
                exc_info=True,
            )
            raise RuntimeError(f"Failed processing primitive '{name}'") from e

        # --- Handle output shapes for non-function primitives ---
        # (This logic might need refinement based on how function plugin outputs are handled)
        # Register shape info for the output variables of this equation
        # Use the safe dim-to-symbol conversion
        if not is_function_handler:  # Only for leaf primitives?
            for outvar in eqn.outvars:
                if (
                    outvar is not None
                    and hasattr(outvar, "aval")
                    and hasattr(outvar.aval, "shape")
                ):
                    output_name = self.get_name(outvar)  # Get ONNX name
                    # Use the potentially symbolic shape from the aval
                    shape_tuple = tuple(
                        self._dim_to_symbol_safe(d) for d in outvar.aval.shape
                    )
                    dtype = outvar.aval.dtype
                    # Add shape info to the builder
                    self.add_shape_info(output_name, shape_tuple, dtype)
                else:
                    # Handle literals or vars without shape info if necessary
                    pass

    def _handle_cond(self, eqn, params):
        # ─── register all cond inputs so they get value_info ────────────────
        # ...existing code...

        # ─── now build and inline the THEN‐ and ELSE‐subgraphs ───────────────
        then_closed, else_closed = params["branches"]

        # THEN branch (calls stub for now)
        self.builder.subgraph(
            name="then_body",
            invars=[self.get_name(v) for v in eqn.invars[1:]],
            jaxpr=then_closed.jaxpr,
        )

        # ELSE branch (calls stub for now)
        self.builder.subgraph(
            name="else_body",
            invars=[self.get_name(v) for v in eqn.invars[1:]],
            jaxpr=else_closed.jaxpr,
        )

        # Finally emit the ONNX If node (subgraph bodies currently stubbed)
        self.builder.add_node(
            "If",
            inputs=[self.get_name(eqn.invars[0])],
            outputs=[self.get_name(v) for v in eqn.outvars],
            name=self.unique_name("If"),
        )

    def _get_plugin_file_hint(self, primitive) -> Optional[str]:
        """Get a hint about which plugin file might handle this primitive."""
        name = primitive.name

        # Check if it's in our plugin registry
        if name in PLUGIN_REGISTRY:
            plugin = PLUGIN_REGISTRY[name]
            return f"{plugin.__class__.__module__}.{plugin.__class__.__name__}"

        # Check function plugins
        if name in ONNX_FUNCTION_PLUGIN_REGISTRY:
            plugin = ONNX_FUNCTION_PLUGIN_REGISTRY[name]
            return f"{plugin.__class__.__module__}.{plugin.__class__.__name__}"

        # Special case for pjit
        if name == "pjit":
            return "jax2onnx.converter.jaxpr_converter.Jaxpr2OnnxConverter._handle_pjit"

        # No hint found
        return None

    # ------------------------------------------------------------------
    # Sub‐graph helper (stub) on the converter front‐end
    # ------------------------------------------------------------------
    def subgraph(
        self,
        name: str,
        invars: Sequence[str],
        jaxpr: "ClosedJaxpr",
    ) -> "OnnxBuilder":
        """
        Stub passthrough so that plugins can call `converter.subgraph(...)`
        and it simply delegates to the builder.subgraph stub.
        """
        return self.builder.subgraph(name=name, invars=invars, jaxpr=jaxpr)

    def get_var_from_tracer(self, tracer):
        """Get the underlying JAX variable from a tracer.

        This is used to find the source variable that produced a tracer value,
        which is important when dealing with captured variables in closures.
        """
        return tracer._trace.full_raise(tracer)
