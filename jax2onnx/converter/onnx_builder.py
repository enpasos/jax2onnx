# file: jax2onnx/converter/onnx_builder.py

from typing import Any, Dict, Sequence, Union, Optional, List, Tuple, cast

import logging

import numpy as np
import onnx
from jax.extend.core import Literal, ClosedJaxpr
from onnx import (
    FunctionProto,
    GraphProto,
    ModelProto,
    NodeProto,
    TensorProto,
    ValueInfoProto,
    TypeProto,
    TensorShapeProto,
    helper,
    AttributeProto,
)

# === Import name generators ===
from jax2onnx.converter.name_generator import UniqueNameGenerator

logger = logging.getLogger("jax2onnx.converter.onnx_builder")

CUSTOM_DOMAIN = "custom"
CUSTOM_DOMAIN_VERSION = 1

# Define Shape type for type checking
Shape = Union[Tuple[Any, ...], List[Any], None]

# Add a specific type for the value_info_metadata entries
ValueInfoMetadataType = Tuple[Tuple[Any, ...], Any]
ValueInfoMetadataWithOriginType = Tuple[Tuple[Any, ...], Any, Optional[str]]

DIMVAR_STR2SYMBOL: dict[str, str] = {}  # populated by converter


def _as_tuple(x):
    """
    Converts the input into a tuple if it is not already a tuple or list.

    Args:
        x: Input value, which can be a list, tuple, or other type.

    Returns:
        A tuple containing the input value(s).
    """
    return tuple(x) if isinstance(x, (list, tuple)) else (x,)


# You can define this globally (in onnx_builder.py)
ONNX_DTYPE_MAP = {
    np.float32: TensorProto.FLOAT,
    np.dtype("float32"): TensorProto.FLOAT,
    np.float64: TensorProto.DOUBLE,
    np.dtype("float64"): TensorProto.DOUBLE,
    np.int8: TensorProto.INT8,
    np.dtype("int8"): TensorProto.INT8,
    np.uint8: TensorProto.UINT8,
    np.dtype("uint8"): TensorProto.UINT8,
    np.int16: TensorProto.INT16,
    np.dtype("int16"): TensorProto.INT16,
    np.uint16: TensorProto.UINT16,
    np.dtype("uint16"): TensorProto.UINT16,
    np.int32: TensorProto.INT32,
    np.dtype("int32"): TensorProto.INT32,
    np.uint32: TensorProto.UINT32,
    np.dtype("uint32"): TensorProto.UINT32,
    np.int64: TensorProto.INT64,
    np.dtype("int64"): TensorProto.INT64,
    np.uint64: TensorProto.UINT64,
    np.dtype("uint64"): TensorProto.UINT64,
    np.bool_: TensorProto.BOOL,
    np.dtype("bool"): TensorProto.BOOL,
    bool: TensorProto.BOOL,
    "int64": TensorProto.INT64,
    "bool": TensorProto.BOOL,
}


# ─── new util helpers ────────────────────────────────────────────────────────
def _is_unknown_dim(d) -> bool:  # -1 / None / ""  → unknown
    return d in (-1, None, "")


def _is_shape_more_specific(old: tuple, new: tuple) -> bool:
    """
    Return True if `new` refines `old`, e.g. (-1,) → ('B',) or
    contains concrete ints where the old one had -1 / None.
    """
    if len(old) != len(new):
        return True
    for o, n in zip(old, new):
        if o in (-1, None) and n not in (-1, None):
            return True
    return False


# Convert the method to a standalone function that takes an object and dimension
def _symbol_name(obj, dim) -> str:
    """Get a symbolic dimension name from a dimension object.

    Args:
        obj: The object containing var_to_symbol_map (typically OnnxBuilder)
        dim: The dimension object (could be int, str, or DimVar)

    Returns:
        A string representation of the dimension
    """
    name = str(dim) if not isinstance(dim, str) else dim
    if hasattr(obj, "var_to_symbol_map"):
        resolved = obj.var_to_symbol_map.get(dim, obj.var_to_symbol_map.get(str(dim)))
        final = resolved or name
    else:
        final = name
    # ─────────────  DEBUG  ─────────────
    logger.debug("[_symbol_name] dim=%s (%s)  →  %s", dim, type(dim).__name__, final)
    # ──────────────────────────────────
    return final


def _canonical_symbol(builder, dim):
    """Return either an int or a user-friendly symbolic name."""
    if isinstance(dim, int):
        return dim

    # First try to get from builder's var_to_symbol_map
    if hasattr(builder, "var_to_symbol_map"):
        # Try direct lookup
        if dim in builder.var_to_symbol_map:
            return builder.var_to_symbol_map[dim]
        # Try string lookup
        if str(dim) in builder.var_to_symbol_map:
            return builder.var_to_symbol_map[str(dim)]

    # Then try name for dimension by id if available
    if hasattr(builder, "symbol_name_for_dim"):
        name = builder.symbol_name_for_dim.get(id(dim))
        if name is not None:
            return name

    # Fall

    # Fall back to _symbol_name for compatibility
    if hasattr(dim, "symbol") and dim.symbol:
        return str(dim.symbol)

    # For string dimensions, return as is
    if isinstance(dim, str):
        return dim

    # Last resort: convert to string
    return str(dim)


def _resolve_symbol(obj, dim):
    """Resolve a symbolic dimension to its canonical name.

    Args:
        obj: The object containing var_to_symbol_map (typically OnnxBuilder)
        dim: The dimension object (could be int, str, or DimVar)

    Returns:
        The resolved symbolic name or str(dim) if not found
    """
    # first pass through the fast-path
    if hasattr(obj, "var_to_symbol_map"):
        resolved = obj.var_to_symbol_map.get(dim, obj.var_to_symbol_map.get(str(dim)))
        final = resolved or str(dim)
    else:
        final = str(dim)
    # ─────────────  DEBUG  ─────────────
    logger.debug(
        "[_resolve_symbol] %s (%s)  →  %s   (table=%s)",
        dim,
        type(dim).__name__,
        final,
        getattr(obj, "var_to_symbol_map", {}),
    )
    # ──────────────────────────────────
    return final


def _to_dim_proto_val(dim):
    """
    Convert a JAX shape element into (is_param, value_or_name).

    • int           → (False,  int_value)
    • str           → (True,   "B")
    • JAX DimVar    → (True,   dim.symbol)
    """
    if isinstance(dim, int):
        return False, dim
    if isinstance(dim, str):
        return True, dim
    # JAX Dimension variables have a `.symbol` attribute
    if hasattr(dim, "symbol"):
        return True, str(dim.symbol)
    # Fallback
    return True, ""


class OnnxBuilder:
    """
    A builder class for constructing ONNX models, including nodes, inputs, outputs,
    initializers, and metadata.
    """

    def __init__(
        self,
        name_generator: UniqueNameGenerator,
        opset: int = 21,
        model_name: str = "",
        initializers: list[Any] | None = None,
        converter: Any = None,  # <-- Add converter argument
        enable_double_precision: bool = False,  # Add this
    ) -> None:
        # Initialize the ONNX builder with default values and configurations.
        self.name_generator: UniqueNameGenerator = name_generator

        # maps {DimVar-object-or-id → canonical user symbol, e.g. "B"}
        self.var_to_symbol_map: dict[Any, str] = {}

        self.nodes: list[NodeProto] = []
        self.inputs: list[ValueInfoProto] = []
        self.outputs: list[ValueInfoProto] = []
        self.initializers: list[Any] = initializers if initializers is not None else []
        self.value_info: list[ValueInfoProto] = []
        self.opset: int = opset
        self.functions: dict[str, FunctionProto] = {}
        self.model_name: str = model_name
        self.display_name_map: dict[str, str] = {}
        self.enable_double_precision = enable_double_precision  # Store the flag
        self.working_dtype_onnx = (
            onnx.TensorProto.DOUBLE
            if enable_double_precision
            else onnx.TensorProto.FLOAT
        )

        # Metadata for value information.
        # Update type annotations to match the more flexible type needs
        self.value_info_metadata: dict[str, ValueInfoMetadataType] = {}
        self.value_info_metadata_with_origin: dict[
            str, ValueInfoMetadataWithOriginType
        ] = {}
        self.dtype_env: dict[str, onnx.TensorProto.DataType] = {}
        self.value_info_origin: dict[str, str] = {}  # Initialize value_info_origin
        self.dimvar_to_name: Dict[Any, str] = {}  # Initialize mapping explicitly
        self.dimvar_to_name_by_str: Dict[str, str] = (
            {}
        )  # Add mapping by string representation
        self.converter = converter  # <-- Store converter reference
        self.symbolic_shapes: dict[str, tuple[Any, ...]] = {}

    # ------------------------------------------------------------------
    # Symbolic‐dimension origin registry
    # ------------------------------------------------------------------
    def _register_symbol_origin(self, dim: Any, tensor_name: str, axis: int):
        """
        Record that the symbolic dimension `dim` (or its string) comes from
        axis `axis` of the top‐level tensor `tensor_name`, so later
        plugins can look it up via converter.symbolic_dim_to_origin.
        """
        conv = getattr(self, "converter", None)
        if conv is None:
            return
        # Ensure the map exists on the converter
        mapping = getattr(conv, "symbolic_dim_to_origin", None)
        if mapping is None:
            mapping = {}
            setattr(conv, "symbolic_dim_to_origin", mapping)

        # Register both the raw dim object and its str key
        mapping[dim] = (tensor_name, axis)
        try:
            mapping[str(dim)] = (tensor_name, axis)
        except Exception:
            pass

    def make_value_info(self, name: str, shape: Shape, dtype: Any):
        # Ensure shape is always a tuple (handle None case)
        shape_tuple = () if shape is None else _as_tuple(shape)

        from onnx import ValueInfoProto, TensorProto
        import logging

        logger = logging.getLogger("jax2onnx.converter.onnx_builder")

        vi = ValueInfoProto()
        vi.name = name

        tensor_type = TypeProto.Tensor()
        tensor_type.elem_type = (
            dtype
            if isinstance(dtype, int)
            else ONNX_DTYPE_MAP.get(dtype, TensorProto.FLOAT)
        )

        logger.debug(
            f"🔍 make_value_info for '{name}' with shape={shape_tuple}, dtype={dtype}"
        )

        tensor_shape = TensorShapeProto()
        for i, dim in enumerate(shape_tuple):
            dim_proto = TensorShapeProto.Dimension()
            if isinstance(dim, int):
                dim_proto.dim_value = dim
                logger.debug(f"  - dim[{i}] = {dim} (int value)")
            else:
                friendly = _resolve_symbol(self, dim)
                if friendly is None:
                    friendly = _symbol_name(self, dim)  # ← current fallback
                dim_proto.dim_param = friendly
                logger.debug(f"  - dim[{i}] = {dim} (type={type(dim).__name__})")
                logger.debug(f"    → final dim_param = '{friendly}'")

            tensor_shape.dim.append(dim_proto)

        tensor_type.shape.CopyFrom(tensor_shape)
        vi.type.tensor_type.CopyFrom(tensor_type)
        return vi

    def register_value_info_metadata(
        self,
        name: str,
        shape: Shape,
        dtype: Union[np.dtype, int],
        origin: Optional[str] = None,
    ):
        # Ensure shape is always a tuple
        shape_tuple = () if shape is None else _as_tuple(shape)

        """
        Register metadata for a value_info entry, including shape, dtype, and origin.

        Args:
            name: Name of the variable.
            shape: Shape of the variable as a tuple.
            dtype: Data type of the variable (NumPy dtype or ONNX TensorProto enum).
            origin: Optional description of the metadata's origin.
        """
        import logging

        logger = logging.getLogger("jax2onnx.converter.onnx_builder")

        logger.debug(
            f"🔍 [register_value_info_metadata] name={name}, shape={shape_tuple} (type={type(shape_tuple).__name__}), dtype={dtype}"
        )

        # Log each dimension's type to help identify problematic dimensions
        if shape_tuple:
            for i, dim in enumerate(shape_tuple):
                logger.debug(f"  - shape[{i}] = {dim} (type={type(dim).__name__})")

                # Check if dim is in dimvar_to_name mapping
                if hasattr(self, "dimvar_to_name") and dim in self.dimvar_to_name:
                    logger.debug(
                        f"    ✓ Found in dimvar_to_name: {self.dimvar_to_name[dim]}"
                    )

                # Check string-based mapping
                if (
                    hasattr(self, "dimvar_to_name_by_str")
                    and str(dim) in self.dimvar_to_name_by_str
                ):
                    logger.debug(
                        f"    ✓ Found in dimvar_to_name_by_str: {self.dimvar_to_name_by_str[str(dim)]}"
                    )

        # Use symbolic shape if available
        sym = getattr(self, "converter", None)
        if sym and hasattr(sym, "symbolic_shapes"):
            old_shape = shape
            shape = sym.symbolic_shapes.get(name, shape)
            if shape != old_shape:
                logger.debug(
                    f"  → Shape overridden from symbolic_shapes: {old_shape} → {shape}"
                )

        # Cast to the expected types to fix type errors
        self.value_info_metadata[name] = cast(
            ValueInfoMetadataType, (shape_tuple, dtype)
        )
        self.value_info_metadata_with_origin[name] = cast(
            ValueInfoMetadataWithOriginType, (shape_tuple, dtype, origin or "traced")
        )

    def add_initializer_from_scalar(self, name, value):
        from onnx import TensorProto
        import numpy as np

        if isinstance(value, bool):
            dtype = TensorProto.BOOL
            np_value = np.array(value, dtype=np.bool_)
        elif isinstance(value, int):
            dtype = TensorProto.INT64
            np_value = np.array(value, dtype=np.int64)
        else:  # float
            dtype = TensorProto.FLOAT
            np_value = np.array(value, dtype=np.float32)

        # Create the tensor with proper boolean handling
        if np_value.dtype == np.bool_:
            tensor = helper.make_tensor(
                name=name,
                data_type=TensorProto.BOOL,
                dims=np_value.shape,
                # Use bool_data instead of int32_data for boolean values
                vals=np_value.astype(np.bool_).flatten().tolist(),
            )
            self.initializers.append(tensor)
            self.register_value_info_metadata(
                name, shape=tuple(np_value.shape), dtype=TensorProto.BOOL
            )
            return name
        else:
            # Regular handling for non-boolean types
            return self.add_initializer(name, np_value, dtype, [])

    def to_function_proto(self, name):
        return onnx.helper.make_function(
            domain="",
            name=name,
            inputs=self.input_value_infos,
            outputs=self.output_value_infos,
            nodes=self.nodes,
            opset_imports=[onnx.helper.make_opsetid("", self.opset_version)],
        )

    def get_value_info_metadata_with_origin(
        self, name: str
    ) -> tuple[tuple[int, ...], Any, str | None] | None:
        """
        Retrieve metadata (shape, dtype, origin) for a given value_info name.

        Args:
            name: Name of the value_info entry.

        Returns:
            A tuple containing shape, dtype, and origin, or None if not found.
        """
        if name in self.value_info_metadata_with_origin:
            return self.value_info_metadata_with_origin[name]
        if name in self.value_info_metadata:
            shape, dtype = self.value_info_metadata[name]
            return shape, dtype, None  # origin unknown
        return None

    def find_missing_value_info(self) -> list[str]:
        """
        Identify value_info entries that are referenced in nodes but not defined.

        Returns:
            A list of names for missing value_info entries.
        """
        known_names = {vi.name for vi in self.inputs + self.outputs + self.value_info}
        known_names.update(init.name for init in self.initializers)
        node_names = {
            name for n in self.nodes for name in list(n.input) + list(n.output)
        }
        return sorted(name for name in node_names if name not in known_names)

    def get_constant_name(self, val):
        # If val is a JAX Literal, unwrap it to its Python value
        if isinstance(val, Literal):  # Use the correctly imported Literal
            val = val.val

        # Determine the ONNX TensorProto type and prepare np_val
        if isinstance(val, (bool, int, float)):
            # For Python scalars
            if isinstance(val, bool):
                np_val = np.array(val, dtype=np.bool_)
                # onnx_dtype = TensorProto.BOOL # Inferred by helper.make_tensor from np_val.dtype
            elif isinstance(val, int):
                # Always emit Python integer literals as INT64 in ONNX,
                # so that loop‐carried counters (and any other ints) match expected ONNX types.
                np_val = np.array(val, dtype=np.int64)
            else:  # float
                if self.enable_double_precision:
                    np_val = np.array(val, dtype=np.float64)
                    # onnx_dtype = TensorProto.DOUBLE
                else:
                    np_val = np.array(val, dtype=np.float32)
                    # onnx_dtype = TensorProto.FLOAT
        else:
            # For NumPy arrays, JAX arrays, or other array-like objects
            if not isinstance(val, np.ndarray):
                np_val = np.asarray(val)  # Convert JAX arrays etc. to NumPy arrays
            else:
                np_val = val  # It's already a NumPy array

            # Adjust float precision based on enable_double_precision
            if np.issubdtype(np_val.dtype, np.floating):
                if self.enable_double_precision:
                    if np_val.dtype != np.float64:
                        np_val = np_val.astype(np.float64)
                else:  # not enable_double_precision
                    if np_val.dtype != np.float32:
                        # Ensure float32 if it's any other float type (e.g. float64, float16)
                        np_val = np_val.astype(np.float32)
            # For integer or boolean np.ndarray, their existing dtype (e.g., int32, int64, bool) is preserved.

        # Get the ONNX dtype enum from numpy dtype
        dtype_enum = self._numpy_dtype_to_onnx(np_val.dtype)

        name = self.get_unique_instance_name("const")
        tensor = helper.make_tensor(
            name=name,
            data_type=dtype_enum,
            dims=np_val.shape,
            vals=np_val.flatten().tolist(),
        )
        self.initializers.append(tensor)
        self.register_value_info_metadata(
            name,
            shape=tuple(np_val.shape),
            dtype=dtype_enum,
        )
        return name

    def reset(self) -> None:
        self.name_generator = UniqueNameGenerator()
        self.nodes = []
        self.inputs = []
        self.outputs = []
        self.initializers = []
        self.value_info = []
        self.functions.clear()
        self.display_name_map.clear()
        self.value_info_metadata.clear()
        self.value_info_metadata_with_origin.clear()
        self.dtype_env.clear()
        self.value_info_origin.clear()

    def get_unique_name(self, prefix: str = "node") -> str:
        return self.name_generator.get(prefix)

    def get_unique_instance_name(self, base_name: str) -> str:
        return self.name_generator.get(base_name)

    def add_initializer(
        self, name, vals, data_type=helper.TensorProto.INT64, dims=None
    ):
        if dims is None:
            dims = [len(vals)] if isinstance(vals, (list, tuple)) else []
        flat_vals = np.array(vals).flatten().tolist()
        tensor = helper.make_tensor(
            name=name, data_type=data_type, dims=dims, vals=flat_vals
        )
        self.initializers.append(tensor)

        self.register_value_info_metadata(name, shape=tuple(dims), dtype=data_type)

        return name

    def _add_tensor(
        self,
        collection: list[ValueInfoProto],
        name: str,
        shape: Shape,
        dtype: Any,
    ):
        # Ensure shape is always a tuple
        shape_tuple = () if shape is None else _as_tuple(shape)

        # Use our centralized make_value_info function for consistency
        tensor_def = self.make_value_info(name, shape_tuple, dtype)
        collection.append(tensor_def)

    def change_var_name(self, old_name, new_name) -> None:
        """Change the name of a JAX variable."""
        # check  dtype_env
        dtype_env = self.dtype_env.get(old_name)
        self.dtype_env[new_name] = dtype_env
        # correct inputs
        for i, vi in enumerate(self.inputs):
            if vi.name == old_name:
                self.inputs[i].name = new_name
                break
        # correct outputs
        for i, vi in enumerate(self.outputs):
            if vi.name == old_name:
                self.outputs[i].name = new_name
                break

    def add_input(
        self,
        name: str,
        shape: tuple[Any, ...] | None,
        dtype: Any = np.float32,
    ) -> None:
        # ──────────────────────────────────────────────────────────────────
        # Do **not** promote the tensor to a formal graph input when it is
        # already created inside the graph (i.e. it appears in the output
        # list of a node) – or when it is an input already.
        # This prevents duplicate-name errors such as
        #   "Duplicate definition of name (loop_0_iter32_0)".
        # ──────────────────────────────────────────────────────────────────
        if any(name in n.output for n in self.nodes):
            # internal tensor – only record shape information if missing
            self.add_value_info(name, shape, dtype)
            return

        if any(vi.name == name for vi in self.inputs):
            # already a formal input – keep first declaration
            return

        self.dtype_env[name] = dtype
        self._add_tensor(self.inputs, name, shape, dtype)
        # ─── register any symbolic dims on this new input ───────────────
        if shape is not None:
            for axis, dim in enumerate(shape):
                if not isinstance(dim, int):
                    self._register_symbol_origin(dim, name, axis)

    def add_output(
        self,
        name: str,
        shape: tuple[Any, ...] | None,
        dtype: Any = np.float32,  # Fix type annotation
    ) -> str:
        # Do not emit the same graph-output twice
        if any(vi.name == name for vi in self.outputs):
            return name
        self.dtype_env[name] = dtype
        self._add_tensor(self.outputs, name, shape, dtype)
        # ─── register any symbolic dims on this new graph-output ────────
        if shape is not None:
            for ax, d in enumerate(shape):
                if not isinstance(d, int):
                    self._register_symbol_origin(d, name, ax)
        return name

    def add_value_info(
        self,
        name: str,
        shape: Shape,
        dtype: Union[np.dtype, int],
    ):
        # Ensure shape is always a tuple
        shape_tuple = () if shape is None else _as_tuple(shape)

        # Use symbolic shape if registered (override shape as in register_value_info_metadata)
        sym = getattr(self, "converter", None)
        if sym and hasattr(sym, "symbolic_shapes"):
            shape_tuple = sym.symbolic_shapes.get(name, shape_tuple)

        vi = self.make_value_info(name, shape_tuple, dtype)

        # Enrich doc_string if we have origin info
        origin = self.value_info_origin.get(name)
        if origin:
            vi.doc_string = f"origin: {origin}"

        self.value_info.append(vi)

        # ─── determine ONNX enum dtype ─────────────────────────────────
        if isinstance(dtype, int):
            onnx_dtype = dtype
        else:
            onnx_dtype = vi.type.tensor_type.elem_type

        # ─── register metadata ─────────────────────────────────────────
        self.register_value_info_metadata(name, shape_tuple, onnx_dtype)

        # ─── register any symbolic dims on this intermediate tensor ─────
        for ax, d in enumerate(shape_tuple):
            if not isinstance(d, int):
                self._register_symbol_origin(d, name, ax)

    def create_node(
        self, op_type: str, inputs: list[str], outputs: list[str], **kwargs: Any
    ) -> NodeProto:
        return helper.make_node(op_type, inputs, outputs, **kwargs)

    def add_node(self, node: NodeProto) -> None:
        self.nodes.append(node)

    def _register_deterministic_parameters(self, missing_names: list[str]) -> list[str]:
        """
        Automatically register deterministic flags for dropout layers.

        Args:
            missing_names: List of missing value_info names

        Returns:
            List of still missing value_info names after deterministic flags are handled
        """
        remaining_missing = []
        for name in missing_names:
            if name.endswith("_deterministic") or name == "deterministic":
                # Register deterministic flags as boolean tensors (BOOL)
                self.register_value_info_metadata(
                    name=name,
                    shape=(),  # Scalar boolean value
                    dtype=onnx.TensorProto.BOOL,
                    origin="auto-registered deterministic flag",
                )
                # Immediately add the value_info as well
                self.add_value_info(name, shape=(), dtype=onnx.TensorProto.BOOL)
            else:
                remaining_missing.append(name)
        return remaining_missing

    def create_graph(
        self, name: str, is_subgraph: bool = False, empty_inputs: bool = False
    ) -> GraphProto:
        """Creates a GraphProto, passing the is_subgraph flag."""
        return self._build_graph(
            name, is_subgraph=is_subgraph, empty_inputs=empty_inputs
        )

    def _build_graph(
        self, name=None, is_subgraph=False, empty_inputs=False
    ) -> onnx.GraphProto:
        """Build the ONNX graph."""
        name = name or self.model_name
        logger.debug(
            f"Building graph '{name}', is_subgraph={is_subgraph}, empty_inputs={empty_inputs}"
        )
        # 1. Filter unused initializers (safe for subgraphs too)
        self.filter_unused_initializers()

        # 1.a Strict topology check: every node input must have been produced already
        self._assert_topologically_sorted()

        if not is_subgraph:
            # For the main graph, filter redundant inputs.
            self._filter_redundant_inputs()

        missing = self.find_missing_value_info()

        # Automatically handle deterministic flags
        if missing:
            missing = self._register_deterministic_parameters(missing)

        # Filter out any intermediate conv_transpose outputs
        if missing:
            missing = [m for m in missing if not m.startswith("conv_transpose_out")]

        if missing:
            raise RuntimeError(
                f"Missing value_info for: {missing} in graph '{name}'\n\nConsider adding them using `builder.add_value_info(...)` or `register_value_info_metadata(...)`"
            )

        # If empty_inputs is requested, use an empty list for the graph inputs.
        # Otherwise, use the builder's current inputs.
        final_inputs = [] if empty_inputs else self.inputs

        return helper.make_graph(
            nodes=self.nodes,
            name=name,
            inputs=final_inputs,
            outputs=self.outputs,
            initializer=self.initializers,
            value_info=self.value_info,
        )

    def create_model(self, graph: GraphProto) -> ModelProto:
        return self._finalize_model(graph)

    def create_onnx_model(self, model_name: str) -> onnx.ModelProto:
        graph = self._build_graph(model_name)
        return self._finalize_model(graph)

    def _finalize_model(self, graph: GraphProto) -> ModelProto:
        opset_imports = [
            helper.make_opsetid("", self.opset),
            *(
                [helper.make_opsetid(CUSTOM_DOMAIN, CUSTOM_DOMAIN_VERSION)]
                if self.functions
                else []
            ),
        ]

        unique_function_protos = list(
            {f.name: f for f in self.functions.values()}.values()
        )

        names = [f.name for f in unique_function_protos]
        seen, duplicates = set(), set()
        for n in names:
            if n in seen:
                duplicates.add(n)
            seen.add(n)
        if duplicates:
            logging.warning(f"Duplicate ONNX functions detected: {sorted(duplicates)}")
        else:
            logging.debug("✅ No duplicate ONNX function names")

        model = helper.make_model(
            graph,
            opset_imports=opset_imports,
            functions=unique_function_protos,
        )
        return model

    def _numpy_dtype_to_onnx(self, dtype: Any) -> int:
        """
        Convert a numpy dtype to ONNX TensorProto dtype.
        This is a simplified version that leverages the same mapping used in make_value_info.
        """
        # If dtype is already an integer (ONNX enum), return it directly
        if isinstance(dtype, int):
            return dtype

        # Otherwise use the make_value_info logic for consistency
        # Create a dummy tensor and extract its dtype
        dummy_info = self.make_value_info("dummy", (), dtype)
        return dummy_info.type.tensor_type.elem_type

    def add_function(
        self,
        name: str,
        sub_builder: "OnnxBuilder",
        param_input_names: list[str],
        sub_converter=None,
    ) -> str:
        missing = sub_builder.find_missing_value_info()  # Existing code

        # Handle parameters that might be missing from value_info
        if missing:
            from onnx import TensorProto

            # Handle the common case of missing 'deterministic' parameter
            if "deterministic" in missing:
                # Always use BOOL for boolean parameters
                sub_builder.register_value_info_metadata(
                    "deterministic", (), TensorProto.BOOL, origin="function_param_auto"
                )
                sub_builder.add_value_info("deterministic", (), TensorProto.BOOL)
                logging.debug(
                    f"Auto-registered deterministic parameter in function '{name}' as BOOL"
                )
                # Check if we still have missing items
                missing = sub_builder.find_missing_value_info()

        # Raise error if there are still missing items
        if missing:  # Existing code
            raise RuntimeError(  # Existing code
                f"Missing value_info in function '{name}': {missing}\n\nFix the corresponding plugin using `register_value_info_metadata(...)`"
            )

        function_graph = sub_builder.create_graph(name + "_graph")  # Existing code
        # These are the internal names used for function outputs
        internal_output_names = [
            vi.name for vi in function_graph.output
        ]  # Modified variable name for clarity

        # --- START REFINED CHANGE ---
        # Construct the final input names list, handling both generic and descriptive names
        final_input_names = []
        seen_names = set()

        # If we have access to the sub_converter, use it to resolve descriptive names
        if (
            sub_converter is not None
            and hasattr(sub_converter, "jaxpr")
            and hasattr(sub_converter, "var_to_name")
        ):
            logging.debug(
                f"Using sub_converter to deduplicate function inputs for '{name}'"
            )

            # Get the original input variables from the sub_converter's jaxpr
            original_internal_input_vars = sub_converter.jaxpr.invars

            # Map all original input variables to their FINAL descriptive names
            for var in original_internal_input_vars:
                # Use the sub_converter's map to get the potentially renamed final name
                final_name = sub_converter.var_to_name.get(var, None)
                if final_name is None:
                    # Handle cases where a var might not be in the map
                    logging.warning(
                        f"Could not find final name for input var: {var}. Skipping."
                    )
                    continue

                # Ensure uniqueness in the final list
                if final_name not in seen_names:
                    final_input_names.append(final_name)
                    seen_names.add(final_name)

                    # Always ensure deterministic parameter is registered with BOOL
                    if final_name == "deterministic":
                        from onnx import TensorProto

                        sub_builder.register_value_info_metadata(
                            "deterministic",
                            (),
                            TensorProto.BOOL,
                            origin="function_param_forced",
                        )
                        sub_builder.add_value_info(
                            "deterministic", (), TensorProto.BOOL
                        )
                        logging.debug(
                            f"Force-updated deterministic parameter to BOOL in function '{name}'"
                        )
                else:
                    logging.debug(f"Deduplicating function input name: {final_name}")

            # Add any extra parameter inputs (like weights/constants)
            for param_name in param_input_names:
                if param_name not in seen_names:
                    # Generalize: always register user-supplied scalar parameters as scalar inputs
                    # Check if we have metadata for this parameter
                    try:
                        shape, dtype_enum = self.get_shape_dtype(param_name)
                        # If scalar (shape == ()), register as scalar input
                        if shape == ():
                            sub_builder.add_scalar_input(param_name, dtype_enum)
                        else:
                            # For non-scalars, add as normal input
                            sub_builder.add_input(param_name, shape, dtype_enum)
                    except Exception:
                        # If metadata is missing, fallback to add as scalar input with default float32
                        from onnx import TensorProto

                        sub_builder.add_scalar_input(param_name, TensorProto.FLOAT)
                    final_input_names.append(param_name)
                    seen_names.add(param_name)

            logging.debug(
                f"Final computed input names for function '{name}': {final_input_names}"
            )
        else:
            # Fallback to the original approach if sub_converter is not available
            internal_data_input_names = [vi.name for vi in function_graph.input]
            final_input_names = internal_data_input_names + param_input_names

        # 1. Get ValueInfo for intermediate/output tensors from the sub-builder
        intermediate_and_output_value_info = sub_builder.value_info

        # 2. Create ValueInfo for the function's inputs
        input_value_infos = []

        for input_name in final_input_names:
            try:
                # Look up shape/dtype in the main builder's metadata
                shape, dtype_enum = self.get_shape_dtype(input_name)

                # If this is the deterministic parameter, always use BOOL
                if input_name == "deterministic":
                    from onnx import TensorProto

                    dtype_enum = TensorProto.BOOL

                # Create ValueInfoProto for this input
                vi = helper.make_tensor_value_info(input_name, dtype_enum, shape)
                input_value_infos.append(vi)
            except ValueError:
                pass

        # 3. Combine input ValueInfo with intermediate/output ValueInfo
        combined_value_info_dict = {vi.name: vi for vi in input_value_infos}
        for vi in intermediate_and_output_value_info:
            if vi.name not in combined_value_info_dict:
                combined_value_info_dict[vi.name] = vi

        # Special handling for 'deterministic' parameter - CRITICAL FIX
        # Override any existing deterministic ValueInfo to ensure it uses BOOL
        if "deterministic" in combined_value_info_dict:
            from onnx import TensorProto

            deterministic_vi = helper.make_tensor_value_info(
                "deterministic", TensorProto.BOOL, ()
            )
            combined_value_info_dict["deterministic"] = deterministic_vi
            logging.debug(
                f"Forced deterministic parameter to BOOL type in function '{name}'"
            )

        final_function_value_info = list(combined_value_info_dict.values())

        function_proto = helper.make_function(
            domain=CUSTOM_DOMAIN,
            fname=name,
            inputs=final_input_names,
            outputs=internal_output_names,
            nodes=function_graph.node,
            opset_imports=[
                helper.make_opsetid("", self.opset),
                helper.make_opsetid(CUSTOM_DOMAIN, CUSTOM_DOMAIN_VERSION),
            ],
            value_info=final_function_value_info,
        )

        self.functions[name] = function_proto

        return name

    def _get_shape(self, vi):
        if hasattr(vi, "type") and hasattr(vi.type, "tensor_type"):
            shape_proto = vi.type.tensor_type.shape
            return [
                d.dim_value if d.HasField("dim_value") else None
                for d in shape_proto.dim
            ]
        return None

    def _get_dtype(self, vi):
        if hasattr(vi, "type") and hasattr(vi.type, "tensor_type"):
            return vi.type.tensor_type.elem_type
        return TensorProto.FLOAT  # default fallback

    def _register_value_info_for_function_inputs_outputs_and_intermediates(
        self, func: onnx.FunctionProto, input_names: list[str], output_names: list[str]
    ):

        # Inputs
        for func_input_name, outer_input_name in zip(
            func.input, input_names, strict=False
        ):
            vi = next((v for v in self.value_info if v.name == outer_input_name), None)
            if vi:
                self.add_value_info(
                    func_input_name, self._get_shape(vi), self._get_dtype(vi)
                )
            elif outer_input_name in self.value_info_metadata:
                shape, dtype = self.value_info_metadata[outer_input_name]
                self.add_value_info(func_input_name, shape, dtype)

        # Outputs
        for func_output_name, outer_output_name in zip(
            func.output, output_names, strict=False
        ):
            vi = next((v for v in self.value_info if v.name == outer_output_name), None)
            if vi:
                self.add_value_info(
                    func_output_name, self._get_shape(vi), self._get_dtype(vi)
                )
            elif outer_output_name in self.value_info_metadata:
                shape, dtype = self.value_info_metadata[outer_output_name]
                self.add_value_info(func_output_name, shape, dtype)

        # Intermediates
        all_known = set(func.input) | set(func.output)
        for node in func.node:
            for name in list(node.input) + list(node.output):
                if (
                    name
                    and name not in all_known
                    and name not in self.value_info_metadata
                ):
                    # Ensure shape is not None by providing a default empty tuple
                    self.add_value_info(name, (), TensorProto.FLOAT)

    def _register_value_info_if_missing(self, name: str):
        if name not in self.value_info:
            if name not in self.value_info_metadata:
                raise RuntimeError(f"[STRICT] Missing value_info_metadata for '{name}'")
            shape, dtype = self.value_info_metadata[name]

            if shape is None:
                # fallback for debugging
                logging.warn(f"[WARN] Missing metadata for: {name} — using fallback")
                shape = ()  # or None
            # print(
            #    f"[INFO] Registering value_info: {name}, shape={shape}, dtype={dtype}"
            # )
            self.add_value_info(name, shape, dtype)

    def _auto_fix_constant_value_info(self, name: str, value: np.ndarray):
        if name in self.value_info_metadata:
            return  # ✅ NEVER overwrite already correctly set metadata
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        shape = tuple(value.shape)
        onnx_dtype = self._numpy_dtype_to_onnx(value.dtype)
        self.register_value_info_metadata(name, shape=shape, dtype=onnx_dtype)

    def merge_functions_from(self, other: "OnnxBuilder"):
        for name, func in other.functions.items():
            if name not in self.functions:
                self.functions[name] = func

    def get_shape_dtype(self, var_name: str) -> tuple[tuple[int, ...], int]:
        metadata = self.value_info_metadata.get(var_name)
        if metadata is None:
            raise ValueError(
                f"[❌] Variable '{var_name}' not found in value_info_metadata."
            )
        shape, dtype = metadata
        return shape, dtype

    def add_function_call_node(
        self,
        function_name: str,
        input_names: list[str],
        output_names: list[str],
        node_name: str | None = None,
        op_type: str | None = None,
        user_display_name: str | None = None,
    ):
        if node_name is None:
            readable_base = (user_display_name or function_name).split(".")[-1]
            node_name = self.get_unique_instance_name(readable_base)
        else:
            node_name = node_name.split(".")[-1]

        # ✅ Create function call node
        node = helper.make_node(
            op_type=op_type or node_name,
            inputs=input_names,
            outputs=output_names,
            name=node_name,
            domain=CUSTOM_DOMAIN,
        )

        self.nodes.append(node)

    def _adjust_tensor_shape(self, tensor, shape_hint, batch_dims):
        if not tensor.type.HasField(
            "tensor_type"
        ) or not tensor.type.tensor_type.HasField("shape"):
            return
        tensor_dims = tensor.type.tensor_type.shape.dim
        num_tensor_dims = len(tensor_dims)
        for idx, dim_symbol in enumerate(shape_hint):
            if idx < num_tensor_dims and dim_symbol == "B":
                if tensor_dims[idx].HasField("dim_value"):
                    tensor_dims[idx].ClearField("dim_value")
                tensor_dims[idx].dim_param = "B"
        for idx in batch_dims:
            if idx < num_tensor_dims:
                if tensor_dims[idx].HasField("dim_value"):
                    tensor_dims[idx].ClearField("dim_value")
                tensor_dims[idx].dim_param = "B"

    def adjust_dynamic_batch_dimensions(self, input_shapes):
        # Identify which dimensions should be dynamic (marked as 'B')
        batch_dims = {
            idx for shape in input_shapes for idx, dim in enumerate(shape) if dim == "B"
        }
        if not batch_dims:
            return

        logging.debug(f"Making dimensions {batch_dims} dynamic in the ONNX model")

        # First, identify which inputs are tensor inputs vs scalar parameter inputs
        tensor_inputs = []
        param_inputs = []

        for inp in self.inputs:
            # Check if this input has dimensions
            has_dims = (
                inp.type.HasField("tensor_type")
                and inp.type.tensor_type.HasField("shape")
                and inp.type.tensor_type.shape.dim
            )

            if has_dims:
                tensor_inputs.append(inp)
            else:
                param_inputs.append(inp)

        logging.debug(
            f"Found {len(tensor_inputs)} tensor inputs and {len(param_inputs)} parameter inputs"
        )

        # Apply dynamic dimensions to all tensor inputs
        for i, tensor in enumerate(tensor_inputs):
            if i < len(input_shapes):
                logging.debug(f"Making dimensions dynamic for input: {tensor.name}")
                self._adjust_tensor_shape(tensor, input_shapes[i], batch_dims)
            else:
                logging.warn(f"No shape hint available for input: {tensor.name}")

        # Make all outputs dynamic as well
        for tensor in self.outputs:
            self._adjust_tensor_shape(tensor, [], batch_dims)

        # Also update all value_info to make batch dimensions dynamic
        for value_info in self.value_info:
            self._adjust_tensor_shape(value_info, [], batch_dims)

    def filter_unused_initializers(self):
        used_inputs = {inp for node in self.nodes for inp in node.input}
        for func_proto in self.functions.values():
            for node in func_proto.node:
                used_inputs.update(node.input)

        # Also preserve any initializer that *is* a graph output
        output_names = {out.name for out in self.outputs}
        self.initializers = [
            init
            for init in self.initializers
            if init.name in used_inputs or init.name in output_names
        ]

    def get_value_info_origins(self) -> dict[str, str]:
        """
        Returns a dictionary mapping each value name to its metadata origin.
        Example:
            {
                "var_0": "traced",
                "var_1": "recovered",
                ...
            }
        """
        if hasattr(self, "value_info_origin"):
            return dict(self.value_info_origin)
        return {}

    def print_value_info_summary(self) -> None:
        """
        Debug utility: prints all registered value_info entries with shape, dtype, and origin.
        """
        print("\n[🔎] ONNX ValueInfo Summary:")
        for name in sorted(self.value_info_metadata):
            shape, dtype = self.value_info_metadata[name]
            origin = self.value_info_origin.get(name, "unknown")
            print(f" - {name:30} shape={shape}, dtype={dtype}, origin={origin}")

    def merge_value_info_metadata_from(self, other: "OnnxBuilder"):
        """
        Merges value_info metadata from another OnnxBuilder into this one.

        Only adds metadata if the name is not already present.
        If a name already exists with a different shape or dtype, logs a warning.

        Args:
            other: Another OnnxBuilder instance whose metadata should be merged in.
        """
        for name, (shape, dtype) in other.value_info_metadata.items():
            if name not in self.value_info_metadata:
                self.value_info_metadata[name] = (shape, dtype)
            else:
                existing = self.value_info_metadata[name]
                if existing != (shape, dtype):
                    logging.warning(
                        f"⚠️ [merge] Mismatch in value_info for '{name}': "
                        f"existing={existing}, new={(shape, dtype)}"
                    )

    def _propagate_nested_functions(self, sub_builder: "OnnxBuilder"):
        """
        Merge all nested function definitions from a sub_builder into the current builder.
        This ensures that functions defined within a function are preserved in the top-level model.
        """
        for name, func in sub_builder.functions.items():
            if name not in self.functions:
                self.functions[name] = func
            else:
                logging.warning(
                    f"⚠️ [Duplicate function] Skipping already-registered function '{name}'"
                )

    def add_scalar_input(self, name: str, dtype: int):
        """
        Adds a scalar (0-dimensional) input to the ONNX model, typically for call-time parameters such as flags.

        Args:
            name: Name of the scalar input parameter.
            dtype: ONNX TensorProto data type (e.g., TensorProto.BOOL).

        Returns:
            The name of the registered scalar input.
        """
        shape = ()
        value_info = self.make_value_info(name, shape, dtype)
        self.inputs.append(value_info)
        self.register_value_info_metadata(name, shape, dtype, origin="call_parameter")
        logging.debug(f"Added scalar parameter input: {name} (dtype: {dtype})")
        return name

    def _dim_to_symbol(self, d):
        if isinstance(d, int):
            return d
        s = self.dimvar_to_name_by_str.get(str(d))
        if s:  # found via string key
            return s
        if hasattr(d, "symbol") and d.symbol:
            return str(d.symbol)
        return _symbol_name(self, d)  # final fallback

    def _assert_topologically_sorted(self):
        """Assert that the nodes are topologically sorted.

        This ensures that for every node, all its inputs have been defined earlier
        in the graph, either as inputs, initializers, or outputs of previous nodes.
        """
        available_tensors = set()

        # Add all graph inputs
        for inp in self.inputs:
            available_tensors.add(inp.name)

        # Add all initializers
        for init in self.initializers:
            available_tensors.add(init.name)

        # Check each node in order
        for node in self.nodes:
            # Check that all inputs to this node are available
            for inp in node.input:
                if inp and inp not in available_tensors:
                    raise RuntimeError(
                        f"Node {node.name} (op={node.op_type}) has an input '{inp}' "
                        f"that hasn't been produced yet. This indicates the graph is not "
                        f"topologically sorted or there's a missing tensor definition."
                    )

            # Add this node's outputs to available tensors
            available_tensors.update(node.output)

    # ------------------------------------------------------------------
    #  Remove any ValueInfo that is *not* referenced by nodes, outputs
    #  or initializers.  This prevents compile-time constants that were
    #  later replaced (e.g. transposed kernels) from surfacing as graph
    #  inputs.
    # ------------------------------------------------------------------
    def _filter_unused_inputs(self):
        used_names: set[str] = set()

        # all node inputs
        for n in self.nodes:
            used_names.update(n.input)

        # graph outputs must stay
        used_names.update(o.name for o in self.outputs)

        # and every initializer is baked into the model
        # Build a mapping from initializer names for quick lookup
        self.initializers_by_name = {init.name: init for init in self.initializers}
        used_names.update(self.initializers_by_name.keys())

        # keep only genuinely used inputs
        before = len(self.inputs)
        self.inputs = [vi for vi in self.inputs if vi.name in used_names]

        if before != len(self.inputs):
            logger.debug(
                "Pruned %d unused graph inputs (constants that became "
                "initializers or were otherwise dropped).",
                before - len(self.inputs),
            )

    # ------------------------------------------------------------------
    # helper
    # ------------------------------------------------------------------
    def _filter_redundant_inputs(self) -> None:
        """Drop every `graph.input` that
        * is also produced by some node **or**
        * duplicates an initializer **or**
        * is not consumed by any node (including nodes in subgraphs).
        """
        node_in, node_out = set(), set()
        for n in self.nodes:
            node_in.update([t for t in n.input if t])
            node_out.update([t for t in n.output if t])
            # Recursively find inputs in subgraphs
            for attr in n.attribute:
                if attr.type == AttributeProto.GRAPH:
                    for sub_node in attr.g.node:
                        node_in.update([t for t in sub_node.input if t])
                elif attr.type == AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        for sub_node in g.node:
                            node_in.update([t for t in sub_node.input if t])

        # Build initializers dictionary if not already done
        if not hasattr(self, "initializers_by_name"):
            self.initializers_by_name = {init.name: init for init in self.initializers}

        inits = set(self.initializers_by_name.keys())
        g_outs = set(o.name for o in self.outputs)

        before = len(self.inputs)
        self.inputs = [
            vi
            for vi in self.inputs
            if (
                # must still be needed
                vi.name in node_in
                or vi.name in g_outs
            )
            and (
                # …but not produced inside
                vi.name
                not in node_out
            )
            and (
                # …and not shadow an initializer
                vi.name
                not in inits
            )
        ]

        if before != len(self.inputs):
            logger.debug("Pruned %d redundant graph inputs.", before - len(self.inputs))

    def subgraph(
        self,
        name: str,
        invars: Sequence[str],
        jaxpr: "ClosedJaxpr",
    ) -> "OnnxBuilder":
        """
        Lightweight stub so that experimental control-flow code can call
        `builder.subgraph()` without breaking the current stable path.

        * Returns **self** for now – i.e. the caller keeps using the parent
          builder context.
        * Adds **no** nodes, **no** IO, **no** metadata.
        * Logs a DEBUG line so we know if it ever gets hit in production
          before the real implementation lands.
        """
        logger.debug("subgraph(%s) called in stub mode – no graph emitted", name)
        return self
