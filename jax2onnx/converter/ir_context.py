# jax2onnx/converter/ir_context.py

from __future__ import annotations
from typing import Any, Sequence, Dict, Tuple, Optional, TYPE_CHECKING
import numpy as np
import onnx_ir as ir
from onnx_ir import Attr, AttributeType
from .ir_builder import IRBuilder, _dtype_to_ir

from jax.extend import core as jcore_ext  # type: ignore

if TYPE_CHECKING:
    from .conversion_api import FunctionRegistry


class _InitializerProxy:
    """List-like view over builder.initializers that is function-safe."""

    def __init__(self, ctx: "IRContext") -> None:  # type: ignore[name-defined]
        self._ctx = ctx
        self._storage = ctx.builder.initializers

    def append(self, value: ir.Value) -> None:
        self._ctx._handle_initializer_append(value)

    def extend(self, values):
        for value in values:
            self.append(value)

    def __iter__(self):
        return iter(self._storage)

    def __len__(self) -> int:
        return len(self._storage)

    def __getitem__(self, item):
        return self._storage[item]

    def __bool__(self) -> bool:
        return bool(self._storage)

    def __getattr__(self, name):
        return getattr(self._storage, name)


# ---- literal + dtype bookkeeping -------------------------------------------------
_LITERAL_TYPES = (jcore_ext.Literal,)
_FLOAT_TYPE_NAMES = ("FLOAT", "DOUBLE", "FLOAT16", "BFLOAT16")
_FLOAT_DTYPES = {
    ir.DataType.__members__[name]
    for name in _FLOAT_TYPE_NAMES
    if name in ir.DataType.__members__
}
_INT_TYPE_NAMES = (
    "INT8",
    "INT16",
    "INT32",
    "INT64",
    "UINT8",
    "UINT16",
    "UINT32",
    "UINT64",
)
_INT_DTYPES = {
    ir.DataType.__members__[name]
    for name in _INT_TYPE_NAMES
    if name in ir.DataType.__members__
}


# ---- shape coercion: int stays int; otherwise stringify (safe for onnx_ir) --------
def _to_ir_shape(dims: Sequence[Any]) -> ir.Shape:
    out: list[int | str] = []
    for d in dims:
        if isinstance(d, (int, np.integer)):
            out.append(int(d))
        else:
            try:
                out.append(int(d))
            except Exception:
                out.append(str(d))
    return ir.Shape(tuple(out))


def _is_float_dtype_enum(enum: ir.DataType) -> bool:
    return enum in _FLOAT_DTYPES


def _is_int_dtype_enum(enum: ir.DataType) -> bool:
    return enum in _INT_DTYPES


class IRContext:
    def __init__(
        self,
        *,
        opset: int,
        enable_double_precision: bool,
        input_specs: Sequence[Any] | None = None,
    ):
        self.builder = IRBuilder(
            opset=opset, enable_double_precision=enable_double_precision
        )
        self.builder._function_mode = False
        self._default_float_dtype = (
            np.float64 if enable_double_precision else np.float32
        )
        # Back-compat views some plugins touch directly
        self._var2val = self.builder._var2val
        self._initializers = _InitializerProxy(self)
        self._nodes = self.builder.nodes
        self._inputs = self.builder.inputs
        # Intermediate ValueInfo staging (mirrors builder.value_info)
        self._value_info = self.builder.value_info
        # Some helpers expect _value_infos (legacy alias)
        self._value_infos = self.builder.value_info
        # Track where each symbolic dim came from (object if hashable, and always string)
        self._sym_origin: dict[object, tuple[ir.Value, int]] = {}
        self._sym_origin_str: dict[str, tuple[ir.Value, int]] = {}
        # Name counters for fresh_name(); keep a typed attribute so mypy is happy.
        # Using dict[str, int] since we only ever index by the base string.
        self._name_counters: dict[str, int] = {}
        self._function_mode: bool = False
        self._function_registry: Optional["FunctionRegistry"] = None
        self._ir_functions: list[ir.Function] = []
        # name -> {attr_name: python_value or TensorProto}
        self._attr_overrides: Dict[str, Dict[str, Any]] = {}
        # Set by FunctionScope while emitting FunctionProto
        self._inside_function_scope: bool = False
        self._keep_function_float32: bool = False

    @property
    def opset(self) -> int:
        return self.builder.opset

    @property
    def enable_double_precision(self) -> bool:
        return self.builder.enable_double_precision

    # ------------------------------- Function registry helpers ------------------

    def get_function_registry(self) -> Optional["FunctionRegistry"]:
        return self._function_registry

    def set_function_registry(self, registry: "FunctionRegistry") -> None:
        self._function_registry = registry

    # ------------------------------- IR functions bucket ------------------------

    @property
    def ir_functions(self) -> list[ir.Function]:
        return self._ir_functions

    # ------------------------------- Attr overrides -----------------------------

    @property
    def attr_overrides(self) -> Dict[str, Dict[str, Any]]:
        return self._attr_overrides

    @property
    def value_infos(self) -> Sequence[ir.Value]:
        return self._value_info

    def _promote_float_array(self, arr: np.ndarray) -> np.ndarray:
        if (
            self.builder.enable_double_precision
            and np.issubdtype(arr.dtype, np.floating)
            and arr.dtype != np.float64
        ):
            return arr.astype(np.float64, copy=False)
        return arr

    def fresh_name(self, base: str) -> str:
        # Counter dict is initialized in __init__; no lazy setup needed.
        i = self._name_counters.get(base, 0)
        self._name_counters[base] = i + 1
        # Use underscore-separated numeric suffixes: in_0, out_0, Reshape_0, ...
        sep = "" if base.endswith(("_", "/")) else "_"
        return f"{base}{sep}{i}"

    # Record an attribute override to be applied on the final ModelProto
    def add_node_attr_override(self, node_name: str, attrs: dict[str, object]) -> None:
        if not node_name:
            return
        current = self._attr_overrides.get(node_name)
        if current is None:
            self._attr_overrides[node_name] = dict(attrs or {})
        else:
            current.update(attrs or {})

    # ------------------------------------------------------------------
    # Helper: set attributes in a way that works for both:
    #   - function bodies (need onnx_ir Attr objects)
    #   - top-level graphs (stash raw values, applied later in to_onnx)
    # ------------------------------------------------------------------
    def set_node_attrs(self, node: Any, attrs: Dict[str, Any]) -> None:
        # make sure there is a stable name to key overrides
        if isinstance(node, ir.Node):
            name = node.name
            if not name:
                prefix = node.op_type or "node"
                node.name = self.builder.fresh_name(prefix)
                name = node.name
        else:
            name = node.name if hasattr(node, "name") else None  # type: ignore[attr-defined]
            if not name:
                prefix = node.op_type if hasattr(node, "op_type") else "node"  # type: ignore[attr-defined]
                name = self.builder.fresh_name(prefix)
                setattr(node, "name", name)
        merged = dict(self._attr_overrides.get(name, {}))
        merged.update(attrs or {})
        self._attr_overrides[name] = merged

    def get_node_attrs(self, node: Any) -> Dict[str, Any]:
        if isinstance(node, ir.Node):
            name = node.name
        else:
            name = node.name if hasattr(node, "name") else ""  # type: ignore[attr-defined]
        return self._attr_overrides.get(name, {})

    # ---------- Scope-agnostic external flag as graph input (top) or local value (function)
    def ensure_external_flag(self, name: str, var: Any):
        """Top-level: return/create a BOOL[] graph input `name`.
        Function body: return the Value for `var` (function input or literal)."""
        if self._inside_function_scope:
            if var is None:
                lookup = self.__dict__.get("_call_param_value_by_name")
                if isinstance(lookup, dict) and name in lookup:
                    return lookup[name]
                raise RuntimeError(
                    f"Call parameter '{name}' does not have a dynamic value in function scope"
                )
            return self.get_value_for_var(var, name_hint=name)
        # top-level graph input (reuse if already present)
        for vi in self.builder.inputs:
            if (vi.name or "") == name:
                return vi
        v = ir.Value(
            name=name, type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape(())
        )
        self.builder.inputs.append(v)
        return v

    def ensure_training_mode(self, flag_name: str, var: Any) -> Any:
        """
        Return a BOOL[] Value for `training_mode`.
        - Inside a function: if `var` is a JAX literal (has a `.val`), fold to a constant
          training_mode = not(var.val) and feed Dropout directly (NO Not).
        - Otherwise: route the flag through a Not: flag → Not → training_mode.
          (Top-level uses a single graph input named `flag_name`; function uses the local wire.)
        """
        # If we're inside a function body and the flag is a literal, fold now.
        if self._inside_function_scope:
            lit_obj = var.val if hasattr(var, "val") else None  # type: ignore[attr-defined]
            # accept native bool, np.bool_, scalar array etc.
            if lit_obj is not None:
                lit = bool(np.asarray(lit_obj).item())
                return ir.Value(
                    name=self.builder.fresh_name("training_mode"),
                    type=ir.TensorType(ir.DataType.BOOL),
                    shape=ir.Shape(()),
                    const_value=ir.tensor(np.array(not lit, dtype=np.bool_)),
                )
            det_val = self.get_value_for_var(var, name_hint=flag_name)
        else:
            det_val = self.ensure_external_flag(flag_name, var)

        # Dynamic path: build Not(det_val) → training_mode
        tm = ir.Value(
            name=self.builder.fresh_name("training_mode"),
            type=ir.TensorType(ir.DataType.BOOL),
            shape=ir.Shape(()),
        )
        node = ir.Node(
            op_type="Not",
            domain="",
            inputs=[det_val],
            outputs=[tm],
            name=self.builder.fresh_name("Not"),
        )
        self.add_node(node)
        return tm

    def add_node(self, node: ir.Node, inputs=None, outputs=None):
        # maintain legacy signature; plugins pass a constructed ir.Node
        self.builder.nodes.append(node)
        return node

    # ---------- initializer management ----------
    def _handle_initializer_append(self, value: ir.Value) -> None:
        if self._inside_function_scope or self._function_mode:
            tensor = value.const_value
            if tensor is None:
                # Fallback: store as initializer to avoid data loss.
                self.builder.initializers.append(value)
                return
            self._materialize_constant_value(value, tensor)
            return
        self.builder.initializers.append(value)

    def _materialize_constant_value(self, value: ir.Value, tensor) -> None:
        attributes = [Attr("value", AttributeType.TENSOR, tensor)]
        node = ir.Node(
            op_type="Constant",
            domain="",
            inputs=[],
            outputs=[value],
            name=self.fresh_name("Constant"),
            attributes=attributes,
        )
        self.add_node(node)

    def cast_like(
        self, tensor: ir.Value, exemplar: ir.Value, *, name_hint: Optional[str] = None
    ) -> ir.Value:
        out = ir.Value(
            name=self.fresh_name(name_hint or f"{tensor.name}_cast"),
            type=exemplar.type,
            shape=tensor.shape,
        )
        self.add_node(
            ir.Node(
                op_type="CastLike",
                domain="",
                inputs=[tensor, exemplar],
                outputs=[out],
                name=self.fresh_name("CastLike"),
            )
        )
        return out

    def bind_const_for_var(self, var: Any, np_array: np.ndarray) -> ir.Value:
        array = (
            np.asarray(np_array) if not isinstance(np_array, np.ndarray) else np_array
        )
        promote_flag = self.builder.enable_double_precision
        keep_float32 = self._keep_function_float32
        if self._function_mode:
            aval = var.aval if hasattr(var, "aval") else None  # type: ignore[attr-defined]
            aval_np_dtype: np.dtype | None = None
            if aval is not None and hasattr(aval, "dtype"):
                try:
                    aval_np_dtype = np.dtype(aval.dtype)  # type: ignore[arg-type]
                except TypeError:
                    aval_np_dtype = None
            if (
                keep_float32
                and aval_np_dtype is not None
                and np.issubdtype(aval_np_dtype, np.floating)
                and aval_np_dtype != np.float64
            ):
                if array.dtype != aval_np_dtype:
                    array = np.asarray(array, dtype=aval_np_dtype)
                promote_flag = False
            else:
                array = self._promote_float_array(array)
        else:
            array = self._promote_float_array(array)

        tensor = ir.tensor(array)
        value_name = "const_val" if self._function_mode else "const"
        value = ir.Value(
            name=self.fresh_name(value_name),
            type=ir.TensorType(_dtype_to_ir(array.dtype, promote_flag)),
            shape=_to_ir_shape(array.shape),
            const_value=tensor,
        )

        if self._function_mode:
            self.add_node(
                ir.Node(
                    op_type="Constant",
                    domain="",
                    inputs=[],
                    outputs=[value],
                    name=self.fresh_name("Constant"),
                    attributes=[Attr("value", AttributeType.TENSOR, tensor)],
                )
            )
        else:
            self.builder.initializers.append(value)

        try:
            self.builder._var2val[var] = value
        except TypeError:
            pass
        return value

    # Bind an existing IR Value to a JAX var (no new Value created).
    # Used by FunctionPlugin to tie function-scope inputs to inner jaxpr invars.
    def bind_value_for_var(self, var: object, value: ir.Value) -> None:
        try:
            self.builder._var2val[var] = value
        except TypeError:
            # Some JAX Literal objects are unhashable; skip caching in that case.
            pass

    def add_input_for_invar(self, var: Any, index: int) -> ir.Value:
        aval = var.aval
        shp = tuple(aval.shape)
        aval_dtype = np.dtype(aval.dtype)
        promote_flag = self.builder.enable_double_precision
        if (
            self._function_mode
            and self._keep_function_float32
            and np.issubdtype(aval_dtype, np.floating)
            and aval_dtype != np.float64
        ):
            promote_flag = False
        val = ir.Value(
            name=f"in_{index}",
            type=ir.TensorType(_dtype_to_ir(aval_dtype, promote_flag)),
            shape=_to_ir_shape(shp),
        )
        self.builder._var2val[var] = val
        self.builder.inputs.append(val)
        # Remember which input/axis supplies each symbolic dim
        for ax, d in enumerate(shp):
            if not isinstance(d, (int, np.integer)):
                try:
                    self._sym_origin[d] = (val, ax)  # for hashable DimExprs
                except TypeError:
                    pass
                self._sym_origin_str[str(d)] = (val, ax)
        return val

    def get_symbolic_dim_origin(self, dim: object) -> Optional[tuple[ir.Value, int]]:
        if dim in self._sym_origin:
            return self._sym_origin[dim]
        return self._sym_origin_str.get(str(dim))

    def get_value_for_var(
        self,
        var: Any,
        *,
        name_hint: Optional[str] = None,
        prefer_np_dtype: Optional[np.dtype] = None,
    ) -> ir.Value:
        # Literals show up directly in eqn.invars for things like add_const
        if _LITERAL_TYPES and isinstance(var, _LITERAL_TYPES):
            arr = np.asarray(var.val)
            aval = var.aval if hasattr(var, "aval") else None  # type: ignore[attr-defined]
            literal_dtype = None
            if aval is not None:
                try:
                    literal_dtype = np.dtype(aval.dtype if hasattr(aval, "dtype") else arr.dtype)  # type: ignore[attr-defined]
                except TypeError:
                    literal_dtype = None
            if prefer_np_dtype is not None:
                try:
                    prefer_dtype = np.dtype(prefer_np_dtype)
                    literal_dtype = prefer_dtype
                except TypeError:
                    pass

            if np.issubdtype(arr.dtype, np.floating):
                tgt = literal_dtype or np.dtype(self._default_float_dtype)
                arr = arr.astype(tgt, copy=False)
            elif np.issubdtype(arr.dtype, np.integer) and literal_dtype is not None:
                arr = arr.astype(literal_dtype, copy=False)
            return self.bind_const_for_var(var, arr)

        if var in self.builder._var2val:
            return self.builder._var2val[var]
        aval = var.aval if hasattr(var, "aval") else None  # type: ignore[attr-defined]
        if aval is None:
            raise TypeError(f"Unsupported var type: {type(var)}")
        aval_dtype = np.dtype(aval.dtype)
        if (
            not self.builder.enable_double_precision
            and np.issubdtype(aval_dtype, np.floating)
            and aval_dtype != np.dtype(self._default_float_dtype)
        ):
            aval_dtype = np.dtype(self._default_float_dtype)
        promote_flag = self.builder.enable_double_precision
        if (
            self._function_mode
            and self._keep_function_float32
            and promote_flag
            and np.issubdtype(aval_dtype, np.floating)
            and aval_dtype != np.float64
        ):
            promote_flag = False
        v = ir.Value(
            name=name_hint or self.fresh_name("v"),
            type=ir.TensorType(_dtype_to_ir(aval_dtype, promote_flag)),
            shape=_to_ir_shape(tuple(aval.shape)),
        )
        self.builder._var2val[var] = v
        return v

    def add_outputs_from_vars(self, outvars: Sequence[Any]) -> None:
        for i, var in enumerate(outvars):
            v = self.get_value_for_var(var, name_hint=f"out_{i}")
            target_enum = None
            aval = var.aval if hasattr(var, "aval") else None  # type: ignore[attr-defined]
            if aval is not None:
                aval_dtype = aval.dtype if hasattr(aval, "dtype") else None  # type: ignore[attr-defined]
                if aval_dtype is not None:
                    try:
                        np_dtype = np.dtype(aval_dtype)
                    except TypeError:
                        np_dtype = None
                    else:
                        target_enum = _dtype_to_ir(
                            np_dtype, self.builder.enable_double_precision
                        )
            current_type = v.type
            current_enum = (
                current_type.dtype if isinstance(current_type, ir.TensorType) else None
            )
            if target_enum is not None and current_enum is not None:
                if target_enum != current_enum:
                    promote_float = (
                        self.builder.enable_double_precision
                        and _is_float_dtype_enum(target_enum)
                        and _is_float_dtype_enum(current_enum)
                    )
                    downcast_float = (
                        not self.builder.enable_double_precision
                        and _is_float_dtype_enum(target_enum)
                        and _is_float_dtype_enum(current_enum)
                    )
                    keep_int64 = (
                        _is_int_dtype_enum(target_enum)
                        and _is_int_dtype_enum(current_enum)
                        and current_enum == ir.DataType.INT64
                    )
                    if promote_float:
                        target_enum = current_enum
                    elif downcast_float:
                        target_enum = current_enum
                    elif keep_int64:
                        target_enum = current_enum
                    else:
                        cast_val = ir.Value(
                            name=self.fresh_name("output_cast"),
                            type=ir.TensorType(target_enum),
                            shape=v.shape,
                        )
                        self.add_node(
                            ir.Node(
                                op_type="Cast",
                                domain="",
                                inputs=[v],
                                outputs=[cast_val],
                                name=self.fresh_name("Cast"),
                                attributes=[
                                    Attr(
                                        "to",
                                        AttributeType.INT,
                                        int(target_enum.value),
                                    )
                                ],
                            )
                        )
                        v = cast_val
            elif target_enum is not None and current_enum is None:
                v.type = ir.TensorType(target_enum)
            self.builder.outputs.append(v)

    # Convenience: make sure the model declares an opset import for a domain
    def ensure_opset_import(self, domain: str, version: int = 1) -> None:
        if hasattr(self.builder, "ensure_opset_import"):
            self.builder.ensure_opset_import(domain, version)
        elif hasattr(self.builder, "add_opset_import"):
            self.builder.add_opset_import(domain, version)

    def to_model_proto(self, *, name: str, ir_version: int = 10):
        if hasattr(self.builder, "to_model_proto"):
            return self.builder.to_model_proto(name=name, ir_version=ir_version)
        return self.builder.to_ir_model(name=name, ir_version=ir_version)


EMPTY_SHAPE: Tuple[Any, ...] = ()
