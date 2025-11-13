# jax2onnx/converter/ir_context.py

from __future__ import annotations
from typing import (
    Any,
    Sequence,
    Dict,
    Tuple,
    Optional,
    TYPE_CHECKING,
    Iterable,
    Iterator,
    Union,
    Callable,
    cast,
    Final,
)
from typing import overload
from collections.abc import MutableSequence
import os
import numpy as np
import onnx_ir as ir
from onnx_ir import Attr, AttributeType
from .ir_builder import IRBuilder, _dtype_to_ir
from .ir_constants import ConstantFolder
from .lower_dimexpr import LowerDimExpr
from .typing_support import SymbolicDimOrigin
from jax.extend import core as jcore_ext
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .conversion_api import FunctionRegistry


class _InitializerProxy(MutableSequence[ir.Value]):
    """Typed view over builder.initializers that preserves IR builder invariants."""

    def __init__(self, ctx: "IRContext") -> None:
        self._ctx = ctx
        self._storage = ctx.builder.initializers

    def append(self, value: ir.Value) -> None:
        self._ctx._handle_initializer_append(value)

    def extend(self, values: Iterable[ir.Value]) -> None:
        for value in values:
            self.append(value)

    @overload
    def __getitem__(self, index: int) -> ir.Value: ...

    @overload
    def __getitem__(self, index: slice) -> MutableSequence[ir.Value]: ...

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[ir.Value, MutableSequence[ir.Value]]:
        return self._storage[index]

    @overload
    def __setitem__(self, index: int, value: ir.Value) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[ir.Value]) -> None: ...

    def __setitem__(
        self, index: Union[int, slice], value: Union[ir.Value, Iterable[ir.Value]]
    ) -> None:
        if isinstance(index, slice):
            if not isinstance(value, Iterable):
                raise TypeError("Slice assignment expects an iterable of ir.Value")
            self._storage[index] = list(value)
        else:
            if not isinstance(value, ir.Value):
                raise TypeError("Expected ir.Value for item assignment")
            self._storage[index] = value

    @overload
    def __delitem__(self, index: int) -> None: ...

    @overload
    def __delitem__(self, index: slice) -> None: ...

    def __delitem__(self, index: Union[int, slice]) -> None:
        del self._storage[index]

    def __len__(self) -> int:
        return len(self._storage)

    def insert(self, index: int, value: ir.Value) -> None:
        if index >= len(self._storage):
            self.append(value)
            return
        self._storage.insert(index, value)

    def __iter__(self) -> Iterator[ir.Value]:
        return iter(self._storage)

    def __contains__(self, value: object) -> bool:
        return value in self._storage

    def clear(self) -> None:
        self._storage.clear()

    def pop(self, index: int = -1) -> ir.Value:
        return self._storage.pop(index)

    def __bool__(self) -> bool:
        return bool(self._storage)

    def __repr__(self) -> str:
        return f"_InitializerProxy({self._storage!r})"


# ---- literal + dtype bookkeeping -------------------------------------------------
_LITERAL_TYPES: tuple[type[jcore_ext.Literal], ...] = (jcore_ext.Literal,)


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


def _maybe_attr(obj: Any, attr: str) -> Any | None:
    if obj is None:
        return None
    if not hasattr(obj, attr):
        return None
    try:
        return object.__getattribute__(obj, attr)
    except AttributeError:
        return None
    except Exception:
        try:
            return getattr(obj, attr)
        except Exception:
            return None


def _maybe_literal_value(literal: Any) -> Any | None:
    return _maybe_attr(literal, "val")


def _maybe_aval(obj: Any) -> Any | None:
    return _maybe_attr(obj, "aval")


def _maybe_dtype(aval: Any) -> Optional[np.dtype[Any]]:
    dtype_obj = _maybe_attr(aval, "dtype")
    if dtype_obj is None:
        return None
    try:
        return cast(np.dtype[Any], np.dtype(dtype_obj))
    except TypeError:
        return None


def _maybe_shape(aval: Any) -> Optional[Tuple[Any, ...]]:
    shape_obj = _maybe_attr(aval, "shape")
    if shape_obj is None:
        return None
    try:
        return tuple(shape_obj)
    except TypeError:
        return None


_STACKTRACE_METADATA_ENV: Final[str] = "JAX2ONNX_ENABLE_STACKTRACE_METADATA"


class IRContext:
    def __init__(
        self,
        *,
        opset: int,
        enable_double_precision: bool,
        input_specs: Sequence[Any] | None = None,
        stacktrace_metadata: Optional[bool] = None,
    ):
        if stacktrace_metadata is None:
            env_flag = os.getenv(_STACKTRACE_METADATA_ENV)
            if env_flag is None:
                stacktrace_metadata = False
            else:
                stacktrace_metadata = env_flag.strip().lower() not in {
                    "0",
                    "false",
                    "no",
                    "",
                }
        self.builder: IRBuilder = IRBuilder(
            opset=opset,
            enable_double_precision=enable_double_precision,
            enable_stacktrace_metadata=bool(stacktrace_metadata),
        )
        self._stacktrace_metadata_enabled: bool = bool(stacktrace_metadata)
        if self._stacktrace_metadata_enabled:
            detail_mode = (
                os.getenv("JAX2ONNX_STACKTRACE_DETAIL", "minimal").strip().lower()
            )
            self.builder.set_stacktrace_mode(detail_mode)
        self.builder._function_mode = False
        self.dim_expr_lowerer = LowerDimExpr(self)
        self._default_float_dtype = (
            np.float64 if enable_double_precision else np.float32
        )
        # Back-compat views some plugins touch directly
        self._var2val = self.builder._var2val
        self._initializers = _InitializerProxy(self)
        self._nodes = self.builder.nodes
        self._inputs = self.builder.inputs
        self.record_primitive_calls_file: str | None = None
        # Track where each symbolic dim came from (object if hashable, and always string)
        self._sym_origin: dict[object, SymbolicDimOrigin] = {}
        self._sym_origin_str: dict[str, SymbolicDimOrigin] = {}
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
        # Function plugin bookkeeping mirrors (always initialised for typed access)
        self._func_name_counters: dict[str, int] = {}
        self._call_input_param_names: set[str] = set()
        self._call_input_param_literals: dict[str, Any] = {}
        self._call_param_value_by_name: dict[str, ir.Value] = {}
        self._const_folder = ConstantFolder()
        self._current_eqn: Any = None

    def register_constant_evaluator(
        self, primitive: Any, handler: Callable[..., Any] | None = None
    ) -> None:
        if isinstance(primitive, str):
            prim_name: str = primitive
        else:
            prim_name_obj = getattr(primitive, "name", None)
            if not isinstance(prim_name_obj, str):
                raise TypeError(
                    "register_constant_evaluator expects a primitive or primitive name"
                )
            prim_name = prim_name_obj
        if handler is None:
            bind = getattr(primitive, "bind", None)
            if not callable(bind):
                raise TypeError(
                    "register_constant_evaluator requires a handler when primitive has no 'bind'"
                )
            handler = cast(Callable[..., Any], bind)
        self._const_folder.register_handler(prim_name, handler)

    def try_evaluate_const(self, var: Any) -> Optional[NDArray[np.generic]]:
        return self._const_folder.try_evaluate(var)

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
    def set_node_attrs(self, node: ir.Node, attrs: Dict[str, Any]) -> None:
        node_name = node.name
        if not node_name:
            prefix = node.op_type or "node"
            node_name = self.builder.fresh_name(prefix)
            node.name = node_name
        merged = dict(self._attr_overrides.get(node_name, {}))
        merged.update(attrs)
        self._attr_overrides[node_name] = merged

    def get_node_attrs(self, node: ir.Node) -> Dict[str, Any]:
        return self._attr_overrides.get(node.name or "", {})

    # ---------- Scope-agnostic external flag as graph input (top) or local value (function)
    def ensure_external_flag(self, name: str, var: Any) -> ir.Value:
        """Top-level: return/create a BOOL[] graph input `name`.
        Function body: return the Value for `var` (function input or literal)."""
        if self._inside_function_scope:
            if var is None:
                lookup = self.__dict__.get("_call_param_value_by_name")
                if isinstance(lookup, dict) and name in lookup:
                    value_obj = lookup[name]
                    if isinstance(value_obj, ir.Value):
                        return value_obj
                    raise TypeError(
                        f"Dynamic call parameter '{name}' is not an ir.Value"
                    )
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

    def ensure_training_mode(self, flag_name: str, var: Any) -> ir.Value:
        """
        Return a BOOL[] Value for `training_mode`.
        - Inside a function: if `var` is a JAX literal (has a `.val`), fold to a constant
          training_mode = not(var.val) and feed Dropout directly (NO Not).
        - Otherwise: route the flag through a Not: flag → Not → training_mode.
          (Top-level uses a single graph input named `flag_name`; function uses the local wire.)
        """
        # If we're inside a function body and the flag is a literal, fold now.
        if self._inside_function_scope:
            lit_obj = _maybe_literal_value(var)
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

    def add_node(
        self,
        node: ir.Node,
        inputs: Optional[Sequence[ir.Value]] = None,
        outputs: Optional[Sequence[ir.Value]] = None,
    ) -> ir.Node:
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

    def _materialize_constant_value(self, value: ir.Value, tensor: Any) -> None:
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
        self._const_folder.register_const(var, array)
        promote_flag = self.builder.enable_double_precision
        keep_float32 = self._keep_function_float32
        if self._function_mode:
            aval = _maybe_aval(var)
            aval_np_dtype = _maybe_dtype(aval)
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
        aval = _maybe_aval(var)
        if aval is None:
            raise TypeError("Expected var with 'aval' attribute")
        shp = _maybe_shape(aval)
        if shp is None:
            raise TypeError("Aval missing shape")
        aval_dtype = _maybe_dtype(aval)
        if aval_dtype is None:
            raise TypeError("Aval missing dtype")
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
                    self._sym_origin[d] = SymbolicDimOrigin(value=val, axis=ax)
                except TypeError:
                    pass
                self._sym_origin_str[str(d)] = SymbolicDimOrigin(value=val, axis=ax)
        return val

    def get_symbolic_dim_origin(self, dim: object) -> Optional[SymbolicDimOrigin]:
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
            literal_val = _maybe_literal_value(var)
            if literal_val is None:
                raise TypeError("Literal missing value")
            arr = np.asarray(literal_val)
            aval = _maybe_aval(var)
            literal_dtype = _maybe_dtype(aval)
            if literal_dtype is None:
                try:
                    literal_dtype = np.dtype(arr.dtype)
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
        aval = _maybe_aval(var)
        if aval is None:
            raise TypeError(f"Unsupported var type: {type(var)}")
        aval_dtype = _maybe_dtype(aval)
        if aval_dtype is None:
            raise TypeError("Aval missing dtype")
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
        shape_tuple = _maybe_shape(aval)
        if shape_tuple is None:
            raise TypeError("Aval missing shape")
        v = ir.Value(
            name=name_hint or self.fresh_name("v"),
            type=ir.TensorType(_dtype_to_ir(aval_dtype, promote_flag)),
            shape=_to_ir_shape(shape_tuple),
        )
        self.builder._var2val[var] = v
        return v

    def add_outputs_from_vars(self, outvars: Sequence[Any]) -> None:
        for i, var in enumerate(outvars):
            v = self.get_value_for_var(var, name_hint=f"out_{i}")
            target_enum: Optional[ir.DataType] = None
            aval = _maybe_aval(var)
            np_dtype = _maybe_dtype(aval)
            if np_dtype is not None and np.issubdtype(np_dtype, np.complexfloating):
                base_np_dtype = np.float64 if np_dtype == np.complex128 else np.float32
                target_enum = _dtype_to_ir(
                    np.dtype(base_np_dtype), self.builder.enable_double_precision
                )
                dims = list(_maybe_shape(aval) or ())
                dims.append(2)
                v.type = ir.TensorType(target_enum)
                v.shape = _to_ir_shape(tuple(dims))
                self.builder.outputs.append(v)
                continue
            if np_dtype is not None:
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
                        and target_enum.is_floating_point()
                        and current_enum.is_floating_point()
                    )
                    downcast_float = (
                        not self.builder.enable_double_precision
                        and target_enum.is_floating_point()
                        and current_enum.is_floating_point()
                    )
                    keep_int64 = (
                        target_enum.is_integer() and current_enum == ir.DataType.INT64
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

    def to_model_proto(
        self,
        *,
        name: str,
        ir_version: int = 10,
        protective_clone: bool = True,
    ) -> ir.Model:
        return self.builder.to_ir_model(
            name=name,
            ir_version=ir_version,
            protective_clone=protective_clone,
        )


EMPTY_SHAPE: Tuple[Any, ...] = ()
