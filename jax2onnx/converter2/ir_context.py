# file: jax2onnx/converter2/ir_context.py

from __future__ import annotations
from typing import Any, Sequence, Dict, Tuple, Optional
import numpy as np
import onnx_ir as ir
from .ir_builder import IRBuilder, _dtype_to_ir

from jax.extend import core as jcore_ext  # type: ignore


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


_LITERAL_TYPES = (jcore_ext.Literal,)


# ---- shape coercion: int stays int; otherwise stringify (safe for onnx_ir) ---
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
        self._function_registry = None  # filled by conversion_api
        # name -> {attr_name: python_value or TensorProto}
        self._attr_overrides: Dict[str, Dict[str, Any]] = {}
        # Set by FunctionScope while emitting FunctionProto
        self._inside_function_scope: bool = False

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
        name = getattr(node, "name", None)
        if not name:
            prefix = getattr(node, "op_type", "node")
            name = self.builder.fresh_name(prefix)
            setattr(node, "name", name)
        merged = dict(self._attr_overrides.get(name, {}))
        merged.update(attrs or {})
        self._attr_overrides[name] = merged

    def get_node_attrs(self, node: Any) -> Dict[str, Any]:
        name = getattr(node, "name", "")
        return self._attr_overrides.get(name, {})

    # ---------- Scope-agnostic external flag as graph input (top) or local value (function)
    def ensure_external_flag(self, name: str, var: Any):
        """Top-level: return/create a BOOL[] graph input `name`.
        Function body: return the Value for `var` (function input or literal)."""
        if getattr(self, "_inside_function_scope", False):
            return self.get_value_for_var(var, name_hint=name)
        # top-level graph input (reuse if already present)
        for vi in getattr(self.builder, "inputs", []):
            if getattr(vi, "name", "") == name:
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
        if getattr(self, "_inside_function_scope", False):
            lit_obj = getattr(var, "val", None)
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
        # Prefer ctx.add_node if present; otherwise builder.add_node
        add = getattr(self, "add_node", None)
        if callable(add):
            add(node)
        else:
            self.builder.add_node(
                op_type="Not",
                inputs=[det_val],
                outputs=[tm],
                attributes=[],
                name=node.name,
            )
        return tm

    def add_node(self, node: ir.Node, inputs=None, outputs=None):
        # maintain legacy signature; plugins pass a constructed ir.Node
        self.builder.nodes.append(node)
        return node

    # ---------- initializer management ----------
    def _handle_initializer_append(self, value: ir.Value) -> None:
        if getattr(self, "_inside_function_scope", False) or getattr(
            self, "_function_mode", False
        ):
            tensor = getattr(value, "const_value", None)
            if tensor is None:
                # Fallback: store as initializer to avoid data loss.
                self.builder.initializers.append(value)
                return
            self._materialize_constant_value(value, tensor)
            return
        self.builder.initializers.append(value)

    def _materialize_constant_value(self, value: ir.Value, tensor) -> None:
        Attr = getattr(ir, "Attr", getattr(ir, "Attribute", None))
        AttrType = getattr(ir, "AttributeType", getattr(ir, "AttrType", None))
        attributes: list[Any] = []
        if Attr is not None:
            try:
                if hasattr(Attr, "t"):
                    attributes.append(Attr.t("value", tensor))
                elif AttrType is not None:
                    attributes.append(Attr("value", AttrType.TENSOR, tensor))
                else:
                    attributes.append(Attr("value", tensor))
            except Exception:
                pass
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
        if getattr(self, "_function_mode", False):
            # In functions, use a Constant node (no model-level initializer)
            # Ensure array is properly handled
            if not isinstance(np_array, np.ndarray):
                np_array = np.asarray(np_array)

            # Create a Value with the tensor stored in const_value for later reference
            v = ir.Value(
                name=self.fresh_name("const_val"),
                type=ir.TensorType(
                    _dtype_to_ir(np_array.dtype, self.builder.enable_double_precision)
                ),
                shape=_to_ir_shape(np_array.shape),
                const_value=ir.tensor(
                    np_array
                ),  # Store tensor here for FunctionProto serialization
            )

            # Create the Constant node that will produce this value
            const_attr = ir.Attr("value", ir.AttributeType.TENSOR, ir.tensor(np_array))
            self.add_node(
                ir.Node(
                    op_type="Constant",
                    domain="",
                    inputs=[],
                    outputs=[v],
                    name=self.fresh_name("Constant"),
                    attributes=[const_attr],
                )
            )
            try:
                self.builder._var2val[var] = v
            except TypeError:
                pass
            return v
        else:
            # normal path: initializer
            v = ir.Value(
                name=self.fresh_name("const"),
                type=ir.TensorType(
                    _dtype_to_ir(np_array.dtype, self.builder.enable_double_precision)
                ),
                shape=_to_ir_shape(np_array.shape),
                const_value=ir.tensor(np_array),
            )
            self.builder.initializers.append(v)
            try:
                self.builder._var2val[var] = v
            except TypeError:
                pass
            return v

    # Bind an existing IR Value to a JAX var (no new Value created).
    # Used by FunctionPlugin to tie function-scope inputs to inner jaxpr invars.
    def bind_value_for_var(self, var: object, value: ir.Value) -> None:
        self.builder._var2val[var] = value

    def add_input_for_invar(self, var: Any, index: int) -> ir.Value:
        aval = var.aval
        shp = tuple(aval.shape)
        val = ir.Value(
            name=f"in_{index}",
            type=ir.TensorType(
                _dtype_to_ir(np.dtype(aval.dtype), self.builder.enable_double_precision)
            ),
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
            if np.issubdtype(arr.dtype, np.floating):
                tgt = (
                    np.dtype(prefer_np_dtype)
                    if prefer_np_dtype is not None
                    else np.dtype(self._default_float_dtype)
                )
                arr = arr.astype(tgt, copy=False)
            return self.bind_const_for_var(var, arr)

        if var in self.builder._var2val:
            return self.builder._var2val[var]
        aval = getattr(var, "aval", None)
        if aval is None:
            raise TypeError(f"Unsupported var type: {type(var)}")
        v = ir.Value(
            name=name_hint or self.fresh_name("v"),
            type=ir.TensorType(
                _dtype_to_ir(np.dtype(aval.dtype), self.builder.enable_double_precision)
            ),
            shape=_to_ir_shape(tuple(aval.shape)),
        )
        self.builder._var2val[var] = v
        return v

    def add_outputs_from_vars(self, outvars: Sequence[Any]) -> None:
        for i, var in enumerate(outvars):
            v = self.get_value_for_var(var, name_hint=f"out_{i}")
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
