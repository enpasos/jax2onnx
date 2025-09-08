from __future__ import annotations
from typing import Any, Optional, Sequence, Tuple
import numpy as np
import onnx_ir as ir
from .ir_builder import IRBuilder, _dtype_to_ir

from jax.extend import core as jcore_ext  # type: ignore

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
        self._default_float_dtype = (
            np.float64 if enable_double_precision else np.float32
        )
        # Back-compat views some plugins touch directly
        self._var2val = self.builder._var2val
        self._initializers = self.builder.initializers
        self._nodes = self.builder.nodes
        self._inputs = self.builder.inputs
        # Track where each symbolic dim came from (object if hashable, and always string)
        self._sym_origin: dict[object, tuple[ir.Value, int]] = {}
        self._sym_origin_str: dict[str, tuple[ir.Value, int]] = {}
        # Name counters for fresh_name(); keep a typed attribute so mypy is happy.
        # Using dict[str, int] since we only ever index by the base string.
        self._name_counters: dict[str, int] = {}
        self._function_mode: bool = False
        self._function_registry = None  # filled by conversion_api
        # Late attribute overrides keyed by node.name -> {attr_name: python value}
        # Used for both top-level and function bodies; converted to AttributeProto later.
        self._attr_overrides: dict[str, dict[str, object]] = {}

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
    def set_node_attrs(
        self, node: ir.Node, attrs: dict[str, object] | None = None, **kwargs
    ) -> None:
        """
        Record attributes for a node without constructing onnx_ir.Attr.
        We always stash raw Python values into _attr_overrides and let the
        serializer (top-level or FunctionProto builder) turn them into real
        AttributeProto entries. This keeps behavior consistent across
        onnx_ir versions and in function mode.
        """
        values: dict[str, object] = dict(attrs or {})
        values.update(kwargs)
        self._attr_overrides.setdefault(node.name, {}).update(values)
        return

    def add_node(self, node: ir.Node, inputs=None, outputs=None):
        # maintain legacy signature; plugins pass a constructed ir.Node
        self.builder.nodes.append(node)
        return node

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

    def to_model_proto(self, *, name: str, ir_version: int = 10):
        return self.builder.to_model_proto(name=name, ir_version=ir_version)

    # Convenience: make sure the model declares an opset import for a domain
    def ensure_opset_import(self, domain: str, version: int = 1) -> None:
        if hasattr(self.builder, "ensure_opset_import"):
            self.builder.ensure_opset_import(domain, version)
        elif hasattr(self.builder, "add_opset_import"):
            self.builder.add_opset_import(domain, version)


EMPTY_SHAPE: Tuple[Any, ...] = ()
