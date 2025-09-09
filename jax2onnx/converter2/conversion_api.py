# file: jax2onnx/converter2/conversion_api.py


from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager, ExitStack
from jax import export as jax_export

from jax2onnx.converter2.ir_pretty import print_ir_model
from jax2onnx.plugins2.plugin_system import (
    PLUGIN_REGISTRY2,
    PrimitiveLeafPlugin,
    apply_monkey_patches,
    import_all_plugins,
)
import onnx_ir as ir
import numpy as np
import jax
import jax.numpy as jnp

from jax2onnx.plugins2 import plugin_system as ps2

# new imports
from .ir_context import IRContext
from .ir_builder import IRBuilder
from .ir_optimizations import (
    remove_redundant_transpose_pairs_ir,
    remove_redundant_reshape_pairs_in_functions,
)
from .function_scope import FunctionRegistry

# ---- JAX 0.6.x: bind from jax.extend.core only ------------------------------
# We officially support JAX 0.6.x; never touch jax.core.Literal on this path.
from jax.extend import core as jcore_ext  # type: ignore

_LITERAL_TYPES = (jcore_ext.Literal,)

# NOTE: onnx_ir: https://github.com/onnx/ir-py
# We use it as the builder backend.

# For optional type hints; avoid jax.core on 0.6.x

# plugin2 registry (old registry stays untouched)

# Keep ORT-compatible IR version (ORT 1.18.x supports IR v10 max in many wheels)
_ORT_SAFE_IR_VERSION = 10

# ---------------------------
# Helpers
# ---------------------------


def _np_float_dtype(enable_double_precision: bool):
    return np.float64 if enable_double_precision else np.float32


def _to_ir_dtype_from_np(np_dtype: np.dtype) -> "ir.DataType":
    np_dtype = np.dtype(np_dtype)
    if np.issubdtype(np_dtype, np.floating):
        return ir.DataType.DOUBLE if np_dtype == np.float64 else ir.DataType.FLOAT
    if np.issubdtype(np_dtype, np.integer):
        # choose common default width mappings
        return {
            np.dtype(np.int64): ir.DataType.INT64,
            np.dtype(np.int32): ir.DataType.INT32,
            np.dtype(np.int16): ir.DataType.INT16,
            np.dtype(np.int8): ir.DataType.INT8,
            np.dtype(np.uint64): ir.DataType.UINT64,
            np.dtype(np.uint32): ir.DataType.UINT32,
            np.dtype(np.uint16): ir.DataType.UINT16,
            np.dtype(np.uint8): ir.DataType.UINT8,
        }.get(np_dtype, ir.DataType.INT64)
    if np_dtype == np.bool_:
        return ir.DataType.BOOL
    # fallback
    return ir.DataType.FLOAT


def _to_ir_shape(shape_tuple) -> "ir.Shape":
    # allow ints or symbolic-like objects → stringify non-ints
    dims: Tuple[Union[int, str], ...] = tuple(
        int(d) if isinstance(d, (int, np.integer)) else str(d) for d in shape_tuple
    )
    return ir.Shape(dims)


def _as_sds_list(
    inputs: List[Any], enable_double_precision: bool
) -> List["jax.ShapeDtypeStruct"]:
    """Normalize user 'inputs' to ShapeDtypeStructs for abstract tracing."""
    sds_list: List[jax.ShapeDtypeStruct] = []

    # 1) Collect all symbol names that appear as strings in the input shapes
    symnames: list[str] = []
    for spec in inputs:
        if hasattr(spec, "shape") and hasattr(spec, "dtype"):
            # Already a ShapeDtypeStruct / ShapedArray; nothing to collect here.
            continue
        if isinstance(spec, (list, tuple)):
            for d in spec:
                if isinstance(d, str) and d not in symnames:
                    symnames.append(d)

    # 2) Create JAX symbolic DimSize objects one-by-one (strings only)
    #    and remember them by name so equal names map to the same symbol.
    name2sym: dict[str, object] = {n: jax_export.symbolic_shape(n)[0] for n in symnames}

    # 3) Build the ShapeDtypeStructs
    for spec in inputs:
        if hasattr(spec, "shape") and hasattr(spec, "dtype"):
            sds_list.append(jax.ShapeDtypeStruct(tuple(spec.shape), spec.dtype))
        elif isinstance(spec, (list, tuple)):
            dt = jnp.float64 if enable_double_precision else jnp.float32
            dims = tuple(name2sym[d] if isinstance(d, str) else int(d) for d in spec)
            sds_list.append(jax.ShapeDtypeStruct(dims, dt))
        else:
            raise TypeError(f"Unsupported input spec: {type(spec)}")
    return sds_list


# ---------------------------
# Minimal IR Build Context
# ---------------------------
class _IRBuildContext:
    def __init__(self, *, opset: int, default_float_dtype: np.dtype):
        self.opset = opset
        self._default_float_dtype = np.dtype(default_float_dtype)
        self._var2val: Dict[Any, ir.Value] = {}
        self._inputs: List[ir.Value] = []
        self._initializers: List[ir.Value] = []
        self._nodes: List[ir.Node] = []
        self._name_counter = 0
        # Map a symbolic dimension to the input Value/axis that defines it.
        # Keep both the actual dim object (preferred) and its string repr as a fallback.
        self._symdim_origin: dict[object, tuple[ir.Value, int]] = {}
        self._symdim_origin_str: dict[str, tuple[ir.Value, int]] = {}

    # API used by plugins2
    def fresh_name(self, prefix: str) -> str:
        self._name_counter += 1
        return f"{prefix}_{self._name_counter}"

    def add_node(self, node, inputs=None, outputs=None):
        # Accept both (node) and (node, inputs, outputs) forms
        self._nodes.append(node)
        return node

    def get_value_for_var(
        self,
        var,
        *,
        name_hint: Optional[str] = None,
        prefer_np_dtype: Optional[np.dtype] = None,
    ) -> "ir.Value":
        # Handle Literals first; don't touch the dict (Literals are unhashable).
        if _LITERAL_TYPES and isinstance(var, _LITERAL_TYPES):
            arr = np.asarray(var.val)
            # For floating literals, align to either caller's preferred dtype
            # (e.g., the other operand) or our default float dtype.
            if np.issubdtype(arr.dtype, np.floating):
                target = (
                    np.dtype(prefer_np_dtype)
                    if prefer_np_dtype is not None
                    else self._default_float_dtype
                )
                arr = np.asarray(var.val, dtype=target)
            c_ir = ir.Value(
                name=name_hint or self.fresh_name("const"),
                type=ir.TensorType(_to_ir_dtype_from_np(arr.dtype)),
                shape=_to_ir_shape(arr.shape),
                const_value=ir.tensor(arr),
            )
            self._initializers.append(c_ir)
            return c_ir
        # If we've already materialized it, return it.
        if var in self._var2val:
            return self._var2val[var]
        # Otherwise create a fresh Value (outvar or intermediate)
        aval = getattr(var, "aval", None)
        if aval is None:
            raise TypeError(f"Unsupported var type: {type(var)}")
        v = ir.Value(
            name=name_hint or self.fresh_name("v"),
            type=ir.TensorType(
                _to_ir_dtype_from_np(getattr(aval, "dtype", np.float32))
            ),
            shape=_to_ir_shape(getattr(aval, "shape", ())),
        )
        self._var2val[var] = v
        return v

    # helpers for converter
    def add_input_for_invar(self, var: Any, index: int) -> ir.Value:
        aval = var.aval
        val = ir.Value(
            name=f"in_{index}",
            type=ir.TensorType(_to_ir_dtype_from_np(np.dtype(aval.dtype))),
            shape=_to_ir_shape(tuple(aval.shape)),
        )
        self._var2val[var] = val
        self._inputs.append(val)

        # Remember which input/axis supplies each symbolic dim
        for ax, d in enumerate(getattr(aval, "shape", ())):
            if not isinstance(d, (int, np.integer)):
                self._symdim_origin[d] = (val, ax)
                self._symdim_origin_str[str(d)] = (val, ax)
        return val

    def get_symbolic_dim_origin(self, dim: object) -> Optional[tuple[ir.Value, int]]:
        """Return (Value, axis) that provides the given symbolic dim."""
        if dim in self._symdim_origin:
            return self._symdim_origin[dim]
        return self._symdim_origin_str.get(str(dim))

    # ------------------------------------------------------------------
    # Tiny helper: Cast one tensor to the element dtype of another.
    # Keeps the original shape; creates a CastLike node.
    # ------------------------------------------------------------------
    def cast_like(
        self, tensor: ir.Value, exemplar: ir.Value, *, name_hint: Optional[str] = None
    ) -> ir.Value:
        """
        Return a new Value that is `tensor` cast to the element dtype of `exemplar`
        using ONNX CastLike. Shape is preserved from `tensor`.
        """
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


@contextmanager
def _activate_plugin_worlds():
    # Ensure all plugins are imported so the registry is populated
    import_all_plugins()

    with ExitStack() as stack:
        # LEGACY: still support older plugins that provide patch_info()
        # Apply legacy first so plugins2 can override where both exist.
        stack.enter_context(apply_monkey_patches())

        # NEW: activate centralized per-plugin bindings (override legacy where needed)
        for plugin_instance in PLUGIN_REGISTRY2.values():
            if isinstance(plugin_instance, PrimitiveLeafPlugin):
                stack.enter_context(plugin_instance.__class__.plugin_binding())

        yield


def to_onnx(
    *,
    fn: Any,
    inputs: List[Any],
    input_params: Optional[Dict[str, Any]],
    model_name: str,
    opset: int,
    enable_double_precision: bool,
    loosen_internal_shapes: bool,
    record_primitive_calls_file: Optional[str],
) -> ir.Model:
    # 1) Prepare abstract inputs for tracing (respect x64 policy)
    default_float = np.float64 if enable_double_precision else np.float32
    sds_list = _as_sds_list(inputs, enable_double_precision)

    # 2) Trace to ClosedJaxpr (input_params threaded; plugin worlds already handled above)
    def _wrapped(*xs):
        return fn(*xs, **(input_params or {}))

    with _activate_plugin_worlds():
        closed = jax.make_jaxpr(_wrapped)(*sds_list)
    jpr = closed.jaxpr
    print(f"JAXPR: {jpr.pretty_print()}")  # TODO remove

    # 3) Build IR context & bind inputs/consts
    ctx = IRContext(
        opset=opset,
        enable_double_precision=enable_double_precision,
        input_specs=sds_list,
    )
    # Avoid mypy complaining about attribute type on IRContext; attach dynamically.
    # Ensure a function registry exists, but don't clobber if a caller already set one
    if getattr(ctx, "_function_registry", None) is None:
        setattr(ctx, "_function_registry", FunctionRegistry())

    # map constvars
    for cv, cval in zip(jpr.constvars, closed.consts):
        np_c = np.asarray(cval)
        if np.issubdtype(np_c.dtype, np.floating):
            np_c = np_c.astype(default_float, copy=False)
        ctx.bind_const_for_var(cv, np_c)

    # invars → graph inputs
    for i, v in enumerate(jpr.invars):
        ctx.add_input_for_invar(v, i)

    # 4) Walk equations & dispatch to plugin.lower
    _seen_prims: list[str] = []

    # expose builder *and* context to plugins (e.g., FunctionPlugin)
    class _ConverterFacade:
        builder: IRBuilder
        ctx: IRContext

    converter = _ConverterFacade()
    converter.builder = ctx.builder
    converter.ctx = ctx

    for eqn in jpr.eqns:
        prim_name = eqn.primitive.name
        _seen_prims.append(prim_name)
        plugin_ref = PLUGIN_REGISTRY2.get(prim_name)
        if plugin_ref is None:
            raise NotImplementedError(
                f"[converter2] No plugins2 registered for primitive '{prim_name}'"
            )
        # leaf primitive plugins expose .lower(ctx, eqn)
        # function plugins use FunctionPlugin's handler via plugin_system
        if hasattr(plugin_ref, "lower"):
            # Backwards-compatible dispatch:
            # - old leaf plugins: lower(self, ctx, eqn)
            # - new leaf plugins: lower(self, ctx, eqn, params)
            lower = plugin_ref.lower
            try:
                import inspect as _ins

                has_params = "params" in _ins.signature(lower).parameters
            except Exception:
                has_params = False
            if has_params:
                lower(ctx, eqn, getattr(eqn, "params", None))
            else:
                lower(ctx, eqn)
        elif hasattr(plugin_ref, "get_handler"):
            handler = plugin_ref.get_handler(converter)
            handler(converter, eqn, eqn.params)
        else:
            raise NotImplementedError(
                f"[converter2] Unsupported plugin type for '{prim_name}'"
            )

    # 5) Outputs
    ctx.add_outputs_from_vars(jpr.outvars)

    # 6) IR-level graph optimizations (safe structure-only)
    remove_redundant_transpose_pairs_ir(ctx)

    # 7) IR model (attach ir.Functions)
    ir_model = ctx.builder.to_ir_model(name=model_name, ir_version=_ORT_SAFE_IR_VERSION)

    # Optional debug print; never fail conversion if pretty-print raises
    try:
        from .ir_pretty import print_ir_model
        import os as _os
        if _os.environ.get("J2O_PRINT_IR", "0") == "1":
            print_ir_model(ir_model)
    except Exception:
        pass

    # Attach any native ir.Functions collected on ctx
    ir_funcs = list(getattr(ctx, "_ir_functions", []) or [])
    if ir_funcs:
        # add function domain imports on the top graph
        imports = dict(getattr(ir_model.graph, "opset_imports", {}) or {})
        for fn in ir_funcs:
            if fn.domain and fn.domain not in imports:
                imports[fn.domain] = 1  # version 1 namespace
        ir_model.graph.opset_imports = imports
        # attach functions to IR model
        if not hasattr(ir_model, "functions") or ir_model.functions is None:
            ir_model.functions = []
        ir_model.functions.extend(ir_funcs)

    # IR-level: apply late attribute overrides directly on IR nodes
    def _apply_ir_attr_overrides(ir_model: ir.Model, overrides: dict[str, dict[str, object]]):
        if not overrides:
            return
        # some onnx_ir builds expose graph._nodes instead of graph.nodes
        g = ir_model.graph
        nodes_ref = getattr(g, "nodes", None)
        if nodes_ref is None:
            nodes_ref = getattr(g, "_nodes", None)
        nodes_ref = nodes_ref or []

        # Build a name -> node map without materializing copies
        name2node = {}
        for n in nodes_ref:
            nm = getattr(n, "name", "")
            if nm:
                name2node[nm] = n

        def _append_ir_attr(attr_list, k: str, v: object) -> None:
            """
            Append/replace an attribute on the node.
            Accept Python scalars; for tensors use ir.tensor(...) automatically.
            """
            # Remove any existing attr with same name
            try:
                # attr_list is a real list object owned by the node
                for i in range(len(attr_list) - 1, -1, -1):
                    a = attr_list[i]
                    aname = getattr(a, "name", getattr(a, "key", None))
                    if aname == k:
                        del attr_list[i]
            except Exception:
                # ignore if not a mutable list
                pass

            # Best-effort construction of ir.Attr
            attr_obj = None
            try:
                # First try: scalar form
                attr_obj = ir.Attr(k, v)
            except Exception:
                # If it's array-like, wrap as tensor
                try:
                    import numpy as _np
                    attr_obj = ir.Attr(k, ir.tensor(_np.asarray(v)))
                except Exception:
                    attr_obj = None
            if attr_obj is not None:
                try:
                    attr_list.append(attr_obj)
                except Exception:
                    # If not appendable, ignore silently (better to proceed than crash)
                    pass

        for nm, kv in overrides.items():
            n = name2node.get(nm)
            if not n:
                continue
            # attributes may be dict-like or a list of ir.Attr. We MUST mutate, not assign.
            current = getattr(n, "attributes", None)

            # Dict-like container → safe to update in-place.
            if hasattr(current, "update"):
                try:
                    current.update(kv or {})
                except Exception:
                    # Fall back to list mutation if update fails unexpectedly
                    pass
                continue

            # List/tuple of ir.Attr → mutate the real list
            attr_list = current
            # Some builds may expose a tuple; try to grab the real list via protected slot
            if not isinstance(attr_list, list):
                # Try known internals (defensive)
                attr_list = getattr(n, "_attributes", attr_list)
            if isinstance(attr_list, list):
                for k_attr, v_attr in (kv or {}).items():
                    _append_ir_attr(attr_list, k_attr, v_attr)
            # else: nothing we can mutate; skip gracefully
    _apply_ir_attr_overrides(ir_model, getattr(ctx, "_attr_overrides", {}))

    # ---- IR post-pass: ensure required attrs on operators (e.g., Concat.axis) ----
    def _ensure_required_attrs(ir_model: ir.Model) -> None:
        g = ir_model.graph
        nodes_ref = getattr(g, "nodes", None) or getattr(g, "_nodes", None) or []
        for n in nodes_ref:
            try:
                if n.op_type == "Concat":
                    # fetch attribute list (list of ir.Attr)
                    attrs = getattr(n, "attributes", None)
                    if not isinstance(attrs, list):
                        attrs = getattr(n, "_attributes", None)
                    # if we can't mutate, skip
                    if not isinstance(attrs, list):
                        continue
                    # already has axis?
                    has_axis = any(getattr(a, "name", getattr(a, "key", "")) == "axis" for a in attrs)
                    if not has_axis:
                        attrs.append(ir.Attr("axis", 0))
            except Exception:
                # never fail the conversion on best-effort pass
                pass
    _ensure_required_attrs(ir_model)

    # --- Do not emit placeholders for @onnx_function hits ---
    # Consume/clear any recorded hits to avoid cross-run leakage, but don't create dummies.
    try:
        _ = getattr(ps2, "_consume_onnx_function_hits")()
    except Exception:
        pass

    # (Optional) keep the old registry for test bookkeeping; it no longer serializes protobuf
    # freg = getattr(ctx, "_function_registry", None)

    # IR-only return; protobuf conversion happens outside converter2
    return ir_model