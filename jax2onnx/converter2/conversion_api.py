# jax2onnx/converter2/conversion_api.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager, ExitStack
import inspect as _ins
import os

import jax
import jax.numpy as jnp
from jax import export as jax_export
import numpy as np
import onnx_ir as ir

from jax2onnx.plugins2 import plugin_system as ps2
from jax2onnx.plugins2.plugin_system import (
    PLUGIN_REGISTRY2,
    PrimitiveLeafPlugin,
    apply_monkey_patches,
    import_all_plugins,
)

from .ir_context import IRContext
from .ir_builder import IRBuilder
from .ir_optimizations import optimize_graph
from .function_scope import FunctionRegistry

# ---- JAX 0.6.x: bind from jax.extend.core only ------------------------------
# We officially support JAX 0.6.x; never touch jax.core.Literal on this path.
from jax.extend import core as jcore_ext  # type: ignore

_LITERAL_TYPES = (jcore_ext.Literal,)

# Keep ORT-compatible IR version (ORT ~1.18 supports IR v10 broadly)
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
    return ir.DataType.FLOAT


def _to_ir_shape(shape_tuple) -> "ir.Shape":
    dims: Tuple[Union[int, str], ...] = tuple(
        int(d) if isinstance(d, (int, np.integer)) else str(d) for d in shape_tuple
    )
    return ir.Shape(dims)


def _as_sds_list(
    inputs: List[Any], enable_double_precision: bool
) -> List["jax.ShapeDtypeStruct"]:
    """Normalize user 'inputs' to ShapeDtypeStructs for abstract tracing."""
    sds_list: List[jax.ShapeDtypeStruct] = []

    # 1) gather string symbols
    symnames: list[str] = []
    for spec in inputs:
        if hasattr(spec, "shape") and hasattr(spec, "dtype"):
            continue
        if isinstance(spec, (list, tuple)):
            for d in spec:
                if isinstance(d, str) and d not in symnames:
                    symnames.append(d)

    # 2) create symbolic sizes
    name2sym: dict[str, object] = {n: jax_export.symbolic_shape(n)[0] for n in symnames}

    # 3) build SDS list
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
# Minimal IR Build Context facade (for plugins)
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
        self._symdim_origin: dict[object, tuple[ir.Value, int]] = {}
        self._symdim_origin_str: dict[str, tuple[ir.Value, int]] = {}

    def fresh_name(self, prefix: str) -> str:
        self._name_counter += 1
        return f"{prefix}_{self._name_counter}"

    def add_node(self, node, inputs=None, outputs=None):
        self._nodes.append(node)
        return node

    def get_value_for_var(
        self,
        var,
        *,
        name_hint: Optional[str] = None,
        prefer_np_dtype: Optional[np.dtype] = None,
    ) -> "ir.Value":
        if _LITERAL_TYPES and isinstance(var, _LITERAL_TYPES):
            arr = np.asarray(var.val)
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

        if var in self._var2val:
            return self._var2val[var]

        aval = getattr(var, "aval", None)
        if aval is None:
            raise TypeError(f"Unsupported var type: {type(var)}")
        v = ir.Value(
            name=name_hint or self.fresh_name("v"),
            type=ir.TensorType(_to_ir_dtype_from_np(getattr(aval, "dtype", np.float32))),
            shape=_to_ir_shape(getattr(aval, "shape", ())),
        )
        self._var2val[var] = v
        return v

    def add_input_for_invar(self, var: Any, index: int) -> ir.Value:
        aval = var.aval
        val = ir.Value(
            name=f"in_{index}",
            type=ir.TensorType(_to_ir_dtype_from_np(np.dtype(aval.dtype))),
            shape=_to_ir_shape(tuple(aval.shape)),
        )
        self._var2val[var] = val
        self._inputs.append(val)

        # Track symbolic dim origins
        for ax, d in enumerate(getattr(aval, "shape", ())):
            if not isinstance(d, (int, np.integer)):
                self._symdim_origin[d] = (val, ax)
                self._symdim_origin_str[str(d)] = (val, ax)
        return val

    def get_symbolic_dim_origin(self, dim: object) -> Optional[tuple[ir.Value, int]]:
        if dim in self._symdim_origin:
            return self._symdim_origin[dim]
        return self._symdim_origin_str.get(str(dim))

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


@contextmanager
def _activate_plugin_worlds():
    # Ensure plugin registry is populated
    import_all_plugins()
    with ExitStack() as stack:
        # Legacy patches first
        stack.enter_context(apply_monkey_patches())
        # New-style leaf bindings
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
    """
    Build an ONNX-IR model in three phases:
    1) Trace JAX to ClosedJaxpr with symbolic shapes.
    2) Lower to onnx_ir (plugins; function bodies allowed).
    3) Run a single IR-wide optimization pass (cross-node cleanups).
    """
    # 1) Abstract inputs
    default_float = _np_float_dtype(enable_double_precision)
    sds_list = _as_sds_list(inputs, enable_double_precision)

    # 2) JAXPR (optionally print for debugging)
    def _wrapped(*xs):
        return fn(*xs, **(input_params or {}))

    with _activate_plugin_worlds():
        closed = jax.make_jaxpr(_wrapped)(*sds_list)
    if os.environ.get("J2O_PRINT_JAXPR", "0") == "1":
        try:
            print(f"JAXPR: {closed.jaxpr.pretty_print()}")
        except Exception:
            pass
    jpr = closed.jaxpr

    # 3) IR context & inputs/consts
    ctx = IRContext(
        opset=opset,
        enable_double_precision=enable_double_precision,
        input_specs=sds_list,
    )
    # Expose knobs for downstream (optional)
    setattr(ctx, "loosen_internal_shapes", bool(loosen_internal_shapes))
    if record_primitive_calls_file:
        setattr(ctx, "record_primitive_calls_file", str(record_primitive_calls_file))

    if getattr(ctx, "_function_registry", None) is None:
        setattr(ctx, "_function_registry", FunctionRegistry())

    # Map constvars
    for cv, cval in zip(jpr.constvars, closed.consts):
        np_c = np.asarray(cval)
        if np.issubdtype(np_c.dtype, np.floating):
            np_c = np_c.astype(default_float, copy=False)
        ctx.bind_const_for_var(cv, np_c)

    # Inputs
    for i, v in enumerate(jpr.invars):
        ctx.add_input_for_invar(v, i)

    # Lower equations
    class _ConverterFacade:
        builder: IRBuilder
        ctx: IRContext

    converter = _ConverterFacade()
    converter.builder = ctx.builder
    converter.ctx = ctx

    for eqn in jpr.eqns:
        prim_name = eqn.primitive.name
        plugin_ref = PLUGIN_REGISTRY2.get(prim_name)
        if plugin_ref is None:
            raise NotImplementedError(
                f"[converter2] No plugins2 registered for primitive '{prim_name}'"
            )
        if hasattr(plugin_ref, "lower"):
            lower = plugin_ref.lower
            try:
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

    # Outputs
    ctx.add_outputs_from_vars(jpr.outvars)

    # Build IR model
    ir_model = ctx.builder.to_ir_model(name=model_name, ir_version=_ORT_SAFE_IR_VERSION)

    # Attach any native ir.Functions collected on ctx
    ir_funcs = list(getattr(ctx, "_ir_functions", []) or [])
    if ir_funcs:
        fstore = getattr(ir_model, "functions", None)
        if fstore is None:
            try:
                ir_model.functions = {}
                fstore = ir_model.functions
            except Exception:
                ir_model.functions = []
                fstore = ir_model.functions

        if isinstance(fstore, dict):
            for fn_ir in ir_funcs:
                key = (
                    getattr(fn_ir, "id", None)
                    or getattr(fn_ir, "identifier", None)
                    or (
                        (getattr(fn_ir, "domain", "") or ""),
                        (getattr(fn_ir, "name", "") or ""),
                        (getattr(fn_ir, "overload", "") or ""),
                    )
                )
                fstore[key] = fn_ir
        elif isinstance(fstore, list):
            def _fid(f):
                return (
                    getattr(f, "domain", "") or "",
                    getattr(f, "name", "") or "",
                    getattr(f, "overload", "") or "",
                )
            existing = {_fid(f) for f in fstore}
            for fn_ir in ir_funcs:
                if _fid(fn_ir) not in existing:
                    fstore.append(fn_ir)
        else:
            ir_model.functions = list(ir_funcs)

        # Ensure model-level opset imports cover default "" and each function domain
        model_imports: Dict[str, int] = dict(getattr(ir_model, "opset_imports", {}) or {})
        if "" not in model_imports:
            try:
                default_opset = int(getattr(getattr(ctx, "builder", None), "opset", 21))
            except Exception:
                default_opset = 21
            model_imports[""] = default_opset or 21
        for fn_ir in ir_funcs:
            dom = (getattr(fn_ir, "domain", "") or "").strip()
            if dom and dom not in model_imports:
                model_imports[dom] = 1
        try:
            ir_model.opset_imports = model_imports
        except Exception:
            try:
                getattr(ir_model, "opset_imports", {}).update(model_imports)
            except Exception:
                pass

    # ---- Single IR-wide optimization pass (centralized cleanups) ----
    try:
        optimize_graph(ir_model)
    except Exception as _e:
        import logging as _logging
        _logging.getLogger("jax2onnx.converter2.ir_optimizations").debug(
            "optimize_graph skipped: %s", _e
        )

    # ---- Late attribute overrides (polish; not structural rewrites) ----

    def _ir_attr_int(name: str, val: int):
        Attr = getattr(ir, "Attr", None)
        AttrType = getattr(ir, "AttributeType", getattr(ir, "AttrType", None))
        ival = int(val)
        if Attr is None:
            raise RuntimeError("onnx_ir.Attr is not available")
        if hasattr(Attr, "i"):
            return Attr.i(name, ival)
        if AttrType is not None:
            return Attr(name, AttrType.INT, ival)
        return Attr(name, ival)

    def _apply_ir_attr_overrides_to_graph(gr: "ir.Graph", overrides: dict[str, dict[str, object]]):
        if not overrides or gr is None:
            return
        nodes_ref = getattr(gr, "nodes", None) or getattr(gr, "_nodes", None) or []
        name2node: Dict[str, ir.Node] = {}
        for n in nodes_ref:
            nm = getattr(n, "name", "")
            if nm:
                name2node[nm] = n

        def _mutate_attr_list(attr_list: list, k: str, v: object) -> None:
            for i in range(len(attr_list) - 1, -1, -1):
                a = attr_list[i]
                aname = getattr(a, "name", getattr(a, "key", None))
                if aname == k:
                    del attr_list[i]
            try:
                attr_obj = ir.Attr(k, v)
            except Exception:
                try:
                    import numpy as _np
                    attr_obj = ir.Attr(k, ir.tensor(_np.asarray(v)))
                except Exception:
                    return
            attr_list.append(attr_obj)

        for nm, kv in (overrides or {}).items():
            n = name2node.get(nm)
            if not n:
                continue
            current = getattr(n, "attributes", None)
            if hasattr(current, "update"):
                try:
                    current.update(kv or {})
                except Exception:
                    pass
                continue
            attr_list = current if isinstance(current, list) else getattr(n, "_attributes", None)
            if isinstance(attr_list, list):
                for k_attr, v_attr in (kv or {}).items():
                    _mutate_attr_list(attr_list, k_attr, v_attr)

    def _fix_concat_axis_in_graph(gr: "ir.Graph") -> None:
        nodes_ref = getattr(gr, "nodes", None) or getattr(gr, "_nodes", None) or []
        for n in nodes_ref:
            if getattr(n, "op_type", "") != "Concat":
                continue
            attrs = getattr(n, "attributes", None)
            if not isinstance(attrs, list):
                attrs = getattr(n, "_attributes", None)
            if not isinstance(attrs, list):
                continue
            if any(getattr(a, "name", getattr(a, "key", "")) == "axis" for a in attrs):
                continue
            try:
                attrs.append(_ir_attr_int("axis", 0))
            except Exception:
                pass

    # Apply overrides/fixes to top graph
    _apply_ir_attr_overrides_to_graph(ir_model.graph, getattr(ctx, "_attr_overrides", {}))
    _fix_concat_axis_in_graph(ir_model.graph)
    # …and to all function bodies (if any)
    _func_container = getattr(ir_model, "functions", None) or []
    _func_values = (_func_container.values() if isinstance(_func_container, dict) else _func_container)
    for fn in _func_values:
        try:
            _apply_ir_attr_overrides_to_graph(fn.graph, getattr(ctx, "_attr_overrides", {}))
            _fix_concat_axis_in_graph(fn.graph)
        except Exception:
            pass

    # Avoid emitting placeholders for onnx_function hits across runs
    try:
        _ = getattr(ps2, "_consume_onnx_function_hits")()
    except Exception:
        pass

    return ir_model
