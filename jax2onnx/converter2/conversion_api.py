# file: jax2onnx/converter2/conversion_api.py


from __future__ import annotations
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager, ExitStack
from jax import export as jax_export

import numpy as np
import onnx
import jax
import jax.numpy as jnp
from jax import config as jax_config
from jax import export as jax_export   

# ---- JAX 0.6.x: bind from jax.extend.core only ------------------------------
# We officially support JAX 0.6.x; never touch jax.core.Literal on this path.
from jax.extend import core as jcore_ext  # type: ignore
_LITERAL_TYPES = (jcore_ext.Literal,)

# NOTE: onnx_ir: https://github.com/onnx/ir-py
# We use it as the builder backend.
import onnx_ir as ir

# For optional type hints; avoid jax.core on 0.6.x
from jax.extend.core import Literal as JaxLiteral, Var as JaxVar  # type: ignore

# plugin2 registry (old registry stays untouched)
from jax2onnx.plugins2.plugin_system import PLUGIN_REGISTRY2

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

def _as_sds_list(inputs: List[Any], enable_double_precision: bool) -> List["jax.ShapeDtypeStruct"]:
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
                target = np.dtype(prefer_np_dtype) if prefer_np_dtype is not None else self._default_float_dtype
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
            type=ir.TensorType(_to_ir_dtype_from_np(getattr(aval, "dtype", np.float32))),
            shape=_to_ir_shape(getattr(aval, "shape", ())),
        )
        self._var2val[var] = v
        return v

    # helpers for converter
    def add_input_for_invar(self, var: Any, index: int) -> ir.Value:
        aval = var.aval
        val = ir.Value(
            name=f"in{index}",
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
    def cast_like(self, tensor: ir.Value, exemplar: ir.Value, *, name_hint: Optional[str] = None) -> ir.Value:
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

    # ... rest of _IRBuildContext ...

@contextmanager
def _activate_plugin_worlds():
    """
    Enter all plugin-provided 'world activation' scopes (e.g., temporarily
    assign framework-level variables like flax.nnx.linear_p, apply monkey
    patches), and restore everything afterwards.
    """
    with ExitStack() as stack:
        for plugin_ref in PLUGIN_REGISTRY2.items():
            # plugin_ref may be (name, ref) if iterating items(); support both
            ref = plugin_ref[1] if isinstance(plugin_ref, tuple) else plugin_ref
            cls = ref if isinstance(ref, type) else ref.__class__
            cm_fn = getattr(cls, "world_activation", None)
            if cm_fn is not None:
                stack.enter_context(cm_fn())
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
) -> onnx.ModelProto:
    """
    Minimal jaxpr → onnx_ir lowering loop:
      - trace ClosedJaxpr with abstract inputs
      - map invars to graph inputs
      - dispatch each eqn to its plugins2.lower(ctx, eqn)
      - collect outvars as graph outputs
    """
    if ir is None:
        raise ImportError("onnx_ir is required for converter2 but could not be imported") from _IR_IMPORT_ERROR

    # 1) Prepare abstract inputs for tracing
    sds_list = _as_sds_list(inputs, enable_double_precision)

    # 2) Trace to a ClosedJaxpr (thread input_params during tracing)
    def _wrapped(*xs):
        return fn(*xs, **(input_params or {}))

    # Temporarily enable x64 for tracing when double precision was requested.
    prev_x64 = None
    if enable_double_precision:
        try:
            prev_x64 = jax_config.read("jax_enable_x64")
        except Exception:
            prev_x64 = None
        if prev_x64 is False:
            jax_config.update("jax_enable_x64", True)
    try:
        # Apply high-level bindings/patches *only* for the trace.
        with _activate_plugin_worlds():
            closed = jax.make_jaxpr(_wrapped)(*sds_list)  # ClosedJaxpr
    finally:
        if enable_double_precision and prev_x64 is False:
            jax_config.update("jax_enable_x64", False)
    jpr = closed.jaxpr

    # 3) Build IR context and materialize graph inputs (+ consts if any)
    ctx = _IRBuildContext(
        opset=opset,
        default_float_dtype=np.float64 if enable_double_precision else np.float32,
    )

    # consts (rare in tanh path), map constvars → initializers
    for cv, cval in zip(jpr.constvars, closed.consts):
        np_c = np.asarray(cval)
        if np.issubdtype(np_c.dtype, np.floating):
            np_c = np_c.astype(ctx._default_float_dtype, copy=False)
        c_ir = ir.Value(
            name=ctx.fresh_name("const"),
            type=ir.TensorType(_to_ir_dtype_from_np(np_c.dtype)),
            shape=_to_ir_shape(np_c.shape),
            const_value=ir.tensor(np_c),
        )
        ctx._initializers.append(c_ir)
        ctx._var2val[cv] = c_ir

    # invars → graph inputs
    for i, v in enumerate(jpr.invars):
        ctx.add_input_for_invar(v, i)

    # 4) Walk equations and dispatch to plugin.lower
    _seen_prims: list[str] = []
    for eqn in jpr.eqns:
        prim_name = eqn.primitive.name
        _seen_prims.append(prim_name)
        plugin_ref = PLUGIN_REGISTRY2.get(prim_name)
        if plugin_ref is None:
            raise NotImplementedError(f"[converter2] No plugins2 registered for primitive '{prim_name}'")
        # Registry may store a class or an instance. Support both.
        plugin = plugin_ref() if isinstance(plugin_ref, type) else plugin_ref
        lower = getattr(plugin, "lower", None)
        if lower is None:
            ref_name = getattr(plugin_ref, "__name__", plugin_ref.__class__.__name__)
            raise NotImplementedError(f"[converter2] Plugin '{ref_name}' lacks a 'lower(ctx, eqn)' method")
        lower(ctx, eqn)

    # 5) Collect outputs
    out_vals: List[ir.Value] = [ctx.get_value_for_var(v, name_hint=f"out{i}") for i, v in enumerate(jpr.outvars)]

    # 6) Assemble graph & model
    graph = ir.Graph(
        inputs=list(ctx._inputs),
        outputs=out_vals,
        nodes=list(ctx._nodes),
        initializers=list(ctx._initializers),
        name=model_name or "jax2onnx_ir_graph",
        opset_imports={"": opset},
    )
    model = ir.Model(graph, ir_version=_ORT_SAFE_IR_VERSION)

    # 7) Return an onnx.ModelProto (serialize via onnx_ir then load with onnx)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            tmp_path = f.name
        ir.save(model, tmp_path)
        onnx_model = onnx.load_model(tmp_path)

        if record_primitive_calls_file:
            try:
                with open(record_primitive_calls_file, "w", encoding="utf-8") as fh:
                    for p in _seen_prims:
                        fh.write(f"{p}\n")
            except Exception:
                pass  # best-effort

        return onnx_model
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        ir.save(model, tmp_path)
        return onnx.load_model(tmp_path)

