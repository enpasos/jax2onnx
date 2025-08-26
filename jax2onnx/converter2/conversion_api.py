from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import tempfile

import numpy as np
import onnx
import numpy as np
import jax
import jax.numpy as jnp
from jax import core as jcore

# NOTE: onnx_ir: https://github.com/onnx/ir-py
# We use it as the builder backend.
try:
    import onnx_ir as ir
except Exception as e:  # pragma: no cover
    ir = None
    _IR_IMPORT_ERROR = e

# --- JAX core type compatibility (JAX >= 0.6 moved these) --------------------
try:  # JAX 0.6+
    from jax.extend.core import Literal as JaxLiteral, Var as JaxVar
except Exception:  # Older JAX
    from jax.core import Literal as JaxLiteral, Var as JaxVar

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

def _as_sds_list(
    inputs: List[Any], enable_double_precision: bool
) -> List["jax.ShapeDtypeStruct"]:
    """Normalize user 'inputs' to ShapeDtypeStructs for abstract tracing."""
    sds_list: List[jax.ShapeDtypeStruct] = []
    for spec in inputs:
        if hasattr(spec, "shape") and hasattr(spec, "dtype"):
            # ShapedArray / ShapeDtypeStruct already
            sds_list.append(jax.ShapeDtypeStruct(tuple(spec.shape), spec.dtype))
        elif isinstance(spec, (list, tuple)):
            # plain shape → choose float dtype based on flag
            dt = jnp.float64 if enable_double_precision else jnp.float32
            sds_list.append(jax.ShapeDtypeStruct(tuple(spec), dt))
        else:
            raise TypeError(f"Unsupported input spec: {type(spec)}")
    return sds_list


# ---------------------------
# Minimal IR Build Context
# ---------------------------
class _IRBuildContext:
    def __init__(self, *, opset: int):
        self.opset = opset
        self._var2val: Dict[Any, ir.Value] = {}
        self._inputs: List[ir.Value] = []
        self._initializers: List[ir.Value] = []
        self._nodes: List[ir.Node] = []
        self._name_counter = 0

    # API used by plugins2
    def fresh_name(self, prefix: str) -> str:
        self._name_counter += 1
        return f"{prefix}_{self._name_counter}"

    def add_node(self, op_type: str, inputs: List[ir.Value], outputs: List[ir.Value], **attrs):
        # attributes dict can be empty
        node = ir.node(op_type=op_type, inputs=inputs, outputs=outputs, attributes=attrs or None)
        self._nodes.append(node)
        return node

    def get_value_for_var(self, var, *, name_hint: Optional[str] = None) -> "ir.Value":
        # If we've already materialized it, return it.
        if var in self._var2val:
            return self._var2val[var]
        # Literals become constants
        if isinstance(var, JaxLiteral):
            np_c = np.asarray(var.val)
            v = ir.Value(
                name=self.fresh_name("const") if name_hint is None else name_hint,
                type=ir.TensorType(_to_ir_dtype_from_np(np_c.dtype)),
                shape=_to_ir_shape(np_c.shape),
                const_value=ir.tensor(np_c),
            )
            self._initializers.append(v)
            self._var2val[var] = v
            return v
        # Regular JAX Var with aval describes shape & dtype
        aval = getattr(var, "aval", None)
        if aval is None:
            raise TypeError(f"Unsupported var without aval: {var!r}")
        v = ir.Value(
            name=self.fresh_name("v") if name_hint is None else name_hint,
            type=ir.TensorType(_to_ir_dtype_from_np(getattr(aval, "dtype", None))),
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
        return val


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

    closed = jax.make_jaxpr(_wrapped)(*sds_list)  # ClosedJaxpr
    jpr = closed.jaxpr

    # 3) Build IR context and materialize graph inputs (+ consts if any)
    ctx = _IRBuildContext(opset=opset)

    # consts (rare in tanh path), map constvars → initializers
    for cv, cval in zip(jpr.constvars, closed.consts):
        np_c = np.asarray(cval)
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
    for eqn in jpr.eqns:
        prim_name = eqn.primitive.name
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
        return onnx.load_model(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            try:
                os.remove(tmp_path)
            except OSError:
                pass
