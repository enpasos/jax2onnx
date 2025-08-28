# file: jax2onnx/converter2/conversion_api.py


from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager, ExitStack
from jax import export as jax_export

from jax2onnx.plugins2.plugin_system import (
    PLUGIN_REGISTRY2,
    PrimitiveLeafPlugin,
    apply_monkey_patches,
    import_all_plugins,
)
import onnx_ir as ir
import numpy as np
import onnx
import jax
import jax.numpy as jnp

# new imports
from .ir_context import IRContext
from .ir_builder import IRBuilder

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
        # NEW: activate centralized per-plugin bindings
        for plugin_instance in PLUGIN_REGISTRY2.values():
            if isinstance(plugin_instance, PrimitiveLeafPlugin):
                stack.enter_context(plugin_instance.__class__.plugin_binding())

        # LEGACY: still support older plugins that provide patch_info()
        stack.enter_context(apply_monkey_patches())

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
    # 1) Prepare abstract inputs for tracing (respect x64 policy)
    import numpy as np

    default_float = np.float64 if enable_double_precision else np.float32
    sds_list = _as_sds_list(inputs, enable_double_precision)

    # 2) Trace to ClosedJaxpr (input_params threaded; plugin worlds already handled above)
    def _wrapped(*xs):
        return fn(*xs, **(input_params or {}))

    with _activate_plugin_worlds():
        closed = jax.make_jaxpr(_wrapped)(*sds_list)
    jpr = closed.jaxpr

    # 3) Build IR context & bind inputs/consts
    ctx = IRContext(
        opset=opset,
        enable_double_precision=enable_double_precision,
        input_specs=sds_list,
    )

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

    # expose builder to FunctionPlugin via a tiny facade
    class _ConverterFacade:
        builder: IRBuilder

    converter = _ConverterFacade()
    converter.builder = ctx.builder

    for eqn in jpr.eqns:
        prim_name = eqn.primitive.name
        _seen_prims.append(prim_name)
        plugin_ref = PLUGIN_REGISTRY2.get(prim_name)
        if plugin_ref is None:
            raise NotImplementedError(
                f"[converter2] No plugins2 registered for primitive '{prim_name}'"
            )
        # leaf primitive plugins expose .lower(ctx, eqn)
        # function plugins use FunctionPlugin’s handler via plugin_system
        if hasattr(plugin_ref, "lower"):
            plugin_ref.lower(ctx, eqn)
        elif hasattr(plugin_ref, "get_handler"):
            handler = plugin_ref.get_handler(converter)
            handler(converter, eqn, eqn.params)
        else:
            raise NotImplementedError(
                f"[converter2] Unsupported plugin type for '{prim_name}'"
            )

    # 5) Outputs
    ctx.add_outputs_from_vars(jpr.outvars)

    # 6) Model proto
    model = ctx.to_model_proto(name=model_name)
    return model
