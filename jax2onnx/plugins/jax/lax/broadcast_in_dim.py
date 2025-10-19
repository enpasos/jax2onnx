# jax2onnx/plugins/jax/lax/broadcast_in_dim.py

from typing import TYPE_CHECKING, Any, Final, Optional, Set
import jax
import jax.numpy as jnp
from jax import lax
from jax._src.export.shape_poly import _DimExpr
import numpy as np
import onnx_ir as ir

# from onnx_ir import Attribute as IRAttr
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.converter.ir_optimizations import _get_attr as _iro_get_attr
from jax2onnx.converter.ir_optimizations import _node_inputs as _iro_node_inputs

if TYPE_CHECKING:
    pass  # for hints

_IR_TO_NP_DTYPE: Final[dict[ir.DataType | None, np.dtype[Any]]] = {
    getattr(ir.DataType, "FLOAT16", None): np.float16,
    getattr(ir.DataType, "BFLOAT16", None): getattr(np, "bfloat16", np.float16),
    ir.DataType.FLOAT: np.float32,
    getattr(ir.DataType, "DOUBLE", None): np.float64,
    ir.DataType.INT8: np.int8,
    ir.DataType.INT16: np.int16,
    ir.DataType.INT32: np.int32,
    ir.DataType.INT64: np.int64,
    getattr(ir.DataType, "UINT8", None): np.uint8,
    getattr(ir.DataType, "UINT16", None): np.uint16,
    getattr(ir.DataType, "UINT32", None): np.uint32,
    getattr(ir.DataType, "UINT64", None): np.uint64,
    ir.DataType.BOOL: np.bool_,
}


def _dynamic_or_constant(specs, *, symbols=None):
    dynamic_checker = EG(specs, symbols=symbols, no_unused_inputs=True)
    constant_checker = EG([])

    def _check(model):
        return dynamic_checker(model) or constant_checker(model)

    return _check


# def _np_dtype_from_ir(enum) -> Optional[np.dtype]:
#     if isinstance(enum, ir.DataType):
#         return _IR_TO_NP_DTYPE.get(enum)
#     if isinstance(enum, (int, np.integer)):
#         try:
#             return _IR_TO_NP_DTYPE.get(ir.DataType(enum))
#         except Exception:
#             return None
#     return None


def _value_to_numpy(val: ir.Value | None):
    if val is None:
        return None
    for attr in ("const_value", "_const_value", "value", "data", "numpy"):
        payload = getattr(val, attr, None)
        if payload is None:
            continue
        try:
            return np.asarray(payload)
        except Exception:
            try:
                return np.asarray(payload())
            except Exception:
                continue
    return None


def _static_shape_tuple(shape_tuple):
    dims = []
    for dim in shape_tuple:
        if isinstance(dim, (int, np.integer)):
            dims.append(int(dim))
        else:
            return None
    return tuple(dims)

def _eval_primitive(primitive, *args, **kwargs):
    return primitive.bind(*args, **kwargs)

def _maybe_inline_constant_broadcast(ctx, out_var, x_val, shape, bdims, op_shape):
    const_arr = ctx.try_evaluate_const(x_val, _eval_primitive)
    if const_arr is None:
        return False

    static_shape = _static_shape_tuple(shape)
    if static_shape is None:
        return False

    reshape_tuple = None
    if len(op_shape) != len(shape):
        dims = [1] * len(shape)
        ok = True
        for src_axis, out_axis in enumerate(bdims):
            if src_axis >= len(op_shape):
                ok = False
                break
            dim_size = op_shape[src_axis]
            if not isinstance(dim_size, (int, np.integer)):
                ok = False
                break
            dims[out_axis] = int(dim_size)
        if not ok:
            return False
        reshape_tuple = tuple(dims)

    arr = np.asarray(const_arr)
    try:
        if reshape_tuple is not None and tuple(arr.shape) != reshape_tuple:
            arr = np.reshape(arr, reshape_tuple)
        broadcasted = np.broadcast_to(arr, static_shape)
    except Exception:
        return False

    target_dtype = None
    aval = getattr(out_var, "aval", None)
    if aval is not None:
        try:
            target_dtype = np.dtype(getattr(aval, "dtype", arr.dtype))
        except TypeError:
            target_dtype = None
    if target_dtype is not None and broadcasted.dtype != target_dtype:
        broadcasted = np.asarray(broadcasted, dtype=target_dtype)

    ctx.bind_const_for_var(out_var, np.asarray(broadcasted))
    return True


@register_primitive(
    jaxpr_primitive=jax.lax.broadcast_in_dim_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/jax-primitives.html",
    onnx=[
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        },
        {
            "component": "Expand",
            "doc": "https://onnx.ai/onnx/operators/onnx__Expand.html",
        },
        {  # Added Identity for completeness
            "component": "Identity",
            "doc": "https://onnx.ai/onnx/operators/onnx__Identity.html",
        },
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="broadcast_in_dim",
    testcases=[
        {
            "testcase": "broadcast_in_dim",
            "callable": lambda x: jax.lax.broadcast_in_dim(
                x, (3,), broadcast_dimensions=(0,)
            ),
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                [],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "broadcast_in_dim_2d_to_3d",
            "callable": lambda x: jax.lax.broadcast_in_dim(
                x, (2, 3, 4), broadcast_dimensions=(1, 2)
            ),
            "input_shapes": [(3, 4)],
            "post_check_onnx_graph": EG(
                ["Reshape:1x3x4 -> Expand:2x3x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "broadcast_in_dim_scalar",
            "callable": lambda x: jax.lax.broadcast_in_dim(
                x, (2, 3, 4), broadcast_dimensions=()
            ),
            "input_shapes": [()],
            # switch to value-based numeric testing
            "input_values": [0.5],
            "post_check_onnx_graph": EG(
                ["Expand:2x3x4"],
                no_unused_inputs=True,
            ),
        },
        {
            # ------------------------------------------------------------------
            # Re‑creates the "broadcast (1,1,D) → (B,1,D)" pattern that broke
            # when `shape` contained the symbolic batch dimension  B.
            # ------------------------------------------------------------------
            "testcase": "broadcast_in_dim_batch",
            "callable": lambda x: jnp.broadcast_to(  # ⤵ uses lax.broadcast_in_dim
                jnp.zeros((1, 1, x.shape[-1]), dtype=x.dtype),  #   token (1,1,D)
                (x.shape[0], 1, x.shape[-1]),  # → (B,1,D)
            ),
            "input_shapes": [
                ("B", 49, 256)
            ],  # Use a concrete batch for non-dynamic test
            "expected_output_shapes": [("B", 1, 256)],
            "post_check_onnx_graph": _dynamic_or_constant(
                ["Shape -> Gather -> Concat -> Expand:Bx1x256"],
                symbols={"B": None},
            ),
        },
        # ------------------------------------------------------------------
        # dynamic-batch test: symbolic B
        {
            "testcase": "broadcast_in_dim_dynamic_B",
            "callable": lambda x: lax.broadcast_in_dim(
                0.5, shape=(x.shape[0], 3, 4), broadcast_dimensions=()
            ),
            "input_shapes": [("B",)],  # symbolic batch dim
            "post_check_onnx_graph": _dynamic_or_constant(
                [
                    {
                        "inputs": {0: {"const": 0.5}},
                        "path": "Shape -> Gather -> Concat -> Expand:Bx3x4",
                    }
                ],
                symbols={"B": None},
            ),
        },
    ],
)
class BroadcastInDimPlugin(PrimitiveLeafPlugin):
    """
    Lower jax.lax.broadcast_in_dim(x, shape, broadcast_dimensions) to:
        (optional) Reshape(x, reshape_shape) -> Expand(…, target_shape)
    where reshape_shape inserts 1s in the non-mapped result axes.
    """

    def lower(self, ctx, eqn):
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError(
                "IR build context missing builder for broadcast_in_dim lowering"
            )

        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]
        shape = tuple(eqn.params["shape"])
        bdims = tuple(eqn.params["broadcast_dimensions"])

        hints = getattr(ctx, "_scatter_window_hints", None)
        allow_hints = bool(bdims)

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("bcast_in"))
        op_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))

        if _maybe_inline_constant_broadcast(
            ctx, out_var, x_val, shape, bdims, op_shape
        ):
            return

        def _peek_scatter_hint(axis: int) -> ir.Value | None:
            if not isinstance(hints, dict):
                return None
            values = hints.get(axis)
            if not values:
                return None
            return values[-1]

        modified_target_shape: list[ir.Value | _DimExpr | int] = []
        for axis, d in enumerate(shape):
            override_val = _peek_scatter_hint(axis)
            if allow_hints and hints and axis not in bdims and override_val is not None:
                modified_target_shape.append(override_val)
            else:
                modified_target_shape.append(d)

        # If operand is a scalar or the number of dimensions does not change, we can skip the Reshape and go straight to Expand.
        need_reshape = len(op_shape) > 0 and len(shape) != len(op_shape)

        rrank = len(shape)
        reshape_dims: Optional[list[int]] = [1] * rrank
        
        # Build reshape_shape by placing operand dims into their mapped result axes, 1 elsewhere.
        for i, r_axis in enumerate(bdims):
            reshape_dims[r_axis] = op_shape[i]
        
        if need_reshape:
            reshape_dim_vals = ctx.dim_expr_lowerer(reshape_dims)
            
            reshaped_val = builder.Reshape(
                x_val,
                reshape_dim_vals,
                _outputs=[ctx.fresh_name("bcast_reshape_out")],
            )
            _stamp_type_and_shape(reshaped_val, tuple(reshape_dims))
            if getattr(x_val, "type", None) is not None:
                reshaped_val.type = x_val.type
            _ensure_value_metadata(ctx, reshaped_val)
            expand_input = reshaped_val
        else:
            expand_input = x_val  # scalar or already aligned

        # Final expanded tensor should match the outvar's jax aval.
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("bcast_out"))
        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        out_dtype = getattr(getattr(out_spec, "type", None), "dtype", None)

        if getattr(out_spec, "producer", lambda: None)() is not None:
            desired_name = ctx.fresh_name("Expand")  # Already produced, need unique name
        else:
            desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("Expand")

        # Only do the Expand if we actually need to Broadcast. The lax.broadcast_in_dim is often used for simple reshaping by JAX
        if modified_target_shape != reshape_dims:
            modified_target_shape_val = ctx.dim_expr_lowerer(modified_target_shape)
            
            #Final Expand
            expanded_out = builder.Expand(
                expand_input,
                modified_target_shape_val,
                _outputs=[desired_name],
            )
            final_dtype = out_dtype or getattr(
                getattr(expand_input, "type", None), "dtype", None
            )
            if final_dtype is not None:
                expanded_out.type = ir.TensorType(final_dtype)
            _stamp_type_and_shape(expanded_out, out_shape)
            _ensure_value_metadata(ctx, expanded_out)
            ctx.bind_value_for_var(out_var, expanded_out)
        else:
            ctx.bind_value_for_var(out_var, expand_input)
    
