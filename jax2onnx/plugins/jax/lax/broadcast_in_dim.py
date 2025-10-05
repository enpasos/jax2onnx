# jax2onnx/plugins/jax/lax/broadcast_in_dim.py

from typing import TYPE_CHECKING, Optional, Set
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import onnx_ir as ir

# from onnx_ir import Attribute as IRAttr  # NEW: proper Attr objects
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from onnx_ir import (
    Attr as IRAttr,
    AttributeType as IRAttrType,
)  # FIX: correct attr types
from jax2onnx.converter.ir_optimizations import _get_attr as _iro_get_attr
from jax2onnx.converter.ir_optimizations import _node_inputs as _iro_node_inputs

if TYPE_CHECKING:
    pass  # for hints

_IR_TO_NP_DTYPE = {
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


def _np_dtype_from_ir(enum) -> Optional[np.dtype]:
    if isinstance(enum, ir.DataType):
        return _IR_TO_NP_DTYPE.get(enum)
    if isinstance(enum, (int, np.integer)):
        try:
            return _IR_TO_NP_DTYPE.get(ir.DataType(enum))
        except Exception:
            return None
    return None


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


def _node_constant_array(ctx, node, target_value, seen: Set[object]):
    op_type = getattr(node, "op_type", "")
    inputs = _iro_node_inputs(node)
    if op_type == "Cast" and inputs:
        arr = _materialize_constant_array(ctx, inputs[0], seen)
        if arr is None:
            return None
        target_enum = getattr(getattr(target_value, "type", None), "dtype", None)
        dtype = _np_dtype_from_ir(target_enum)
        if dtype is not None:
            return np.asarray(arr, dtype=dtype)
        return arr
    if op_type == "CastLike" and len(inputs) >= 2:
        arr = _materialize_constant_array(ctx, inputs[0], seen)
        like_arr = _materialize_constant_array(ctx, inputs[1], seen)
        if arr is None or like_arr is None:
            return None
        return np.asarray(arr, dtype=like_arr.dtype)
    if op_type == "Reshape" and len(inputs) >= 2:
        data_arr = _materialize_constant_array(ctx, inputs[0], seen)
        shape_arr = _materialize_constant_array(ctx, inputs[1], seen)
        if data_arr is None or shape_arr is None:
            return None
        try:
            target_shape = tuple(int(x) for x in np.asarray(shape_arr).reshape(-1))
        except Exception:
            return None
        try:
            return np.reshape(data_arr, target_shape)
        except Exception:
            return None
    return None


def _materialize_constant_array(ctx, value, seen: Optional[Set[object]] = None):
    arr = _value_to_numpy(value)
    if arr is not None:
        return arr
    name = getattr(value, "name", None)
    if name:
        inits = []
        for attr in ("initializers", "_initializers"):
            seq = getattr(ctx.builder, attr, None)
            if seq is None:
                continue
            try:
                inits.extend(list(seq))
            except Exception:
                try:
                    inits.extend(iter(seq))
                except Exception:
                    pass
        for init in inits:
            if getattr(init, "name", None) == name:
                arr = _value_to_numpy(init)
                if arr is not None:
                    return arr

    producer_fn = getattr(value, "producer", None)
    node = None
    if callable(producer_fn):
        try:
            node = producer_fn()
        except Exception:
            node = None
    if node is None:
        return None
    if seen is None:
        seen = set()
    if node in seen:
        return None
    seen.add(node)
    arr = _node_constant_array(ctx, node, value, seen)
    if arr is not None:
        return arr
    # Fallback: Constant node attributes
    if getattr(node, "op_type", "") == "Constant":
        attr = _iro_get_attr(node, "value")
        if attr is not None:
            return _value_to_numpy(attr)
    return None


def _maybe_inline_constant_broadcast(ctx, out_var, x_val, shape, bdims, op_shape):
    const_arr = _materialize_constant_array(ctx, x_val)
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
        },
        {
            "testcase": "broadcast_in_dim_2d_to_3d",
            "callable": lambda x: jax.lax.broadcast_in_dim(
                x, (2, 3, 4), broadcast_dimensions=(1, 2)
            ),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "broadcast_in_dim_scalar",
            "callable": lambda x: jax.lax.broadcast_in_dim(
                x, (2, 3, 4), broadcast_dimensions=()
            ),
            "input_shapes": [()],
            # switch to value-based numeric testing
            "input_values": [0.5],
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
        },
        # ------------------------------------------------------------------
        # dynamic-batch test: symbolic B
        {
            "testcase": "broadcast_in_dim_dynamic_B",
            "callable": lambda x: lax.broadcast_in_dim(
                0.5, shape=(x.shape[0], 3, 4), broadcast_dimensions=()
            ),
            "input_shapes": [("B",)],  # symbolic batch dim
            "post_check_onnx_graph": lambda m: (
                __import__("onnx").checker.check_model(m) or True
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

        # Build target shape as a 1-D INT64 tensor, supporting symbolic dims.
        # Each dimension becomes a length-1 vector; we Concat along axis=0.
        dim_pieces: list[ir.Value] = []
        # We'll keep a reference tensor with the *desired* float dtype (usually from the
        # origin of a symbolic dim like 'B') so we can CastLike the operand if needed.
        like_ref_val: ir.Value | None = None
        for axis, d in enumerate(shape):
            if allow_hints and hints and axis not in bdims:
                override_val = _peek_scatter_hint(axis)
                if override_val is not None:
                    dim_pieces.append(override_val)
                    continue
            if isinstance(d, (int, np.integer)):
                c = ir.Value(
                    name=ctx.fresh_name("bcast_dim_c"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((1,)),
                    const_value=ir.tensor(np.array([int(d)], dtype=np.int64)),
                )
                ctx._initializers.append(c)
                dim_pieces.append(c)
            else:
                # Dynamic/symbolic dimension: fetch from its recorded origin.
                origin = getattr(ctx, "get_symbolic_dim_origin", None)
                if origin is None:
                    raise NotImplementedError(
                        "symbolic dims require ctx.get_symbolic_dim_origin"
                    )
                src = origin(d)
                if src is None:
                    raise NotImplementedError(
                        f"no origin recorded for symbolic dim '{d}'"
                    )
                src_val, axis = src
                # Save the first available tensor as a dtype reference.
                if like_ref_val is None:
                    like_ref_val = src_val
                # Shape(src) → Gather(…, [axis]) → length-1 vector
                src_rank = len(
                    getattr(getattr(src_val, "shape", None), "dims", ()) or ()
                )
                shp = ir.Value(
                    name=ctx.fresh_name("bcast_src_shape"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((src_rank,)),
                )
                shape_node = ir.Node(
                    op_type="Shape",
                    domain="",
                    inputs=[src_val],
                    outputs=[shp],
                    name=ctx.fresh_name("Shape"),
                )
                ctx.add_node(shape_node)

                idx = ir.Value(
                    name=ctx.fresh_name("bcast_idx"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((1,)),
                    const_value=ir.tensor(np.array([int(axis)], dtype=np.int64)),
                )
                ctx._initializers.append(idx)

                dim1 = ir.Value(
                    name=ctx.fresh_name("bcast_dim_dyn"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((1,)),
                )
                gather_node = ir.Node(
                    op_type="Gather",
                    domain="",
                    inputs=[shp, idx],
                    outputs=[dim1],
                    name=ctx.fresh_name("Gather"),
                    attributes=[IRAttr("axis", IRAttrType.INT, 0)],  # FIX
                )
                ctx.add_node(gather_node)
                dim_pieces.append(dim1)

        tgt_shape_val = ir.Value(
            name=ctx.fresh_name("bcast_target_shape"),
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape((len(shape),)),
        )
        concat_node = ir.Node(
            op_type="Concat",
            domain="",
            inputs=dim_pieces,
            outputs=[tgt_shape_val],
            name=ctx.fresh_name("Concat"),
            attributes=[IRAttr("axis", IRAttrType.INT, 0)],  # FIX
        )
        ctx.add_node(concat_node)

        # If operand is a scalar, we can skip the Reshape and go straight to Expand.
        need_reshape = len(op_shape) > 0 and len(shape) != len(op_shape)

        if need_reshape:
            # Build reshape_shape by placing operand dims into their mapped result axes, 1 elsewhere.
            rrank = len(shape)
            reshape_dims: list[int] = [1] * rrank
            for i, r_axis in enumerate(bdims):
                # guard if aval dims are unknown (shouldn't happen on tests)
                dim = (
                    int(op_shape[i])
                    if i < len(op_shape) and isinstance(op_shape[i], (int, np.integer))
                    else 1
                )
                reshape_dims[r_axis] = dim

            reshape_dim_pieces: list[ir.Value] = []
            for axis, dim in enumerate(reshape_dims):
                override_val = None
                if allow_hints and hints and axis not in bdims:
                    override_val = _peek_scatter_hint(axis)
                if override_val is not None:
                    reshape_dim_pieces.append(override_val)
                    continue
                const_val = ir.Value(
                    name=ctx.fresh_name("bcast_reshape_dim"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((1,)),
                    const_value=ir.tensor(np.asarray([int(dim)], dtype=np.int64)),
                )
                ctx._initializers.append(const_val)
                reshape_dim_pieces.append(const_val)

            rs_val = ir.Value(
                name=ctx.fresh_name("bcast_reshape_shape"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape((rrank,)),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Concat",
                    domain="",
                    inputs=reshape_dim_pieces,
                    outputs=[rs_val],
                    name=ctx.fresh_name("Concat"),
                    attributes=[IRAttr("axis", IRAttrType.INT, 0)],
                )
            )

            reshaped_val = ir.Value(
                name=ctx.fresh_name("bcast_reshape_out"),
                type=x_val.type,  # same dtype
                shape=ir.Shape(tuple(reshape_dims)),  # rank == result rank
            )
            reshape_node = ir.Node(
                op_type="Reshape",
                domain="",
                inputs=[x_val, rs_val],
                outputs=[reshaped_val],
                name=ctx.fresh_name("Reshape"),
            )
            ctx.add_node(reshape_node)
            expand_input = reshaped_val
        else:
            expand_input = x_val  # scalar or already aligned

        # Final expanded tensor should match the outvar's jax aval; let ctx create it.
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("bcast_out"))
        # onnx_ir refuses to assign a new producer to a Value that already
        # belongs to another node. When the same JAX var is materialised via
        # multiple broadcast ops (common inside nnx.Sequential), rebinding the
        # cached Value avoids "producer already set" errors.
        if (
            callable(getattr(out_val, "producer", None))
            and out_val.producer() is not None
        ):
            out_val = ir.Value(
                name=ctx.fresh_name("bcast_out"),
                type=out_val.type,
                shape=out_val.shape,
            )
            ctx.builder._var2val[out_var] = out_val

        # --- DTYPE UNIFICATION --------------------------------------------------
        # If the operand (possibly a Python float literal → fp64) doesn't match the
        # expected output dtype (usually fp32 in x32 mode), cast it before Expand.
        in_dt = getattr(getattr(expand_input, "type", None), "dtype", None)
        out_dt = getattr(getattr(out_val, "type", None), "dtype", None)
        if (in_dt is not None) and (out_dt is not None) and (in_dt != out_dt):
            # Use a "like" tensor ONLY if it already matches the target dtype.
            like_for_cast = None
            lr_dt = getattr(getattr(like_ref_val, "type", None), "dtype", None)
            if like_ref_val is not None and lr_dt == out_dt:
                like_for_cast = like_ref_val

            # Otherwise synthesize a tiny constant with the target dtype.
            if like_for_cast is None:
                _np_dt = {
                    ir.DataType.FLOAT: np.float32,
                    ir.DataType.DOUBLE: np.float64,
                    # (extend here if you ever hit other types)
                }.get(out_dt, None)
                if _np_dt is None:
                    # If we can't materialize a "like", skip the cast safely.
                    # (Shouldn't happen in these tests.)
                    pass
                else:
                    like_for_cast = ir.Value(
                        name=ctx.fresh_name("bcast_like_c"),
                        type=ir.TensorType(out_dt),
                        shape=ir.Shape((1,)),
                        const_value=ir.tensor(np.zeros((1,), dtype=_np_dt)),
                    )
                    ctx._initializers.append(like_for_cast)
            if like_for_cast is not None:
                casted = ir.Value(
                    name=ctx.fresh_name("bcast_in_cast"),
                    type=ir.TensorType(out_dt),
                    shape=expand_input.shape,
                )
                ctx.add_node(
                    ir.Node(
                        op_type="CastLike",
                        domain="",
                        inputs=[expand_input, like_for_cast],
                        outputs=[casted],
                        name=ctx.fresh_name("CastLike"),
                    )
                )
                expand_input = casted
        # -----------------------------------------------------------------------

        expand_node = ir.Node(
            op_type="Expand",
            domain="",
            inputs=[expand_input, tgt_shape_val],
            outputs=[out_val],
            name=ctx.fresh_name("Expand"),
        )
        ctx.add_node(expand_node)
