# jax2onnx/plugins/jax/lax/broadcast_in_dim.py

import os
from typing import TYPE_CHECKING, Any, Final, Optional, Set
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import onnx_ir as ir

# from onnx_ir import Attribute as IRAttr
from jax2onnx.converter.typing_support import (
    LoweringContextProtocol,
    SymbolicDimOrigin,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import (
    _ensure_value_metadata,
    _stamp_type_and_shape,
    _to_ir_dim_for_shape,
)
from jax2onnx.plugins._loop_extent_meta import get_axis0_override, set_axis0_override
from jax2onnx.plugins._axis0_utils import ensure_axis0_extent, _static_dim_as_int
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.converter.ir_optimizations import _get_attr as _iro_get_attr
from jax2onnx.converter.ir_optimizations import _node_inputs as _iro_node_inputs

if TYPE_CHECKING:
    pass  # for hints

_IR_TO_NP_DTYPE: Final[dict[ir.DataType, np.dtype[Any]]] = {
    ir.DataType.FLOAT16: np.dtype(np.float16),
    ir.DataType.BFLOAT16: np.dtype(getattr(np, "bfloat16", np.float16)),
    ir.DataType.FLOAT: np.dtype(np.float32),
    ir.DataType.DOUBLE: np.dtype(np.float64),
    ir.DataType.INT8: np.dtype(np.int8),
    ir.DataType.INT16: np.dtype(np.int16),
    ir.DataType.INT32: np.dtype(np.int32),
    ir.DataType.INT64: np.dtype(np.int64),
    ir.DataType.UINT8: np.dtype(np.uint8),
    ir.DataType.UINT16: np.dtype(np.uint16),
    ir.DataType.UINT32: np.dtype(np.uint32),
    ir.DataType.UINT64: np.dtype(np.uint64),
    ir.DataType.BOOL: np.dtype(np.bool_),
}


def _dynamic_or_constant(specs, *, symbols=None):
    dynamic_checker = EG(specs, symbols=symbols, no_unused_inputs=True)
    constant_checker = EG([])

    def _check(model):
        return dynamic_checker(model) or constant_checker(model)

    return _check


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
    const_arr = ctx.try_evaluate_const(x_val)
    if const_arr is None:
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

    def lower(self, ctx: LoweringContextProtocol, eqn):
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError(
                "IR build context missing builder for broadcast_in_dim lowering"
            )

        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]
        shape = tuple(eqn.params["shape"])
        target_shape_dims: list[Any] = list(shape)
        bdims = tuple(eqn.params["broadcast_dimensions"])
        axis0_in_bdims = 0 in bdims

        hints = getattr(ctx, "_scatter_window_hints", None)
        use_loop_hints = bool(getattr(ctx, "_loop_extent_hints_enabled", False))
        loop_extents = (
            getattr(ctx, "_loop_extent_hints", None) if use_loop_hints else None
        )

        def _loop_hint(axis: int) -> ir.Value | None:
            if not isinstance(loop_extents, dict):
                return None
            if axis != 0:
                return None
            values = loop_extents.get(axis)
            if not values:
                return None
            if isinstance(values, list):
                return values[-1]
            return values

        allow_hints = bool(bdims)
        allow_loop_hints = bool(loop_extents)

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("bcast_in"))
        op_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("bcast_out"))
        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        out_axis0_static = (
            int(out_shape[0])
            if out_shape
            and isinstance(out_shape[0], (int, np.integer))
            and int(out_shape[0]) >= 0
            else None
        )

        rrank = len(shape)
        reshape_dims: list[object] = [1] * rrank
        for i, r_axis in enumerate(bdims):
            dim = op_shape[i] if i < len(op_shape) else 1
            reshape_dims[r_axis] = dim

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
        override_candidates = [
            get_axis0_override(x_val),
            get_axis0_override(out_spec),
        ]
        if axis0_in_bdims:
            override_candidates.append(getattr(ctx, "_static_loop_extent_axis0", None))
        meta_override_axis0 = next(
            (
                int(val)
                for val in override_candidates
                if isinstance(val, (int, np.integer)) and int(val) > 0
            ),
            None,
        )
        if meta_override_axis0 is None:
            fallback_ctx = getattr(ctx, "_static_loop_extent_axis0", None)
            if isinstance(fallback_ctx, (int, np.integer)) and int(fallback_ctx) > 0:
                meta_override_axis0 = int(fallback_ctx)
        if (
            isinstance(meta_override_axis0, (int, np.integer))
            and meta_override_axis0 > 0
        ):
            meta_override_axis0 = int(meta_override_axis0)
            if out_shape:
                out_shape = (meta_override_axis0,) + out_shape[1:]
            target_shape_dims[0] = meta_override_axis0
        debug = os.environ.get("J2O_DEBUG_BCAST_HINTS") == "1"
        for axis, d in enumerate(shape):
            if axis == 0 and out_axis0_static is not None and axis0_in_bdims:
                dim_pieces.append(
                    _const_i64(
                        ctx,
                        np.asarray([out_axis0_static], dtype=np.int64),
                        ctx.fresh_name("bcast_dim_axis0"),
                    )
                )
                continue
            if axis not in bdims and (allow_hints or allow_loop_hints):
                force_loop_axis0 = bool(
                    getattr(ctx, "_force_loop_extent_axis0", False)
                    and axis == 0
                    and axis0_in_bdims
                )
                override_val = None
                if force_loop_axis0:
                    override_val = _loop_hint(axis)
                    if override_val is None and hints:
                        override_val = _peek_scatter_hint(axis)
                    if debug:
                        print(
                            "[broadcast_hint_check]",
                            axis,
                            bdims,
                            override_val is not None,
                            getattr(ctx, "_force_loop_extent_axis0", False),
                            getattr(ctx, "_static_loop_extent_axis0", None),
                            flush=True,
                        )
                else:
                    override_val = _peek_scatter_hint(axis) if hints else None
                    if override_val is None:
                        override_val = _loop_hint(axis)
                    if debug:
                        print(
                            "[broadcast_hint_check]",
                            axis,
                            bdims,
                            override_val is not None,
                            getattr(ctx, "_force_loop_extent_axis0", False),
                            getattr(ctx, "_static_loop_extent_axis0", None),
                            flush=True,
                        )
                if override_val is not None:
                    if os.environ.get("J2O_DEBUG_BCAST_HINTS") == "1":
                        print(
                            "[broadcast_hint]",
                            axis,
                            bdims,
                            getattr(ctx, "_loop_extent_hints_enabled", False),
                            flush=True,
                        )
                    dim_pieces.append(override_val)
                    continue
            if (
                axis == 0
                and isinstance(meta_override_axis0, (int, np.integer))
                and int(meta_override_axis0) > 0
            ):
                dim_pieces.append(
                    _const_i64(
                        ctx,
                        np.asarray([int(meta_override_axis0)], dtype=np.int64),
                        ctx.fresh_name("bcast_dim_override"),
                    )
                )
                continue
            if isinstance(d, (int, np.integer)):
                dim_pieces.append(
                    _const_i64(
                        ctx,
                        np.asarray([int(d)], dtype=np.int64),
                        ctx.fresh_name("bcast_dim_c"),
                    )
                )
                target_shape_dims[axis] = int(d)
            else:
                # Dynamic/symbolic dimension: fetch from its recorded origin.
                origin_lookup = getattr(ctx, "get_symbolic_dim_origin", None)
                if origin_lookup is None:
                    raise NotImplementedError(
                        "symbolic dims require ctx.get_symbolic_dim_origin"
                    )
                origin = SymbolicDimOrigin.resolve(origin_lookup, d)
                if origin is None:
                    raise NotImplementedError(
                        f"no origin recorded for symbolic dim '{d}'"
                    )
                src_val = origin.value
                src_axis = int(origin.axis)
                # Shape(src) → Gather(…, [axis]) → length-1 vector
                src_rank = len(
                    getattr(getattr(src_val, "shape", None), "dims", ()) or ()
                )
                shp = builder.Shape(
                    src_val,
                    _outputs=[ctx.fresh_name("bcast_src_shape")],
                )
                _stamp_type_and_shape(shp, (src_rank,))
                _ensure_value_metadata(ctx, shp)

                idx = _const_i64(
                    ctx,
                    np.asarray([src_axis], dtype=np.int64),
                    ctx.fresh_name("bcast_idx"),
                )

                dim1 = builder.Gather(
                    shp,
                    idx,
                    axis=0,
                    _outputs=[ctx.fresh_name("bcast_dim_dyn")],
                )
                _stamp_type_and_shape(dim1, (1,))
                _ensure_value_metadata(ctx, dim1)
                dim_pieces.append(dim1)

        tgt_shape_val = builder.Concat(
            *dim_pieces,
            axis=0,
            _outputs=[ctx.fresh_name("bcast_target_shape")],
        )
        _stamp_type_and_shape(tgt_shape_val, (len(shape),))
        tgt_shape_val.type = ir.TensorType(ir.DataType.INT64)
        _ensure_value_metadata(ctx, tgt_shape_val)

        # If operand is a scalar, we can skip the Reshape and go straight to Expand.
        need_reshape = len(op_shape) > 0 and len(shape) != len(op_shape)

        if need_reshape:
            # Build reshape_shape by placing operand dims into their mapped result axes, 1 elsewhere.
            reshape_dim_pieces: list[ir.Value] = []
            for axis, dim in enumerate(reshape_dims):
                axis_hint_allowed = not (
                    axis == 0 and axis not in bdims and out_axis0_static is not None
                )
                override_val = None
                if (
                    axis not in bdims
                    and axis_hint_allowed
                    and (allow_hints or allow_loop_hints)
                ):
                    override_val = _peek_scatter_hint(axis) if hints else None
                    if override_val is None:
                        override_val = _loop_hint(axis)
                if override_val is not None:
                    reshape_dim_pieces.append(override_val)
                    continue
                if isinstance(dim, (int, np.integer)):
                    reshape_dim_pieces.append(
                        _const_i64(
                            ctx,
                            np.asarray([int(dim)], dtype=np.int64),
                            ctx.fresh_name("bcast_reshape_dim"),
                        )
                    )
                    continue
                origin_lookup = getattr(ctx, "get_symbolic_dim_origin", None)
                if origin_lookup is None:
                    raise NotImplementedError(
                        "symbolic dims require ctx.get_symbolic_dim_origin"
                    )
                origin = SymbolicDimOrigin.resolve(origin_lookup, dim)
                if origin is None:
                    raise NotImplementedError(
                        f"no origin recorded for symbolic dim '{dim}'"
                    )
                src_val = origin.value
                src_axis = int(origin.axis)
                src_rank = len(
                    getattr(getattr(src_val, "shape", None), "dims", ()) or ()
                )
                shp = builder.Shape(
                    src_val,
                    _outputs=[ctx.fresh_name("bcast_reshape_sym_shape")],
                )
                _stamp_type_and_shape(shp, (src_rank,))
                _ensure_value_metadata(ctx, shp)
                idx = _const_i64(
                    ctx,
                    np.asarray([src_axis], dtype=np.int64),
                    ctx.fresh_name("bcast_reshape_sym_idx"),
                )
                dim_val = builder.Gather(
                    shp,
                    idx,
                    axis=0,
                    _outputs=[ctx.fresh_name("bcast_reshape_sym_dim")],
                )
                _stamp_type_and_shape(dim_val, (1,))
                _ensure_value_metadata(ctx, dim_val)
                reshape_dim_pieces.append(dim_val)

            rs_val = builder.Concat(
                *reshape_dim_pieces,
                axis=0,
                _outputs=[ctx.fresh_name("bcast_reshape_shape")],
            )
            _stamp_type_and_shape(rs_val, (rrank,))
            rs_val.type = ir.TensorType(ir.DataType.INT64)
            _ensure_value_metadata(ctx, rs_val)

            reshaped_val = builder.Reshape(
                x_val,
                rs_val,
                _outputs=[ctx.fresh_name("bcast_reshape_out")],
            )
            _stamp_type_and_shape(reshaped_val, tuple(reshape_dims))
            if getattr(x_val, "type", None) is not None:
                reshaped_val.type = x_val.type
            _ensure_value_metadata(ctx, reshaped_val)
            reshape_override = next(
                (
                    int(val)
                    for val in (
                        get_axis0_override(x_val),
                        get_axis0_override(reshaped_val),
                        meta_override_axis0,
                    )
                    if isinstance(val, (int, np.integer)) and int(val) > 0
                ),
                None,
            )
            if reshape_override is not None:
                set_axis0_override(reshaped_val, reshape_override)
            expand_input = reshaped_val
        else:
            expand_input = x_val  # scalar or already aligned

        # Final expanded tensor should match the outvar's jax aval.
        if meta_override_axis0 is None:
            input_override = get_axis0_override(expand_input)
            if (
                isinstance(input_override, (int, np.integer))
                and int(input_override) > 0
            ):
                meta_override_axis0 = int(input_override)

        if meta_override_axis0 is None:
            fallback_override = next(
                (
                    int(val)
                    for val in (
                        get_axis0_override(out_spec),
                        getattr(ctx, "_static_loop_extent_axis0", None),
                    )
                    if isinstance(val, (int, np.integer)) and int(val) > 0
                ),
                None,
            )
            meta_override_axis0 = fallback_override
        out_dtype = getattr(getattr(out_spec, "type", None), "dtype", None)

        if os.environ.get("J2O_DEBUG_BCAST_SHAPE") == "1":
            print(
                "[broadcast_shape]",
                getattr(out_spec, "name", None),
                out_shape,
                bdims,
                flush=True,
            )

        in_dt = getattr(getattr(expand_input, "type", None), "dtype", None)
        if in_dt is not None and out_dtype is not None and in_dt != out_dtype:
            casted = builder.Cast(
                expand_input,
                _outputs=[ctx.fresh_name("bcast_in_cast")],
                to=int(out_dtype.value),
            )
            expand_input_shape = tuple(
                getattr(getattr(expand_input, "shape", None), "dims", ()) or out_shape
            )
            _stamp_type_and_shape(casted, expand_input_shape)
            _ensure_value_metadata(ctx, casted)
            expand_input = casted

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("Expand")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("Expand")

        expanded_out = builder.Expand(
            expand_input,
            tgt_shape_val,
            _outputs=[desired_name],
        )
        final_dtype = out_dtype or getattr(
            getattr(expand_input, "type", None), "dtype", None
        )
        if final_dtype is not None:
            expanded_out.type = ir.TensorType(final_dtype)
        _stamp_type_and_shape(expanded_out, out_shape)
        _ensure_value_metadata(ctx, expanded_out)
        if meta_override_axis0 is None and target_shape_dims:
            first_dim = target_shape_dims[0]
            first_dim_int = _static_dim_as_int(first_dim)
            if isinstance(first_dim_int, int) and first_dim_int > 1:
                meta_override_axis0 = first_dim_int
        if (
            isinstance(meta_override_axis0, (int, np.integer))
            and int(meta_override_axis0) >= 0
        ):
            override_int = int(meta_override_axis0)
            set_axis0_override(expanded_out, override_int)
            if override_int > 1:
                expanded_out = ensure_axis0_extent(
                    ctx, expanded_out, override_int, reference=out_spec
                )
                out_shape = (override_int,) + out_shape[1:] if out_shape else out_shape
                target_shape_dims[0] = override_int
        if target_shape_dims:
            try:
                stamped_dims = tuple(
                    _to_ir_dim_for_shape(dim) if not isinstance(dim, ir.Value) else None
                    for dim in target_shape_dims
                )
                if any(dim is not None for dim in stamped_dims):
                    _stamp_type_and_shape(expanded_out, stamped_dims)
                    _stamp_type_and_shape(out_spec, stamped_dims)
                    _ensure_value_metadata(ctx, expanded_out)
            except Exception:
                pass
        ctx.bind_value_for_var(out_var, expanded_out)
