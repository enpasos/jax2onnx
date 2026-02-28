# jax2onnx/plugins/jax/lax/_cum_extrema.py

from __future__ import annotations

from typing import Any

import numpy as np
import onnx_ir as ir

from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.jax.lax._index_utils import _const_i64


def _stamp_meta(
    ctx: Any,
    value: ir.Value,
    *,
    ref: ir.Value | None = None,
    dtype: ir.DataType | None = None,
    shape: tuple[object, ...] | None = None,
) -> None:
    if dtype is not None:
        value.type = ir.TensorType(dtype)
    elif ref is not None and getattr(ref, "type", None) is not None:
        value.type = ref.type
    if shape is not None:
        _stamp_type_and_shape(value, shape)
    _ensure_value_metadata(ctx, value)


def lower_cum_extrema(ctx: Any, eqn: Any, *, mode: str) -> None:
    """Lower ``cummax`` / ``cummin`` via MaxPool along a flattened axis."""

    if mode not in {"max", "min"}:
        raise ValueError(f"Unsupported cum extrema mode '{mode}'.")

    x_var = eqn.invars[0]
    out_var = eqn.outvars[0]
    params = dict(getattr(eqn, "params", {}) or {})

    axis_param = int(params.get("axis", 0))
    reverse = bool(params.get("reverse", False))

    x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()) or ())
    rank = len(x_shape)
    if rank == 0:
        raise ValueError(f"cum{mode} requires rank >= 1 input")

    axis = axis_param % rank if axis_param < 0 else axis_param
    if axis < 0 or axis >= rank:
        raise ValueError(f"cum{mode} axis {axis_param} out of range for rank {rank}")

    axis_extent = x_shape[axis]
    if not isinstance(axis_extent, (int, np.integer)):
        raise NotImplementedError(
            f"cum{mode} currently requires static axis extent, got '{axis_extent}'"
        )
    axis_extent_i = int(axis_extent)
    if axis_extent_i < 0:
        raise ValueError(f"cum{mode} axis extent must be non-negative")

    x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name(f"cum{mode}_in"))
    out_spec = ctx.get_value_for_var(
        out_var, name_hint=ctx.fresh_name(f"cum{mode}_out")
    )

    desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(f"cum{mode}_out")
    producer = getattr(out_spec, "producer", lambda: None)
    if callable(producer) and producer() is not None:
        desired_name = ctx.fresh_name(f"cum{mode}_out")

    # Empty-axis cum-extrema is vacuous; preserve the input tensor.
    if axis_extent_i == 0:
        result = ctx.builder.Identity(x_val, _outputs=[desired_name])
        _stamp_meta(ctx, result, ref=out_spec)
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        elif getattr(x_val, "shape", None) is not None:
            result.shape = x_val.shape
        ctx.bind_value_for_var(out_var, result)
        return

    dtype_enum = getattr(getattr(x_val, "type", None), "dtype", None)

    perm: list[int] | None = None
    inv_perm: list[int] | None = None
    x_work = x_val
    work_shape: tuple[object, ...] = x_shape
    if axis != rank - 1:
        perm = [i for i in range(rank) if i != axis] + [axis]
        inv_perm = [perm.index(i) for i in range(rank)]
        work_shape = tuple(x_shape[i] for i in perm)
        x_work = ctx.builder.Transpose(
            x_val,
            perm=perm,
            _outputs=[ctx.fresh_name(f"cum{mode}_perm")],
        )
        _stamp_meta(ctx, x_work, ref=x_val, shape=work_shape)

    shape_tensor = ctx.builder.Shape(
        x_work,
        _outputs=[ctx.fresh_name(f"cum{mode}_shape")],
    )
    _stamp_meta(ctx, shape_tensor, dtype=ir.DataType.INT64, shape=(rank,))

    axes0 = _const_i64(ctx, np.asarray([0], dtype=np.int64), f"cum{mode}_axes0")

    one_scalar = _const_i64(ctx, np.asarray(1, dtype=np.int64), f"cum{mode}_outer_one")
    if rank == 1:
        outer_size = one_scalar
    else:
        starts = _const_i64(ctx, np.asarray([0], dtype=np.int64), f"cum{mode}_starts")
        ends = _const_i64(
            ctx, np.asarray([rank - 1], dtype=np.int64), f"cum{mode}_ends"
        )
        steps = _const_i64(ctx, np.asarray([1], dtype=np.int64), f"cum{mode}_steps")
        outer_dims = ctx.builder.Slice(
            shape_tensor,
            starts,
            ends,
            axes0,
            steps,
            _outputs=[ctx.fresh_name(f"cum{mode}_outer_dims")],
        )
        _stamp_meta(ctx, outer_dims, dtype=ir.DataType.INT64, shape=(rank - 1,))
        outer_size = ctx.builder.ReduceProd(
            outer_dims,
            keepdims=0,
            _outputs=[ctx.fresh_name(f"cum{mode}_outer_size")],
        )
        _stamp_meta(ctx, outer_size, dtype=ir.DataType.INT64, shape=())

    one_vec = ctx.builder.Unsqueeze(
        one_scalar,
        axes0,
        _outputs=[ctx.fresh_name(f"cum{mode}_one_vec")],
    )
    _stamp_meta(ctx, one_vec, dtype=ir.DataType.INT64, shape=(1,))
    outer_vec = ctx.builder.Unsqueeze(
        outer_size,
        axes0,
        _outputs=[ctx.fresh_name(f"cum{mode}_outer_vec")],
    )
    _stamp_meta(ctx, outer_vec, dtype=ir.DataType.INT64, shape=(1,))
    axis_vec = _const_i64(
        ctx,
        np.asarray([axis_extent_i], dtype=np.int64),
        f"cum{mode}_axis_extent",
    )

    reshape_to_3d = ctx.builder.Concat(
        one_vec,
        outer_vec,
        axis_vec,
        axis=0,
        _outputs=[ctx.fresh_name(f"cum{mode}_reshape3d")],
    )
    _stamp_meta(ctx, reshape_to_3d, dtype=ir.DataType.INT64, shape=(3,))

    outer_size_static: int | None = 1
    for dim in work_shape[:-1]:
        if isinstance(dim, (int, np.integer)) and outer_size_static is not None:
            outer_size_static = outer_size_static * int(dim)
        else:
            outer_size_static = None
            break
    pooled_shape: tuple[object, ...] = (1, outer_size_static, axis_extent_i)

    x_3d = ctx.builder.Reshape(
        x_work,
        reshape_to_3d,
        _outputs=[ctx.fresh_name(f"cum{mode}_3d")],
    )
    _stamp_meta(ctx, x_3d, ref=x_work, shape=pooled_shape)

    pool_input = x_3d
    if mode == "min":
        pool_input = ctx.builder.Neg(
            x_3d,
            _outputs=[ctx.fresh_name("cummin_neg_in")],
        )
        _stamp_meta(ctx, pool_input, ref=x_3d, shape=pooled_shape)

    pads = (0, axis_extent_i - 1) if reverse else (axis_extent_i - 1, 0)
    pooled = ctx.builder.MaxPool(
        pool_input,
        kernel_shape=(axis_extent_i,),
        strides=(1,),
        pads=pads,
        _outputs=[ctx.fresh_name(f"cum{mode}_pooled")],
    )
    _stamp_meta(ctx, pooled, ref=pool_input, shape=pooled_shape)

    restored_3d = pooled
    if mode == "min":
        restored_3d = ctx.builder.Neg(
            pooled,
            _outputs=[ctx.fresh_name("cummin_neg_out")],
        )
        _stamp_meta(ctx, restored_3d, ref=pooled, shape=pooled_shape)

    x_work_out = ctx.builder.Reshape(
        restored_3d,
        shape_tensor,
        _outputs=[ctx.fresh_name(f"cum{mode}_work_out")],
    )
    _stamp_meta(ctx, x_work_out, ref=x_work, shape=work_shape)

    result = x_work_out
    if perm is not None and inv_perm is not None:
        result = ctx.builder.Transpose(
            x_work_out,
            perm=inv_perm,
            _outputs=[desired_name],
        )
        _stamp_meta(ctx, result, ref=x_val, shape=x_shape)
    else:
        result = ctx.builder.Identity(
            x_work_out,
            _outputs=[desired_name],
        )
        _stamp_meta(ctx, result, ref=x_work_out, shape=work_shape)

    if getattr(out_spec, "type", None) is not None:
        result.type = out_spec.type
    elif dtype_enum is not None:
        result.type = ir.TensorType(dtype_enum)
    if getattr(out_spec, "shape", None) is not None:
        result.shape = out_spec.shape
    elif getattr(x_val, "shape", None) is not None:
        result.shape = x_val.shape
    _ensure_value_metadata(ctx, result)
    ctx.bind_value_for_var(out_var, result)
