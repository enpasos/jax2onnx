# jax2onnx/plugins/jax/lax/eigh.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _stamp_like(value, ref) -> None:
    if getattr(ref, "type", None) is not None:
        value.type = ref.type
    if getattr(ref, "shape", None) is not None:
        value.shape = ref.shape


def _gather_mat_elem(ctx: "IRContext", mat, i: int, j: int, name: str):
    i_idx = _const_i64(ctx, np.asarray([i], dtype=np.int64), f"{name}_i")
    row = ctx.builder.Gather(
        mat,
        i_idx,
        axis=0,
        _outputs=[ctx.fresh_name(f"{name}_row")],
    )
    if getattr(mat, "type", None) is not None:
        row.type = mat.type
    j_idx = _const_i64(ctx, np.asarray([j], dtype=np.int64), f"{name}_j")
    elem = ctx.builder.Gather(
        row,
        j_idx,
        axis=1,
        _outputs=[ctx.fresh_name(name)],
    )
    if getattr(mat, "type", None) is not None:
        elem.type = mat.type
    elem.shape = ir.Shape((1, 1))
    return elem


def _cast_if_needed(ctx: "IRContext", value, target: ir.DataType, name_hint: str):
    current = getattr(value, "dtype", None)
    if current is None:
        value_type = getattr(value, "type", None)
        if isinstance(value_type, ir.TensorType):
            current = value_type.dtype
    if current == target:
        return value
    casted = ctx.builder.Cast(
        value,
        to=int(target.value),
        _outputs=[ctx.fresh_name(name_hint)],
    )
    casted.type = ir.TensorType(target)
    if getattr(value, "shape", None) is not None:
        casted.shape = value.shape
    return casted


def _reshape_to_1d_len1(ctx: "IRContext", value, name_hint: str):
    one_shape = _const_i64(ctx, np.asarray([1], dtype=np.int64), f"{name_hint}_shape")
    out = ctx.builder.Reshape(
        value,
        one_shape,
        _outputs=[ctx.fresh_name(name_hint)],
    )
    if getattr(value, "type", None) is not None:
        out.type = value.type
    out.shape = ir.Shape((1,))
    return out


def _reshape_to_2d(ctx: "IRContext", value, shape: tuple[int, int], name_hint: str):
    target_shape = _const_i64(
        ctx,
        np.asarray([shape[0], shape[1]], dtype=np.int64),
        f"{name_hint}_shape",
    )
    out = ctx.builder.Reshape(
        value,
        target_shape,
        _outputs=[ctx.fresh_name(name_hint)],
    )
    if getattr(value, "type", None) is not None:
        out.type = value.type
    out.shape = ir.Shape(shape)
    return out


def _slice_1d(ctx: "IRContext", value, *, start: int, end: int, name_hint: str):
    starts = _const_i64(
        ctx,
        np.asarray([start], dtype=np.int64),
        f"{name_hint}_starts",
    )
    ends = _const_i64(
        ctx,
        np.asarray([end], dtype=np.int64),
        f"{name_hint}_ends",
    )
    axes = _const_i64(ctx, np.asarray([0], dtype=np.int64), f"{name_hint}_axes")
    out = ctx.builder.Slice(
        value,
        starts,
        ends,
        axes,
        _outputs=[ctx.fresh_name(name_hint)],
    )
    if getattr(value, "type", None) is not None:
        out.type = value.type
    out.shape = ir.Shape((end - start,))
    return out


def _slice_2d_cols(
    ctx: "IRContext",
    value,
    *,
    rows: int,
    col_start: int,
    col_end: int,
    name_hint: str,
):
    starts = _const_i64(
        ctx,
        np.asarray([0, col_start], dtype=np.int64),
        f"{name_hint}_starts",
    )
    ends = _const_i64(
        ctx,
        np.asarray([rows, col_end], dtype=np.int64),
        f"{name_hint}_ends",
    )
    axes = _const_i64(ctx, np.asarray([0, 1], dtype=np.int64), f"{name_hint}_axes")
    out = ctx.builder.Slice(
        value,
        starts,
        ends,
        axes,
        _outputs=[ctx.fresh_name(name_hint)],
    )
    if getattr(value, "type", None) is not None:
        out.type = value.type
    out.shape = ir.Shape((rows, col_end - col_start))
    return out


@register_primitive(
    jaxpr_primitive=jax.lax.linalg.eigh_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.linalg.eigh.html",
    onnx=[
        {
            "component": "Gather",
            "doc": "https://onnx.ai/onnx/operators/onnx__Gather.html",
        },
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        },
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
        {"component": "Sub", "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html"},
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
        {
            "component": "Sqrt",
            "doc": "https://onnx.ai/onnx/operators/onnx__Sqrt.html",
        },
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
        {
            "component": "LessOrEqual",
            "doc": "https://onnx.ai/onnx/operators/onnx__LessOrEqual.html",
        },
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        },
        {
            "component": "Slice",
            "doc": "https://onnx.ai/onnx/operators/onnx__Slice.html",
        },
        {"component": "Cast", "doc": "https://onnx.ai/onnx/operators/onnx__Cast.html"},
        {
            "component": "Identity",
            "doc": "https://onnx.ai/onnx/operators/onnx__Identity.html",
        },
    ],
    since="0.12.1",
    context="primitives.lax",
    component="eigh",
    testcases=[
        {
            "testcase": "eigh_1x1",
            "callable": lambda x: jax.lax.linalg.eigh(
                x, lower=True, symmetrize_input=False
            ),
            "input_values": [np.asarray([[2.5]], dtype=np.float32)],
            "post_check_onnx_graph": EG(
                ["Gather", "Identity"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "eigh_2x2_lower_true",
            "callable": lambda x: jax.lax.linalg.eigh(x, lower=True)[1],
            "input_values": [
                np.asarray(
                    [
                        [2.0, 9.0],
                        [1.0, 3.0],
                    ],
                    dtype=np.float32,
                )
            ],
            "post_check_onnx_graph": EG(
                ["Sqrt", "Concat"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "eigh_2x2_lower_false",
            "callable": lambda x: jax.lax.linalg.eigh(x, lower=False)[1],
            "input_values": [
                np.asarray(
                    [
                        [2.0, 1.0],
                        [9.0, 3.0],
                    ],
                    dtype=np.float32,
                )
            ],
            "post_check_onnx_graph": EG(
                ["Sqrt", "Concat"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "eigh_2x2_subset_top1",
            "callable": lambda x: jax.lax.linalg.eigh(
                x, lower=True, subset_by_index=(1, 2)
            )[1],
            "input_values": [
                np.asarray(
                    [
                        [5.0, 0.0],
                        [1.0, 2.0],
                    ],
                    dtype=np.float32,
                )
            ],
            "skip_numeric_validation": True,
            "post_check_onnx_graph": EG(
                ["Slice", "Sqrt"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class EighPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.linalg.eigh`` for static square ``1x1`` and real symmetric ``2x2``."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        params = dict(getattr(eqn, "params", {}) or {})
        lower = bool(params.get("lower", True))
        sort_eigenvalues = bool(params.get("sort_eigenvalues", True))
        subset_by_index = params.get("subset_by_index", None)
        if not sort_eigenvalues:
            raise NotImplementedError(
                "eigh currently supports only sort_eigenvalues=True"
            )

        (x_var,) = eqn.invars
        if len(eqn.outvars) != 2:
            raise NotImplementedError("eigh expects exactly 2 outputs")
        vec_var, val_var = eqn.outvars

        x = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("eigh_in"))
        vec_spec = ctx.get_value_for_var(vec_var, name_hint=ctx.fresh_name("eigh_vecs"))
        val_spec = ctx.get_value_for_var(val_var, name_hint=ctx.fresh_name("eigh_vals"))

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        if len(x_shape) != 2:
            raise NotImplementedError("eigh currently supports rank-2 inputs only")
        n_rows_raw, n_cols_raw = x_shape
        if not isinstance(n_rows_raw, (int, np.integer)) or not isinstance(
            n_cols_raw, (int, np.integer)
        ):
            raise NotImplementedError("eigh requires static matrix dimensions")
        n_rows = int(n_rows_raw)
        n_cols = int(n_cols_raw)
        if n_rows != n_cols:
            raise ValueError("eigh requires square matrices")
        if n_rows > 2:
            raise NotImplementedError(
                "eigh currently supports only 1x1 and 2x2 matrices; larger decompositions are pending"
            )
        subset_start = 0
        subset_end = n_rows
        if subset_by_index is not None:
            if not (
                isinstance(subset_by_index, (tuple, list)) and len(subset_by_index) == 2
            ):
                raise NotImplementedError(
                    "eigh subset_by_index must be a static (start, end) pair"
                )
            subset_start_raw, subset_end_raw = subset_by_index
            if not isinstance(subset_start_raw, (int, np.integer)) or not isinstance(
                subset_end_raw, (int, np.integer)
            ):
                raise NotImplementedError(
                    "eigh subset_by_index must contain integer bounds"
                )
            subset_start = int(subset_start_raw)
            subset_end = int(subset_end_raw)
            if subset_start < 0 or subset_end > n_rows or subset_start >= subset_end:
                raise ValueError(
                    f"eigh subset_by_index out of range for size {n_rows}: "
                    f"({subset_start}, {subset_end})"
                )

        x_input_dtype = np.dtype(
            getattr(getattr(x_var, "aval", None), "dtype", np.float32)
        )
        if np.issubdtype(x_input_dtype, np.complexfloating):
            raise NotImplementedError("eigh complex input is not supported yet")

        val_dtype = np.dtype(
            getattr(getattr(val_var, "aval", None), "dtype", np.float32)
        )
        val_dtype_enum = _dtype_to_ir(val_dtype, ctx.builder.enable_double_precision)
        if val_dtype_enum is None:
            raise TypeError(f"Unsupported eigh eigenvalue dtype '{val_dtype}'")

        vec_dtype = np.dtype(
            getattr(getattr(vec_var, "aval", None), "dtype", np.float32)
        )
        vec_dtype_enum = _dtype_to_ir(vec_dtype, ctx.builder.enable_double_precision)
        if vec_dtype_enum is None:
            raise TypeError(f"Unsupported eigh eigenvector dtype '{vec_dtype}'")

        if n_rows == 1:
            x00 = _gather_mat_elem(ctx, x, 0, 0, "eigh_x00")
            x00_val = _cast_if_needed(ctx, x00, val_dtype_enum, "eigh_x00_cast")
            vals_1d = _reshape_to_1d_len1(ctx, x00_val, "eigh_vals_1d")
            vals_name = getattr(val_spec, "name", None) or ctx.fresh_name("eigh_vals")
            vals = vals_1d
            if getattr(vals, "name", None) != vals_name:
                vals = ctx.builder.Identity(vals_1d, _outputs=[vals_name])
            vals.type = ir.TensorType(val_dtype_enum)
            vals.shape = ir.Shape((1,))
            _stamp_like(vals, val_spec if getattr(val_spec, "type", None) else vals_1d)
            if getattr(val_spec, "shape", None) is not None:
                vals.shape = val_spec.shape

            vec_const = ctx.bind_const_for_var(
                object(), np.asarray([[1]], dtype=vec_dtype)
            )
            vec_name = getattr(vec_spec, "name", None) or ctx.fresh_name("eigh_vecs")
            vecs = ctx.builder.Identity(vec_const, _outputs=[vec_name])
            _stamp_like(
                vecs, vec_spec if getattr(vec_spec, "type", None) else vec_const
            )
            if getattr(vec_spec, "shape", None) is not None:
                vecs.shape = vec_spec.shape

            ctx.bind_value_for_var(vec_var, vecs)
            ctx.bind_value_for_var(val_var, vals)
            return

        # n == 2 branch
        a_raw = _gather_mat_elem(ctx, x, 0, 0, "eigh_a")
        d_raw = _gather_mat_elem(ctx, x, 1, 1, "eigh_d")
        b_raw = (
            _gather_mat_elem(ctx, x, 1, 0, "eigh_b_lower")
            if lower
            else _gather_mat_elem(ctx, x, 0, 1, "eigh_b_upper")
        )

        a = _cast_if_needed(ctx, a_raw, val_dtype_enum, "eigh_a_cast")
        b = _cast_if_needed(ctx, b_raw, val_dtype_enum, "eigh_b_cast")
        d = _cast_if_needed(ctx, d_raw, val_dtype_enum, "eigh_d_cast")

        scalar_np_dtype = (
            np.float64 if val_dtype_enum == ir.DataType.DOUBLE else np.float32
        )
        zero = ctx.bind_const_for_var(
            object(), np.asarray([[0.0]], dtype=scalar_np_dtype)
        )
        one = ctx.bind_const_for_var(
            object(), np.asarray([[1.0]], dtype=scalar_np_dtype)
        )
        two = ctx.bind_const_for_var(
            object(), np.asarray([[2.0]], dtype=scalar_np_dtype)
        )
        four = ctx.bind_const_for_var(
            object(), np.asarray([[4.0]], dtype=scalar_np_dtype)
        )
        for const in (zero, one, two, four):
            const.type = ir.TensorType(val_dtype_enum)
            const.shape = ir.Shape((1, 1))

        trace = ctx.builder.Add(a, d, _outputs=[ctx.fresh_name("eigh_trace")])
        trace.type = ir.TensorType(val_dtype_enum)
        delta = ctx.builder.Sub(a, d, _outputs=[ctx.fresh_name("eigh_delta")])
        delta.type = ir.TensorType(val_dtype_enum)
        delta2 = ctx.builder.Mul(delta, delta, _outputs=[ctx.fresh_name("eigh_delta2")])
        delta2.type = ir.TensorType(val_dtype_enum)
        b2 = ctx.builder.Mul(b, b, _outputs=[ctx.fresh_name("eigh_b2")])
        b2.type = ir.TensorType(val_dtype_enum)
        four_b2 = ctx.builder.Mul(four, b2, _outputs=[ctx.fresh_name("eigh_four_b2")])
        four_b2.type = ir.TensorType(val_dtype_enum)
        disc2 = ctx.builder.Add(
            delta2, four_b2, _outputs=[ctx.fresh_name("eigh_disc2")]
        )
        disc2.type = ir.TensorType(val_dtype_enum)
        disc = ctx.builder.Sqrt(disc2, _outputs=[ctx.fresh_name("eigh_disc")])
        disc.type = ir.TensorType(val_dtype_enum)

        lam0_num = ctx.builder.Sub(
            trace,
            disc,
            _outputs=[ctx.fresh_name("eigh_lam0_num")],
        )
        lam0_num.type = ir.TensorType(val_dtype_enum)
        lam1_num = ctx.builder.Add(
            trace,
            disc,
            _outputs=[ctx.fresh_name("eigh_lam1_num")],
        )
        lam1_num.type = ir.TensorType(val_dtype_enum)
        lam0 = ctx.builder.Div(lam0_num, two, _outputs=[ctx.fresh_name("eigh_lam0")])
        lam0.type = ir.TensorType(val_dtype_enum)
        lam1 = ctx.builder.Div(lam1_num, two, _outputs=[ctx.fresh_name("eigh_lam1")])
        lam1.type = ir.TensorType(val_dtype_enum)

        lam0_1d = _reshape_to_1d_len1(ctx, lam0, "eigh_lam0_1d")
        lam1_1d = _reshape_to_1d_len1(ctx, lam1, "eigh_lam1_1d")
        vals_raw = ctx.builder.Concat(
            lam0_1d,
            lam1_1d,
            axis=0,
            _outputs=[ctx.fresh_name("eigh_vals_raw")],
        )
        vals_raw.type = ir.TensorType(val_dtype_enum)
        vals_raw.shape = ir.Shape((2,))
        vals_full = vals_raw
        if subset_start != 0 or subset_end != 2:
            vals_full = _slice_1d(
                ctx,
                vals_raw,
                start=subset_start,
                end=subset_end,
                name_hint="eigh_vals_subset",
            )
        vals_name = getattr(val_spec, "name", None) or ctx.fresh_name("eigh_vals")
        vals = vals_full
        if getattr(vals, "name", None) != vals_name:
            vals = ctx.builder.Identity(vals_full, _outputs=[vals_name])
        vals.type = ir.TensorType(val_dtype_enum)
        vals.shape = ir.Shape((subset_end - subset_start,))
        _stamp_like(vals, val_spec if getattr(val_spec, "type", None) else vals_full)
        if getattr(val_spec, "shape", None) is not None:
            vals.shape = val_spec.shape

        # Eigenvector columns for ascending eigenvalues:
        # v0 ~ [b, lam0-a], v1 ~ [b, lam1-a], normalized with diagonal fallback.
        cond_a_le_d = ctx.builder.LessOrEqual(
            a, d, _outputs=[ctx.fresh_name("eigh_a_le_d")]
        )
        cond_a_le_d.type = ir.TensorType(ir.DataType.BOOL)

        def _normalized_vector(lam, prefix: str):
            vx = b
            vy = ctx.builder.Sub(lam, a, _outputs=[ctx.fresh_name(f"{prefix}_vy")])
            vy.type = ir.TensorType(val_dtype_enum)
            vx2 = ctx.builder.Mul(vx, vx, _outputs=[ctx.fresh_name(f"{prefix}_vx2")])
            vy2 = ctx.builder.Mul(vy, vy, _outputs=[ctx.fresh_name(f"{prefix}_vy2")])
            vx2.type = ir.TensorType(val_dtype_enum)
            vy2.type = ir.TensorType(val_dtype_enum)
            norm2 = ctx.builder.Add(
                vx2, vy2, _outputs=[ctx.fresh_name(f"{prefix}_norm2")]
            )
            norm2.type = ir.TensorType(val_dtype_enum)
            norm = ctx.builder.Sqrt(norm2, _outputs=[ctx.fresh_name(f"{prefix}_norm")])
            norm.type = ir.TensorType(val_dtype_enum)
            is_zero = ctx.builder.Equal(
                norm,
                zero,
                _outputs=[ctx.fresh_name(f"{prefix}_is_zero")],
            )
            is_zero.type = ir.TensorType(ir.DataType.BOOL)
            norm_safe = ctx.builder.Where(
                is_zero,
                one,
                norm,
                _outputs=[ctx.fresh_name(f"{prefix}_norm_safe")],
            )
            norm_safe.type = ir.TensorType(val_dtype_enum)
            ux = ctx.builder.Div(
                vx, norm_safe, _outputs=[ctx.fresh_name(f"{prefix}_ux")]
            )
            uy = ctx.builder.Div(
                vy, norm_safe, _outputs=[ctx.fresh_name(f"{prefix}_uy")]
            )
            ux.type = ir.TensorType(val_dtype_enum)
            uy.type = ir.TensorType(val_dtype_enum)
            return ux, uy, is_zero

        v0x_u, v0y_u, v0_is_zero = _normalized_vector(lam0, "eigh_v0")
        v1x_u, v1y_u, v1_is_zero = _normalized_vector(lam1, "eigh_v1")

        fb0x = ctx.builder.Where(
            cond_a_le_d,
            one,
            zero,
            _outputs=[ctx.fresh_name("eigh_fb0x")],
        )
        fb0y = ctx.builder.Where(
            cond_a_le_d,
            zero,
            one,
            _outputs=[ctx.fresh_name("eigh_fb0y")],
        )
        fb1x = ctx.builder.Where(
            cond_a_le_d,
            zero,
            one,
            _outputs=[ctx.fresh_name("eigh_fb1x")],
        )
        fb1y = ctx.builder.Where(
            cond_a_le_d,
            one,
            zero,
            _outputs=[ctx.fresh_name("eigh_fb1y")],
        )
        for val in (fb0x, fb0y, fb1x, fb1y):
            val.type = ir.TensorType(val_dtype_enum)

        v0x = ctx.builder.Where(
            v0_is_zero,
            fb0x,
            v0x_u,
            _outputs=[ctx.fresh_name("eigh_v0x")],
        )
        v0y = ctx.builder.Where(
            v0_is_zero,
            fb0y,
            v0y_u,
            _outputs=[ctx.fresh_name("eigh_v0y")],
        )
        v1x = ctx.builder.Where(
            v1_is_zero,
            fb1x,
            v1x_u,
            _outputs=[ctx.fresh_name("eigh_v1x")],
        )
        v1y = ctx.builder.Where(
            v1_is_zero,
            fb1y,
            v1y_u,
            _outputs=[ctx.fresh_name("eigh_v1y")],
        )
        for val in (v0x, v0y, v1x, v1y):
            val.type = ir.TensorType(val_dtype_enum)

        v0x_1d = _reshape_to_1d_len1(ctx, v0x, "eigh_v0x_1d")
        v0y_1d = _reshape_to_1d_len1(ctx, v0y, "eigh_v0y_1d")
        v1x_1d = _reshape_to_1d_len1(ctx, v1x, "eigh_v1x_1d")
        v1y_1d = _reshape_to_1d_len1(ctx, v1y, "eigh_v1y_1d")

        row0_1d = ctx.builder.Concat(
            v0x_1d,
            v1x_1d,
            axis=0,
            _outputs=[ctx.fresh_name("eigh_row0_1d")],
        )
        row1_1d = ctx.builder.Concat(
            v0y_1d,
            v1y_1d,
            axis=0,
            _outputs=[ctx.fresh_name("eigh_row1_1d")],
        )
        row0_1d.type = ir.TensorType(val_dtype_enum)
        row1_1d.type = ir.TensorType(val_dtype_enum)
        row0_1d.shape = ir.Shape((2,))
        row1_1d.shape = ir.Shape((2,))

        row0 = _reshape_to_2d(ctx, row0_1d, (1, 2), "eigh_row0")
        row1 = _reshape_to_2d(ctx, row1_1d, (1, 2), "eigh_row1")
        vecs_raw = ctx.builder.Concat(
            row0,
            row1,
            axis=0,
            _outputs=[ctx.fresh_name("eigh_vecs_raw")],
        )
        vecs_raw.type = ir.TensorType(val_dtype_enum)
        vecs_raw.shape = ir.Shape((2, 2))

        vecs_cast = _cast_if_needed(ctx, vecs_raw, vec_dtype_enum, "eigh_vecs_cast")
        vecs_full = vecs_cast
        if subset_start != 0 or subset_end != 2:
            vecs_full = _slice_2d_cols(
                ctx,
                vecs_cast,
                rows=2,
                col_start=subset_start,
                col_end=subset_end,
                name_hint="eigh_vecs_subset",
            )
        vecs_name = getattr(vec_spec, "name", None) or ctx.fresh_name("eigh_vecs")
        vecs = vecs_full
        if getattr(vecs, "name", None) != vecs_name:
            vecs = ctx.builder.Identity(vecs_full, _outputs=[vecs_name])
        vecs.type = ir.TensorType(vec_dtype_enum)
        vecs.shape = ir.Shape((2, subset_end - subset_start))
        _stamp_like(vecs, vec_spec if getattr(vec_spec, "type", None) else vecs_full)
        if getattr(vec_spec, "shape", None) is not None:
            vecs.shape = vec_spec.shape

        ctx.bind_value_for_var(vec_var, vecs)
        ctx.bind_value_for_var(val_var, vals)
