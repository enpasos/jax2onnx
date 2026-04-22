# jax2onnx/plugins/jax/lax/lu.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.ir_utils import numpy_dtype_to_ir
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _stamp_like(value: Any, ref: Any) -> None:
    if getattr(ref, "type", None) is not None:
        value.type = ref.type
    if getattr(ref, "shape", None) is not None:
        value.shape = ref.shape


def _gather_mat_elem(
    ctx: "IRContext", mat: ir.Value, i: int, j: int, name: str
) -> ir.Value:
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


def _scatter_mat_elem(
    ctx: "IRContext",
    mat: ir.Value,
    i: int,
    j: int,
    value: ir.Value,
    name: str,
) -> ir.Value:
    idx = _const_i64(
        ctx,
        np.asarray([[[i, j]]], dtype=np.int64),
        f"{name}_idx",
    )
    out = ctx.builder.ScatterND(mat, idx, value, _outputs=[ctx.fresh_name(name)])
    _stamp_like(out, mat)
    return out


def _scatter_row_static(
    ctx: "IRContext",
    mat: ir.Value,
    row_idx: int,
    row_value: ir.Value,
    name: str,
) -> ir.Value:
    idx = _const_i64(
        ctx,
        np.asarray([[row_idx]], dtype=np.int64),
        f"{name}_idx",
    )
    out = ctx.builder.ScatterND(mat, idx, row_value, _outputs=[ctx.fresh_name(name)])
    _stamp_like(out, mat)
    return out


def _scatter_row_dynamic(
    ctx: "IRContext",
    mat: ir.Value,
    row_idx: ir.Value,
    row_value: ir.Value,
    name: str,
) -> ir.Value:
    axes = _const_i64(ctx, np.asarray([0], dtype=np.int64), f"{name}_axes")
    idx = ctx.builder.Unsqueeze(row_idx, axes, _outputs=[ctx.fresh_name(f"{name}_idx")])
    idx.type = ir.TensorType(ir.DataType.INT64)
    idx.shape = ir.Shape((1, 1))
    out = ctx.builder.ScatterND(mat, idx, row_value, _outputs=[ctx.fresh_name(name)])
    _stamp_like(out, mat)
    return out


def _cast_int_vector_to_out_dtype(
    ctx: "IRContext",
    vec_i64: ir.Value,
    *,
    out_dtype: np.dtype[Any],
    out_name: str,
    out_shape: tuple[int, ...],
) -> ir.Value:
    out_dtype_enum = numpy_dtype_to_ir(out_dtype)

    if out_dtype_enum == ir.DataType.INT64:
        out = ctx.builder.Identity(vec_i64, _outputs=[out_name])
        out.type = ir.TensorType(ir.DataType.INT64)
        out.shape = ir.Shape(out_shape)
        return out

    out = ctx.builder.Cast(
        vec_i64,
        to=int(out_dtype_enum.value),
        _outputs=[out_name],
    )
    out.type = ir.TensorType(out_dtype_enum)
    out.shape = ir.Shape(out_shape)
    return out


@register_primitive(
    jaxpr_primitive=jax.lax.linalg.lu_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.linalg.lu.html",
    onnx=[
        {"component": "Abs", "doc": "https://onnx.ai/onnx/operators/onnx__Abs.html"},
        {
            "component": "ArgMax",
            "doc": "https://onnx.ai/onnx/operators/onnx__ArgMax.html",
        },
        {
            "component": "ScatterND",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterND.html",
        },
        {
            "component": "ScatterElements",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterElements.html",
        },
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {"component": "Sub", "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html"},
    ],
    since="0.12.1",
    context="primitives.lax",
    component="lu",
    testcases=[
        {
            "testcase": "lu_square_3x3",
            "callable": lambda x: jax.lax.linalg.lu(x),
            "input_values": [
                np.asarray(
                    [
                        [3.0, 1.0, 2.0],
                        [6.0, 3.0, 4.0],
                        [3.0, 1.0, 5.0],
                    ],
                    dtype=np.float32,
                )
            ],
            "expected_output_dtypes": [np.float32, np.int32, np.int32],
            "post_check_onnx_graph": EG(
                ["ArgMax", "ScatterND", "ScatterElements"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "lu_rectangular_4x3",
            "callable": lambda x: jax.lax.linalg.lu(x),
            "input_values": [
                np.asarray(
                    [
                        [2.0, 3.0, 1.0],
                        [4.0, 1.0, 2.0],
                        [1.0, 5.0, 0.5],
                        [3.0, 2.0, 4.0],
                    ],
                    dtype=np.float32,
                )
            ],
            "expected_output_dtypes": [np.float32, np.int32, np.int32],
            "post_check_onnx_graph": EG(
                ["ScatterND", "Div", "Sub"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class LuPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.linalg.lu`` with static unrolled partial pivoting."""

    def lower(self, ctx: "IRContext", eqn: Any) -> None:
        (x_var,) = eqn.invars
        lu_var, pivots_var, perm_var = eqn.outvars

        x = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("lu_in"))
        lu_spec = ctx.get_value_for_var(lu_var, name_hint=ctx.fresh_name("lu_out"))
        pivots_spec = ctx.get_value_for_var(
            pivots_var, name_hint=ctx.fresh_name("lu_pivots")
        )
        perm_spec = ctx.get_value_for_var(perm_var, name_hint=ctx.fresh_name("lu_perm"))

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        if len(x_shape) != 2:
            raise NotImplementedError("lu currently supports rank-2 inputs only")
        m_raw, n_raw = x_shape
        if not isinstance(m_raw, (int, np.integer)) or not isinstance(
            n_raw, (int, np.integer)
        ):
            raise NotImplementedError("lu requires static matrix dimensions")
        m = int(m_raw)
        n = int(n_raw)
        if m < 0 or n < 0:
            raise ValueError("lu matrix dimensions must be non-negative")

        pivots_shape = tuple(getattr(getattr(pivots_var, "aval", None), "shape", ()))
        perm_shape = tuple(getattr(getattr(perm_var, "aval", None), "shape", ()))
        if len(pivots_shape) != 1 or len(perm_shape) != 1:
            raise NotImplementedError("lu currently supports vector pivots/permutation")
        k = min(m, n)
        if int(pivots_shape[0]) != k:
            raise ValueError("lu pivots length must equal min(m, n)")
        if int(perm_shape[0]) != m:
            raise ValueError("lu permutation length must equal m")

        lu_cur = x
        pivots_i64 = _const_i64(ctx, np.zeros((k,), dtype=np.int64), "lu_pivots_init")
        pivots_i64.type = ir.TensorType(ir.DataType.INT64)
        pivots_i64.shape = ir.Shape((k,))
        perm_i64 = _const_i64(ctx, np.arange(m, dtype=np.int64), "lu_perm_init")
        perm_i64.type = ir.TensorType(ir.DataType.INT64)
        perm_i64.shape = ir.Shape((m,))

        for i in range(k):
            col_idx = _const_i64(
                ctx, np.asarray([i], dtype=np.int64), f"lu_col_idx_{i}"
            )
            col = ctx.builder.Gather(
                lu_cur,
                col_idx,
                axis=1,
                _outputs=[ctx.fresh_name(f"lu_col_{i}_2d")],
            )
            if getattr(lu_cur, "type", None) is not None:
                col.type = lu_cur.type
            col.shape = ir.Shape((m, 1))

            sq_axes = _const_i64(
                ctx, np.asarray([1], dtype=np.int64), f"lu_sq_axes_{i}"
            )
            col = ctx.builder.Squeeze(
                col,
                sq_axes,
                _outputs=[ctx.fresh_name(f"lu_col_{i}")],
            )
            if getattr(lu_cur, "type", None) is not None:
                col.type = lu_cur.type
            col.shape = ir.Shape((m,))

            trailing_col = col
            if i > 0:
                starts = _const_i64(
                    ctx,
                    np.asarray([i], dtype=np.int64),
                    f"lu_slice_starts_{i}",
                )
                ends = _const_i64(
                    ctx, np.asarray([m], dtype=np.int64), f"lu_slice_ends_{i}"
                )
                axes = _const_i64(
                    ctx, np.asarray([0], dtype=np.int64), f"lu_slice_axes_{i}"
                )
                trailing_col = ctx.builder.Slice(
                    col,
                    starts,
                    ends,
                    axes,
                    _outputs=[ctx.fresh_name(f"lu_col_tail_{i}")],
                )
                if getattr(col, "type", None) is not None:
                    trailing_col.type = col.type
                trailing_col.shape = ir.Shape((m - i,))

            abs_col = ctx.builder.Abs(
                trailing_col,
                _outputs=[ctx.fresh_name(f"lu_abs_col_{i}")],
            )
            if getattr(trailing_col, "type", None) is not None:
                abs_col.type = trailing_col.type
            abs_col.shape = trailing_col.shape

            pivot_rel = ctx.builder.ArgMax(
                abs_col,
                axis=0,
                keepdims=1,
                _outputs=[ctx.fresh_name(f"lu_pivot_rel_{i}")],
            )
            pivot_rel.type = ir.TensorType(ir.DataType.INT64)
            pivot_rel.shape = ir.Shape((1,))

            pivot_idx = pivot_rel
            if i > 0:
                offset = _const_i64(
                    ctx,
                    np.asarray([i], dtype=np.int64),
                    f"lu_pivot_off_{i}",
                )
                pivot_idx = ctx.builder.Add(
                    pivot_rel,
                    offset,
                    _outputs=[ctx.fresh_name(f"lu_pivot_idx_{i}")],
                )
                pivot_idx.type = ir.TensorType(ir.DataType.INT64)
                pivot_idx.shape = ir.Shape((1,))

            i_idx = _const_i64(ctx, np.asarray([i], dtype=np.int64), f"lu_i_idx_{i}")
            pivots_i64 = ctx.builder.ScatterElements(
                pivots_i64,
                i_idx,
                pivot_idx,
                axis=0,
                _outputs=[ctx.fresh_name(f"lu_set_pivot_{i}")],
            )
            pivots_i64.type = ir.TensorType(ir.DataType.INT64)
            pivots_i64.shape = ir.Shape((k,))

            row_i = ctx.builder.Gather(
                lu_cur,
                i_idx,
                axis=0,
                _outputs=[ctx.fresh_name(f"lu_row_i_{i}")],
            )
            _stamp_like(row_i, lu_cur)
            row_i.shape = ir.Shape((1, n))

            row_p = ctx.builder.Gather(
                lu_cur,
                pivot_idx,
                axis=0,
                _outputs=[ctx.fresh_name(f"lu_row_p_{i}")],
            )
            _stamp_like(row_p, lu_cur)
            row_p.shape = ir.Shape((1, n))

            lu_swapped = _scatter_row_static(
                ctx,
                lu_cur,
                i,
                row_p,
                f"lu_swap_row_i_{i}",
            )
            lu_cur = _scatter_row_dynamic(
                ctx,
                lu_swapped,
                pivot_idx,
                row_i,
                f"lu_swap_row_p_{i}",
            )

            perm_val_i = ctx.builder.Gather(
                perm_i64,
                i_idx,
                axis=0,
                _outputs=[ctx.fresh_name(f"lu_perm_val_i_{i}")],
            )
            perm_val_i.type = ir.TensorType(ir.DataType.INT64)
            perm_val_i.shape = ir.Shape((1,))
            perm_val_p = ctx.builder.Gather(
                perm_i64,
                pivot_idx,
                axis=0,
                _outputs=[ctx.fresh_name(f"lu_perm_val_p_{i}")],
            )
            perm_val_p.type = ir.TensorType(ir.DataType.INT64)
            perm_val_p.shape = ir.Shape((1,))

            perm_step = ctx.builder.ScatterElements(
                perm_i64,
                i_idx,
                perm_val_p,
                axis=0,
                _outputs=[ctx.fresh_name(f"lu_perm_swap_i_{i}")],
            )
            perm_step.type = ir.TensorType(ir.DataType.INT64)
            perm_step.shape = ir.Shape((m,))
            perm_i64 = ctx.builder.ScatterElements(
                perm_step,
                pivot_idx,
                perm_val_i,
                axis=0,
                _outputs=[ctx.fresh_name(f"lu_perm_swap_p_{i}")],
            )
            perm_i64.type = ir.TensorType(ir.DataType.INT64)
            perm_i64.shape = ir.Shape((m,))

            pii = _gather_mat_elem(ctx, lu_cur, i, i, f"lu_pii_{i}")
            for j in range(i + 1, m):
                pji = _gather_mat_elem(ctx, lu_cur, j, i, f"lu_pji_{i}_{j}")
                lij = ctx.builder.Div(
                    pji,
                    pii,
                    _outputs=[ctx.fresh_name(f"lu_lij_{i}_{j}")],
                )
                _stamp_like(lij, pji)
                lu_cur = _scatter_mat_elem(
                    ctx,
                    lu_cur,
                    j,
                    i,
                    lij,
                    f"lu_set_lij_{i}_{j}",
                )

                for col_j in range(i + 1, n):
                    pjc = _gather_mat_elem(
                        ctx, lu_cur, j, col_j, f"lu_pjc_{i}_{j}_{col_j}"
                    )
                    pic = _gather_mat_elem(
                        ctx, lu_cur, i, col_j, f"lu_pic_{i}_{j}_{col_j}"
                    )
                    mul = ctx.builder.Mul(
                        lij,
                        pic,
                        _outputs=[ctx.fresh_name(f"lu_mul_{i}_{j}_{col_j}")],
                    )
                    _stamp_like(mul, pic)
                    upd = ctx.builder.Sub(
                        pjc,
                        mul,
                        _outputs=[ctx.fresh_name(f"lu_upd_{i}_{j}_{col_j}")],
                    )
                    _stamp_like(upd, pjc)
                    lu_cur = _scatter_mat_elem(
                        ctx,
                        lu_cur,
                        j,
                        col_j,
                        upd,
                        f"lu_set_upd_{i}_{j}_{col_j}",
                    )

        lu_name = getattr(lu_spec, "name", None) or ctx.fresh_name("lu")
        lu_out = lu_cur
        if getattr(lu_out, "name", None) != lu_name:
            lu_out = ctx.builder.Identity(lu_cur, _outputs=[lu_name])
        _stamp_like(lu_out, lu_spec if getattr(lu_spec, "type", None) else lu_cur)
        if getattr(lu_spec, "shape", None) is not None:
            lu_out.shape = lu_spec.shape

        pivots_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(pivots_var, "aval", None), "dtype", np.int32)
        )
        pivots_name = getattr(pivots_spec, "name", None) or ctx.fresh_name("lu_pivots")
        pivots_out = _cast_int_vector_to_out_dtype(
            ctx,
            pivots_i64,
            out_dtype=pivots_dtype,
            out_name=pivots_name,
            out_shape=(k,),
        )
        _stamp_like(
            pivots_out,
            pivots_spec if getattr(pivots_spec, "type", None) else pivots_out,
        )
        if getattr(pivots_spec, "shape", None) is not None:
            pivots_out.shape = pivots_spec.shape

        perm_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(perm_var, "aval", None), "dtype", np.int32)
        )
        perm_name = getattr(perm_spec, "name", None) or ctx.fresh_name("lu_perm")
        perm_out = _cast_int_vector_to_out_dtype(
            ctx,
            perm_i64,
            out_dtype=perm_dtype,
            out_name=perm_name,
            out_shape=(m,),
        )
        _stamp_like(
            perm_out, perm_spec if getattr(perm_spec, "type", None) else perm_out
        )
        if getattr(perm_spec, "shape", None) is not None:
            perm_out.shape = perm_spec.shape

        ctx.bind_value_for_var(lu_var, lu_out)
        ctx.bind_value_for_var(pivots_var, pivots_out)
        ctx.bind_value_for_var(perm_var, perm_out)
