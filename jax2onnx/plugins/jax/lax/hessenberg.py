# jax2onnx/plugins/jax/lax/hessenberg.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir

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


def _slice_matrix(
    ctx: "IRContext",
    mat,
    *,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    out_shape: tuple[int, int],
    name: str,
):
    starts = _const_i64(
        ctx,
        np.asarray([row_start, col_start], dtype=np.int64),
        f"{name}_starts",
    )
    ends = _const_i64(
        ctx,
        np.asarray([row_end, col_end], dtype=np.int64),
        f"{name}_ends",
    )
    axes = _const_i64(ctx, np.asarray([0, 1], dtype=np.int64), f"{name}_axes")
    out = ctx.builder.Slice(
        mat,
        starts,
        ends,
        axes,
        _outputs=[ctx.fresh_name(name)],
    )
    if getattr(mat, "type", None) is not None:
        out.type = mat.type
    out.shape = ir.Shape(out_shape)
    return out


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


def _scatter_mat_elem(ctx: "IRContext", mat, i: int, j: int, value, name: str):
    idx = _const_i64(
        ctx,
        np.asarray([[[i, j]]], dtype=np.int64),
        f"{name}_idx",
    )
    out = ctx.builder.ScatterND(mat, idx, value, _outputs=[ctx.fresh_name(name)])
    _stamp_like(out, mat)
    return out


def _scatter_block(
    ctx: "IRContext",
    base,
    block,
    *,
    row_start: int,
    col_start: int,
    rows: int,
    cols: int,
    name: str,
):
    idx_data: list[list[int]] = []
    for r in range(rows):
        for c in range(cols):
            idx_data.append([row_start + r, col_start + c])
    idx = _const_i64(ctx, np.asarray(idx_data, dtype=np.int64), f"{name}_idx")

    flat_shape = _const_i64(
        ctx,
        np.asarray([rows * cols], dtype=np.int64),
        f"{name}_flat_shape",
    )
    flat = ctx.builder.Reshape(
        block, flat_shape, _outputs=[ctx.fresh_name(f"{name}_flat")]
    )
    if getattr(block, "type", None) is not None:
        flat.type = block.type
    flat.shape = ir.Shape((rows * cols,))

    out = ctx.builder.ScatterND(base, idx, flat, _outputs=[ctx.fresh_name(name)])
    _stamp_like(out, base)
    return out


def _gather_vec_elem(ctx: "IRContext", vec, i: int, name: str):
    idx = _const_i64(ctx, np.asarray([i], dtype=np.int64), f"{name}_idx")
    out = ctx.builder.Gather(
        vec,
        idx,
        axis=0,
        _outputs=[ctx.fresh_name(name)],
    )
    if getattr(vec, "type", None) is not None:
        out.type = vec.type
    out.shape = ir.Shape((1,))
    return out


def _scatter_vec_elem(ctx: "IRContext", vec, i: int, value, name: str):
    idx = _const_i64(ctx, np.asarray([[i]], dtype=np.int64), f"{name}_idx")
    out = ctx.builder.ScatterND(vec, idx, value, _outputs=[ctx.fresh_name(name)])
    _stamp_like(out, vec)
    return out


@register_primitive(
    jaxpr_primitive=jax.lax.linalg.hessenberg_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.linalg.hessenberg.html",
    onnx=[
        {
            "component": "MatMul",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
        },
        {"component": "Sqrt", "doc": "https://onnx.ai/onnx/operators/onnx__Sqrt.html"},
        {
            "component": "ScatterND",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterND.html",
        },
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
    ],
    since="0.12.1",
    context="primitives.lax",
    component="hessenberg",
    testcases=[
        {
            "testcase": "hessenberg_square_4x4",
            "callable": lambda x: jax.lax.linalg.hessenberg(x),
            "input_values": [
                np.asarray(
                    [
                        [1.0, 2.0, 3.0, -1.0],
                        [2.5, -0.5, 1.2, 0.3],
                        [0.7, 1.1, 0.8, 2.2],
                        [1.4, -1.2, 0.5, 0.6],
                    ],
                    dtype=np.float32,
                )
            ],
            "post_check_onnx_graph": EG(
                ["MatMul", "ScatterND"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "hessenberg_diagonal",
            "callable": lambda x: jax.lax.linalg.hessenberg(x),
            "input_values": [
                np.asarray(
                    np.diag([2.0, -1.0, 3.0, 4.0]),
                    dtype=np.float32,
                )
            ],
            "post_check_onnx_graph": EG(
                ["Sqrt"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class HessenbergPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.linalg.hessenberg`` for static rank-2 square inputs."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        (x_var,) = eqn.invars
        if len(eqn.outvars) != 2:
            raise NotImplementedError("hessenberg expects exactly 2 outputs")
        a_var, taus_var = eqn.outvars

        x = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("hess_in"))
        a_spec = ctx.get_value_for_var(a_var, name_hint=ctx.fresh_name("hess_a"))
        taus_spec = ctx.get_value_for_var(
            taus_var, name_hint=ctx.fresh_name("hess_taus")
        )

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        if len(x_shape) != 2:
            raise NotImplementedError(
                "hessenberg currently supports rank-2 inputs only"
            )
        n_rows_raw, n_cols_raw = x_shape
        if not isinstance(n_rows_raw, (int, np.integer)) or not isinstance(
            n_cols_raw, (int, np.integer)
        ):
            raise NotImplementedError("hessenberg requires static matrix shape")
        n_rows = int(n_rows_raw)
        n_cols = int(n_cols_raw)
        if n_rows != n_cols:
            raise ValueError("hessenberg requires square matrices")
        n = n_rows

        taus_shape = tuple(getattr(getattr(taus_var, "aval", None), "shape", ()))
        if len(taus_shape) != 1:
            raise NotImplementedError("hessenberg taus output must be rank-1")
        if int(taus_shape[0]) != max(n - 1, 0):
            raise ValueError("hessenberg taus length mismatch")

        np_dtype = np.dtype(getattr(getattr(x_var, "aval", None), "dtype", np.float32))
        a_cur = x
        taus_cur = ctx.bind_const_for_var(
            object(),
            np.zeros((max(n - 1, 0),), dtype=np_dtype),
        )
        if getattr(x, "type", None) is not None:
            taus_cur.type = x.type
        taus_cur.shape = ir.Shape((max(n - 1, 0),))

        one = ctx.bind_const_for_var(object(), np.asarray([[1.0]], dtype=np_dtype))
        neg_one = ctx.bind_const_for_var(object(), np.asarray([[-1.0]], dtype=np_dtype))
        zero = ctx.bind_const_for_var(object(), np.asarray([[0.0]], dtype=np_dtype))
        for const_val in (one, neg_one, zero):
            if getattr(x, "type", None) is not None:
                const_val.type = x.type
            const_val.shape = ir.Shape((1, 1))

        for i in range(max(n - 1, 0)):
            p = n - i - 1
            if p <= 0:
                continue
            if p == 1:
                tau_zero = ctx.bind_const_for_var(
                    object(), np.asarray([0.0], dtype=np_dtype)
                )
                if getattr(x, "type", None) is not None:
                    tau_zero.type = x.type
                tau_zero.shape = ir.Shape((1,))
                taus_cur = _scatter_vec_elem(
                    ctx,
                    taus_cur,
                    i,
                    tau_zero,
                    f"hess_tau_zero_{i}",
                )
                continue

            x_col = _slice_matrix(
                ctx,
                a_cur,
                row_start=i + 1,
                row_end=n,
                col_start=i,
                col_end=i + 1,
                out_shape=(p, 1),
                name=f"hess_x_col_{i}",
            )
            x_tail = _slice_matrix(
                ctx,
                x_col,
                row_start=1,
                row_end=p,
                col_start=0,
                col_end=1,
                out_shape=(p - 1, 1),
                name=f"hess_x_tail_{i}",
            )

            alpha = _gather_mat_elem(ctx, a_cur, i + 1, i, f"hess_alpha_{i}")
            alpha_sq = ctx.builder.Mul(
                alpha,
                alpha,
                _outputs=[ctx.fresh_name(f"hess_alpha_sq_{i}")],
            )
            _stamp_like(alpha_sq, alpha)

            x_tail_t = ctx.builder.Transpose(
                x_tail,
                perm=[1, 0],
                _outputs=[ctx.fresh_name(f"hess_x_tail_t_{i}")],
            )
            if getattr(x_tail, "type", None) is not None:
                x_tail_t.type = x_tail.type
            x_tail_t.shape = ir.Shape((1, p - 1))
            tail_norm_sq = ctx.builder.MatMul(
                x_tail_t,
                x_tail,
                _outputs=[ctx.fresh_name(f"hess_tail_norm_sq_{i}")],
            )
            _stamp_like(tail_norm_sq, alpha)
            tail_is_zero = ctx.builder.Equal(
                tail_norm_sq,
                zero,
                _outputs=[ctx.fresh_name(f"hess_tail_is_zero_{i}")],
            )
            tail_is_zero.type = ir.TensorType(ir.DataType.BOOL)
            tail_is_zero.shape = ir.Shape((1, 1))

            x_norm_sq = ctx.builder.Add(
                alpha_sq,
                tail_norm_sq,
                _outputs=[ctx.fresh_name(f"hess_x_norm_sq_{i}")],
            )
            _stamp_like(x_norm_sq, alpha)
            x_norm = ctx.builder.Sqrt(
                x_norm_sq,
                _outputs=[ctx.fresh_name(f"hess_x_norm_{i}")],
            )
            _stamp_like(x_norm, alpha)

            less_zero = ctx.builder.Less(
                alpha,
                zero,
                _outputs=[ctx.fresh_name(f"hess_less_zero_{i}")],
            )
            less_zero.type = ir.TensorType(ir.DataType.BOOL)
            less_zero.shape = ir.Shape((1, 1))
            sign = ctx.builder.Where(
                less_zero,
                one,
                neg_one,
                _outputs=[ctx.fresh_name(f"hess_sign_{i}")],
            )
            _stamp_like(sign, alpha)
            beta = ctx.builder.Mul(
                sign,
                x_norm,
                _outputs=[ctx.fresh_name(f"hess_beta_{i}")],
            )
            _stamp_like(beta, alpha)
            beta_eff = ctx.builder.Where(
                tail_is_zero,
                alpha,
                beta,
                _outputs=[ctx.fresh_name(f"hess_beta_eff_{i}")],
            )
            _stamp_like(beta_eff, alpha)

            denom = ctx.builder.Sub(
                alpha,
                beta_eff,
                _outputs=[ctx.fresh_name(f"hess_denom_{i}")],
            )
            _stamp_like(denom, alpha)
            denom_eff = ctx.builder.Where(
                tail_is_zero,
                one,
                denom,
                _outputs=[ctx.fresh_name(f"hess_denom_eff_{i}")],
            )
            _stamp_like(denom_eff, alpha)

            zero_tail = ctx.bind_const_for_var(
                object(), np.zeros((p - 1, 1), dtype=np_dtype)
            )
            if getattr(x, "type", None) is not None:
                zero_tail.type = x.type
            zero_tail.shape = ir.Shape((p - 1, 1))

            v_tail_raw = ctx.builder.Div(
                x_tail,
                denom_eff,
                _outputs=[ctx.fresh_name(f"hess_v_tail_raw_{i}")],
            )
            if getattr(x_tail, "type", None) is not None:
                v_tail_raw.type = x_tail.type
            v_tail_raw.shape = ir.Shape((p - 1, 1))
            v_tail = ctx.builder.Where(
                tail_is_zero,
                zero_tail,
                v_tail_raw,
                _outputs=[ctx.fresh_name(f"hess_v_tail_{i}")],
            )
            if getattr(x_tail, "type", None) is not None:
                v_tail.type = x_tail.type
            v_tail.shape = ir.Shape((p - 1, 1))

            tau_num = ctx.builder.Sub(
                beta_eff,
                alpha,
                _outputs=[ctx.fresh_name(f"hess_tau_num_{i}")],
            )
            _stamp_like(tau_num, alpha)
            beta_div = ctx.builder.Where(
                tail_is_zero,
                one,
                beta_eff,
                _outputs=[ctx.fresh_name(f"hess_beta_div_{i}")],
            )
            _stamp_like(beta_div, alpha)
            tau_raw = ctx.builder.Div(
                tau_num,
                beta_div,
                _outputs=[ctx.fresh_name(f"hess_tau_raw_{i}")],
            )
            _stamp_like(tau_raw, alpha)
            tau = ctx.builder.Where(
                tail_is_zero,
                zero,
                tau_raw,
                _outputs=[ctx.fresh_name(f"hess_tau_{i}")],
            )
            _stamp_like(tau, alpha)

            tau_1d_shape = _const_i64(
                ctx,
                np.asarray([1], dtype=np.int64),
                f"hess_tau_reshape_shape_{i}",
            )
            tau_1d = ctx.builder.Reshape(
                tau,
                tau_1d_shape,
                _outputs=[ctx.fresh_name(f"hess_tau_1d_{i}")],
            )
            if getattr(tau, "type", None) is not None:
                tau_1d.type = tau.type
            tau_1d.shape = ir.Shape((1,))
            taus_cur = _scatter_vec_elem(ctx, taus_cur, i, tau_1d, f"hess_set_tau_{i}")

            v_full = ctx.bind_const_for_var(object(), np.zeros((p, 1), dtype=np_dtype))
            if getattr(x, "type", None) is not None:
                v_full.type = x.type
            v_full.shape = ir.Shape((p, 1))
            v_full = _scatter_mat_elem(ctx, v_full, 0, 0, one, f"hess_set_v0_{i}")
            for r in range(1, p):
                vr = _gather_mat_elem(ctx, v_tail, r - 1, 0, f"hess_vtail_elem_{i}_{r}")
                v_full = _scatter_mat_elem(ctx, v_full, r, 0, vr, f"hess_set_v_{i}_{r}")

            v_t = ctx.builder.Transpose(
                v_full,
                perm=[1, 0],
                _outputs=[ctx.fresh_name(f"hess_v_t_{i}")],
            )
            if getattr(v_full, "type", None) is not None:
                v_t.type = v_full.type
            v_t.shape = ir.Shape((1, p))

            a_left = _slice_matrix(
                ctx,
                a_cur,
                row_start=i + 1,
                row_end=n,
                col_start=i,
                col_end=n,
                out_shape=(p, n - i),
                name=f"hess_a_left_{i}",
            )
            vt_a = ctx.builder.MatMul(
                v_t,
                a_left,
                _outputs=[ctx.fresh_name(f"hess_vt_a_{i}")],
            )
            if getattr(a_left, "type", None) is not None:
                vt_a.type = a_left.type
            vt_a.shape = ir.Shape((1, n - i))
            tau_vt_a = ctx.builder.Mul(
                tau,
                vt_a,
                _outputs=[ctx.fresh_name(f"hess_tau_vt_a_{i}")],
            )
            if getattr(vt_a, "type", None) is not None:
                tau_vt_a.type = vt_a.type
            tau_vt_a.shape = ir.Shape((1, n - i))
            left_delta = ctx.builder.MatMul(
                v_full,
                tau_vt_a,
                _outputs=[ctx.fresh_name(f"hess_left_delta_{i}")],
            )
            if getattr(a_left, "type", None) is not None:
                left_delta.type = a_left.type
            left_delta.shape = ir.Shape((p, n - i))
            a_left_new = ctx.builder.Sub(
                a_left,
                left_delta,
                _outputs=[ctx.fresh_name(f"hess_a_left_new_{i}")],
            )
            if getattr(a_left, "type", None) is not None:
                a_left_new.type = a_left.type
            a_left_new.shape = ir.Shape((p, n - i))
            a_cur = _scatter_block(
                ctx,
                a_cur,
                a_left_new,
                row_start=i + 1,
                col_start=i,
                rows=p,
                cols=n - i,
                name=f"hess_left_update_{i}",
            )

            a_right = _slice_matrix(
                ctx,
                a_cur,
                row_start=0,
                row_end=n,
                col_start=i + 1,
                col_end=n,
                out_shape=(n, p),
                name=f"hess_a_right_{i}",
            )
            a_v = ctx.builder.MatMul(
                a_right,
                v_full,
                _outputs=[ctx.fresh_name(f"hess_a_v_{i}")],
            )
            if getattr(a_right, "type", None) is not None:
                a_v.type = a_right.type
            a_v.shape = ir.Shape((n, 1))
            tau_a_v = ctx.builder.Mul(
                tau,
                a_v,
                _outputs=[ctx.fresh_name(f"hess_tau_a_v_{i}")],
            )
            if getattr(a_v, "type", None) is not None:
                tau_a_v.type = a_v.type
            tau_a_v.shape = ir.Shape((n, 1))
            right_delta = ctx.builder.MatMul(
                tau_a_v,
                v_t,
                _outputs=[ctx.fresh_name(f"hess_right_delta_{i}")],
            )
            if getattr(a_right, "type", None) is not None:
                right_delta.type = a_right.type
            right_delta.shape = ir.Shape((n, p))
            a_right_new = ctx.builder.Sub(
                a_right,
                right_delta,
                _outputs=[ctx.fresh_name(f"hess_a_right_new_{i}")],
            )
            if getattr(a_right, "type", None) is not None:
                a_right_new.type = a_right.type
            a_right_new.shape = ir.Shape((n, p))
            a_cur = _scatter_block(
                ctx,
                a_cur,
                a_right_new,
                row_start=0,
                col_start=i + 1,
                rows=n,
                cols=p,
                name=f"hess_right_update_{i}",
            )

            a_cur = _scatter_mat_elem(
                ctx, a_cur, i + 1, i, beta_eff, f"hess_set_beta_{i}"
            )
            for r in range(i + 2, n):
                vr = _gather_mat_elem(
                    ctx, v_tail, r - (i + 2), 0, f"hess_store_vr_{i}_{r}"
                )
                a_cur = _scatter_mat_elem(ctx, a_cur, r, i, vr, f"hess_store_v_{i}_{r}")

        a_name = getattr(a_spec, "name", None) or ctx.fresh_name("hess_a")
        a_out = a_cur
        if getattr(a_out, "name", None) != a_name:
            a_out = ctx.builder.Identity(a_cur, _outputs=[a_name])
        _stamp_like(a_out, a_spec if getattr(a_spec, "type", None) else a_cur)
        if getattr(a_spec, "shape", None) is not None:
            a_out.shape = a_spec.shape

        taus_name = getattr(taus_spec, "name", None) or ctx.fresh_name("hess_taus")
        taus_out = taus_cur
        if getattr(taus_out, "name", None) != taus_name:
            taus_out = ctx.builder.Identity(taus_cur, _outputs=[taus_name])
        _stamp_like(
            taus_out, taus_spec if getattr(taus_spec, "type", None) else taus_cur
        )
        if getattr(taus_spec, "shape", None) is not None:
            taus_out.shape = taus_spec.shape

        ctx.bind_value_for_var(a_var, a_out)
        ctx.bind_value_for_var(taus_var, taus_out)
