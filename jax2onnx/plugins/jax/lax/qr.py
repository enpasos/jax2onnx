# jax2onnx/plugins/jax/lax/qr.py

from __future__ import annotations

from typing import Any, cast

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


def _stamp_like(value: Any, ref: Any) -> None:
    if getattr(ref, "type", None) is not None:
        value.type = ref.type
    if getattr(ref, "shape", None) is not None:
        value.shape = ref.shape


def _slice_matrix(
    ctx: LoweringContextProtocol,
    mat: ir.Value,
    *,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    out_shape: tuple[int, int],
    name: str,
) -> ir.Value:
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
    out = cast(
        ir.Value,
        ctx.builder.Slice(
            mat,
            starts,
            ends,
            axes,
            _outputs=[ctx.fresh_name(name)],
        ),
    )
    if getattr(mat, "type", None) is not None:
        out.type = mat.type
    out.shape = ir.Shape(out_shape)
    return out


def _gather_mat_elem(
    ctx: LoweringContextProtocol, mat: ir.Value, i: int, j: int, name: str
) -> ir.Value:
    i_idx = _const_i64(ctx, np.asarray([i], dtype=np.int64), f"{name}_i")
    row = cast(
        ir.Value,
        ctx.builder.Gather(
            mat,
            i_idx,
            axis=0,
            _outputs=[ctx.fresh_name(f"{name}_row")],
        ),
    )
    if getattr(mat, "type", None) is not None:
        row.type = mat.type

    j_idx = _const_i64(ctx, np.asarray([j], dtype=np.int64), f"{name}_j")
    elem = cast(
        ir.Value,
        ctx.builder.Gather(
            row,
            j_idx,
            axis=1,
            _outputs=[ctx.fresh_name(name)],
        ),
    )
    if getattr(mat, "type", None) is not None:
        elem.type = mat.type
    elem.shape = ir.Shape((1, 1))
    return elem


def _scatter_mat_elem(
    ctx: LoweringContextProtocol,
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
    out = cast(
        ir.Value,
        ctx.builder.ScatterND(mat, idx, value, _outputs=[ctx.fresh_name(name)]),
    )
    _stamp_like(out, mat)
    return out


def _scatter_block(
    ctx: LoweringContextProtocol,
    base: ir.Value,
    block: ir.Value,
    *,
    row_start: int,
    col_start: int,
    rows: int,
    cols: int,
    name: str,
) -> ir.Value:
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
    flat = cast(
        ir.Value,
        ctx.builder.Reshape(
            block, flat_shape, _outputs=[ctx.fresh_name(f"{name}_flat")]
        ),
    )
    if getattr(block, "type", None) is not None:
        flat.type = block.type
    flat.shape = ir.Shape((rows * cols,))

    out = cast(
        ir.Value,
        ctx.builder.ScatterND(base, idx, flat, _outputs=[ctx.fresh_name(name)]),
    )
    _stamp_like(out, base)
    return out


@register_primitive(
    jaxpr_primitive=jax.lax.linalg.qr_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.linalg.qr.html",
    onnx=[
        {
            "component": "MatMul",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
        },
        {"component": "Sqrt", "doc": "https://onnx.ai/onnx/operators/onnx__Sqrt.html"},
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {"component": "Sub", "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html"},
        {
            "component": "ScatterND",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterND.html",
        },
    ],
    since="0.12.1",
    context="primitives.lax",
    component="qr",
    testcases=[
        {
            "testcase": "qr_reduced_tall",
            "callable": lambda x: jax.lax.linalg.qr(
                x, pivoting=False, full_matrices=False
            ),
            "input_values": [
                np.asarray(
                    [
                        [1.0, 2.0, 3.0],
                        [2.0, -1.0, 0.5],
                        [0.5, 1.5, -2.0],
                        [3.0, 0.2, 1.0],
                    ],
                    dtype=np.float32,
                )
            ],
            "post_check_onnx_graph": EG(
                ["MatMul", "Sqrt", "ScatterND"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "qr_reduced_wide",
            "callable": lambda x: jax.lax.linalg.qr(
                x, pivoting=False, full_matrices=False
            ),
            "input_values": [
                np.asarray(
                    [
                        [1.0, 2.0, 3.0, 1.5],
                        [2.0, -1.0, 0.5, 2.5],
                        [0.5, 1.5, -2.0, -0.5],
                    ],
                    dtype=np.float32,
                )
            ],
            "post_check_onnx_graph": EG(
                ["MatMul", "Sub"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "qr_full_tall",
            "callable": lambda x: jax.lax.linalg.qr(
                x, pivoting=False, full_matrices=True
            ),
            "input_values": [
                np.asarray(
                    [
                        [1.0, 2.0, 3.0],
                        [2.0, -1.0, 0.5],
                        [0.5, 1.5, -2.0],
                        [3.0, 0.2, 1.0],
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
            "testcase": "qr_full_wide",
            "callable": lambda x: jax.lax.linalg.qr(
                x, pivoting=False, full_matrices=True
            ),
            "input_values": [
                np.asarray(
                    [
                        [1.0, 2.0, 3.0, 1.5],
                        [2.0, -1.0, 0.5, 2.5],
                        [0.5, 1.5, -2.0, -0.5],
                    ],
                    dtype=np.float32,
                )
            ],
            "post_check_onnx_graph": EG(
                ["MatMul", "Sub"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class QrPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.linalg.qr`` for ``pivoting=False`` (both reduced/full modes)."""

    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        params = dict(getattr(eqn, "params", {}) or {})
        pivoting = bool(params.get("pivoting", False))
        full_matrices = bool(params.get("full_matrices", True))
        if pivoting:
            raise NotImplementedError("qr with pivoting=True is not supported yet")

        (x_var,) = eqn.invars
        if len(eqn.outvars) != 2:
            raise NotImplementedError(
                "qr with full_matrices=False and pivoting=False must return 2 outputs"
            )
        q_var, r_var = eqn.outvars

        x = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("qr_in"))
        q_spec = ctx.get_value_for_var(q_var, name_hint=ctx.fresh_name("qr_q"))
        r_spec = ctx.get_value_for_var(r_var, name_hint=ctx.fresh_name("qr_r"))

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        if len(x_shape) != 2:
            raise NotImplementedError("qr currently supports rank-2 inputs only")
        m_raw, n_raw = x_shape
        if not isinstance(m_raw, (int, np.integer)) or not isinstance(
            n_raw, (int, np.integer)
        ):
            raise NotImplementedError("qr requires static matrix dimensions")
        m = int(m_raw)
        n = int(n_raw)
        if m < 0 or n < 0:
            raise ValueError("qr matrix dimensions must be non-negative")
        k = min(m, n)

        np_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(x_var, "aval", None), "dtype", np.float32)
        )
        r_cur = x
        q_cur = ctx.bind_const_for_var(object(), np.eye(m, dtype=np_dtype))
        _stamp_like(q_cur, x)
        q_cur.shape = ir.Shape((m, m))

        one = ctx.bind_const_for_var(object(), np.asarray([[1.0]], dtype=np_dtype))
        neg_one = ctx.bind_const_for_var(object(), np.asarray([[-1.0]], dtype=np_dtype))
        two = ctx.bind_const_for_var(object(), np.asarray([[2.0]], dtype=np_dtype))
        zero = ctx.bind_const_for_var(object(), np.asarray([[0.0]], dtype=np_dtype))
        for const_val in (one, neg_one, two, zero):
            if getattr(x, "type", None) is not None:
                const_val.type = x.type
            const_val.shape = ir.Shape((1, 1))

        iterations = k if m > n else max(k - 1, 0)
        for j in range(iterations):
            sub_rows = m - j
            sub_cols_r = n - j
            sub_cols_q = m - j

            x_col = _slice_matrix(
                ctx,
                r_cur,
                row_start=j,
                row_end=m,
                col_start=j,
                col_end=j + 1,
                out_shape=(sub_rows, 1),
                name=f"qr_x_col_{j}",
            )

            x_col_t = cast(
                ir.Value,
                ctx.builder.Transpose(
                    x_col,
                    perm=[1, 0],
                    _outputs=[ctx.fresh_name(f"qr_x_col_t_{j}")],
                ),
            )
            if getattr(x_col, "type", None) is not None:
                x_col_t.type = x_col.type
            x_col_t.shape = ir.Shape((1, sub_rows))

            norm_sq = cast(
                ir.Value,
                ctx.builder.MatMul(
                    x_col_t,
                    x_col,
                    _outputs=[ctx.fresh_name(f"qr_norm_sq_{j}")],
                ),
            )
            if getattr(x_col, "type", None) is not None:
                norm_sq.type = x_col.type
            norm_sq.shape = ir.Shape((1, 1))
            norm = cast(
                ir.Value,
                ctx.builder.Sqrt(norm_sq, _outputs=[ctx.fresh_name(f"qr_norm_{j}")]),
            )
            if getattr(norm_sq, "type", None) is not None:
                norm.type = norm_sq.type
            norm.shape = ir.Shape((1, 1))

            x0 = _gather_mat_elem(ctx, r_cur, j, j, f"qr_x0_{j}")
            less_zero = cast(
                ir.Value,
                ctx.builder.Less(
                    x0,
                    zero,
                    _outputs=[ctx.fresh_name(f"qr_less_zero_{j}")],
                ),
            )
            less_zero.type = ir.TensorType(ir.DataType.BOOL)
            less_zero.shape = ir.Shape((1, 1))

            alpha_sign = cast(
                ir.Value,
                ctx.builder.Where(
                    less_zero,
                    one,
                    neg_one,
                    _outputs=[ctx.fresh_name(f"qr_alpha_sign_{j}")],
                ),
            )
            if getattr(x, "type", None) is not None:
                alpha_sign.type = x.type
            alpha_sign.shape = ir.Shape((1, 1))

            alpha = cast(
                ir.Value,
                ctx.builder.Mul(
                    alpha_sign,
                    norm,
                    _outputs=[ctx.fresh_name(f"qr_alpha_{j}")],
                ),
            )
            if getattr(norm, "type", None) is not None:
                alpha.type = norm.type
            alpha.shape = ir.Shape((1, 1))

            v0 = cast(
                ir.Value,
                ctx.builder.Sub(x0, alpha, _outputs=[ctx.fresh_name(f"qr_v0_{j}")]),
            )
            if getattr(x0, "type", None) is not None:
                v0.type = x0.type
            v0.shape = ir.Shape((1, 1))

            v = _scatter_mat_elem(ctx, x_col, 0, 0, v0, f"qr_set_v0_{j}")
            v.shape = ir.Shape((sub_rows, 1))

            v_t = cast(
                ir.Value,
                ctx.builder.Transpose(
                    v,
                    perm=[1, 0],
                    _outputs=[ctx.fresh_name(f"qr_v_t_{j}")],
                ),
            )
            if getattr(v, "type", None) is not None:
                v_t.type = v.type
            v_t.shape = ir.Shape((1, sub_rows))

            beta = cast(
                ir.Value,
                ctx.builder.MatMul(v_t, v, _outputs=[ctx.fresh_name(f"qr_beta_{j}")]),
            )
            if getattr(v, "type", None) is not None:
                beta.type = v.type
            beta.shape = ir.Shape((1, 1))

            tau_raw = cast(
                ir.Value,
                ctx.builder.Div(
                    two,
                    beta,
                    _outputs=[ctx.fresh_name(f"qr_tau_raw_{j}")],
                ),
            )
            if getattr(beta, "type", None) is not None:
                tau_raw.type = beta.type
            tau_raw.shape = ir.Shape((1, 1))

            beta_zero = cast(
                ir.Value,
                ctx.builder.Equal(
                    beta,
                    zero,
                    _outputs=[ctx.fresh_name(f"qr_beta_zero_{j}")],
                ),
            )
            beta_zero.type = ir.TensorType(ir.DataType.BOOL)
            beta_zero.shape = ir.Shape((1, 1))
            tau = cast(
                ir.Value,
                ctx.builder.Where(
                    beta_zero,
                    zero,
                    tau_raw,
                    _outputs=[ctx.fresh_name(f"qr_tau_{j}")],
                ),
            )
            if getattr(beta, "type", None) is not None:
                tau.type = beta.type
            tau.shape = ir.Shape((1, 1))

            r_sub = _slice_matrix(
                ctx,
                r_cur,
                row_start=j,
                row_end=m,
                col_start=j,
                col_end=n,
                out_shape=(sub_rows, sub_cols_r),
                name=f"qr_r_sub_{j}",
            )
            vt_r = cast(
                ir.Value,
                ctx.builder.MatMul(
                    v_t,
                    r_sub,
                    _outputs=[ctx.fresh_name(f"qr_vt_r_{j}")],
                ),
            )
            if getattr(r_sub, "type", None) is not None:
                vt_r.type = r_sub.type
            vt_r.shape = ir.Shape((1, sub_cols_r))
            w = cast(
                ir.Value,
                ctx.builder.Mul(tau, vt_r, _outputs=[ctx.fresh_name(f"qr_w_{j}")]),
            )
            if getattr(vt_r, "type", None) is not None:
                w.type = vt_r.type
            w.shape = ir.Shape((1, sub_cols_r))
            vw = cast(
                ir.Value,
                ctx.builder.MatMul(v, w, _outputs=[ctx.fresh_name(f"qr_vw_{j}")]),
            )
            if getattr(r_sub, "type", None) is not None:
                vw.type = r_sub.type
            vw.shape = ir.Shape((sub_rows, sub_cols_r))
            r_sub_new = cast(
                ir.Value,
                ctx.builder.Sub(
                    r_sub,
                    vw,
                    _outputs=[ctx.fresh_name(f"qr_r_sub_new_{j}")],
                ),
            )
            if getattr(r_sub, "type", None) is not None:
                r_sub_new.type = r_sub.type
            r_sub_new.shape = ir.Shape((sub_rows, sub_cols_r))
            r_cur = _scatter_block(
                ctx,
                r_cur,
                r_sub_new,
                row_start=j,
                col_start=j,
                rows=sub_rows,
                cols=sub_cols_r,
                name=f"qr_r_update_{j}",
            )

            q_tail = _slice_matrix(
                ctx,
                q_cur,
                row_start=0,
                row_end=m,
                col_start=j,
                col_end=m,
                out_shape=(m, sub_cols_q),
                name=f"qr_q_tail_{j}",
            )
            qv = cast(
                ir.Value,
                ctx.builder.MatMul(
                    q_tail,
                    v,
                    _outputs=[ctx.fresh_name(f"qr_qv_{j}")],
                ),
            )
            if getattr(q_tail, "type", None) is not None:
                qv.type = q_tail.type
            qv.shape = ir.Shape((m, 1))
            tau_vt = cast(
                ir.Value,
                ctx.builder.Mul(
                    tau,
                    v_t,
                    _outputs=[ctx.fresh_name(f"qr_tau_vt_{j}")],
                ),
            )
            if getattr(v_t, "type", None) is not None:
                tau_vt.type = v_t.type
            tau_vt.shape = ir.Shape((1, sub_cols_q))
            q_delta = cast(
                ir.Value,
                ctx.builder.MatMul(
                    qv,
                    tau_vt,
                    _outputs=[ctx.fresh_name(f"qr_q_delta_{j}")],
                ),
            )
            if getattr(q_tail, "type", None) is not None:
                q_delta.type = q_tail.type
            q_delta.shape = ir.Shape((m, sub_cols_q))
            q_tail_new = cast(
                ir.Value,
                ctx.builder.Sub(
                    q_tail,
                    q_delta,
                    _outputs=[ctx.fresh_name(f"qr_q_tail_new_{j}")],
                ),
            )
            if getattr(q_tail, "type", None) is not None:
                q_tail_new.type = q_tail.type
            q_tail_new.shape = ir.Shape((m, sub_cols_q))
            q_cur = _scatter_block(
                ctx,
                q_cur,
                q_tail_new,
                row_start=0,
                col_start=j,
                rows=m,
                cols=sub_cols_q,
                name=f"qr_q_update_{j}",
            )

        if full_matrices:
            q_final = q_cur
            r_final = r_cur
        else:
            q_final = _slice_matrix(
                ctx,
                q_cur,
                row_start=0,
                row_end=m,
                col_start=0,
                col_end=k,
                out_shape=(m, k),
                name="qr_q_reduced",
            )
            r_final = _slice_matrix(
                ctx,
                r_cur,
                row_start=0,
                row_end=k,
                col_start=0,
                col_end=n,
                out_shape=(k, n),
                name="qr_r_reduced",
            )

        q_name = getattr(q_spec, "name", None) or ctx.fresh_name("qr_q")
        q_out = q_final
        if getattr(q_out, "name", None) != q_name:
            q_out = cast(ir.Value, ctx.builder.Identity(q_final, _outputs=[q_name]))
        _stamp_like(q_out, q_spec if getattr(q_spec, "type", None) else q_final)
        if getattr(q_spec, "shape", None) is not None:
            q_out.shape = q_spec.shape

        r_name = getattr(r_spec, "name", None) or ctx.fresh_name("qr_r")
        r_out = r_final
        if getattr(r_out, "name", None) != r_name:
            r_out = cast(ir.Value, ctx.builder.Identity(r_final, _outputs=[r_name]))
        _stamp_like(r_out, r_spec if getattr(r_spec, "type", None) else r_final)
        if getattr(r_spec, "shape", None) is not None:
            r_out.shape = r_spec.shape

        ctx.bind_value_for_var(q_var, q_out)
        ctx.bind_value_for_var(r_var, r_out)
