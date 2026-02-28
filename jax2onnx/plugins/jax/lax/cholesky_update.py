# jax2onnx/plugins/jax/lax/cholesky_update.py

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


def _gather_mat_elem(ctx: "IRContext", mat, i: int, j: int, name: str):
    i_idx = _const_i64(ctx, np.asarray([i], dtype=np.int64), f"{name}_i")
    row = ctx.builder.Gather(
        mat,
        i_idx,
        axis=0,
        _outputs=[ctx.fresh_name(f"{name}_row")],
    )
    _stamp_like(row, mat)
    row.shape = ir.Shape(
        (
            1,
            (
                getattr(mat.shape, "dims", [None, None])[-1]
                if getattr(mat, "shape", None) is not None
                and hasattr(mat.shape, "dims")
                else None
            ),
        )
    )

    j_idx = _const_i64(ctx, np.asarray([j], dtype=np.int64), f"{name}_j")
    elem = ctx.builder.Gather(
        row,
        j_idx,
        axis=1,
        _outputs=[ctx.fresh_name(name)],
    )
    _stamp_like(elem, row)
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


def _gather_vec_elem(ctx: "IRContext", vec, i: int, name: str):
    idx = _const_i64(ctx, np.asarray([i], dtype=np.int64), f"{name}_idx")
    out = ctx.builder.Gather(
        vec,
        idx,
        axis=0,
        _outputs=[ctx.fresh_name(name)],
    )
    _stamp_like(out, vec)
    out.shape = ir.Shape((1,))
    return out


def _scatter_vec_elem(ctx: "IRContext", vec, i: int, value, name: str):
    idx = _const_i64(
        ctx,
        np.asarray([[[i]]], dtype=np.int64),
        f"{name}_idx",
    )
    updates = value
    if getattr(value, "shape", None) is None or tuple(
        getattr(value.shape, "dims", ())
    ) != (1,):
        # Normalize to rank-1 update payload for vector scatter.
        axes = _const_i64(ctx, np.asarray([0], dtype=np.int64), f"{name}_unsq_axes")
        updates = ctx.builder.Unsqueeze(
            value, axes, _outputs=[ctx.fresh_name(f"{name}_unsq")]
        )
    out = ctx.builder.ScatterND(vec, idx, updates, _outputs=[ctx.fresh_name(name)])
    _stamp_like(out, vec)
    return out


@register_primitive(
    jaxpr_primitive=jax.lax.linalg.cholesky_update_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.linalg.cholesky_update.html",
    onnx=[
        {"component": "Sqrt", "doc": "https://onnx.ai/onnx/operators/onnx__Sqrt.html"},
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
        {"component": "Sub", "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html"},
        {
            "component": "ScatterND",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterND.html",
        },
    ],
    since="0.12.1",
    context="primitives.lax",
    component="cholesky_update",
    testcases=[
        {
            "testcase": "cholesky_update_upper_3x3",
            "callable": lambda r, w: jax.lax.linalg.cholesky_update(r, w),
            "input_values": [
                np.asarray(
                    [
                        [2.0, 0.5, 0.4],
                        [0.0, 1.8, -0.2],
                        [0.0, 0.0, 1.5],
                    ],
                    dtype=np.float32,
                ),
                np.asarray([0.2, -0.4, 0.3], dtype=np.float32),
            ],
            "post_check_onnx_graph": EG(["Sqrt", "ScatterND"], no_unused_inputs=True),
        },
        {
            "testcase": "cholesky_update_identity",
            "callable": lambda r, w: jax.lax.linalg.cholesky_update(r, w),
            "input_values": [
                np.asarray(np.eye(4), dtype=np.float32),
                np.asarray([0.1, 0.2, -0.1, 0.3], dtype=np.float32),
            ],
            "post_check_onnx_graph": EG(["Div", "Mul", "Sub"], no_unused_inputs=True),
        },
    ],
)
class CholeskyUpdatePlugin(PrimitiveLeafPlugin):
    """Lower ``lax.linalg.cholesky_update`` for static rank-2 upper factors."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        r_var, w_var = eqn.invars
        out_var = eqn.outvars[0]

        r_mat = ctx.get_value_for_var(r_var, name_hint=ctx.fresh_name("cholupd_r"))
        w_vec = ctx.get_value_for_var(w_var, name_hint=ctx.fresh_name("cholupd_w"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("cholupd_out")
        )

        r_shape = tuple(getattr(getattr(r_var, "aval", None), "shape", ()))
        w_shape = tuple(getattr(getattr(w_var, "aval", None), "shape", ()))
        if len(r_shape) != 2:
            raise NotImplementedError(
                "cholesky_update currently supports rank-2 `r_matrix` only"
            )
        if len(w_shape) != 1:
            raise NotImplementedError(
                "cholesky_update currently supports rank-1 `w_vector` only"
            )
        n_rows, n_cols = r_shape
        if not isinstance(n_rows, (int, np.integer)) or not isinstance(
            n_cols, (int, np.integer)
        ):
            raise NotImplementedError(
                "cholesky_update requires static matrix dimensions"
            )
        n_rows = int(n_rows)
        n_cols = int(n_cols)
        if n_rows != n_cols:
            raise ValueError("cholesky_update requires square `r_matrix`")
        if int(w_shape[0]) != n_rows:
            raise ValueError(
                "cholesky_update requires `w_vector` length == matrix size"
            )
        n = n_rows

        r_cur = r_mat
        w_cur = w_vec

        for k in range(n):
            rkk = _gather_mat_elem(ctx, r_cur, k, k, f"cholupd_rkk_{k}")
            wk = _gather_vec_elem(ctx, w_cur, k, f"cholupd_wk_{k}")

            rkk_sq = ctx.builder.Mul(
                rkk, rkk, _outputs=[ctx.fresh_name(f"cholupd_rkk_sq_{k}")]
            )
            _stamp_like(rkk_sq, rkk)
            wk_sq = ctx.builder.Mul(
                wk, wk, _outputs=[ctx.fresh_name(f"cholupd_wk_sq_{k}")]
            )
            _stamp_like(wk_sq, wk)
            r_sq = ctx.builder.Add(
                rkk_sq,
                wk_sq,
                _outputs=[ctx.fresh_name(f"cholupd_r_sq_{k}")],
            )
            _stamp_like(r_sq, rkk)
            r_new = ctx.builder.Sqrt(
                r_sq, _outputs=[ctx.fresh_name(f"cholupd_r_new_{k}")]
            )
            _stamp_like(r_new, rkk)

            c = ctx.builder.Div(r_new, rkk, _outputs=[ctx.fresh_name(f"cholupd_c_{k}")])
            _stamp_like(c, rkk)
            s = ctx.builder.Div(wk, rkk, _outputs=[ctx.fresh_name(f"cholupd_s_{k}")])
            _stamp_like(s, wk)

            r_cur = _scatter_mat_elem(ctx, r_cur, k, k, r_new, f"cholupd_set_diag_{k}")

            for j in range(k + 1, n):
                rkj = _gather_mat_elem(ctx, r_cur, k, j, f"cholupd_rkj_{k}_{j}")
                wj = _gather_vec_elem(ctx, w_cur, j, f"cholupd_wj_{k}_{j}")

                s_mul_wj = ctx.builder.Mul(
                    s,
                    wj,
                    _outputs=[ctx.fresh_name(f"cholupd_s_mul_wj_{k}_{j}")],
                )
                _stamp_like(s_mul_wj, wj)
                num = ctx.builder.Add(
                    rkj,
                    s_mul_wj,
                    _outputs=[ctx.fresh_name(f"cholupd_num_{k}_{j}")],
                )
                _stamp_like(num, rkj)
                rkj_new = ctx.builder.Div(
                    num,
                    c,
                    _outputs=[ctx.fresh_name(f"cholupd_rkj_new_{k}_{j}")],
                )
                _stamp_like(rkj_new, rkj)
                r_cur = _scatter_mat_elem(
                    ctx, r_cur, k, j, rkj_new, f"cholupd_set_rkj_{k}_{j}"
                )

                c_mul_wj = ctx.builder.Mul(
                    c,
                    wj,
                    _outputs=[ctx.fresh_name(f"cholupd_c_mul_wj_{k}_{j}")],
                )
                _stamp_like(c_mul_wj, wj)
                s_mul_rnew = ctx.builder.Mul(
                    s,
                    rkj_new,
                    _outputs=[ctx.fresh_name(f"cholupd_s_mul_rnew_{k}_{j}")],
                )
                _stamp_like(s_mul_rnew, rkj_new)
                wj_new = ctx.builder.Sub(
                    c_mul_wj,
                    s_mul_rnew,
                    _outputs=[ctx.fresh_name(f"cholupd_wj_new_{k}_{j}")],
                )
                _stamp_like(wj_new, wj)
                w_cur = _scatter_vec_elem(
                    ctx, w_cur, j, wj_new, f"cholupd_set_wj_{k}_{j}"
                )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
            "cholesky_update"
        )
        result = r_cur
        if getattr(result, "name", None) != desired_name:
            result = ctx.builder.Identity(result, _outputs=[desired_name])
        _stamp_like(result, out_spec if getattr(out_spec, "type", None) else r_cur)
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
