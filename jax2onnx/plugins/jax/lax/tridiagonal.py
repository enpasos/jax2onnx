# jax2onnx/plugins/jax/lax/tridiagonal.py

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


def _scatter_vec_elem(ctx: "IRContext", vec, i: int, value_1d, name: str):
    idx = _const_i64(ctx, np.asarray([[i]], dtype=np.int64), f"{name}_idx")
    out = ctx.builder.ScatterND(vec, idx, value_1d, _outputs=[ctx.fresh_name(name)])
    _stamp_like(out, vec)
    return out


@register_primitive(
    jaxpr_primitive=jax.lax.linalg.tridiagonal_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.linalg.tridiagonal.html",
    onnx=[
        {
            "component": "Gather",
            "doc": "https://onnx.ai/onnx/operators/onnx__Gather.html",
        },
        {
            "component": "ScatterND",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterND.html",
        },
        {
            "component": "Identity",
            "doc": "https://onnx.ai/onnx/operators/onnx__Identity.html",
        },
    ],
    since="0.12.1",
    context="primitives.lax",
    component="tridiagonal",
    testcases=[
        {
            "testcase": "tridiagonal_2x2_lower_true",
            "callable": lambda x: jax.lax.linalg.tridiagonal(x, lower=True),
            "input_values": [
                np.asarray(
                    [
                        [1.0, 7.0],
                        [3.0, 4.0],
                    ],
                    dtype=np.float32,
                )
            ],
            "post_check_onnx_graph": EG(
                ["Gather", "ScatterND"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "tridiagonal_2x2_lower_false",
            "callable": lambda x: jax.lax.linalg.tridiagonal(x, lower=False),
            "input_values": [
                np.asarray(
                    [
                        [2.0, -5.0],
                        [9.0, 1.0],
                    ],
                    dtype=np.float32,
                )
            ],
            "post_check_onnx_graph": EG(
                ["Gather", "Identity"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class TridiagonalPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.linalg.tridiagonal`` exactly for square matrices with ``n <= 2``."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        params = dict(getattr(eqn, "params", {}) or {})
        lower = bool(params.get("lower", True))

        (x_var,) = eqn.invars
        if len(eqn.outvars) != 4:
            raise NotImplementedError("tridiagonal expects exactly 4 outputs")
        a_var, d_var, e_var, taus_var = eqn.outvars

        x = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("tridiag_in"))
        a_spec = ctx.get_value_for_var(a_var, name_hint=ctx.fresh_name("tridiag_a"))
        d_spec = ctx.get_value_for_var(d_var, name_hint=ctx.fresh_name("tridiag_d"))
        e_spec = ctx.get_value_for_var(e_var, name_hint=ctx.fresh_name("tridiag_e"))
        taus_spec = ctx.get_value_for_var(
            taus_var, name_hint=ctx.fresh_name("tridiag_taus")
        )

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        if len(x_shape) != 2:
            raise NotImplementedError(
                "tridiagonal currently supports rank-2 inputs only"
            )
        n_rows_raw, n_cols_raw = x_shape
        if not isinstance(n_rows_raw, (int, np.integer)) or not isinstance(
            n_cols_raw, (int, np.integer)
        ):
            raise NotImplementedError("tridiagonal requires static matrix dimensions")
        n_rows = int(n_rows_raw)
        n_cols = int(n_cols_raw)
        if n_rows != n_cols:
            raise ValueError("tridiagonal requires square matrices")
        n = n_rows
        if n > 2:
            raise NotImplementedError(
                "tridiagonal currently supports n <= 2 only; larger reductions are pending"
            )

        np_dtype = np.dtype(getattr(getattr(x_var, "aval", None), "dtype", np.float32))
        a_name = getattr(a_spec, "name", None) or ctx.fresh_name("tridiag_a")
        a_out = ctx.builder.Identity(x, _outputs=[a_name])
        _stamp_like(a_out, a_spec if getattr(a_spec, "type", None) else x)
        if getattr(a_spec, "shape", None) is not None:
            a_out.shape = a_spec.shape

        d_vec = ctx.bind_const_for_var(object(), np.zeros((n,), dtype=np_dtype))
        if getattr(x, "type", None) is not None:
            d_vec.type = x.type
        d_vec.shape = ir.Shape((n,))
        for i in range(n):
            d_ii = _gather_mat_elem(ctx, x, i, i, f"tridiag_dii_{i}")
            one_shape = _const_i64(
                ctx, np.asarray([1], dtype=np.int64), f"tridiag_dii_shape_{i}"
            )
            d_ii_1d = ctx.builder.Reshape(
                d_ii,
                one_shape,
                _outputs=[ctx.fresh_name(f"tridiag_dii_1d_{i}")],
            )
            if getattr(d_ii, "type", None) is not None:
                d_ii_1d.type = d_ii.type
            d_ii_1d.shape = ir.Shape((1,))
            d_vec = _scatter_vec_elem(ctx, d_vec, i, d_ii_1d, f"tridiag_set_d_{i}")

        e_len = max(n - 1, 0)
        e_vec = ctx.bind_const_for_var(object(), np.zeros((e_len,), dtype=np_dtype))
        if getattr(x, "type", None) is not None:
            e_vec.type = x.type
        e_vec.shape = ir.Shape((e_len,))
        if n == 2:
            ei = (
                _gather_mat_elem(ctx, x, 1, 0, "tridiag_e0")
                if lower
                else _gather_mat_elem(ctx, x, 0, 1, "tridiag_e0")
            )
            one_shape = _const_i64(
                ctx, np.asarray([1], dtype=np.int64), "tridiag_e_shape"
            )
            ei_1d = ctx.builder.Reshape(
                ei,
                one_shape,
                _outputs=[ctx.fresh_name("tridiag_e0_1d")],
            )
            if getattr(ei, "type", None) is not None:
                ei_1d.type = ei.type
            ei_1d.shape = ir.Shape((1,))
            e_vec = _scatter_vec_elem(ctx, e_vec, 0, ei_1d, "tridiag_set_e0")

        taus_vec = ctx.bind_const_for_var(object(), np.zeros((e_len,), dtype=np_dtype))
        if getattr(x, "type", None) is not None:
            taus_vec.type = x.type
        taus_vec.shape = ir.Shape((e_len,))

        d_name = getattr(d_spec, "name", None) or ctx.fresh_name("tridiag_d")
        d_out = d_vec
        if getattr(d_out, "name", None) != d_name:
            d_out = ctx.builder.Identity(d_vec, _outputs=[d_name])
        _stamp_like(d_out, d_spec if getattr(d_spec, "type", None) else d_vec)
        if getattr(d_spec, "shape", None) is not None:
            d_out.shape = d_spec.shape

        e_name = getattr(e_spec, "name", None) or ctx.fresh_name("tridiag_e")
        e_out = e_vec
        if getattr(e_out, "name", None) != e_name:
            e_out = ctx.builder.Identity(e_vec, _outputs=[e_name])
        _stamp_like(e_out, e_spec if getattr(e_spec, "type", None) else e_vec)
        if getattr(e_spec, "shape", None) is not None:
            e_out.shape = e_spec.shape

        taus_name = getattr(taus_spec, "name", None) or ctx.fresh_name("tridiag_taus")
        taus_out = taus_vec
        if getattr(taus_out, "name", None) != taus_name:
            taus_out = ctx.builder.Identity(taus_vec, _outputs=[taus_name])
        _stamp_like(
            taus_out, taus_spec if getattr(taus_spec, "type", None) else taus_vec
        )
        if getattr(taus_spec, "shape", None) is not None:
            taus_out.shape = taus_spec.shape

        ctx.bind_value_for_var(a_var, a_out)
        ctx.bind_value_for_var(d_var, d_out)
        ctx.bind_value_for_var(e_var, e_out)
        ctx.bind_value_for_var(taus_var, taus_out)
