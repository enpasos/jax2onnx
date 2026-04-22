# jax2onnx/plugins/jax/lax/ormqr.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax.householder_product import (
    _gather_mat_elem,
    _gather_vec_elem,
    _scatter_mat_elem,
    _stamp_like,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _make_householder_vector(
    ctx: "IRContext",
    a_val: ir.Value,
    *,
    idx: int,
    m: int,
    np_dtype: np.dtype[Any],
) -> ir.Value:
    v = ctx.bind_const_for_var(object(), np.zeros((m, 1), dtype=np_dtype))
    if getattr(a_val, "type", None) is not None:
        v.type = a_val.type
    v.shape = ir.Shape((m, 1))

    one = ctx.bind_const_for_var(object(), np.asarray([[1.0]], dtype=np_dtype))
    if getattr(a_val, "type", None) is not None:
        one.type = a_val.type
    one.shape = ir.Shape((1, 1))

    v = _scatter_mat_elem(ctx, v, idx, 0, one, f"ormqr_set_vdiag_{idx}")
    for row in range(idx + 1, m):
        a_elem = _gather_mat_elem(ctx, a_val, row, idx, f"ormqr_aelem_{idx}_{row}")
        v = _scatter_mat_elem(ctx, v, row, 0, a_elem, f"ormqr_set_v_{idx}_{row}")
    return v


def _transpose_vector(ctx: "IRContext", v: ir.Value, *, m: int, idx: int) -> ir.Value:
    v_t = ctx.builder.Transpose(
        v,
        perm=[1, 0],
        _outputs=[ctx.fresh_name(f"ormqr_vt_{idx}")],
    )
    if getattr(v, "type", None) is not None:
        v_t.type = v.type
    v_t.shape = ir.Shape((1, m))
    return v_t


@register_primitive(
    jaxpr_primitive=jax.lax.linalg.ormqr_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.linalg.ormqr.html",
    onnx=[
        {
            "component": "MatMul",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
        },
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {"component": "Sub", "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html"},
        {
            "component": "ScatterND",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterND.html",
        },
        {
            "component": "Transpose",
            "doc": "https://onnx.ai/onnx/operators/onnx__Transpose.html",
        },
    ],
    since="0.13.0",
    context="primitives.lax",
    component="ormqr",
    testcases=[
        {
            "testcase": "ormqr_left",
            "callable": lambda a, taus, c: jax.lax.linalg.ormqr(a, taus, c),
            "input_values": [
                np.asarray(
                    [
                        [1.0, 2.0],
                        [0.5, 4.0],
                        [-1.0, 0.3],
                    ],
                    dtype=np.float32,
                ),
                np.asarray([0.7, -0.25], dtype=np.float32),
                np.asarray(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.5, -0.5],
                    ],
                    dtype=np.float32,
                ),
            ],
            "post_check_onnx_graph": EG(
                ["MatMul", "Sub", "ScatterND"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "ormqr_left_transpose",
            "callable": lambda a, taus, c: jax.lax.linalg.ormqr(
                a, taus, c, transpose=True
            ),
            "input_values": [
                np.asarray(
                    [
                        [1.0, 2.0],
                        [0.5, 4.0],
                        [-1.0, 0.3],
                    ],
                    dtype=np.float32,
                ),
                np.asarray([0.7, -0.25], dtype=np.float32),
                np.asarray(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.5, -0.5],
                    ],
                    dtype=np.float32,
                ),
            ],
            "post_check_onnx_graph": EG(
                ["MatMul", "Sub", "ScatterND"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "ormqr_right",
            "callable": lambda a, taus, c: jax.lax.linalg.ormqr(a, taus, c, left=False),
            "input_values": [
                np.asarray(
                    [
                        [1.0, 2.0],
                        [0.5, 4.0],
                        [-1.0, 0.3],
                    ],
                    dtype=np.float32,
                ),
                np.asarray([0.7, -0.25], dtype=np.float32),
                np.asarray(
                    [
                        [1.0, 0.0, 1.5],
                        [0.0, 1.0, -0.5],
                    ],
                    dtype=np.float32,
                ),
            ],
            "post_check_onnx_graph": EG(
                ["MatMul", "Sub", "ScatterND"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class OrmqrPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.linalg.ormqr`` for static rank-2 real inputs."""

    def lower(self, ctx: "IRContext", eqn: Any) -> None:
        params = dict(getattr(eqn, "params", {}) or {})
        left = bool(params.get("left", True))
        transpose = bool(params.get("transpose", False))

        a_var, taus_var, c_var = eqn.invars
        out_var = eqn.outvars[0]

        a_val = ctx.get_value_for_var(a_var, name_hint=ctx.fresh_name("ormqr_a"))
        taus_val = ctx.get_value_for_var(
            taus_var, name_hint=ctx.fresh_name("ormqr_taus")
        )
        c_val = ctx.get_value_for_var(c_var, name_hint=ctx.fresh_name("ormqr_c"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("ormqr_out"))

        a_shape = tuple(getattr(getattr(a_var, "aval", None), "shape", ()))
        taus_shape = tuple(getattr(getattr(taus_var, "aval", None), "shape", ()))
        c_shape = tuple(getattr(getattr(c_var, "aval", None), "shape", ()))
        if len(a_shape) != 2 or len(taus_shape) != 1 or len(c_shape) != 2:
            raise NotImplementedError(
                "ormqr currently supports rank-2 `a`/`c` and rank-1 `taus` only"
            )

        m_raw, n_raw = a_shape
        k_raw = taus_shape[0]
        c_rows_raw, c_cols_raw = c_shape
        static_dims = (m_raw, n_raw, k_raw, c_rows_raw, c_cols_raw)
        if not all(isinstance(dim, (int, np.integer)) for dim in static_dims):
            raise NotImplementedError("ormqr requires static matrix dimensions")

        m = int(m_raw)
        n = int(n_raw)
        k = int(k_raw)
        c_rows = int(c_rows_raw)
        c_cols = int(c_cols_raw)
        if k < 0 or k > min(m, n):
            raise ValueError("ormqr requires 0 <= len(taus) <= min(a.shape)")
        if left and c_rows != m:
            raise ValueError("ormqr with left=True requires c.shape[0] == a.shape[0]")
        if not left and c_cols != m:
            raise ValueError("ormqr with left=False requires c.shape[1] == a.shape[0]")

        np_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(a_var, "aval", None), "dtype", np.float32)
        )
        if np.issubdtype(np_dtype, np.complexfloating):
            raise NotImplementedError("ormqr complex inputs are not supported yet")

        result = c_val
        sequence = range(k - 1, -1, -1) if left != transpose else range(k)
        for idx in sequence:
            v = _make_householder_vector(
                ctx,
                a_val,
                idx=idx,
                m=m,
                np_dtype=np_dtype,
            )
            v_t = _transpose_vector(ctx, v, m=m, idx=idx)
            tau_i = _gather_vec_elem(ctx, taus_val, idx, f"ormqr_tau_{idx}")
            tau_i.shape = ir.Shape((1,))

            if left:
                vtc = ctx.builder.MatMul(
                    v_t,
                    result,
                    _outputs=[ctx.fresh_name(f"ormqr_vtc_{idx}")],
                )
                if getattr(result, "type", None) is not None:
                    vtc.type = result.type
                vtc.shape = ir.Shape((1, c_cols))

                update = ctx.builder.MatMul(
                    v,
                    vtc,
                    _outputs=[ctx.fresh_name(f"ormqr_update_{idx}")],
                )
                if getattr(result, "type", None) is not None:
                    update.type = result.type
                update.shape = ir.Shape((m, c_cols))
            else:
                cv = ctx.builder.MatMul(
                    result,
                    v,
                    _outputs=[ctx.fresh_name(f"ormqr_cv_{idx}")],
                )
                if getattr(result, "type", None) is not None:
                    cv.type = result.type
                cv.shape = ir.Shape((c_rows, 1))

                update = ctx.builder.MatMul(
                    cv,
                    v_t,
                    _outputs=[ctx.fresh_name(f"ormqr_update_{idx}")],
                )
                if getattr(result, "type", None) is not None:
                    update.type = result.type
                update.shape = ir.Shape((c_rows, m))

            scaled = ctx.builder.Mul(
                tau_i,
                update,
                _outputs=[ctx.fresh_name(f"ormqr_scaled_update_{idx}")],
            )
            if getattr(update, "type", None) is not None:
                scaled.type = update.type
            scaled.shape = update.shape

            result = ctx.builder.Sub(
                result,
                scaled,
                _outputs=[ctx.fresh_name(f"ormqr_result_{idx}")],
            )
            _stamp_like(result, c_val)

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("ormqr")
        if getattr(result, "name", None) != desired_name:
            result = ctx.builder.Identity(result, _outputs=[desired_name])
        _stamp_like(result, out_spec if getattr(out_spec, "type", None) else c_val)
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
