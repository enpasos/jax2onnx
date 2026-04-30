# jax2onnx/plugins/jax/lax/householder_product.py

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


def _gather_vec_elem(
    ctx: LoweringContextProtocol, vec: ir.Value, i: int, name: str
) -> ir.Value:
    idx = _const_i64(ctx, np.asarray([i], dtype=np.int64), f"{name}_idx")
    out = cast(
        ir.Value,
        ctx.builder.Gather(
            vec,
            idx,
            axis=0,
            _outputs=[ctx.fresh_name(name)],
        ),
    )
    if getattr(vec, "type", None) is not None:
        out.type = vec.type
    out.shape = ir.Shape((1,))
    return out


@register_primitive(
    jaxpr_primitive=jax.lax.linalg.householder_product_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.linalg.householder_product.html",
    onnx=[
        {
            "component": "MatMul",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
        },
        {
            "component": "Transpose",
            "doc": "https://onnx.ai/onnx/operators/onnx__Transpose.html",
        },
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {"component": "Sub", "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html"},
        {
            "component": "ScatterND",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterND.html",
        },
    ],
    since="0.12.1",
    context="primitives.lax",
    component="householder_product",
    testcases=[
        {
            "testcase": "householder_product_basic",
            "callable": lambda a, taus: jax.lax.linalg.householder_product(a, taus),
            "input_values": [
                np.asarray(
                    [
                        [1.0, 2.0, 3.0],
                        [0.5, 4.0, 5.0],
                        [-1.0, 0.3, 6.0],
                        [0.2, -0.4, 1.5],
                    ],
                    dtype=np.float32,
                ),
                np.asarray([0.7, -0.25, 0.5], dtype=np.float32),
            ],
            "post_check_onnx_graph": EG(
                ["MatMul", "ScatterND"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "householder_product_k0",
            "callable": lambda a, taus: jax.lax.linalg.householder_product(a, taus),
            "input_values": [
                np.asarray(
                    [
                        [2.0, 1.0],
                        [0.5, 3.0],
                        [4.0, -2.0],
                    ],
                    dtype=np.float32,
                ),
                np.asarray([], dtype=np.float32),
            ],
            "post_check_onnx_graph": EG(["Identity"]),
        },
    ],
)
class HouseholderProductPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.linalg.householder_product`` for static rank-2 inputs."""

    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        a_var, taus_var = eqn.invars
        out_var = eqn.outvars[0]

        a_val = ctx.get_value_for_var(a_var, name_hint=ctx.fresh_name("hhp_a"))
        taus_val = ctx.get_value_for_var(taus_var, name_hint=ctx.fresh_name("hhp_taus"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("hhp_out"))

        a_shape = tuple(getattr(getattr(a_var, "aval", None), "shape", ()))
        taus_shape = tuple(getattr(getattr(taus_var, "aval", None), "shape", ()))
        if len(a_shape) != 2:
            raise NotImplementedError(
                "householder_product currently supports rank-2 `a` only"
            )
        if len(taus_shape) != 1:
            raise NotImplementedError(
                "householder_product currently supports rank-1 `taus` only"
            )

        m_raw, n_raw = a_shape
        if not isinstance(m_raw, (int, np.integer)) or not isinstance(
            n_raw, (int, np.integer)
        ):
            raise NotImplementedError(
                "householder_product requires static matrix shape"
            )
        m = int(m_raw)
        n = int(n_raw)
        if m < n:
            raise ValueError(
                "householder_product requires number of rows >= number of columns"
            )

        k_raw = taus_shape[0]
        if not isinstance(k_raw, (int, np.integer)):
            raise NotImplementedError("householder_product requires static taus length")
        k = int(k_raw)
        if k < 0 or k > min(m, n):
            raise ValueError("householder_product requires 0 <= len(taus) <= min(m, n)")

        np_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(a_var, "aval", None), "dtype", np.float32)
        )
        q_cur = ctx.bind_const_for_var(object(), np.eye(m, n, dtype=np_dtype))
        _stamp_like(q_cur, a_val)
        q_cur.shape = ir.Shape((m, n))

        if k == 0:
            desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
                "householder_product"
            )
            result = cast(
                ir.Value, ctx.builder.Identity(q_cur, _outputs=[desired_name])
            )
            _stamp_like(result, out_spec if getattr(out_spec, "type", None) else q_cur)
            if getattr(out_spec, "shape", None) is not None:
                result.shape = out_spec.shape
            ctx.bind_value_for_var(out_var, result)
            return

        eye_m = ctx.bind_const_for_var(object(), np.eye(m, dtype=np_dtype))
        if getattr(a_val, "type", None) is not None:
            eye_m.type = a_val.type
        eye_m.shape = ir.Shape((m, m))

        one = ctx.bind_const_for_var(object(), np.asarray([[1.0]], dtype=np_dtype))
        if getattr(a_val, "type", None) is not None:
            one.type = a_val.type
        one.shape = ir.Shape((1, 1))

        for i in range(k - 1, -1, -1):
            v = ctx.bind_const_for_var(object(), np.zeros((m, 1), dtype=np_dtype))
            if getattr(a_val, "type", None) is not None:
                v.type = a_val.type
            v.shape = ir.Shape((m, 1))

            v = _scatter_mat_elem(ctx, v, i, 0, one, f"hhp_set_vdiag_{i}")
            for r in range(i + 1, m):
                a_elem = _gather_mat_elem(ctx, a_val, r, i, f"hhp_aelem_{i}_{r}")
                v = _scatter_mat_elem(ctx, v, r, 0, a_elem, f"hhp_set_v_{i}_{r}")

            vt = cast(
                ir.Value,
                ctx.builder.Transpose(
                    v,
                    perm=[1, 0],
                    _outputs=[ctx.fresh_name(f"hhp_vt_{i}")],
                ),
            )
            if getattr(v, "type", None) is not None:
                vt.type = v.type
            vt.shape = ir.Shape((1, m))

            vvt = cast(
                ir.Value,
                ctx.builder.MatMul(v, vt, _outputs=[ctx.fresh_name(f"hhp_vvt_{i}")]),
            )
            if getattr(a_val, "type", None) is not None:
                vvt.type = a_val.type
            vvt.shape = ir.Shape((m, m))

            tau_i = _gather_vec_elem(ctx, taus_val, i, f"hhp_tau_{i}")
            tau_axes = _const_i64(
                ctx, np.asarray([1], dtype=np.int64), f"hhp_tau_unsq_axes_{i}"
            )
            tau_i = cast(
                ir.Value,
                ctx.builder.Unsqueeze(
                    tau_i,
                    tau_axes,
                    _outputs=[ctx.fresh_name(f"hhp_tau_2d_{i}")],
                ),
            )
            if getattr(a_val, "type", None) is not None:
                tau_i.type = a_val.type
            tau_i.shape = ir.Shape((1, 1))

            tau_vvt = cast(
                ir.Value,
                ctx.builder.Mul(
                    tau_i,
                    vvt,
                    _outputs=[ctx.fresh_name(f"hhp_tau_vvt_{i}")],
                ),
            )
            if getattr(vvt, "type", None) is not None:
                tau_vvt.type = vvt.type
            tau_vvt.shape = ir.Shape((m, m))

            h_i = cast(
                ir.Value,
                ctx.builder.Sub(
                    eye_m,
                    tau_vvt,
                    _outputs=[ctx.fresh_name(f"hhp_h_{i}")],
                ),
            )
            if getattr(eye_m, "type", None) is not None:
                h_i.type = eye_m.type
            h_i.shape = ir.Shape((m, m))

            q_cur = cast(
                ir.Value,
                ctx.builder.MatMul(
                    h_i,
                    q_cur,
                    _outputs=[ctx.fresh_name(f"hhp_q_{i}")],
                ),
            )
            if getattr(a_val, "type", None) is not None:
                q_cur.type = a_val.type
            q_cur.shape = ir.Shape((m, n))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
            "householder_product"
        )
        result = q_cur
        if getattr(result, "name", None) != desired_name:
            result = cast(
                ir.Value, ctx.builder.Identity(result, _outputs=[desired_name])
            )
        _stamp_like(result, out_spec if getattr(out_spec, "type", None) else q_cur)
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
