# jax2onnx/plugins/jax/lax/cholesky.py

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


def _gather_elem(
    ctx: LoweringContextProtocol, mat: ir.Value, i: int, j: int, name: str
) -> ir.Value:
    i_idx = _const_i64(ctx, np.asarray([i], dtype=np.int64), f"{name}_i")
    row = cast(
        ir.Value,
        ctx.builder.Gather(
            mat, i_idx, axis=0, _outputs=[ctx.fresh_name(f"{name}_row")]
        ),
    )
    if getattr(mat, "type", None) is not None:
        row.type = mat.type
    j_idx = _const_i64(ctx, np.asarray([j], dtype=np.int64), f"{name}_j")
    elem = cast(
        ir.Value,
        ctx.builder.Gather(row, j_idx, axis=1, _outputs=[ctx.fresh_name(name)]),
    )
    if getattr(mat, "type", None) is not None:
        elem.type = mat.type
    elem.shape = ir.Shape((1, 1))
    return elem


def _scatter_set(
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


def _lower_single_cholesky(
    ctx: LoweringContextProtocol,
    x: ir.Value,
    *,
    n: int,
    np_dtype: np.dtype[Any],
    name_prefix: str,
) -> ir.Value:
    l_mat = ctx.bind_const_for_var(object(), np.zeros((n, n), dtype=np_dtype))
    _stamp_like(l_mat, x)

    for i in range(n):
        for j in range(i + 1):
            s = ctx.bind_const_for_var(object(), np.zeros((1, 1), dtype=np_dtype))
            for k in range(j):
                lik = _gather_elem(ctx, l_mat, i, k, f"{name_prefix}_lik_{i}_{j}_{k}")
                ljk = _gather_elem(ctx, l_mat, j, k, f"{name_prefix}_ljk_{i}_{j}_{k}")
                prod = cast(
                    ir.Value,
                    ctx.builder.Mul(
                        lik,
                        ljk,
                        _outputs=[ctx.fresh_name(f"{name_prefix}_prod_{i}_{j}_{k}")],
                    ),
                )
                _stamp_like(prod, lik)
                s = cast(
                    ir.Value,
                    ctx.builder.Add(
                        s,
                        prod,
                        _outputs=[ctx.fresh_name(f"{name_prefix}_sum_{i}_{j}_{k}")],
                    ),
                )
                _stamp_like(s, prod)

            aij = _gather_elem(ctx, x, i, j, f"{name_prefix}_aij_{i}_{j}")
            num = cast(
                ir.Value,
                ctx.builder.Sub(
                    aij,
                    s,
                    _outputs=[ctx.fresh_name(f"{name_prefix}_num_{i}_{j}")],
                ),
            )
            _stamp_like(num, aij)

            if i == j:
                val = cast(
                    ir.Value,
                    ctx.builder.Sqrt(
                        num,
                        _outputs=[ctx.fresh_name(f"{name_prefix}_diag_{i}")],
                    ),
                )
                _stamp_like(val, num)
            else:
                ljj = _gather_elem(ctx, l_mat, j, j, f"{name_prefix}_ljj_{i}_{j}")
                val = cast(
                    ir.Value,
                    ctx.builder.Div(
                        num,
                        ljj,
                        _outputs=[ctx.fresh_name(f"{name_prefix}_offdiag_{i}_{j}")],
                    ),
                )
                _stamp_like(val, num)

            l_mat = _scatter_set(ctx, l_mat, i, j, val, f"{name_prefix}_set_{i}_{j}")
    return l_mat


@register_primitive(
    jaxpr_primitive=jax.lax.linalg.cholesky_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.linalg.cholesky.html",
    onnx=[
        {
            "component": "Gather",
            "doc": "https://onnx.ai/onnx/operators/onnx__Gather.html",
        },
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {"component": "Sub", "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html"},
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
        {"component": "Sqrt", "doc": "https://onnx.ai/onnx/operators/onnx__Sqrt.html"},
        {
            "component": "ScatterND",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterND.html",
        },
        {
            "component": "Squeeze",
            "doc": "https://onnx.ai/onnx/operators/onnx__Squeeze.html",
        },
        {
            "component": "Unsqueeze",
            "doc": "https://onnx.ai/onnx/operators/onnx__Unsqueeze.html",
        },
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        },
    ],
    since="0.12.1",
    context="primitives.lax",
    component="cholesky",
    testcases=[
        {
            "testcase": "cholesky_spd_3x3",
            "callable": lambda x: jax.lax.linalg.cholesky(x, symmetrize_input=False),
            "input_values": [
                np.asarray(
                    [[4.0, 1.0, 2.0], [1.0, 3.0, 0.5], [2.0, 0.5, 5.0]],
                    dtype=np.float32,
                )
            ],
            "post_check_onnx_graph": EG(["Sqrt", "ScatterND"], no_unused_inputs=True),
        },
        {
            "testcase": "cholesky_diagonal",
            "callable": lambda x: jax.lax.linalg.cholesky(x, symmetrize_input=False),
            "input_values": [np.asarray(np.diag([1.0, 4.0, 9.0]), dtype=np.float32)],
            "post_check_onnx_graph": EG(["Sqrt"], no_unused_inputs=True),
        },
        {
            "testcase": "cholesky_batched_2x3x3",
            "callable": lambda x: jax.lax.linalg.cholesky(x, symmetrize_input=False),
            "input_values": [
                np.asarray(
                    [
                        [[4.0, 1.0, 2.0], [1.0, 3.0, 0.5], [2.0, 0.5, 5.0]],
                        [[9.0, 0.0, 3.0], [0.0, 4.0, 1.0], [3.0, 1.0, 6.0]],
                    ],
                    dtype=np.float32,
                )
            ],
            "post_check_onnx_graph": EG(["Squeeze", "Concat"], no_unused_inputs=True),
        },
    ],
)
class CholeskyPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.linalg.cholesky`` with static unrolled factorization."""

    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars

        x = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("cholesky_in"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("cholesky_out")
        )

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        if len(x_shape) not in {2, 3}:
            raise NotImplementedError(
                "cholesky currently supports rank-2 or rank-3 inputs only"
            )
        if len(x_shape) == 2:
            n_rows_raw, n_cols_raw = x_shape
            batch = None
        else:
            batch_raw, n_rows_raw, n_cols_raw = x_shape
            if not isinstance(batch_raw, (int, np.integer)):
                raise NotImplementedError("cholesky requires static batch dimension")
            batch = int(batch_raw)
            if batch < 0:
                raise ValueError("cholesky batch dimension must be non-negative")

        if not isinstance(n_rows_raw, (int, np.integer)) or not isinstance(
            n_cols_raw, (int, np.integer)
        ):
            raise NotImplementedError("cholesky requires static matrix dimensions")
        n_rows = int(n_rows_raw)
        n_cols = int(n_cols_raw)
        if n_rows != n_cols:
            raise ValueError("cholesky requires square matrices")
        n = n_rows

        np_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(x_var, "aval", None), "dtype", np.float32)
        )
        if n == 0 or (batch is not None and batch == 0):
            desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("cholesky")
            result = cast(ir.Value, ctx.builder.Identity(x, _outputs=[desired_name]))
            _stamp_like(result, out_spec if getattr(out_spec, "type", None) else x)
            if getattr(out_spec, "shape", None) is not None:
                result.shape = out_spec.shape
            ctx.bind_value_for_var(out_var, result)
            return

        if batch is None:
            l_mat = _lower_single_cholesky(
                ctx, x, n=n, np_dtype=np_dtype, name_prefix="chol"
            )
        else:
            squeeze_axes = _const_i64(
                ctx, np.asarray([0], dtype=np.int64), "chol_batch_sq_axes"
            )
            unsqueeze_axes = _const_i64(
                ctx, np.asarray([0], dtype=np.int64), "chol_batch_unsq_axes"
            )
            rows: list[ir.Value] = []
            for b in range(batch):
                b_idx = _const_i64(
                    ctx, np.asarray([b], dtype=np.int64), f"chol_batch_idx_{b}"
                )
                x_3d = cast(
                    ir.Value,
                    ctx.builder.Gather(
                        x,
                        b_idx,
                        axis=0,
                        _outputs=[ctx.fresh_name(f"chol_batch_in3d_{b}")],
                    ),
                )
                _stamp_like(x_3d, x)
                x_3d.shape = ir.Shape((1, n, n))
                x_mat = cast(
                    ir.Value,
                    ctx.builder.Squeeze(
                        x_3d,
                        squeeze_axes,
                        _outputs=[ctx.fresh_name(f"chol_batch_in2d_{b}")],
                    ),
                )
                _stamp_like(x_mat, x)
                x_mat.shape = ir.Shape((n, n))
                l_mat = _lower_single_cholesky(
                    ctx,
                    x_mat,
                    n=n,
                    np_dtype=np_dtype,
                    name_prefix=f"chol_b{b}",
                )
                l_3d = cast(
                    ir.Value,
                    ctx.builder.Unsqueeze(
                        l_mat,
                        unsqueeze_axes,
                        _outputs=[ctx.fresh_name(f"chol_batch_out3d_{b}")],
                    ),
                )
                _stamp_like(l_3d, x_3d)
                l_3d.shape = ir.Shape((1, n, n))
                rows.append(l_3d)
            l_mat = cast(
                ir.Value,
                ctx.builder.Concat(
                    *rows,
                    axis=0,
                    _outputs=[ctx.fresh_name("chol_batch_concat")],
                ),
            )
            _stamp_like(l_mat, out_spec if getattr(out_spec, "type", None) else x)
            l_mat.shape = ir.Shape((batch, n, n))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("cholesky")
        result = l_mat
        if getattr(result, "name", None) != desired_name:
            result = cast(
                ir.Value, ctx.builder.Identity(result, _outputs=[desired_name])
            )
        _stamp_like(result, out_spec if getattr(out_spec, "type", None) else l_mat)
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
