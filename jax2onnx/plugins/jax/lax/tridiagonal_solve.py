# jax2onnx/plugins/jax/lax/tridiagonal_solve.py

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
    if getattr(vec, "shape", None) is not None:
        out.shape = ir.Shape((1,))
    return out


def _gather_row(ctx: "IRContext", mat, i: int, row_width: int, name: str):
    idx = _const_i64(ctx, np.asarray([i], dtype=np.int64), f"{name}_idx")
    out = ctx.builder.Gather(
        mat,
        idx,
        axis=0,
        _outputs=[ctx.fresh_name(name)],
    )
    if getattr(mat, "type", None) is not None:
        out.type = mat.type
    out.shape = ir.Shape((1, row_width))
    return out


@register_primitive(
    jaxpr_primitive=jax.lax.linalg.tridiagonal_solve_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.linalg.tridiagonal_solve.html",
    onnx=[
        {
            "component": "Gather",
            "doc": "https://onnx.ai/onnx/operators/onnx__Gather.html",
        },
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {"component": "Sub", "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html"},
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        },
    ],
    since="0.12.1",
    context="primitives.lax",
    component="tridiagonal_solve",
    testcases=[
        {
            "testcase": "tridiagonal_solve_single_rhs",
            "callable": lambda dl, d, du, b: jax.lax.linalg.tridiagonal_solve(
                dl, d, du, b
            ),
            "input_values": [
                np.asarray([0.0, 1.0, 1.0, 1.0], dtype=np.float32),
                np.asarray([4.0, 4.0, 4.0, 4.0], dtype=np.float32),
                np.asarray([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
                np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32),
            ],
            "post_check_onnx_graph": EG(
                ["Gather", "Div", "Concat"], no_unused_inputs=True
            ),
        },
        {
            "testcase": "tridiagonal_solve_multi_rhs",
            "callable": lambda dl, d, du, b: jax.lax.linalg.tridiagonal_solve(
                dl, d, du, b
            ),
            "input_values": [
                np.asarray([0.0, 2.0, 2.0], dtype=np.float32),
                np.asarray([5.0, 6.0, 7.0], dtype=np.float32),
                np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
                np.asarray([[2.0, 1.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
            ],
            "post_check_onnx_graph": EG(["Concat"], no_unused_inputs=True),
        },
    ],
)
class TridiagonalSolvePlugin(PrimitiveLeafPlugin):
    """Lower ``lax.linalg.tridiagonal_solve`` with a static Thomas algorithm."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        dl_var, d_var, du_var, b_var = eqn.invars
        out_var = eqn.outvars[0]

        dl = ctx.get_value_for_var(dl_var, name_hint=ctx.fresh_name("tridiag_dl"))
        d = ctx.get_value_for_var(d_var, name_hint=ctx.fresh_name("tridiag_d"))
        du = ctx.get_value_for_var(du_var, name_hint=ctx.fresh_name("tridiag_du"))
        b = ctx.get_value_for_var(b_var, name_hint=ctx.fresh_name("tridiag_b"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("tridiag_out")
        )

        dl_shape = tuple(getattr(getattr(dl_var, "aval", None), "shape", ()))
        d_shape = tuple(getattr(getattr(d_var, "aval", None), "shape", ()))
        du_shape = tuple(getattr(getattr(du_var, "aval", None), "shape", ()))
        b_shape = tuple(getattr(getattr(b_var, "aval", None), "shape", ()))
        out_shape = tuple(getattr(getattr(out_var, "aval", None), "shape", ()))

        if len(dl_shape) != 1 or len(d_shape) != 1 or len(du_shape) != 1:
            raise NotImplementedError(
                "tridiagonal_solve currently supports rank-1 diagonals only"
            )
        if len(b_shape) != 2:
            raise NotImplementedError(
                "tridiagonal_solve currently supports rank-2 RHS only"
            )
        if not (dl_shape == d_shape == du_shape):
            raise ValueError(
                "tridiagonal_solve requires all diagonal inputs to have the same shape"
            )
        n = dl_shape[0]
        if not isinstance(n, (int, np.integer)):
            raise NotImplementedError("tridiagonal_solve requires static diagonal size")
        n = int(n)
        if n < 0:
            raise ValueError("tridiagonal_solve diagonal size must be non-negative")
        if int(b_shape[0]) != n:
            raise ValueError(
                f"tridiagonal_solve RHS leading dimension must match diagonal size ({n})"
            )
        rhs_cols = int(b_shape[1])

        # Degenerate system: return RHS unchanged.
        if n == 0:
            desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
                "tridiag_out"
            )
            result = ctx.builder.Identity(b, _outputs=[desired_name])
            _stamp_like(result, out_spec if getattr(out_spec, "type", None) else b)
            if getattr(out_spec, "shape", None) is not None:
                result.shape = out_spec.shape
            ctx.bind_value_for_var(out_var, result)
            return

        # Forward sweep.
        c_prime: list = []
        d_prime: list = []

        d0 = _gather_vec_elem(ctx, d, 0, "tridiag_d0")
        du0 = _gather_vec_elem(ctx, du, 0, "tridiag_du0")
        b0 = _gather_row(ctx, b, 0, rhs_cols, "tridiag_b0")
        c0 = ctx.builder.Div(du0, d0, _outputs=[ctx.fresh_name("tridiag_c0")])
        _stamp_like(c0, d0)
        d0_rhs = ctx.builder.Div(b0, d0, _outputs=[ctx.fresh_name("tridiag_d0_rhs")])
        _stamp_like(d0_rhs, b0)
        c_prime.append(c0)
        d_prime.append(d0_rhs)

        for i in range(1, n):
            dl_i = _gather_vec_elem(ctx, dl, i, f"tridiag_dl_{i}")
            d_i = _gather_vec_elem(ctx, d, i, f"tridiag_d_{i}")
            b_i = _gather_row(ctx, b, i, rhs_cols, f"tridiag_b_{i}")

            dl_mul_c = ctx.builder.Mul(
                dl_i,
                c_prime[i - 1],
                _outputs=[ctx.fresh_name(f"tridiag_dl_mul_c_{i}")],
            )
            _stamp_like(dl_mul_c, d_i)
            denom = ctx.builder.Sub(
                d_i,
                dl_mul_c,
                _outputs=[ctx.fresh_name(f"tridiag_denom_{i}")],
            )
            _stamp_like(denom, d_i)

            if i < n - 1:
                du_i = _gather_vec_elem(ctx, du, i, f"tridiag_du_{i}")
                c_i = ctx.builder.Div(
                    du_i,
                    denom,
                    _outputs=[ctx.fresh_name(f"tridiag_c_{i}")],
                )
                _stamp_like(c_i, du_i)
            else:
                c_i = ctx.bind_const_for_var(
                    object(),
                    np.asarray([0.0], dtype=np.dtype(getattr(dl_var.aval, "dtype"))),
                )
            c_prime.append(c_i)

            dl_mul_dprev = ctx.builder.Mul(
                dl_i,
                d_prime[i - 1],
                _outputs=[ctx.fresh_name(f"tridiag_dl_mul_dprev_{i}")],
            )
            _stamp_like(dl_mul_dprev, b_i)
            rhs_num = ctx.builder.Sub(
                b_i,
                dl_mul_dprev,
                _outputs=[ctx.fresh_name(f"tridiag_rhs_num_{i}")],
            )
            _stamp_like(rhs_num, b_i)
            d_i_rhs = ctx.builder.Div(
                rhs_num,
                denom,
                _outputs=[ctx.fresh_name(f"tridiag_d_rhs_{i}")],
            )
            _stamp_like(d_i_rhs, b_i)
            d_prime.append(d_i_rhs)

        # Back substitution.
        x_rows: list = [d_prime[-1]]
        for i in range(n - 2, -1, -1):
            c_mul_next = ctx.builder.Mul(
                c_prime[i],
                x_rows[0],
                _outputs=[ctx.fresh_name(f"tridiag_c_mul_next_{i}")],
            )
            _stamp_like(c_mul_next, d_prime[i])
            x_i = ctx.builder.Sub(
                d_prime[i],
                c_mul_next,
                _outputs=[ctx.fresh_name(f"tridiag_x_{i}")],
            )
            _stamp_like(x_i, d_prime[i])
            x_rows.insert(0, x_i)

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("tridiag_out")
        result = ctx.builder.Concat(
            *x_rows,
            axis=0,
            _outputs=[desired_name],
        )
        _stamp_like(result, out_spec if getattr(out_spec, "type", None) else b)
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        elif out_shape:
            result.shape = ir.Shape(out_shape)
        ctx.bind_value_for_var(out_var, result)
