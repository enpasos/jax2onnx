# jax2onnx/plugins/jax/lax/symmetric_product.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _stamp_like(value, ref) -> None:
    if getattr(ref, "type", None) is not None:
        value.type = ref.type
    if getattr(ref, "shape", None) is not None:
        value.shape = ref.shape


@register_primitive(
    jaxpr_primitive=jax.lax.linalg.symmetric_product_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.linalg.symmetric_product.html",
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
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
    ],
    since="0.12.1",
    context="primitives.lax",
    component="symmetric_product",
    testcases=[
        {
            "testcase": "symmetric_product_default",
            "callable": lambda a, c: jax.lax.linalg.symmetric_product(a, c),
            "input_values": [
                np.asarray(
                    [[1.0, 2.0], [0.5, -1.0], [3.0, 4.0]],
                    dtype=np.float32,
                ),
                np.asarray(
                    [[2.0, 0.1, 0.2], [0.1, 3.0, -0.3], [0.2, -0.3, 1.5]],
                    dtype=np.float32,
                ),
            ],
            "post_check_onnx_graph": EG(
                ["Transpose", "MatMul"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "symmetric_product_alpha_beta",
            "callable": lambda a, c: jax.lax.linalg.symmetric_product(
                a, c, alpha=2.5, beta=0.3
            ),
            "input_values": [
                np.asarray(
                    [[1.0, -2.0], [0.5, 1.0], [-3.0, 4.0]],
                    dtype=np.float32,
                ),
                np.asarray(
                    [[1.5, 0.2, -0.1], [0.2, 2.0, 0.3], [-0.1, 0.3, 1.0]],
                    dtype=np.float32,
                ),
            ],
            "post_check_onnx_graph": EG(
                ["MatMul", "Mul", "Add"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class SymmetricProductPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.linalg.symmetric_product`` as ``alpha*A*A^T + beta*C``."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        a_var, c_var = eqn.invars
        out_var = eqn.outvars[0]
        params = dict(getattr(eqn, "params", {}) or {})
        alpha = float(params.get("alpha", 1.0))
        beta = float(params.get("beta", 0.0))

        a = ctx.get_value_for_var(a_var, name_hint=ctx.fresh_name("symprod_a"))
        c = ctx.get_value_for_var(c_var, name_hint=ctx.fresh_name("symprod_c"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("symprod_out")
        )

        a_shape = tuple(getattr(getattr(a_var, "aval", None), "shape", ()))
        if len(a_shape) < 2:
            raise ValueError("symmetric_product requires rank >= 2 for `a_matrix`.")
        perm = list(range(len(a_shape)))
        perm[-1], perm[-2] = perm[-2], perm[-1]

        a_t = ctx.builder.Transpose(
            a, perm=perm, _outputs=[ctx.fresh_name("symprod_at")]
        )
        _stamp_like(a_t, a)
        aa_t = ctx.builder.MatMul(a, a_t, _outputs=[ctx.fresh_name("symprod_aat")])
        _stamp_like(aa_t, c if getattr(c, "shape", None) is not None else a)

        out = aa_t
        if alpha != 1.0:
            alpha_c = ctx.bind_const_for_var(
                object(),
                np.asarray(
                    alpha, dtype=np.dtype(getattr(a_var.aval, "dtype", np.float32))
                ),
            )
            out = ctx.builder.Mul(
                out,
                alpha_c,
                _outputs=[ctx.fresh_name("symprod_alpha_scaled")],
            )
            _stamp_like(out, aa_t)

        beta_c = ctx.bind_const_for_var(
            object(),
            np.asarray(beta, dtype=np.dtype(getattr(c_var.aval, "dtype", np.float32))),
        )
        c_term = c
        if beta != 1.0:
            c_term = ctx.builder.Mul(
                c,
                beta_c,
                _outputs=[ctx.fresh_name("symprod_beta_scaled")],
            )
            _stamp_like(c_term, c)
        out = ctx.builder.Add(
            out,
            c_term,
            _outputs=[ctx.fresh_name("symprod_out_raw")],
        )
        _stamp_like(out, c)

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
            "symmetric_product"
        )
        if getattr(out, "name", None) != desired_name:
            out = ctx.builder.Identity(out, _outputs=[desired_name])
        _stamp_like(out, out_spec if getattr(out_spec, "type", None) else c)
        if getattr(out_spec, "shape", None) is not None:
            out.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, out)
