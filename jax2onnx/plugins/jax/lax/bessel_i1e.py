# jax2onnx/plugins/jax/lax/bessel_i1e.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import numpy as np

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _stamp_like(value: Any, ref: Any) -> None:
    if getattr(ref, "type", None) is not None:
        value.type = ref.type
    if getattr(ref, "shape", None) is not None:
        value.shape = ref.shape


@register_primitive(
    jaxpr_primitive=jax.lax.bessel_i1e_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.bessel_i1e.html",
    onnx=[
        {"component": "Abs", "doc": "https://onnx.ai/onnx/operators/onnx__Abs.html"},
        {"component": "Sign", "doc": "https://onnx.ai/onnx/operators/onnx__Sign.html"},
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
    ],
    since="0.12.1",
    context="primitives.lax",
    component="bessel_i1e",
    testcases=[
        {
            "testcase": "bessel_i1e_basic",
            "callable": lambda x: jax.lax.bessel_i1e(x),
            "input_values": [
                np.asarray([-10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0], dtype=np.float32)
            ],
            "post_check_onnx_graph": EG(
                ["Abs", "Sign", "Where"], no_unused_inputs=True
            ),
        }
    ],
)
class BesselI1ePlugin(PrimitiveLeafPlugin):
    """Lower ``lax.bessel_i1e`` with classic Cephes-style piecewise approximations."""

    def lower(self, ctx: "IRContext", eqn: Any) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("bessel_i1e_in"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("bessel_i1e_out")
        )
        np_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(x_var, "aval", None), "dtype", np.float32)
        )

        ax = ctx.builder.Abs(x, _outputs=[ctx.fresh_name("bessel_i1e_abs")])
        _stamp_like(ax, x)
        sign_x = ctx.builder.Sign(x, _outputs=[ctx.fresh_name("bessel_i1e_sign")])
        _stamp_like(sign_x, x)

        c_3p75 = ctx.bind_const_for_var(object(), np.asarray(3.75, dtype=np_dtype))

        t_small = ctx.builder.Div(
            x,
            c_3p75,
            _outputs=[ctx.fresh_name("bessel_i1e_t_small")],
        )
        _stamp_like(t_small, x)
        t2_small = ctx.builder.Mul(
            t_small,
            t_small,
            _outputs=[ctx.fresh_name("bessel_i1e_t2_small")],
        )
        _stamp_like(t2_small, x)

        # Small branch: exp(-|x|) * x * poly(t^2), t = x/3.75
        p = ctx.bind_const_for_var(object(), np.asarray(0.00032411, dtype=np_dtype))
        for coef in [0.00301532, 0.02658733, 0.15084934, 0.51498869, 0.87890594, 0.5]:
            p = ctx.builder.Mul(
                p,
                t2_small,
                _outputs=[ctx.fresh_name("bessel_i1e_small_poly_mul")],
            )
            _stamp_like(p, x)
            p = ctx.builder.Add(
                p,
                ctx.bind_const_for_var(object(), np.asarray(coef, dtype=np_dtype)),
                _outputs=[ctx.fresh_name("bessel_i1e_small_poly_add")],
            )
            _stamp_like(p, x)
        i1_small = ctx.builder.Mul(
            x,
            p,
            _outputs=[ctx.fresh_name("bessel_i1e_small_i1")],
        )
        _stamp_like(i1_small, x)

        neg_ax = ctx.builder.Neg(ax, _outputs=[ctx.fresh_name("bessel_i1e_neg_ax")])
        _stamp_like(neg_ax, x)
        exp_neg_ax = ctx.builder.Exp(
            neg_ax,
            _outputs=[ctx.fresh_name("bessel_i1e_exp_neg_ax")],
        )
        _stamp_like(exp_neg_ax, x)
        small_out = ctx.builder.Mul(
            i1_small,
            exp_neg_ax,
            _outputs=[ctx.fresh_name("bessel_i1e_small_out")],
        )
        _stamp_like(small_out, x)

        # Large branch: sign(x) * poly(3.75/|x|) / sqrt(|x|)
        t_large = ctx.builder.Div(
            c_3p75,
            ax,
            _outputs=[ctx.fresh_name("bessel_i1e_t_large")],
        )
        _stamp_like(t_large, x)
        p_large = ctx.bind_const_for_var(
            object(), np.asarray(-0.00420059, dtype=np_dtype)
        )
        for coef in [
            0.01787654,
            -0.02895312,
            0.02282967,
            -0.01031555,
            0.00163801,
            -0.00362018,
            -0.03988024,
            0.39894228,
        ]:
            mul = ctx.builder.Mul(
                t_large,
                p_large,
                _outputs=[ctx.fresh_name("bessel_i1e_large_poly_mul")],
            )
            _stamp_like(mul, x)
            p_large = ctx.builder.Add(
                ctx.bind_const_for_var(object(), np.asarray(coef, dtype=np_dtype)),
                mul,
                _outputs=[ctx.fresh_name("bessel_i1e_large_poly_add")],
            )
            _stamp_like(p_large, x)

        sqrt_ax = ctx.builder.Sqrt(ax, _outputs=[ctx.fresh_name("bessel_i1e_sqrt_ax")])
        _stamp_like(sqrt_ax, x)
        large_mag = ctx.builder.Div(
            p_large,
            sqrt_ax,
            _outputs=[ctx.fresh_name("bessel_i1e_large_mag")],
        )
        _stamp_like(large_mag, x)
        large_out = ctx.builder.Mul(
            sign_x,
            large_mag,
            _outputs=[ctx.fresh_name("bessel_i1e_large_out")],
        )
        _stamp_like(large_out, x)

        cond_small = ctx.builder.Less(
            ax,
            c_3p75,
            _outputs=[ctx.fresh_name("bessel_i1e_cond_small")],
        )
        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("bessel_i1e")
        result = ctx.builder.Where(
            cond_small,
            small_out,
            large_out,
            _outputs=[desired_name],
        )
        _stamp_like(result, out_spec if getattr(out_spec, "type", None) else x)
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
