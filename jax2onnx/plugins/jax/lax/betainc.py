# jax2onnx/plugins/jax/lax/betainc.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import numpy as np

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax.lgamma import _lanczos_lgamma_positive
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _stamp_like(value: Any, ref: Any) -> None:
    if getattr(ref, "type", None) is not None:
        value.type = ref.type
    if getattr(ref, "shape", None) is not None:
        value.shape = ref.shape


@register_primitive(
    jaxpr_primitive="regularized_incomplete_beta",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.betainc.html",
    onnx=[
        {"component": "Pow", "doc": "https://onnx.ai/onnx/operators/onnx__Pow.html"},
        {"component": "Exp", "doc": "https://onnx.ai/onnx/operators/onnx__Exp.html"},
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
    ],
    since="0.12.1",
    context="primitives.lax",
    component="betainc",
    testcases=[
        {
            "testcase": "betainc_basic",
            "callable": lambda a, b, x: jax.lax.betainc(a, b, x),
            "input_values": [
                np.asarray([1.0, 2.0, 2.5, 5.0], dtype=np.float32),
                np.asarray([1.0, 3.0, 4.0, 2.0], dtype=np.float32),
                np.asarray([0.2, 0.3, 0.6, 0.8], dtype=np.float32),
            ],
            "run_only_f32_variant": True,
            "rtol_f32": 1e-4,
            "atol_f32": 1e-5,
            "post_check_onnx_graph": EG(["Pow", "Where"], no_unused_inputs=True),
        },
        {
            "testcase": "betainc_edge_x",
            "callable": lambda a, b, x: jax.lax.betainc(a, b, x),
            "input_values": [
                np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
                np.asarray([2.0, 3.0, 4.0], dtype=np.float32),
                np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            ],
            "run_only_f32_variant": True,
            "rtol_f32": 1e-4,
            "atol_f32": 1e-5,
            "post_check_onnx_graph": EG(["Where"], no_unused_inputs=True),
        },
    ],
)
class BetaIncPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.betainc`` with normalized midpoint quadrature."""

    def lower(self, ctx: "IRContext", eqn: Any) -> None:
        a_var, b_var, x_var = eqn.invars
        out_var = eqn.outvars[0]

        a = ctx.get_value_for_var(a_var, name_hint=ctx.fresh_name("betainc_a"))
        b = ctx.get_value_for_var(b_var, name_hint=ctx.fresh_name("betainc_b"))
        x = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("betainc_x"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("betainc_out")
        )
        np_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(x_var, "aval", None), "dtype", np.float32)
        )

        zero = ctx.bind_const_for_var(object(), np.asarray(0.0, dtype=np_dtype))
        one = ctx.bind_const_for_var(object(), np.asarray(1.0, dtype=np_dtype))
        half = ctx.bind_const_for_var(object(), np.asarray(0.5, dtype=np_dtype))
        nan_const = ctx.bind_const_for_var(object(), np.asarray(np.nan, dtype=np_dtype))

        a_gt_zero = ctx.builder.Greater(
            a,
            zero,
            _outputs=[ctx.fresh_name("betainc_a_gt_zero")],
        )
        b_gt_zero = ctx.builder.Greater(
            b,
            zero,
            _outputs=[ctx.fresh_name("betainc_b_gt_zero")],
        )
        x_ge_zero = ctx.builder.Not(
            ctx.builder.Less(
                x,
                zero,
                _outputs=[ctx.fresh_name("betainc_x_lt_zero")],
            ),
            _outputs=[ctx.fresh_name("betainc_x_ge_zero")],
        )
        x_le_one = ctx.builder.Not(
            ctx.builder.Greater(
                x,
                one,
                _outputs=[ctx.fresh_name("betainc_x_gt_one")],
            ),
            _outputs=[ctx.fresh_name("betainc_x_le_one")],
        )
        valid = ctx.builder.And(
            a_gt_zero,
            b_gt_zero,
            _outputs=[ctx.fresh_name("betainc_valid_ab")],
        )
        valid = ctx.builder.And(
            valid,
            x_ge_zero,
            _outputs=[ctx.fresh_name("betainc_valid_x0")],
        )
        valid = ctx.builder.And(
            valid,
            x_le_one,
            _outputs=[ctx.fresh_name("betainc_valid")],
        )

        x_eq_zero = ctx.builder.Equal(
            x,
            zero,
            _outputs=[ctx.fresh_name("betainc_x_eq_zero")],
        )
        x_eq_one = ctx.builder.Equal(
            x,
            one,
            _outputs=[ctx.fresh_name("betainc_x_eq_one")],
        )
        x_is_edge = ctx.builder.Or(
            x_eq_zero,
            x_eq_one,
            _outputs=[ctx.fresh_name("betainc_x_is_edge")],
        )
        safe_x = ctx.builder.Where(
            x_is_edge,
            half,
            x,
            _outputs=[ctx.fresh_name("betainc_safe_x")],
        )
        _stamp_like(safe_x, x)

        num_steps = 128
        h = ctx.builder.Div(
            safe_x,
            ctx.bind_const_for_var(
                object(), np.asarray(float(num_steps), dtype=np_dtype)
            ),
            _outputs=[ctx.fresh_name("betainc_h")],
        )
        _stamp_like(h, x)
        a_minus_1 = ctx.builder.Sub(
            a,
            one,
            _outputs=[ctx.fresh_name("betainc_a_minus_1")],
        )
        _stamp_like(a_minus_1, a)
        b_minus_1 = ctx.builder.Sub(
            b,
            one,
            _outputs=[ctx.fresh_name("betainc_b_minus_1")],
        )
        _stamp_like(b_minus_1, b)

        acc = zero
        for i in range(num_steps):
            t = ctx.builder.Mul(
                h,
                ctx.bind_const_for_var(
                    object(), np.asarray(float(i) + 0.5, dtype=np_dtype)
                ),
                _outputs=[ctx.fresh_name(f"betainc_t_{i}")],
            )
            _stamp_like(t, x)
            t_pow = ctx.builder.Pow(
                t,
                a_minus_1,
                _outputs=[ctx.fresh_name(f"betainc_t_pow_{i}")],
            )
            _stamp_like(t_pow, x)
            one_minus_t = ctx.builder.Sub(
                one,
                t,
                _outputs=[ctx.fresh_name(f"betainc_one_minus_t_{i}")],
            )
            _stamp_like(one_minus_t, x)
            omt_pow = ctx.builder.Pow(
                one_minus_t,
                b_minus_1,
                _outputs=[ctx.fresh_name(f"betainc_omt_pow_{i}")],
            )
            _stamp_like(omt_pow, x)
            integrand = ctx.builder.Mul(
                t_pow,
                omt_pow,
                _outputs=[ctx.fresh_name(f"betainc_integrand_{i}")],
            )
            _stamp_like(integrand, x)
            acc = ctx.builder.Add(
                acc,
                integrand,
                _outputs=[ctx.fresh_name(f"betainc_acc_{i}")],
            )
            _stamp_like(acc, x)

        integral = ctx.builder.Mul(
            acc,
            h,
            _outputs=[ctx.fresh_name("betainc_integral")],
        )
        _stamp_like(integral, x)

        lgamma_a = _lanczos_lgamma_positive(ctx, a, np_dtype, "betainc_lgamma_a")
        lgamma_b = _lanczos_lgamma_positive(ctx, b, np_dtype, "betainc_lgamma_b")
        a_plus_b = ctx.builder.Add(
            a,
            b,
            _outputs=[ctx.fresh_name("betainc_a_plus_b")],
        )
        _stamp_like(a_plus_b, a)
        lgamma_ab = _lanczos_lgamma_positive(
            ctx, a_plus_b, np_dtype, "betainc_lgamma_ab"
        )

        log_beta = ctx.builder.Add(
            lgamma_a,
            lgamma_b,
            _outputs=[ctx.fresh_name("betainc_log_beta_ab")],
        )
        _stamp_like(log_beta, x)
        log_beta = ctx.builder.Sub(
            log_beta,
            lgamma_ab,
            _outputs=[ctx.fresh_name("betainc_log_beta")],
        )
        _stamp_like(log_beta, x)
        beta = ctx.builder.Exp(
            log_beta,
            _outputs=[ctx.fresh_name("betainc_beta")],
        )
        _stamp_like(beta, x)

        out = ctx.builder.Div(
            integral,
            beta,
            _outputs=[ctx.fresh_name("betainc_out_raw")],
        )
        _stamp_like(out, x)
        out = ctx.builder.Where(
            x_eq_zero,
            zero,
            out,
            _outputs=[ctx.fresh_name("betainc_out_x0")],
        )
        _stamp_like(out, x)
        out = ctx.builder.Where(
            x_eq_one,
            one,
            out,
            _outputs=[ctx.fresh_name("betainc_out_x1")],
        )
        _stamp_like(out, x)

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("betainc")
        result = ctx.builder.Where(valid, out, nan_const, _outputs=[desired_name])
        _stamp_like(result, out_spec if getattr(out_spec, "type", None) else out)
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
