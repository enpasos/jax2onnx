# jax2onnx/plugins/jax/lax/igamma.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax.lgamma import _lanczos_lgamma_positive
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _stamp_like(value, ref) -> None:
    if getattr(ref, "type", None) is not None:
        value.type = ref.type
    if getattr(ref, "shape", None) is not None:
        value.shape = ref.shape


def _regularized_lower_gamma(
    ctx: "IRContext",
    a,
    x,
    np_dtype,
    name_prefix: str,
    *,
    steps: int = 96,
):
    """Approximate P(a, x) = igamma(a, x) for a>0, x>=0 via midpoint quadrature."""
    zero = ctx.bind_const_for_var(object(), np.asarray(0.0, dtype=np_dtype))
    one = ctx.bind_const_for_var(object(), np.asarray(1.0, dtype=np_dtype))
    steps_f = ctx.bind_const_for_var(object(), np.asarray(float(steps), dtype=np_dtype))

    x_eq_zero = ctx.builder.Equal(
        x,
        zero,
        _outputs=[ctx.fresh_name(f"{name_prefix}_x_eq_zero")],
    )
    safe_x = ctx.builder.Where(
        x_eq_zero,
        one,
        x,
        _outputs=[ctx.fresh_name(f"{name_prefix}_safe_x")],
    )
    _stamp_like(safe_x, x)

    h = ctx.builder.Div(safe_x, steps_f, _outputs=[ctx.fresh_name(f"{name_prefix}_h")])
    _stamp_like(h, x)
    a_minus_one = ctx.builder.Sub(
        a,
        one,
        _outputs=[ctx.fresh_name(f"{name_prefix}_a_minus_one")],
    )
    _stamp_like(a_minus_one, a)

    acc = zero
    for i in range(steps):
        t = ctx.builder.Mul(
            h,
            ctx.bind_const_for_var(
                object(), np.asarray(float(i) + 0.5, dtype=np_dtype)
            ),
            _outputs=[ctx.fresh_name(f"{name_prefix}_t_{i}")],
        )
        _stamp_like(t, x)
        t_pow = ctx.builder.Pow(
            t,
            a_minus_one,
            _outputs=[ctx.fresh_name(f"{name_prefix}_t_pow_{i}")],
        )
        _stamp_like(t_pow, x)
        neg_t = ctx.builder.Neg(
            t, _outputs=[ctx.fresh_name(f"{name_prefix}_neg_t_{i}")]
        )
        _stamp_like(neg_t, x)
        exp_neg_t = ctx.builder.Exp(
            neg_t,
            _outputs=[ctx.fresh_name(f"{name_prefix}_exp_neg_t_{i}")],
        )
        _stamp_like(exp_neg_t, x)
        integrand = ctx.builder.Mul(
            t_pow,
            exp_neg_t,
            _outputs=[ctx.fresh_name(f"{name_prefix}_integrand_{i}")],
        )
        _stamp_like(integrand, x)
        acc = ctx.builder.Add(
            acc,
            integrand,
            _outputs=[ctx.fresh_name(f"{name_prefix}_acc_{i}")],
        )
        _stamp_like(acc, x)

    integral = ctx.builder.Mul(
        acc,
        h,
        _outputs=[ctx.fresh_name(f"{name_prefix}_integral")],
    )
    _stamp_like(integral, x)

    lgamma_a = _lanczos_lgamma_positive(ctx, a, np_dtype, f"{name_prefix}_lgamma")
    gamma_a = ctx.builder.Exp(
        lgamma_a,
        _outputs=[ctx.fresh_name(f"{name_prefix}_gamma_a")],
    )
    _stamp_like(gamma_a, a)

    p = ctx.builder.Div(
        integral,
        gamma_a,
        _outputs=[ctx.fresh_name(f"{name_prefix}_p_raw")],
    )
    _stamp_like(p, x)
    p = ctx.builder.Where(
        x_eq_zero,
        zero,
        p,
        _outputs=[ctx.fresh_name(f"{name_prefix}_p_x0")],
    )
    _stamp_like(p, x)
    p = ctx.builder.Max(
        p,
        zero,
        _outputs=[ctx.fresh_name(f"{name_prefix}_p_clip_lo")],
    )
    _stamp_like(p, x)
    p = ctx.builder.Min(
        p,
        one,
        _outputs=[ctx.fresh_name(f"{name_prefix}_p_clip_hi")],
    )
    _stamp_like(p, x)
    return p


def _valid_domain_mask(ctx: "IRContext", a, x, np_dtype, name_prefix: str):
    zero = ctx.bind_const_for_var(object(), np.asarray(0.0, dtype=np_dtype))
    a_gt_zero = ctx.builder.Greater(
        a,
        zero,
        _outputs=[ctx.fresh_name(f"{name_prefix}_a_gt_zero")],
    )
    x_lt_zero = ctx.builder.Less(
        x,
        zero,
        _outputs=[ctx.fresh_name(f"{name_prefix}_x_lt_zero")],
    )
    x_ge_zero = ctx.builder.Not(
        x_lt_zero,
        _outputs=[ctx.fresh_name(f"{name_prefix}_x_ge_zero")],
    )
    return ctx.builder.And(
        a_gt_zero,
        x_ge_zero,
        _outputs=[ctx.fresh_name(f"{name_prefix}_valid")],
    )


@register_primitive(
    jaxpr_primitive=jax.lax.igamma_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.igamma.html",
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
    component="igamma",
    testcases=[
        {
            "testcase": "igamma_basic",
            "callable": lambda a, x: jax.lax.igamma(a, x),
            "input_values": [
                np.asarray([1.2, 2.5, 5.0, 10.0], dtype=np.float32),
                np.asarray([0.5, 3.0, 5.0, 12.0], dtype=np.float32),
            ],
            "run_only_f32_variant": True,
            "rtol_f32": 5e-4,
            "atol_f32": 5e-5,
            "post_check_onnx_graph": EG(["Pow", "Exp", "Where"], no_unused_inputs=True),
        },
        {
            "testcase": "igamma_zero_x",
            "callable": lambda a, x: jax.lax.igamma(a, x),
            "input_values": [
                np.asarray([1.0, 2.0, 5.0], dtype=np.float32),
                np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            ],
            "run_only_f32_variant": True,
            "rtol_f32": 5e-4,
            "atol_f32": 5e-5,
            "post_check_onnx_graph": EG(["Where"], no_unused_inputs=True),
        },
    ],
)
class IGammaPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.igamma`` with midpoint quadrature for the lower gamma ratio."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        a_var, x_var = eqn.invars
        out_var = eqn.outvars[0]

        a = ctx.get_value_for_var(a_var, name_hint=ctx.fresh_name("igamma_a"))
        x = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("igamma_x"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("igamma_out")
        )
        np_dtype = np.dtype(getattr(getattr(a_var, "aval", None), "dtype", np.float32))

        p = _regularized_lower_gamma(ctx, a, x, np_dtype, "igamma")
        valid = _valid_domain_mask(ctx, a, x, np_dtype, "igamma")
        nan_const = ctx.bind_const_for_var(object(), np.asarray(np.nan, dtype=np_dtype))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("igamma")
        result = ctx.builder.Where(valid, p, nan_const, _outputs=[desired_name])
        _stamp_like(result, out_spec if getattr(out_spec, "type", None) else p)
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)


@register_primitive(
    jaxpr_primitive=jax.lax.igammac_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.igammac.html",
    onnx=[
        {"component": "Sub", "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html"},
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
    ],
    since="0.12.1",
    context="primitives.lax",
    component="igammac",
    testcases=[
        {
            "testcase": "igammac_basic",
            "callable": lambda a, x: jax.lax.igammac(a, x),
            "input_values": [
                np.asarray([1.2, 2.5, 5.0, 10.0], dtype=np.float32),
                np.asarray([0.5, 3.0, 5.0, 12.0], dtype=np.float32),
            ],
            "run_only_f32_variant": True,
            "rtol_f32": 5e-4,
            "atol_f32": 5e-5,
            "post_check_onnx_graph": EG(["Sub", "Where"], no_unused_inputs=True),
        },
        {
            "testcase": "igammac_zero_x",
            "callable": lambda a, x: jax.lax.igammac(a, x),
            "input_values": [
                np.asarray([1.0, 2.0, 5.0], dtype=np.float32),
                np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            ],
            "run_only_f32_variant": True,
            "rtol_f32": 5e-4,
            "atol_f32": 5e-5,
            "post_check_onnx_graph": EG(["Where"], no_unused_inputs=True),
        },
    ],
)
class IGammaCPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.igammac`` as 1 - igamma(a, x) over the valid domain."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        a_var, x_var = eqn.invars
        out_var = eqn.outvars[0]

        a = ctx.get_value_for_var(a_var, name_hint=ctx.fresh_name("igammac_a"))
        x = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("igammac_x"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("igammac_out")
        )
        np_dtype = np.dtype(getattr(getattr(a_var, "aval", None), "dtype", np.float32))

        p = _regularized_lower_gamma(ctx, a, x, np_dtype, "igammac")
        one = ctx.bind_const_for_var(object(), np.asarray(1.0, dtype=np_dtype))
        zero = ctx.bind_const_for_var(object(), np.asarray(0.0, dtype=np_dtype))
        x_eq_zero = ctx.builder.Equal(
            x,
            zero,
            _outputs=[ctx.fresh_name("igammac_x_eq_zero")],
        )
        q = ctx.builder.Sub(one, p, _outputs=[ctx.fresh_name("igammac_q")])
        _stamp_like(q, p)
        q = ctx.builder.Where(
            x_eq_zero,
            one,
            q,
            _outputs=[ctx.fresh_name("igammac_q_x0")],
        )
        _stamp_like(q, p)

        valid = _valid_domain_mask(ctx, a, x, np_dtype, "igammac")
        nan_const = ctx.bind_const_for_var(object(), np.asarray(np.nan, dtype=np_dtype))
        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("igammac")
        result = ctx.builder.Where(valid, q, nan_const, _outputs=[desired_name])
        _stamp_like(result, out_spec if getattr(out_spec, "type", None) else q)
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)


@register_primitive(
    jaxpr_primitive=jax.lax.igamma_grad_a_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.igamma_grad_a.html",
    onnx=[
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
        {"component": "Sub", "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html"},
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
    ],
    since="0.12.1",
    context="primitives.lax",
    component="igamma_grad_a",
    testcases=[
        {
            "testcase": "igamma_grad_a_basic",
            "callable": lambda a, x: jax.lax.igamma_grad_a(a, x),
            "input_values": [
                np.asarray([1.2, 2.5, 5.0, 10.0], dtype=np.float32),
                np.asarray([0.5, 3.0, 5.0, 12.0], dtype=np.float32),
            ],
            "run_only_f32_variant": True,
            "rtol_f32": 2e-3,
            "atol_f32": 2e-4,
            "post_check_onnx_graph": EG(["Sub", "Div", "Where"], no_unused_inputs=True),
        }
    ],
)
class IGammaGradAPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.igamma_grad_a`` with symmetric finite differences on a."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        a_var, x_var = eqn.invars
        out_var = eqn.outvars[0]

        a = ctx.get_value_for_var(a_var, name_hint=ctx.fresh_name("igammagrad_a"))
        x = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("igammagrad_x"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("igammagrad_out")
        )
        np_dtype = np.dtype(getattr(getattr(a_var, "aval", None), "dtype", np.float32))

        eps = ctx.bind_const_for_var(object(), np.asarray(1e-3, dtype=np_dtype))
        a_plus = ctx.builder.Add(a, eps, _outputs=[ctx.fresh_name("igammagrad_ap")])
        _stamp_like(a_plus, a)
        a_minus_raw = ctx.builder.Sub(
            a,
            eps,
            _outputs=[ctx.fresh_name("igammagrad_am_raw")],
        )
        _stamp_like(a_minus_raw, a)
        a_minus = ctx.builder.Max(
            a_minus_raw,
            eps,
            _outputs=[ctx.fresh_name("igammagrad_am")],
        )
        _stamp_like(a_minus, a)

        p_plus = _regularized_lower_gamma(ctx, a_plus, x, np_dtype, "igammagrad_plus")
        p_minus = _regularized_lower_gamma(
            ctx, a_minus, x, np_dtype, "igammagrad_minus"
        )

        num = ctx.builder.Sub(
            p_plus,
            p_minus,
            _outputs=[ctx.fresh_name("igammagrad_num")],
        )
        _stamp_like(num, p_plus)
        den = ctx.builder.Sub(
            a_plus,
            a_minus,
            _outputs=[ctx.fresh_name("igammagrad_den")],
        )
        _stamp_like(den, a_plus)
        grad = ctx.builder.Div(
            num,
            den,
            _outputs=[ctx.fresh_name("igammagrad_grad")],
        )
        _stamp_like(grad, num)

        valid = _valid_domain_mask(ctx, a, x, np_dtype, "igammagrad")
        a_gt_eps = ctx.builder.Greater(
            a,
            eps,
            _outputs=[ctx.fresh_name("igammagrad_a_gt_eps")],
        )
        valid = ctx.builder.And(
            valid,
            a_gt_eps,
            _outputs=[ctx.fresh_name("igammagrad_valid")],
        )
        nan_const = ctx.bind_const_for_var(object(), np.asarray(np.nan, dtype=np_dtype))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
            "igamma_grad_a"
        )
        result = ctx.builder.Where(valid, grad, nan_const, _outputs=[desired_name])
        _stamp_like(result, out_spec if getattr(out_spec, "type", None) else grad)
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
