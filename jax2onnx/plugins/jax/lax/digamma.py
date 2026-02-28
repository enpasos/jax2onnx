# jax2onnx/plugins/jax/lax/digamma.py

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


def _digamma_positive(ctx: "IRContext", x, np_dtype, name_prefix: str):
    one = ctx.bind_const_for_var(object(), np.asarray(1.0, dtype=np_dtype))
    half = ctx.bind_const_for_var(object(), np.asarray(0.5, dtype=np_dtype))
    six = ctx.bind_const_for_var(object(), np.asarray(6.0, dtype=np_dtype))
    c12 = ctx.bind_const_for_var(object(), np.asarray(1.0 / 12.0, dtype=np_dtype))
    c120 = ctx.bind_const_for_var(object(), np.asarray(1.0 / 120.0, dtype=np_dtype))
    c252 = ctx.bind_const_for_var(object(), np.asarray(1.0 / 252.0, dtype=np_dtype))
    zero = ctx.bind_const_for_var(object(), np.asarray(0.0, dtype=np_dtype))

    r = zero
    xw = x
    for i in range(8):
        cond = ctx.builder.Less(
            xw,
            six,
            _outputs=[ctx.fresh_name(f"{name_prefix}_cond_shift_{i}")],
        )
        inv = ctx.builder.Div(
            one,
            xw,
            _outputs=[ctx.fresh_name(f"{name_prefix}_inv_shift_{i}")],
        )
        _stamp_like(inv, x)
        r_new = ctx.builder.Sub(
            r,
            inv,
            _outputs=[ctx.fresh_name(f"{name_prefix}_r_new_{i}")],
        )
        _stamp_like(r_new, x)
        xw_new = ctx.builder.Add(
            xw,
            one,
            _outputs=[ctx.fresh_name(f"{name_prefix}_xw_new_{i}")],
        )
        _stamp_like(xw_new, x)
        r = ctx.builder.Where(
            cond,
            r_new,
            r,
            _outputs=[ctx.fresh_name(f"{name_prefix}_r_{i}")],
        )
        _stamp_like(r, x)
        xw = ctx.builder.Where(
            cond,
            xw_new,
            xw,
            _outputs=[ctx.fresh_name(f"{name_prefix}_xw_{i}")],
        )
        _stamp_like(xw, x)

    inv = ctx.builder.Div(one, xw, _outputs=[ctx.fresh_name(f"{name_prefix}_inv")])
    _stamp_like(inv, x)
    inv2 = ctx.builder.Mul(inv, inv, _outputs=[ctx.fresh_name(f"{name_prefix}_inv2")])
    _stamp_like(inv2, x)

    log_xw = ctx.builder.Log(xw, _outputs=[ctx.fresh_name(f"{name_prefix}_log")])
    _stamp_like(log_xw, x)
    half_inv = ctx.builder.Mul(
        half,
        inv,
        _outputs=[ctx.fresh_name(f"{name_prefix}_half_inv")],
    )
    _stamp_like(half_inv, x)
    term = ctx.builder.Sub(
        log_xw,
        half_inv,
        _outputs=[ctx.fresh_name(f"{name_prefix}_term")],
    )
    _stamp_like(term, x)

    inv2_c252 = ctx.builder.Mul(
        inv2,
        c252,
        _outputs=[ctx.fresh_name(f"{name_prefix}_inv2_c252")],
    )
    _stamp_like(inv2_c252, x)
    inner = ctx.builder.Sub(
        c120,
        inv2_c252,
        _outputs=[ctx.fresh_name(f"{name_prefix}_inner")],
    )
    mid = ctx.builder.Mul(
        inv2,
        inner,
        _outputs=[ctx.fresh_name(f"{name_prefix}_mid")],
    )
    _stamp_like(mid, x)
    outer = ctx.builder.Sub(
        c12,
        mid,
        _outputs=[ctx.fresh_name(f"{name_prefix}_outer")],
    )
    series = ctx.builder.Mul(
        inv2,
        outer,
        _outputs=[ctx.fresh_name(f"{name_prefix}_series")],
    )
    _stamp_like(series, x)

    base = ctx.builder.Add(
        r,
        term,
        _outputs=[ctx.fresh_name(f"{name_prefix}_base")],
    )
    _stamp_like(base, x)
    out = ctx.builder.Sub(
        base,
        series,
        _outputs=[ctx.fresh_name(f"{name_prefix}_out")],
    )
    _stamp_like(out, x)
    return out


@register_primitive(
    jaxpr_primitive=jax.lax.digamma_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.digamma.html",
    onnx=[
        {"component": "Log", "doc": "https://onnx.ai/onnx/operators/onnx__Log.html"},
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
        {"component": "Sin", "doc": "https://onnx.ai/onnx/operators/onnx__Sin.html"},
        {"component": "Cos", "doc": "https://onnx.ai/onnx/operators/onnx__Cos.html"},
    ],
    since="0.12.1",
    context="primitives.lax",
    component="digamma",
    testcases=[
        {
            "testcase": "digamma_positive",
            "callable": lambda x: jax.lax.digamma(x),
            "input_values": [
                np.asarray([0.2, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=np.float32)
            ],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(["Log", "Where"], no_unused_inputs=True),
        },
        {
            "testcase": "digamma_mixed",
            "callable": lambda x: jax.lax.digamma(x),
            "input_values": [
                np.asarray([-2.5, -1.5, -0.5, 0.5, 1.5], dtype=np.float32)
            ],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(["Sin", "Cos"], no_unused_inputs=True),
        },
    ],
)
class DigammaPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.digamma`` using recurrence + asymptotic + reflection."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("digamma_in"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("digamma_out")
        )
        np_dtype = np.dtype(getattr(getattr(x_var, "aval", None), "dtype", np.float32))

        one = ctx.bind_const_for_var(object(), np.asarray(1.0, dtype=np_dtype))
        zero = ctx.bind_const_for_var(object(), np.asarray(0.0, dtype=np_dtype))
        pi = ctx.bind_const_for_var(object(), np.asarray(np.pi, dtype=np_dtype))

        pos_part = _digamma_positive(ctx, x, np_dtype, "digamma_pos")
        one_minus_x = ctx.builder.Sub(one, x, _outputs=[ctx.fresh_name("digamma_1mx")])
        _stamp_like(one_minus_x, x)
        refl_base = _digamma_positive(ctx, one_minus_x, np_dtype, "digamma_refl")

        pi_x = ctx.builder.Mul(pi, x, _outputs=[ctx.fresh_name("digamma_pi_x")])
        _stamp_like(pi_x, x)
        sin_pi_x = ctx.builder.Sin(pi_x, _outputs=[ctx.fresh_name("digamma_sin_pi_x")])
        _stamp_like(sin_pi_x, x)
        cos_pi_x = ctx.builder.Cos(pi_x, _outputs=[ctx.fresh_name("digamma_cos_pi_x")])
        _stamp_like(cos_pi_x, x)
        cot_pi_x = ctx.builder.Div(
            cos_pi_x,
            sin_pi_x,
            _outputs=[ctx.fresh_name("digamma_cot_pi_x")],
        )
        _stamp_like(cot_pi_x, x)
        pi_cot = ctx.builder.Mul(
            pi,
            cot_pi_x,
            _outputs=[ctx.fresh_name("digamma_pi_cot")],
        )
        _stamp_like(pi_cot, x)
        refl = ctx.builder.Sub(
            refl_base,
            pi_cot,
            _outputs=[ctx.fresh_name("digamma_refl_out")],
        )
        _stamp_like(refl, x)

        is_pos = ctx.builder.Greater(
            x,
            zero,
            _outputs=[ctx.fresh_name("digamma_is_pos")],
        )
        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("digamma")
        result = ctx.builder.Where(
            is_pos,
            pos_part,
            refl,
            _outputs=[desired_name],
        )
        _stamp_like(result, out_spec if getattr(out_spec, "type", None) else x)
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
