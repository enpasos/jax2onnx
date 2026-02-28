# jax2onnx/plugins/jax/lax/zeta.py

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


def _zeta_positive(ctx: "IRContext", s, q, np_dtype, name_prefix: str):
    # Hurwitz zeta via truncated series + Euler-Maclaurin tail.
    n_terms = 8
    one = ctx.bind_const_for_var(object(), np.asarray(1.0, dtype=np_dtype))
    half = ctx.bind_const_for_var(object(), np.asarray(0.5, dtype=np_dtype))
    zero = ctx.bind_const_for_var(object(), np.asarray(0.0, dtype=np_dtype))
    n_const = ctx.bind_const_for_var(
        object(), np.asarray(float(n_terms), dtype=np_dtype)
    )

    neg_s = ctx.builder.Neg(s, _outputs=[ctx.fresh_name(f"{name_prefix}_neg_s")])
    _stamp_like(neg_s, s)

    acc = zero
    for k in range(n_terms):
        qk = ctx.builder.Add(
            q,
            ctx.bind_const_for_var(object(), np.asarray(float(k), dtype=np_dtype)),
            _outputs=[ctx.fresh_name(f"{name_prefix}_qk_{k}")],
        )
        _stamp_like(qk, q)
        term = ctx.builder.Pow(
            qk,
            neg_s,
            _outputs=[ctx.fresh_name(f"{name_prefix}_term_{k}")],
        )
        _stamp_like(term, q)
        acc = ctx.builder.Add(
            acc,
            term,
            _outputs=[ctx.fresh_name(f"{name_prefix}_acc_{k}")],
        )
        _stamp_like(acc, q)

    t = ctx.builder.Add(q, n_const, _outputs=[ctx.fresh_name(f"{name_prefix}_t")])
    _stamp_like(t, q)
    s_minus_1 = ctx.builder.Sub(
        s,
        one,
        _outputs=[ctx.fresh_name(f"{name_prefix}_s_minus_1")],
    )
    _stamp_like(s_minus_1, s)
    one_minus_s = ctx.builder.Sub(
        one,
        s,
        _outputs=[ctx.fresh_name(f"{name_prefix}_one_minus_s")],
    )
    _stamp_like(one_minus_s, s)

    # (t^(1-s)) / (s-1)
    tail_1_num = ctx.builder.Pow(
        t,
        one_minus_s,
        _outputs=[ctx.fresh_name(f"{name_prefix}_tail1_num")],
    )
    _stamp_like(tail_1_num, q)
    tail_1 = ctx.builder.Div(
        tail_1_num,
        s_minus_1,
        _outputs=[ctx.fresh_name(f"{name_prefix}_tail1")],
    )
    _stamp_like(tail_1, q)

    # 0.5 * t^-s
    t_neg_s = ctx.builder.Pow(
        t,
        neg_s,
        _outputs=[ctx.fresh_name(f"{name_prefix}_t_neg_s")],
    )
    _stamp_like(t_neg_s, q)
    tail_2 = ctx.builder.Mul(
        half,
        t_neg_s,
        _outputs=[ctx.fresh_name(f"{name_prefix}_tail2")],
    )
    _stamp_like(tail_2, q)

    # + (s/12) * t^(-s-1)
    s_over_12 = ctx.builder.Div(
        s,
        ctx.bind_const_for_var(object(), np.asarray(12.0, dtype=np_dtype)),
        _outputs=[ctx.fresh_name(f"{name_prefix}_s_over_12")],
    )
    _stamp_like(s_over_12, s)
    neg_s_minus_1 = ctx.builder.Sub(
        neg_s,
        one,
        _outputs=[ctx.fresh_name(f"{name_prefix}_neg_s_minus_1")],
    )
    _stamp_like(neg_s_minus_1, s)
    t_pow_3 = ctx.builder.Pow(
        t,
        neg_s_minus_1,
        _outputs=[ctx.fresh_name(f"{name_prefix}_t_pow_3")],
    )
    _stamp_like(t_pow_3, q)
    tail_3 = ctx.builder.Mul(
        s_over_12,
        t_pow_3,
        _outputs=[ctx.fresh_name(f"{name_prefix}_tail3")],
    )
    _stamp_like(tail_3, q)

    # - s(s+1)(s+2)/720 * t^(-s-3)
    s1 = ctx.builder.Add(
        s,
        one,
        _outputs=[ctx.fresh_name(f"{name_prefix}_s1")],
    )
    _stamp_like(s1, s)
    s2 = ctx.builder.Add(
        s,
        ctx.bind_const_for_var(object(), np.asarray(2.0, dtype=np_dtype)),
        _outputs=[ctx.fresh_name(f"{name_prefix}_s2")],
    )
    _stamp_like(s2, s)
    num4 = ctx.builder.Mul(
        s,
        s1,
        _outputs=[ctx.fresh_name(f"{name_prefix}_num4a")],
    )
    _stamp_like(num4, s)
    num4 = ctx.builder.Mul(
        num4,
        s2,
        _outputs=[ctx.fresh_name(f"{name_prefix}_num4b")],
    )
    _stamp_like(num4, s)
    coef4 = ctx.builder.Div(
        num4,
        ctx.bind_const_for_var(object(), np.asarray(-720.0, dtype=np_dtype)),
        _outputs=[ctx.fresh_name(f"{name_prefix}_coef4")],
    )
    _stamp_like(coef4, s)
    neg_s_minus_3 = ctx.builder.Sub(
        neg_s,
        ctx.bind_const_for_var(object(), np.asarray(3.0, dtype=np_dtype)),
        _outputs=[ctx.fresh_name(f"{name_prefix}_neg_s_minus_3")],
    )
    _stamp_like(neg_s_minus_3, s)
    t_pow_4 = ctx.builder.Pow(
        t,
        neg_s_minus_3,
        _outputs=[ctx.fresh_name(f"{name_prefix}_t_pow_4")],
    )
    _stamp_like(t_pow_4, q)
    tail_4 = ctx.builder.Mul(
        coef4,
        t_pow_4,
        _outputs=[ctx.fresh_name(f"{name_prefix}_tail4")],
    )
    _stamp_like(tail_4, q)

    # + s(s+1)(s+2)(s+3)(s+4)/30240 * t^(-s-5)
    s3 = ctx.builder.Add(
        s,
        ctx.bind_const_for_var(object(), np.asarray(3.0, dtype=np_dtype)),
        _outputs=[ctx.fresh_name(f"{name_prefix}_s3")],
    )
    _stamp_like(s3, s)
    s4 = ctx.builder.Add(
        s,
        ctx.bind_const_for_var(object(), np.asarray(4.0, dtype=np_dtype)),
        _outputs=[ctx.fresh_name(f"{name_prefix}_s4")],
    )
    _stamp_like(s4, s)
    num5 = ctx.builder.Mul(
        num4,
        s3,
        _outputs=[ctx.fresh_name(f"{name_prefix}_num5a")],
    )
    _stamp_like(num5, s)
    num5 = ctx.builder.Mul(
        num5,
        s4,
        _outputs=[ctx.fresh_name(f"{name_prefix}_num5b")],
    )
    _stamp_like(num5, s)
    coef5 = ctx.builder.Div(
        num5,
        ctx.bind_const_for_var(object(), np.asarray(30240.0, dtype=np_dtype)),
        _outputs=[ctx.fresh_name(f"{name_prefix}_coef5")],
    )
    _stamp_like(coef5, s)
    neg_s_minus_5 = ctx.builder.Sub(
        neg_s,
        ctx.bind_const_for_var(object(), np.asarray(5.0, dtype=np_dtype)),
        _outputs=[ctx.fresh_name(f"{name_prefix}_neg_s_minus_5")],
    )
    _stamp_like(neg_s_minus_5, s)
    t_pow_5 = ctx.builder.Pow(
        t,
        neg_s_minus_5,
        _outputs=[ctx.fresh_name(f"{name_prefix}_t_pow_5")],
    )
    _stamp_like(t_pow_5, q)
    tail_5 = ctx.builder.Mul(
        coef5,
        t_pow_5,
        _outputs=[ctx.fresh_name(f"{name_prefix}_tail5")],
    )
    _stamp_like(tail_5, q)

    out = ctx.builder.Add(
        acc,
        tail_1,
        _outputs=[ctx.fresh_name(f"{name_prefix}_out_a")],
    )
    _stamp_like(out, q)
    out = ctx.builder.Add(
        out,
        tail_2,
        _outputs=[ctx.fresh_name(f"{name_prefix}_out_b")],
    )
    _stamp_like(out, q)
    out = ctx.builder.Add(
        out,
        tail_3,
        _outputs=[ctx.fresh_name(f"{name_prefix}_out_c")],
    )
    _stamp_like(out, q)
    out = ctx.builder.Add(
        out,
        tail_4,
        _outputs=[ctx.fresh_name(f"{name_prefix}_out_d")],
    )
    _stamp_like(out, q)
    out = ctx.builder.Add(
        out,
        tail_5,
        _outputs=[ctx.fresh_name(f"{name_prefix}_out")],
    )
    _stamp_like(out, q)
    return out


@register_primitive(
    jaxpr_primitive=jax.lax.zeta_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.zeta.html",
    onnx=[
        {"component": "Pow", "doc": "https://onnx.ai/onnx/operators/onnx__Pow.html"},
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
    ],
    since="0.12.1",
    context="primitives.lax",
    component="zeta",
    testcases=[
        {
            "testcase": "zeta_positive",
            "callable": lambda s, q: jax.lax.zeta(s, q),
            "input_values": [
                np.asarray([1.2, 2.0, 3.5], dtype=np.float32),
                np.asarray([0.7, 1.5, 4.0], dtype=np.float32),
            ],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(["Pow", "Where"], no_unused_inputs=True),
        },
        {
            "testcase": "zeta_broadcast",
            "callable": lambda s, q: jax.lax.zeta(s, q),
            "input_values": [
                np.asarray(2.0, dtype=np.float32),
                np.asarray([0.5, 1.0, 2.0, 5.0], dtype=np.float32),
            ],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(["Pow"], no_unused_inputs=True),
        },
    ],
)
class ZetaPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.zeta`` with an Euler-Maclaurin approximation."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        s_var, q_var = eqn.invars
        out_var = eqn.outvars[0]

        s = ctx.get_value_for_var(s_var, name_hint=ctx.fresh_name("zeta_s"))
        q = ctx.get_value_for_var(q_var, name_hint=ctx.fresh_name("zeta_q"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("zeta_out"))
        np_dtype = np.dtype(getattr(getattr(s_var, "aval", None), "dtype", np.float32))

        approx = _zeta_positive(ctx, s, q, np_dtype, "zeta")

        one = ctx.bind_const_for_var(object(), np.asarray(1.0, dtype=np_dtype))
        zero = ctx.bind_const_for_var(object(), np.asarray(0.0, dtype=np_dtype))
        inf_const = ctx.bind_const_for_var(object(), np.asarray(np.inf, dtype=np_dtype))
        nan_const = ctx.bind_const_for_var(object(), np.asarray(np.nan, dtype=np_dtype))

        s_gt_one = ctx.builder.Greater(
            s,
            one,
            _outputs=[ctx.fresh_name("zeta_s_gt_one")],
        )
        q_gt_zero = ctx.builder.Greater(
            q,
            zero,
            _outputs=[ctx.fresh_name("zeta_q_gt_zero")],
        )
        valid = ctx.builder.And(
            s_gt_one,
            q_gt_zero,
            _outputs=[ctx.fresh_name("zeta_valid")],
        )

        s_eq_one = ctx.builder.Equal(
            s,
            one,
            _outputs=[ctx.fresh_name("zeta_s_eq_one")],
        )
        invalid_or_pole = ctx.builder.Where(
            s_eq_one,
            inf_const,
            nan_const,
            _outputs=[ctx.fresh_name("zeta_invalid_or_pole")],
        )
        _stamp_like(invalid_or_pole, approx)

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("zeta")
        result = ctx.builder.Where(
            valid,
            approx,
            invalid_or_pole,
            _outputs=[desired_name],
        )
        _stamp_like(result, out_spec if getattr(out_spec, "type", None) else approx)
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
