# jax2onnx/plugins/jax/lax/lgamma.py

from __future__ import annotations

from typing import Any, cast

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


def _stamp_like(value: ir.Value, ref: ir.Value) -> None:
    if getattr(ref, "type", None) is not None:
        value.type = ref.type
    if getattr(ref, "shape", None) is not None:
        value.shape = ref.shape


def _lanczos_lgamma_positive(
    ctx: LoweringContextProtocol,
    x: ir.Value,
    np_dtype: np.dtype[Any],
    name_prefix: str,
) -> ir.Value:
    # Lanczos approximation, g=7, n=9 (Numerical Recipes coefficients).
    coeffs = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]
    half = ctx.bind_const_for_var(object(), np.asarray(0.5, dtype=np_dtype))
    one = ctx.bind_const_for_var(object(), np.asarray(1.0, dtype=np_dtype))
    g_plus_half = ctx.bind_const_for_var(object(), np.asarray(7.5, dtype=np_dtype))
    log_sqrt_2pi = ctx.bind_const_for_var(
        object(), np.asarray(0.9189385332046727, dtype=np_dtype)
    )

    z = cast(
        ir.Value,
        ctx.builder.Sub(x, one, _outputs=[ctx.fresh_name(f"{name_prefix}_z")]),
    )
    _stamp_like(z, x)

    a = ctx.bind_const_for_var(object(), np.asarray(coeffs[0], dtype=np_dtype))
    for i, c in enumerate(coeffs[1:], start=1):
        zi = cast(
            ir.Value,
            ctx.builder.Add(
                z,
                ctx.bind_const_for_var(object(), np.asarray(float(i), dtype=np_dtype)),
                _outputs=[ctx.fresh_name(f"{name_prefix}_zplus_{i}")],
            ),
        )
        _stamp_like(zi, x)
        term = cast(
            ir.Value,
            ctx.builder.Div(
                ctx.bind_const_for_var(object(), np.asarray(c, dtype=np_dtype)),
                zi,
                _outputs=[ctx.fresh_name(f"{name_prefix}_term_{i}")],
            ),
        )
        _stamp_like(term, x)
        a = cast(
            ir.Value,
            ctx.builder.Add(
                a,
                term,
                _outputs=[ctx.fresh_name(f"{name_prefix}_a_{i}")],
            ),
        )
        _stamp_like(a, x)

    t = cast(
        ir.Value,
        ctx.builder.Add(
            z,
            g_plus_half,
            _outputs=[ctx.fresh_name(f"{name_prefix}_t")],
        ),
    )
    _stamp_like(t, x)
    log_t = cast(
        ir.Value,
        ctx.builder.Log(t, _outputs=[ctx.fresh_name(f"{name_prefix}_log_t")]),
    )
    _stamp_like(log_t, x)
    zph = cast(
        ir.Value,
        ctx.builder.Add(
            z,
            half,
            _outputs=[ctx.fresh_name(f"{name_prefix}_zph")],
        ),
    )
    _stamp_like(zph, x)
    zph_logt = cast(
        ir.Value,
        ctx.builder.Mul(
            zph,
            log_t,
            _outputs=[ctx.fresh_name(f"{name_prefix}_zph_logt")],
        ),
    )
    _stamp_like(zph_logt, x)
    log_a = cast(
        ir.Value,
        ctx.builder.Log(a, _outputs=[ctx.fresh_name(f"{name_prefix}_log_a")]),
    )
    _stamp_like(log_a, x)

    out = cast(
        ir.Value,
        ctx.builder.Add(
            log_sqrt_2pi,
            zph_logt,
            _outputs=[ctx.fresh_name(f"{name_prefix}_tmp_add")],
        ),
    )
    _stamp_like(out, x)
    out = cast(
        ir.Value,
        ctx.builder.Sub(out, t, _outputs=[ctx.fresh_name(f"{name_prefix}_tmp_sub")]),
    )
    _stamp_like(out, x)
    out = cast(
        ir.Value,
        ctx.builder.Add(
            out,
            log_a,
            _outputs=[ctx.fresh_name(f"{name_prefix}_out")],
        ),
    )
    _stamp_like(out, x)
    return out


@register_primitive(
    jaxpr_primitive=jax.lax.lgamma_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.lgamma.html",
    onnx=[
        {"component": "Log", "doc": "https://onnx.ai/onnx/operators/onnx__Log.html"},
        {"component": "Sin", "doc": "https://onnx.ai/onnx/operators/onnx__Sin.html"},
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
    ],
    since="0.12.1",
    context="primitives.lax",
    component="lgamma",
    testcases=[
        {
            "testcase": "lgamma_positive",
            "callable": lambda x: jax.lax.lgamma(x),
            "input_values": [np.asarray([0.2, 0.5, 1.0, 2.5, 5.0], dtype=np.float32)],
            "post_check_onnx_graph": EG(["Log", "Where"], no_unused_inputs=True),
        },
        {
            "testcase": "lgamma_negative_noninteger",
            "callable": lambda x: jax.lax.lgamma(x),
            "input_values": [np.asarray([-2.5, -1.3, -0.5, 0.2], dtype=np.float32)],
            "post_check_onnx_graph": EG(["Sin"], no_unused_inputs=True),
        },
    ],
)
class LGammaPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.lgamma`` via Lanczos + reflection formula."""

    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("lgamma_in"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("lgamma_out")
        )
        np_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(x_var, "aval", None), "dtype", np.float32)
        )

        half = ctx.bind_const_for_var(object(), np.asarray(0.5, dtype=np_dtype))
        one = ctx.bind_const_for_var(object(), np.asarray(1.0, dtype=np_dtype))
        pi = ctx.bind_const_for_var(object(), np.asarray(np.pi, dtype=np_dtype))
        log_pi = ctx.bind_const_for_var(
            object(), np.asarray(np.log(np.pi), dtype=np_dtype)
        )

        pos = _lanczos_lgamma_positive(ctx, x, np_dtype, "lgamma_pos")

        one_minus_x = cast(
            ir.Value,
            ctx.builder.Sub(one, x, _outputs=[ctx.fresh_name("lgamma_1mx")]),
        )
        _stamp_like(one_minus_x, x)
        lg_1mx = _lanczos_lgamma_positive(ctx, one_minus_x, np_dtype, "lgamma_refl")

        pi_x = cast(
            ir.Value,
            ctx.builder.Mul(pi, x, _outputs=[ctx.fresh_name("lgamma_pi_x")]),
        )
        _stamp_like(pi_x, x)
        sin_pi_x = cast(
            ir.Value,
            ctx.builder.Sin(pi_x, _outputs=[ctx.fresh_name("lgamma_sin_pi_x")]),
        )
        _stamp_like(sin_pi_x, x)
        abs_sin = cast(
            ir.Value,
            ctx.builder.Abs(sin_pi_x, _outputs=[ctx.fresh_name("lgamma_abs_sin")]),
        )
        _stamp_like(abs_sin, x)
        log_abs_sin = cast(
            ir.Value,
            ctx.builder.Log(
                abs_sin,
                _outputs=[ctx.fresh_name("lgamma_log_abs_sin")],
            ),
        )
        _stamp_like(log_abs_sin, x)

        refl = cast(
            ir.Value,
            ctx.builder.Sub(
                log_pi,
                log_abs_sin,
                _outputs=[ctx.fresh_name("lgamma_refl_sub1")],
            ),
        )
        _stamp_like(refl, x)
        refl = cast(
            ir.Value,
            ctx.builder.Sub(
                refl,
                lg_1mx,
                _outputs=[ctx.fresh_name("lgamma_refl")],
            ),
        )
        _stamp_like(refl, x)

        use_refl = cast(
            ir.Value,
            ctx.builder.Less(
                x,
                half,
                _outputs=[ctx.fresh_name("lgamma_use_refl")],
            ),
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("lgamma")
        result = cast(
            ir.Value,
            ctx.builder.Where(use_refl, refl, pos, _outputs=[desired_name]),
        )
        _stamp_like(result, out_spec if getattr(out_spec, "type", None) else x)
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
