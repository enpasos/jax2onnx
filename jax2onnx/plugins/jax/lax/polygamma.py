# jax2onnx/plugins/jax/lax/polygamma.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import numpy as np

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax.digamma import _digamma_positive
from jax2onnx.plugins.jax.lax.lgamma import _lanczos_lgamma_positive
from jax2onnx.plugins.jax.lax.zeta import _zeta_positive
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _stamp_like(value: Any, ref: Any) -> None:
    if getattr(ref, "type", None) is not None:
        value.type = ref.type
    if getattr(ref, "shape", None) is not None:
        value.shape = ref.shape


@register_primitive(
    jaxpr_primitive=jax.lax.polygamma_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.polygamma.html",
    onnx=[
        {
            "component": "Round",
            "doc": "https://onnx.ai/onnx/operators/onnx__Round.html",
        },
        {
            "component": "Floor",
            "doc": "https://onnx.ai/onnx/operators/onnx__Floor.html",
        },
        {"component": "Pow", "doc": "https://onnx.ai/onnx/operators/onnx__Pow.html"},
        {"component": "Exp", "doc": "https://onnx.ai/onnx/operators/onnx__Exp.html"},
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
    ],
    since="0.12.1",
    context="primitives.lax",
    component="polygamma",
    testcases=[
        {
            "testcase": "polygamma_orders",
            "callable": lambda n, x: jax.lax.polygamma(n, x),
            "input_values": [
                np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
                np.asarray([0.5, 1.5, 3.0], dtype=np.float32),
            ],
            "run_only_f32_variant": True,
            "rtol_f32": 1e-4,
            "atol_f32": 1e-5,
            "post_check_onnx_graph": EG(
                ["Round", "Pow", "Where"], no_unused_inputs=True
            ),
        },
        {
            "testcase": "polygamma_zero_order",
            "callable": lambda n, x: jax.lax.polygamma(n, x),
            "input_values": [
                np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
                np.asarray([0.5, 1.0, 2.0], dtype=np.float32),
            ],
            "run_only_f32_variant": True,
            "rtol_f32": 2e-4,
            "atol_f32": 2e-5,
            "post_check_onnx_graph": EG(["Log", "Where"], no_unused_inputs=True),
        },
    ],
)
class PolyGammaPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.polygamma`` for integer order on x>0 via zeta identity."""

    def lower(self, ctx: "IRContext", eqn: Any) -> None:
        n_var, x_var = eqn.invars
        out_var = eqn.outvars[0]

        n = ctx.get_value_for_var(n_var, name_hint=ctx.fresh_name("polygamma_n"))
        x = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("polygamma_x"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("polygamma_out")
        )
        np_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(x_var, "aval", None), "dtype", np.float32)
        )

        zero = ctx.bind_const_for_var(object(), np.asarray(0.0, dtype=np_dtype))
        one = ctx.bind_const_for_var(object(), np.asarray(1.0, dtype=np_dtype))
        two = ctx.bind_const_for_var(object(), np.asarray(2.0, dtype=np_dtype))
        neg_one = ctx.bind_const_for_var(object(), np.asarray(-1.0, dtype=np_dtype))
        nan_const = ctx.bind_const_for_var(object(), np.asarray(np.nan, dtype=np_dtype))

        n_round = ctx.builder.Round(n, _outputs=[ctx.fresh_name("polygamma_n_round")])
        _stamp_like(n_round, n)
        n_is_int = ctx.builder.Equal(
            n,
            n_round,
            _outputs=[ctx.fresh_name("polygamma_n_is_int")],
        )
        n_lt_zero = ctx.builder.Less(
            n_round,
            zero,
            _outputs=[ctx.fresh_name("polygamma_n_lt_zero")],
        )
        n_ge_zero = ctx.builder.Not(
            n_lt_zero,
            _outputs=[ctx.fresh_name("polygamma_n_ge_zero")],
        )
        x_gt_zero = ctx.builder.Greater(
            x,
            zero,
            _outputs=[ctx.fresh_name("polygamma_x_gt_zero")],
        )
        valid = ctx.builder.And(
            n_is_int,
            n_ge_zero,
            _outputs=[ctx.fresh_name("polygamma_valid_n")],
        )
        valid = ctx.builder.And(
            valid,
            x_gt_zero,
            _outputs=[ctx.fresh_name("polygamma_valid")],
        )

        # n == 0 -> digamma(x), positive branch.
        n_is_zero = ctx.builder.Equal(
            n_round,
            zero,
            _outputs=[ctx.fresh_name("polygamma_n_is_zero")],
        )
        digamma_x = _digamma_positive(ctx, x, np_dtype, "polygamma_digamma")

        # n >= 1 -> (-1)^(n+1) * Gamma(n+1) * zeta(n+1, x)
        s = ctx.builder.Add(
            n_round,
            one,
            _outputs=[ctx.fresh_name("polygamma_s")],
        )
        _stamp_like(s, n_round)
        zeta_val = _zeta_positive(ctx, s, x, np_dtype, "polygamma_zeta")
        lgamma_s = _lanczos_lgamma_positive(ctx, s, np_dtype, "polygamma_lgamma")
        gamma_s = ctx.builder.Exp(
            lgamma_s,
            _outputs=[ctx.fresh_name("polygamma_gamma_s")],
        )
        _stamp_like(gamma_s, s)

        half_n = ctx.builder.Floor(
            ctx.builder.Div(
                n_round,
                two,
                _outputs=[ctx.fresh_name("polygamma_half_n_raw")],
            ),
            _outputs=[ctx.fresh_name("polygamma_half_n")],
        )
        _stamp_like(half_n, n_round)
        two_half_n = ctx.builder.Mul(
            half_n,
            two,
            _outputs=[ctx.fresh_name("polygamma_two_half_n")],
        )
        _stamp_like(two_half_n, n_round)
        n_is_even = ctx.builder.Equal(
            n_round,
            two_half_n,
            _outputs=[ctx.fresh_name("polygamma_n_is_even")],
        )
        sign = ctx.builder.Where(
            n_is_even,
            neg_one,
            one,
            _outputs=[ctx.fresh_name("polygamma_sign")],
        )
        _stamp_like(sign, n_round)

        out_n_ge_1 = ctx.builder.Mul(
            sign,
            gamma_s,
            _outputs=[ctx.fresh_name("polygamma_out_n_ge_1_a")],
        )
        _stamp_like(out_n_ge_1, n_round)
        out_n_ge_1 = ctx.builder.Mul(
            out_n_ge_1,
            zeta_val,
            _outputs=[ctx.fresh_name("polygamma_out_n_ge_1")],
        )
        _stamp_like(out_n_ge_1, n_round)

        out_valid = ctx.builder.Where(
            n_is_zero,
            digamma_x,
            out_n_ge_1,
            _outputs=[ctx.fresh_name("polygamma_out_valid")],
        )
        _stamp_like(out_valid, out_n_ge_1)

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("polygamma")
        result = ctx.builder.Where(valid, out_valid, nan_const, _outputs=[desired_name])
        _stamp_like(result, out_spec if getattr(out_spec, "type", None) else out_valid)
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
