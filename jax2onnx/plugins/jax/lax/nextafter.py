# jax2onnx/plugins/jax/lax/nextafter.py

from __future__ import annotations

from typing import Any, cast

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.ir_utils import ir_dtype_to_numpy
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


def _stamp_like(value: Any, ref: Any) -> None:
    if getattr(ref, "type", None) is not None:
        value.type = ref.type
    if getattr(ref, "shape", None) is not None:
        value.shape = ref.shape


def _float_layout(np_dtype: np.dtype[Any]) -> tuple[int, int, int] | None:
    # Returns (mantissa_bits, min_normal_exponent, max_exponent).
    if np_dtype == np.dtype(np.float16):
        return (10, -14, 15)
    if np_dtype == np.dtype(np.float32):
        return (23, -126, 127)
    if np_dtype == np.dtype(np.float64):
        return (52, -1022, 1023)
    if np_dtype.name == "bfloat16":
        return (7, -126, 127)
    return None


@register_primitive(
    jaxpr_primitive=jax.lax.nextafter_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.nextafter.html",
    onnx=[
        {"component": "Log", "doc": "https://onnx.ai/onnx/operators/onnx__Log.html"},
        {"component": "Pow", "doc": "https://onnx.ai/onnx/operators/onnx__Pow.html"},
        {"component": "Sign", "doc": "https://onnx.ai/onnx/operators/onnx__Sign.html"},
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
        {
            "component": "IsNaN",
            "doc": "https://onnx.ai/onnx/operators/onnx__IsNaN.html",
        },
    ],
    since="0.12.1",
    context="primitives.lax",
    component="nextafter",
    testcases=[
        {
            "testcase": "nextafter_vector",
            "callable": lambda x, y: jax.lax.nextafter(x, y),
            "input_values": [
                np.asarray([0.0, 1.0, -1.0, 10.0, -10.0], dtype=np.float32),
                np.asarray([1.0, 0.0, 0.0, -20.0, 20.0], dtype=np.float32),
            ],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Log", "Floor", "Pow", "Sign", "Where"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "nextafter_special_values",
            "callable": lambda x, y: jax.lax.nextafter(x, y),
            "input_values": [
                np.asarray(
                    [
                        np.inf,
                        -np.inf,
                        0.0,
                        -0.0,
                        np.finfo(np.float32).tiny,
                        -np.finfo(np.float32).tiny,
                    ],
                    dtype=np.float32,
                ),
                np.asarray([0.0, 0.0, -1.0, 1.0, 0.0, 0.0], dtype=np.float32),
            ],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["IsInf", "IsNaN", "Where"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class NextAfterPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.nextafter`` using ULP arithmetic (no bitcasts required)."""

    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        x_var, y_var = eqn.invars
        out_var = eqn.outvars[0]

        x = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("nextafter_x"))
        y = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("nextafter_y"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("nextafter_out")
        )
        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("nextafter")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("nextafter")

        dtype_enum = getattr(getattr(x, "type", None), "dtype", None)
        np_dtype = ir_dtype_to_numpy(dtype_enum, default=None)
        if np_dtype is None:
            aval = getattr(x_var, "aval", None)
            np_dtype = np.dtype(getattr(aval, "dtype", np.float32))
        layout = _float_layout(np_dtype)
        if layout is None:
            raise TypeError(
                f"nextafter only supports floating dtypes, got '{np_dtype}'"
            )
        mantissa_bits, min_exp, max_exp = layout

        def _const(value: float) -> ir.Value:
            const_value: ir.Value = ctx.bind_const_for_var(
                object(), np.asarray(value, dtype=np_dtype)
            )
            return const_value

        zero = _const(0.0)
        one = _const(1.0)
        two = _const(2.0)
        half = _const(0.5)
        log2 = _const(np.log(2.0))
        tiny_subnormal = _const(
            np.nextafter(
                np.asarray(0.0, dtype=np_dtype),
                np.asarray(1.0, dtype=np_dtype),
            )
        )
        min_normal = _const(np.finfo(np_dtype).tiny)
        max_finite = _const(np.finfo(np_dtype).max)
        mantissa_bits_f = _const(float(mantissa_bits))
        min_exp_f = _const(float(min_exp))
        max_exp_f = _const(float(max_exp))
        nan_const = _const(np.nan)

        abs_x = cast(
            ir.Value, ctx.builder.Abs(x, _outputs=[ctx.fresh_name("nextafter_abs_x")])
        )
        _stamp_like(abs_x, x)
        x_is_zero = cast(
            ir.Value,
            ctx.builder.Equal(
                abs_x,
                zero,
                _outputs=[ctx.fresh_name("nextafter_x_is_zero")],
            ),
        )
        safe_abs = cast(
            ir.Value,
            ctx.builder.Where(
                x_is_zero,
                one,
                abs_x,
                _outputs=[ctx.fresh_name("nextafter_safe_abs")],
            ),
        )
        _stamp_like(safe_abs, x)

        log_abs = cast(
            ir.Value,
            ctx.builder.Log(safe_abs, _outputs=[ctx.fresh_name("nextafter_log")]),
        )
        _stamp_like(log_abs, x)
        log2_abs = cast(
            ir.Value,
            ctx.builder.Div(
                log_abs,
                log2,
                _outputs=[ctx.fresh_name("nextafter_log2")],
            ),
        )
        _stamp_like(log2_abs, x)
        exponent = cast(
            ir.Value,
            ctx.builder.Floor(
                log2_abs,
                _outputs=[ctx.fresh_name("nextafter_exp")],
            ),
        )
        _stamp_like(exponent, x)

        pow_e = cast(
            ir.Value,
            ctx.builder.Pow(
                two, exponent, _outputs=[ctx.fresh_name("nextafter_pow_e")]
            ),
        )
        _stamp_like(pow_e, x)
        ulp_up_exp = cast(
            ir.Value,
            ctx.builder.Sub(
                exponent,
                mantissa_bits_f,
                _outputs=[ctx.fresh_name("nextafter_ulp_exp")],
            ),
        )
        _stamp_like(ulp_up_exp, x)
        ulp_up = cast(
            ir.Value,
            ctx.builder.Pow(
                two,
                ulp_up_exp,
                _outputs=[ctx.fresh_name("nextafter_ulp_up")],
            ),
        )
        _stamp_like(ulp_up, x)
        is_subnormal = cast(
            ir.Value,
            ctx.builder.Less(
                abs_x,
                min_normal,
                _outputs=[ctx.fresh_name("nextafter_is_sub")],
            ),
        )
        ulp = cast(
            ir.Value,
            ctx.builder.Where(
                is_subnormal,
                tiny_subnormal,
                ulp_up,
                _outputs=[ctx.fresh_name("nextafter_ulp")],
            ),
        )
        _stamp_like(ulp, x)

        dir_delta = cast(
            ir.Value,
            ctx.builder.Sub(y, x, _outputs=[ctx.fresh_name("nextafter_delta")]),
        )
        _stamp_like(dir_delta, x)
        dir_sign = cast(
            ir.Value,
            ctx.builder.Sign(
                dir_delta,
                _outputs=[ctx.fresh_name("nextafter_dir_sign")],
            ),
        )
        _stamp_like(dir_sign, x)

        toward_zero_metric = cast(
            ir.Value,
            ctx.builder.Mul(
                x,
                dir_sign,
                _outputs=[ctx.fresh_name("nextafter_toward_zero_metric")],
            ),
        )
        _stamp_like(toward_zero_metric, x)
        toward_zero = cast(
            ir.Value,
            ctx.builder.Less(
                toward_zero_metric,
                zero,
                _outputs=[ctx.fresh_name("nextafter_toward_zero")],
            ),
        )
        is_pow2 = cast(
            ir.Value,
            ctx.builder.Equal(
                abs_x,
                pow_e,
                _outputs=[ctx.fresh_name("nextafter_is_pow2")],
            ),
        )
        exp_gt_min = cast(
            ir.Value,
            ctx.builder.Greater(
                exponent,
                min_exp_f,
                _outputs=[ctx.fresh_name("nextafter_exp_gt_min")],
            ),
        )
        pow2_boundary = cast(
            ir.Value,
            ctx.builder.And(
                is_pow2,
                exp_gt_min,
                _outputs=[ctx.fresh_name("nextafter_pow2_boundary")],
            ),
        )
        use_half_ulp = cast(
            ir.Value,
            ctx.builder.And(
                toward_zero,
                pow2_boundary,
                _outputs=[ctx.fresh_name("nextafter_use_half_ulp")],
            ),
        )
        ulp_half = cast(
            ir.Value,
            ctx.builder.Mul(ulp, half, _outputs=[ctx.fresh_name("nextafter_ulp_half")]),
        )
        _stamp_like(ulp_half, x)
        ulp_eff = cast(
            ir.Value,
            ctx.builder.Where(
                use_half_ulp,
                ulp_half,
                ulp,
                _outputs=[ctx.fresh_name("nextafter_ulp_eff")],
            ),
        )
        _stamp_like(ulp_eff, x)

        step = cast(
            ir.Value,
            ctx.builder.Mul(
                dir_sign, ulp_eff, _outputs=[ctx.fresh_name("nextafter_step")]
            ),
        )
        _stamp_like(step, x)
        candidate = cast(
            ir.Value,
            ctx.builder.Add(x, step, _outputs=[ctx.fresh_name("nextafter_candidate")]),
        )
        _stamp_like(candidate, x)

        x_is_inf = cast(
            ir.Value,
            ctx.builder.IsInf(x, _outputs=[ctx.fresh_name("nextafter_x_is_inf")]),
        )
        x_sign = cast(
            ir.Value,
            ctx.builder.Sign(x, _outputs=[ctx.fresh_name("nextafter_x_sign")]),
        )
        _stamp_like(x_sign, x)
        inf_replacement = cast(
            ir.Value,
            ctx.builder.Mul(
                x_sign,
                max_finite,
                _outputs=[ctx.fresh_name("nextafter_inf_replacement")],
            ),
        )
        _stamp_like(inf_replacement, x)
        candidate = cast(
            ir.Value,
            ctx.builder.Where(
                x_is_inf,
                inf_replacement,
                candidate,
                _outputs=[ctx.fresh_name("nextafter_candidate_inf")],
            ),
        )
        _stamp_like(candidate, x)

        # Optional clamp: keeps approximation bounded for extreme exponents.
        is_over = cast(
            ir.Value,
            ctx.builder.Greater(
                exponent,
                max_exp_f,
                _outputs=[ctx.fresh_name("nextafter_is_over")],
            ),
        )
        candidate = cast(
            ir.Value,
            ctx.builder.Where(
                is_over,
                inf_replacement,
                candidate,
                _outputs=[ctx.fresh_name("nextafter_candidate_over")],
            ),
        )
        _stamp_like(candidate, x)

        x_eq_y = cast(
            ir.Value,
            ctx.builder.Equal(x, y, _outputs=[ctx.fresh_name("nextafter_x_eq_y")]),
        )
        result_no_nan = cast(
            ir.Value,
            ctx.builder.Where(
                x_eq_y,
                y,
                candidate,
                _outputs=[ctx.fresh_name("nextafter_no_nan")],
            ),
        )
        _stamp_like(result_no_nan, x)

        x_is_nan = cast(
            ir.Value,
            ctx.builder.IsNaN(x, _outputs=[ctx.fresh_name("nextafter_x_is_nan")]),
        )
        y_is_nan = cast(
            ir.Value,
            ctx.builder.IsNaN(y, _outputs=[ctx.fresh_name("nextafter_y_is_nan")]),
        )
        any_nan = cast(
            ir.Value,
            ctx.builder.Or(
                x_is_nan,
                y_is_nan,
                _outputs=[ctx.fresh_name("nextafter_any_nan")],
            ),
        )
        result = cast(
            ir.Value,
            ctx.builder.Where(
                any_nan, nan_const, result_no_nan, _outputs=[desired_name]
            ),
        )
        _stamp_like(result, out_spec if getattr(out_spec, "type", None) else x)
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
