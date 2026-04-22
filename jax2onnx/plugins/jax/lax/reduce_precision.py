# jax2onnx/plugins/jax/lax/reduce_precision.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import numpy as np

from jax2onnx.ir_utils import ir_dtype_to_numpy
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _stamp_like(value: Any, ref: Any) -> None:
    if getattr(ref, "type", None) is not None:
        value.type = ref.type
    if getattr(ref, "shape", None) is not None:
        value.shape = ref.shape


def _native_format_bits(np_dtype: np.dtype) -> tuple[int, int] | None:
    if np_dtype == np.dtype(np.float16):
        return (5, 10)
    if np_dtype == np.dtype(np.float32):
        return (8, 23)
    if np_dtype == np.dtype(np.float64):
        return (11, 52)
    if np_dtype.name == "bfloat16":
        return (8, 7)
    return None


@register_primitive(
    jaxpr_primitive=jax.lax.reduce_precision_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_precision.html",
    onnx=[
        {"component": "Log", "doc": "https://onnx.ai/onnx/operators/onnx__Log.html"},
        {
            "component": "Floor",
            "doc": "https://onnx.ai/onnx/operators/onnx__Floor.html",
        },
        {"component": "Pow", "doc": "https://onnx.ai/onnx/operators/onnx__Pow.html"},
        {
            "component": "Round",
            "doc": "https://onnx.ai/onnx/operators/onnx__Round.html",
        },
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
    ],
    since="0.12.1",
    context="primitives.lax",
    component="reduce_precision",
    testcases=[
        {
            "testcase": "reduce_precision_mantissa10",
            "callable": lambda x: jax.lax.reduce_precision(
                x, exponent_bits=8, mantissa_bits=10
            ),
            "input_values": [
                np.asarray([0.3, 1.1, -3.14159, 10.0, 1000.0], dtype=np.float32)
            ],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Log", "Floor", "Pow", "Round", "Where"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reduce_precision_underflow_overflow",
            "callable": lambda x: jax.lax.reduce_precision(
                x, exponent_bits=5, mantissa_bits=10
            ),
            "input_values": [
                np.asarray([-1e-6, 1e-6, 1e4, -1e6, 1e6], dtype=np.float32)
            ],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Sign", "Where"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class ReducePrecisionPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.reduce_precision`` with scalar quantization arithmetic."""

    def lower(self, ctx: "IRContext", eqn: Any) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]
        params = dict(getattr(eqn, "params", {}) or {})

        exponent_bits = int(params.get("exponent_bits", 8))
        mantissa_bits = int(params.get("mantissa_bits", 23))
        if exponent_bits < 2:
            raise ValueError(
                f"reduce_precision requires exponent_bits>=2, got {exponent_bits}"
            )
        if mantissa_bits < 1:
            raise ValueError(
                f"reduce_precision requires mantissa_bits>=1, got {mantissa_bits}"
            )

        x = ctx.get_value_for_var(
            x_var, name_hint=ctx.fresh_name("reduce_precision_in")
        )
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("reduce_precision_out")
        )
        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
            "reduce_precision_out"
        )
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("reduce_precision_out")

        dtype_enum = getattr(getattr(x, "type", None), "dtype", None)
        np_dtype = ir_dtype_to_numpy(dtype_enum, default=None)
        if np_dtype is None:
            aval = getattr(x_var, "aval", None)
            np_dtype = np.dtype(getattr(aval, "dtype", np.float32))
        if not np.issubdtype(np_dtype, np.floating):
            raise TypeError(
                "reduce_precision only supports floating dtypes, " f"got '{np_dtype}'."
            )

        # Exact no-op when requested precision is not reduced for the input dtype.
        native_bits = _native_format_bits(np_dtype)
        if (
            native_bits is not None
            and exponent_bits >= native_bits[0]
            and mantissa_bits >= native_bits[1]
        ):
            result = ctx.builder.Identity(x, _outputs=[desired_name])
            _stamp_like(result, out_spec if getattr(out_spec, "type", None) else x)
            if getattr(out_spec, "shape", None) is not None:
                result.shape = out_spec.shape
            ctx.bind_value_for_var(out_var, result)
            return

        def _const(value: float) -> Any:
            return ctx.bind_const_for_var(object(), np.asarray(value, dtype=np_dtype))

        zero = _const(0.0)
        one = _const(1.0)
        two = _const(2.0)
        ln2 = _const(np.log(2.0))
        mantissa_bits_f = _const(float(mantissa_bits))

        abs_x = ctx.builder.Abs(x, _outputs=[ctx.fresh_name("reduce_precision_abs")])
        _stamp_like(abs_x, x)
        is_zero = ctx.builder.Equal(
            abs_x,
            zero,
            _outputs=[ctx.fresh_name("reduce_precision_is_zero")],
        )
        safe_abs = ctx.builder.Where(
            is_zero,
            one,
            abs_x,
            _outputs=[ctx.fresh_name("reduce_precision_safe_abs")],
        )
        _stamp_like(safe_abs, x)

        log_abs = ctx.builder.Log(
            safe_abs,
            _outputs=[ctx.fresh_name("reduce_precision_log_abs")],
        )
        _stamp_like(log_abs, x)
        log2_abs = ctx.builder.Div(
            log_abs,
            ln2,
            _outputs=[ctx.fresh_name("reduce_precision_log2_abs")],
        )
        _stamp_like(log2_abs, x)
        exponent = ctx.builder.Floor(
            log2_abs,
            _outputs=[ctx.fresh_name("reduce_precision_exponent")],
        )
        _stamp_like(exponent, x)

        step_exp = ctx.builder.Sub(
            exponent,
            mantissa_bits_f,
            _outputs=[ctx.fresh_name("reduce_precision_step_exp")],
        )
        _stamp_like(step_exp, x)
        step = ctx.builder.Pow(
            two,
            step_exp,
            _outputs=[ctx.fresh_name("reduce_precision_step")],
        )
        _stamp_like(step, x)

        scaled = ctx.builder.Div(
            x,
            step,
            _outputs=[ctx.fresh_name("reduce_precision_scaled")],
        )
        _stamp_like(scaled, x)
        rounded = ctx.builder.Round(
            scaled,
            _outputs=[ctx.fresh_name("reduce_precision_rounded")],
        )
        _stamp_like(rounded, x)
        quantized = ctx.builder.Mul(
            rounded,
            step,
            _outputs=[ctx.fresh_name("reduce_precision_quantized")],
        )
        _stamp_like(quantized, x)

        bias = (1 << (exponent_bits - 1)) - 1
        min_exponent = _const(float(1 - bias))
        max_exponent = _const(float(bias))

        sign_x = ctx.builder.Sign(x, _outputs=[ctx.fresh_name("reduce_precision_sign")])
        _stamp_like(sign_x, x)
        signed_zero = ctx.builder.Mul(
            sign_x,
            zero,
            _outputs=[ctx.fresh_name("reduce_precision_signed_zero")],
        )
        _stamp_like(signed_zero, x)
        signed_inf = ctx.builder.Mul(
            sign_x,
            _const(np.inf),
            _outputs=[ctx.fresh_name("reduce_precision_signed_inf")],
        )
        _stamp_like(signed_inf, x)

        is_under = ctx.builder.Less(
            exponent,
            min_exponent,
            _outputs=[ctx.fresh_name("reduce_precision_is_under")],
        )
        quantized = ctx.builder.Where(
            is_under,
            signed_zero,
            quantized,
            _outputs=[ctx.fresh_name("reduce_precision_under_applied")],
        )
        _stamp_like(quantized, x)

        is_over = ctx.builder.Greater(
            exponent,
            max_exponent,
            _outputs=[ctx.fresh_name("reduce_precision_is_over")],
        )
        quantized = ctx.builder.Where(
            is_over,
            signed_inf,
            quantized,
            _outputs=[ctx.fresh_name("reduce_precision_over_applied")],
        )
        _stamp_like(quantized, x)

        result = ctx.builder.Where(
            is_zero,
            x,
            quantized,
            _outputs=[desired_name],
        )
        _stamp_like(result, out_spec if getattr(out_spec, "type", None) else x)
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
