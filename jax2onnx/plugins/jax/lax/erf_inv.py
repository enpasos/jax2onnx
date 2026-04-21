# jax2onnx/plugins/jax/lax/erf_inv.py


import jax
import numpy as np
import onnx_ir as ir
from typing import Any

from jax import core

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.ir_utils import ir_dtype_to_numpy
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


@register_primitive(
    jaxpr_primitive=jax.lax.erf_inv_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.erf_inv.html",
    onnx=[
        {"component": "Log", "doc": "https://onnx.ai/onnx/operators/onnx__Log.html"},
        {
            "component": "Sqrt",
            "doc": "https://onnx.ai/onnx/operators/onnx__Sqrt.html",
        },
        {"component": "Erf", "doc": "https://onnx.ai/onnx/operators/onnx__Erf.html"},
        {"component": "Exp", "doc": "https://onnx.ai/onnx/operators/onnx__Exp.html"},
    ],
    since="0.12.1",
    context="primitives.lax",
    component="erf_inv",
    testcases=[
        {
            "testcase": "erf_inv_midrange",
            "callable": lambda x: jax.lax.erf_inv(x),
            "input_values": [
                np.asarray([-0.9, -0.5, -0.1, 0.0, 0.2, 0.5, 0.9], dtype=np.float32)
            ],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "erf_inv_matrix",
            "callable": lambda x: jax.lax.erf_inv(x),
            "input_values": [
                np.asarray(
                    [[-0.75, -0.25, 0.0], [0.1, 0.3, 0.8]],
                    dtype=np.float32,
                )
            ],
            "run_only_f32_variant": True,
        },
    ],
)
class ErfInvPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.erf_inv`` using Winitzki init + Newton refinement."""

    @staticmethod
    def _stamp_like(value: Any, ref: Any, *, dtype: ir.DataType | None = None) -> None:
        if dtype is not None:
            value.type = ir.TensorType(dtype)
        elif getattr(ref, "type", None) is not None:
            value.type = ref.type
        if getattr(ref, "shape", None) is not None:
            value.shape = ref.shape

    def lower(self, ctx: LoweringContextProtocol, eqn: "core.JaxprEqn") -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("erf_inv_in"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("erf_inv_out")
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("erf_inv_out")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("erf_inv_out")

        dtype_enum = getattr(getattr(x_val, "type", None), "dtype", None)
        np_dtype = ir_dtype_to_numpy(dtype_enum, default=None)
        if np_dtype is None:
            aval = getattr(x_var, "aval", None)
            np_dtype = np.dtype(getattr(aval, "dtype", np.float32))
        ir_dtype = ir.DataType.FLOAT if np_dtype == np.float32 else ir.DataType.DOUBLE

        # Winitzki approximation constants.
        a = ctx.bind_const_for_var(object(), np.asarray(0.147, dtype=np_dtype))
        one = ctx.bind_const_for_var(object(), np.asarray(1.0, dtype=np_dtype))
        half = ctx.bind_const_for_var(object(), np.asarray(0.5, dtype=np_dtype))
        two_over_pi_a = ctx.bind_const_for_var(
            object(), np.asarray(2.0 / (np.pi * 0.147), dtype=np_dtype)
        )
        two_over_sqrt_pi = ctx.bind_const_for_var(
            object(), np.asarray(2.0 / np.sqrt(np.pi), dtype=np_dtype)
        )

        y_sq = ctx.builder.Mul(x_val, x_val, _outputs=[ctx.fresh_name("erf_inv_y_sq")])
        self._stamp_like(y_sq, x_val, dtype=ir_dtype)

        one_minus_y_sq = ctx.builder.Sub(
            one,
            y_sq,
            _outputs=[ctx.fresh_name("erf_inv_one_minus_y_sq")],
        )
        self._stamp_like(one_minus_y_sq, x_val, dtype=ir_dtype)

        ln_term = ctx.builder.Log(
            one_minus_y_sq,
            _outputs=[ctx.fresh_name("erf_inv_ln_term")],
        )
        self._stamp_like(ln_term, x_val, dtype=ir_dtype)

        half_ln = ctx.builder.Mul(
            half,
            ln_term,
            _outputs=[ctx.fresh_name("erf_inv_half_ln")],
        )
        self._stamp_like(half_ln, x_val, dtype=ir_dtype)

        t = ctx.builder.Add(
            two_over_pi_a,
            half_ln,
            _outputs=[ctx.fresh_name("erf_inv_t")],
        )
        self._stamp_like(t, x_val, dtype=ir_dtype)

        t_sq = ctx.builder.Mul(t, t, _outputs=[ctx.fresh_name("erf_inv_t_sq")])
        self._stamp_like(t_sq, x_val, dtype=ir_dtype)

        ln_over_a = ctx.builder.Div(
            ln_term,
            a,
            _outputs=[ctx.fresh_name("erf_inv_ln_over_a")],
        )
        self._stamp_like(ln_over_a, x_val, dtype=ir_dtype)

        inside_outer = ctx.builder.Sub(
            t_sq,
            ln_over_a,
            _outputs=[ctx.fresh_name("erf_inv_inside_outer")],
        )
        self._stamp_like(inside_outer, x_val, dtype=ir_dtype)

        sqrt_outer = ctx.builder.Sqrt(
            inside_outer,
            _outputs=[ctx.fresh_name("erf_inv_sqrt_outer")],
        )
        self._stamp_like(sqrt_outer, x_val, dtype=ir_dtype)

        inside_inner = ctx.builder.Sub(
            sqrt_outer,
            t,
            _outputs=[ctx.fresh_name("erf_inv_inside_inner")],
        )
        self._stamp_like(inside_inner, x_val, dtype=ir_dtype)

        x0_abs = ctx.builder.Sqrt(
            inside_inner,
            _outputs=[ctx.fresh_name("erf_inv_x0_abs")],
        )
        self._stamp_like(x0_abs, x_val, dtype=ir_dtype)

        sign_y = ctx.builder.Sign(x_val, _outputs=[ctx.fresh_name("erf_inv_sign")])
        self._stamp_like(sign_y, x_val, dtype=ir_dtype)

        x_curr = ctx.builder.Mul(
            sign_y,
            x0_abs,
            _outputs=[ctx.fresh_name("erf_inv_x0")],
        )
        self._stamp_like(x_curr, x_val, dtype=ir_dtype)

        # Refine the estimate with Newton iterations.
        for i in range(3):
            erf_x = ctx.builder.Erf(
                x_curr,
                _outputs=[ctx.fresh_name(f"erf_inv_erf_{i}")],
            )
            self._stamp_like(erf_x, x_val, dtype=ir_dtype)

            numerator = ctx.builder.Sub(
                erf_x,
                x_val,
                _outputs=[ctx.fresh_name(f"erf_inv_num_{i}")],
            )
            self._stamp_like(numerator, x_val, dtype=ir_dtype)

            x_sq = ctx.builder.Mul(
                x_curr,
                x_curr,
                _outputs=[ctx.fresh_name(f"erf_inv_x_sq_{i}")],
            )
            self._stamp_like(x_sq, x_val, dtype=ir_dtype)

            neg_x_sq = ctx.builder.Neg(
                x_sq,
                _outputs=[ctx.fresh_name(f"erf_inv_neg_x_sq_{i}")],
            )
            self._stamp_like(neg_x_sq, x_val, dtype=ir_dtype)

            exp_term = ctx.builder.Exp(
                neg_x_sq,
                _outputs=[ctx.fresh_name(f"erf_inv_exp_{i}")],
            )
            self._stamp_like(exp_term, x_val, dtype=ir_dtype)

            denom = ctx.builder.Mul(
                two_over_sqrt_pi,
                exp_term,
                _outputs=[ctx.fresh_name(f"erf_inv_denom_{i}")],
            )
            self._stamp_like(denom, x_val, dtype=ir_dtype)

            delta = ctx.builder.Div(
                numerator,
                denom,
                _outputs=[ctx.fresh_name(f"erf_inv_delta_{i}")],
            )
            self._stamp_like(delta, x_val, dtype=ir_dtype)

            x_curr = ctx.builder.Sub(
                x_curr,
                delta,
                _outputs=[ctx.fresh_name(f"erf_inv_x_{i + 1}")],
            )
            self._stamp_like(x_curr, x_val, dtype=ir_dtype)

        result = ctx.builder.Identity(x_curr, _outputs=[desired_name])
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        elif getattr(x_val, "type", None) is not None:
            result.type = x_val.type
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        elif getattr(x_val, "shape", None) is not None:
            result.shape = x_val.shape
        ctx.bind_value_for_var(out_var, result)
