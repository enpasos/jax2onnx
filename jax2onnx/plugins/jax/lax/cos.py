# jax2onnx/plugins/jax/lax/cos.py

from typing import Any

from jax2onnx._compat.jax import JaxprEqn
import jax
import numpy as np

from jax2onnx.converter.typing_support import LoweringContextProtocol

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape


@register_primitive(
    jaxpr_primitive=jax.lax.cos_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.cos.html",
    onnx=[
        {
            "component": "Cos",
            "doc": "https://onnx.ai/onnx/operators/onnx__Cos.html",
        },
        {
            "component": "Sin",
            "doc": "https://onnx.ai/onnx/operators/onnx__Sin.html",
        },
    ],
    since="0.4.4",
    context="primitives.lax",
    component="cos",
    testcases=[
        {
            "testcase": "cos",
            "callable": lambda x: jax.lax.cos(x),
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                ["Cos:3", "Mul:3 -> Sin:3 -> Mul:3 -> Mul:3 -> Sub:3"],
                mode="any",
                no_unused_inputs=True,
            ),
        }
    ],
)
class CosPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("cos_in"))

        x_dtype: np.dtype[Any] = np.dtype(getattr(x_var.aval, "dtype", np.float32))
        if x_dtype == np.float64:
            # Older supported ORT versions lack DOUBLE Cos. The half-angle
            # identity avoids losing the pi/2 shift for large |x|. Cancellation
            # near cosine zeros limits relative accuracy, but the absolute
            # error stays small (unlike sin(x + pi/2) for large arguments).
            half = ctx.bind_const_for_var(object(), np.asarray(0.5, dtype=x_dtype))
            two = ctx.bind_const_for_var(object(), np.asarray(2.0, dtype=x_dtype))
            one = ctx.bind_const_for_var(object(), np.asarray(1.0, dtype=x_dtype))
            half_x = ctx.builder.Mul(
                x_val, half, _outputs=[ctx.fresh_name("cos_half_x")]
            )
            sin_half = ctx.builder.Sin(
                half_x, _outputs=[ctx.fresh_name("cos_sin_half")]
            )
            squared = ctx.builder.Mul(
                sin_half, sin_half, _outputs=[ctx.fresh_name("cos_sin_squared")]
            )
            twice_squared = ctx.builder.Mul(
                squared, two, _outputs=[ctx.fresh_name("cos_twice_sin_squared")]
            )
            result = ctx.builder.Sub(
                one, twice_squared, _outputs=[ctx.fresh_name("cos_via_sin")]
            )
            for value in (half_x, sin_half, squared, twice_squared, result):
                value.type = x_val.type
                _stamp_type_and_shape(value, getattr(x_var.aval, "shape", ()))
                _ensure_value_metadata(ctx, value)
            ctx.bind_value_for_var(out_var, result)
        else:
            out_spec = ctx.get_value_for_var(
                out_var, name_hint=ctx.fresh_name("cos_out")
            )
            result = ctx.builder.Cos(x_val, _outputs=[out_spec.name])
            result.type = out_spec.type
            result.shape = out_spec.shape
            ctx.bind_value_for_var(out_var, result)
