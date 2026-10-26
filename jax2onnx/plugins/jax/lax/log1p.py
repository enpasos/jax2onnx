# jax2onnx/plugins/jax/lax/log1p.py


from jax2onnx._compat.jax import JaxprEqn
import jax
import numpy as np

from jax2onnx.converter.typing_support import NativeLoweringContext
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._precision_utils import promote_float32_operand
from jax2onnx.plugins.jax._log1p_expm1 import emit_log1p
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)


@register_native_primitive(
    jaxpr_primitive=jax.lax.log1p_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.log1p.html",
    onnx=[
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
        {"component": "Log", "doc": "https://onnx.ai/onnx/operators/onnx__Log.html"},
        {"component": "Sub", "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html"},
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
        {
            "component": "Equal",
            "doc": "https://onnx.ai/onnx/operators/onnx__Equal.html",
        },
        {"component": "Or", "doc": "https://onnx.ai/onnx/operators/onnx__Or.html"},
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
    ],
    since="0.11.0",
    context="primitives.lax",
    component="log1p",
    testcases=[
        {
            "testcase": "log1p",
            "callable": lambda x: jax.lax.log1p(x),
            # Log(1 + x) returns 0 or loses most digits for the small entries.
            "input_values": [
                np.array(
                    [-0.5, -1e-3, -(2.0**-30), 0.0, 2.0**-30, 1e-6, 1e-3, 0.5, 3.0],
                    dtype=np.float32,
                )
            ],
            "rtol_f32": 1e-6,
            "rtol_f64": 1e-12,
            "atol": 0.0,
            "post_check_onnx_graph": EG(
                ["Add:9 -> Sub:9 -> Div:9 -> Where:9 -> Mul:9"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class Log1pPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.log1p`` with a compensated ``Log(1 + x)``."""

    def lower(self, ctx: NativeLoweringContext, eqn: JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.require_value_for_var(x_var)
        out_spec = ctx.allocate_value_for_var(
            out_var, name_hint=ctx.fresh_name("log1p_out")
        )
        x_val = promote_float32_operand(
            ctx, x_val, target_dtype=out_spec.dtype, name_hint="log1p_operand_double"
        )

        result = emit_log1p(ctx, x_val, name_hint="log1p", output=out_spec)
        ctx.bind_value_for_var(out_var, result)
