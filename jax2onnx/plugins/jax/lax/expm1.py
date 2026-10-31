# jax2onnx/plugins/jax/lax/expm1.py


import jax
import numpy as np
from jax2onnx._compat.jax import JaxprEqn

from jax2onnx.converter.typing_support import NativeLoweringContext
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._precision_utils import promote_float32_operand
from jax2onnx.plugins.jax._log1p_expm1 import emit_expm1
from jax2onnx.plugins.jax.lax._source_elementary import lower_source_elementary
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)


@register_native_primitive(
    jaxpr_primitive=jax.lax.expm1_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.expm1.html",
    onnx=[
        {"component": "Exp", "doc": "https://onnx.ai/onnx/operators/onnx__Exp.html"},
        {"component": "Sub", "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html"},
        {"component": "Log", "doc": "https://onnx.ai/onnx/operators/onnx__Log.html"},
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {
            "component": "Equal",
            "doc": "https://onnx.ai/onnx/operators/onnx__Equal.html",
        },
        {"component": "Or", "doc": "https://onnx.ai/onnx/operators/onnx__Or.html"},
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
    ],
    since="0.12.1",
    context="primitives.lax",
    component="expm1",
    testcases=[
        {
            "testcase": "expm1",
            "callable": lambda x: jax.lax.expm1(x),
            # Exp(x) - 1 returns 0 or loses most digits for the small entries.
            "input_values": [
                np.array(
                    [-20.0, -1.0, -1e-3, -(2.0**-30), 0.0, 2.0**-30, 1e-6, 1e-3, 1.0],
                    dtype=np.float32,
                )
            ],
            "rtol_f32": 1e-6,
            "rtol_f64": 1e-12,
            "atol": 0.0,
            "post_check_onnx_graph": EG(
                ["Exp:9 -> Sub:9 -> Mul:9 -> Where:9 -> Where:9"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class Expm1Plugin(PrimitiveLeafPlugin):
    """Lower ``lax.expm1`` with a compensated ``Exp(x) - 1``."""

    def lower(self, ctx: NativeLoweringContext, eqn: JaxprEqn) -> None:
        if lower_source_elementary(ctx, eqn, operation="expm1"):
            return

        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.require_value_for_var(x_var)
        out_spec = ctx.allocate_value_for_var(
            out_var, name_hint=ctx.fresh_name("expm1_out")
        )
        x_val = promote_float32_operand(
            ctx, x_val, target_dtype=out_spec.dtype, name_hint="expm1_operand_double"
        )

        result = emit_expm1(ctx, x_val, name_hint="expm1", output=out_spec)
        ctx.bind_value_for_var(out_var, result)
