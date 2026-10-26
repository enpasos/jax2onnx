# jax2onnx/plugins/jax/lax/sinh.py

import jax
import numpy as np
import onnx_ir as ir
from jax2onnx._compat.jax import JaxprEqn

from jax2onnx.converter.typing_support import NativeLoweringContext
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._precision_utils import promote_float32_operand
from jax2onnx.plugins.jax._log1p_expm1 import emit_sinh
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)


@register_native_primitive(
    jaxpr_primitive=jax.lax.sinh_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.sinh.html",
    onnx=[
        {
            "component": "Sinh",
            "doc": "https://onnx.ai/onnx/operators/onnx__Sinh.html",
        },
        {"component": "Abs", "doc": "https://onnx.ai/onnx/operators/onnx__Abs.html"},
        {"component": "Exp", "doc": "https://onnx.ai/onnx/operators/onnx__Exp.html"},
        {"component": "Sub", "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html"},
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
        {"component": "Log", "doc": "https://onnx.ai/onnx/operators/onnx__Log.html"},
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {
            "component": "Equal",
            "doc": "https://onnx.ai/onnx/operators/onnx__Equal.html",
        },
        {"component": "Less", "doc": "https://onnx.ai/onnx/operators/onnx__Less.html"},
        {"component": "Or", "doc": "https://onnx.ai/onnx/operators/onnx__Or.html"},
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
    ],
    since="0.4.4",
    context="primitives.lax",
    component="sinh",
    testcases=[
        {
            "testcase": "sinh",
            "callable": lambda x: jax.lax.sinh(x),
            # (Exp(x) - Exp(-x)) / 2 returns 0 or loses most digits for the
            # small entries.
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
                ["Sinh:9", "Abs:9 -> Exp:9 -> Sub:9 -> Mul:9 -> Where:9 -> Where:9"],
                mode="any",
                no_unused_inputs=True,
            ),
        }
    ],
)
class SinhPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.sinh`` to ``Sinh``, or a compensated form without a kernel."""

    def lower(self, ctx: NativeLoweringContext, eqn: JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.require_value_for_var(x_var)
        out_spec = ctx.allocate_value_for_var(
            out_var, name_hint=ctx.fresh_name("sinh_out")
        )
        x_val = promote_float32_operand(
            ctx, x_val, target_dtype=out_spec.dtype, name_hint="sinh_operand_double"
        )

        # ONNX Runtime has no float64 or bfloat16 Sinh kernel.
        if out_spec.dtype == ir.DataType.FLOAT:
            result = ctx.builder.op("Sinh", [x_val], output=out_spec)
        else:
            result = emit_sinh(ctx, x_val, name_hint="sinh", output=out_spec)
        ctx.bind_value_for_var(out_var, result)
