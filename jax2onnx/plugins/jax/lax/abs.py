# jax2onnx/plugins/jax/lax/abs.py


from jax2onnx._compat.jax import JaxprEqn
import jax

from jax2onnx.converter.typing_support import LoweringContextProtocol

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._precision_utils import promote_float32_operand
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)


@register_native_primitive(
    jaxpr_primitive=jax.lax.abs_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.abs.html",
    onnx=[
        {
            "component": "Abs",
            "doc": "https://onnx.ai/onnx/operators/onnx__Abs.html",
        }
    ],
    since="0.5.0",
    context="primitives.lax",
    component="abs",
    testcases=[
        {
            "testcase": "abs",
            "callable": lambda x: jax.lax.abs(x),
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                ["Abs:3"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class AbsPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.require_value_for_var(x_var)
        out_spec = ctx.allocate_value_for_var(
            out_var, name_hint=ctx.fresh_name("abs_out")
        )
        x_val = promote_float32_operand(
            ctx, x_val, target_dtype=out_spec.dtype, name_hint="abs_input_double"
        )

        result = ctx.builder.op("Abs", [x_val], output=out_spec)
        ctx.bind_value_for_var(out_var, result)
