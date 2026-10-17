# jax2onnx/plugins/jax/lax/sin.py


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
    jaxpr_primitive=jax.lax.sin_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.sin.html",
    onnx=[
        {
            "component": "Sin",
            "doc": "https://onnx.ai/onnx/operators/onnx__Sin.html",
        }
    ],
    since="0.4.4",
    context="primitives.lax",
    component="sin",
    testcases=[
        {
            "testcase": "sin",
            "callable": lambda x: jax.lax.sin(x),
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                ["Sin:3"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class SinPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.require_value_for_var(x_var)
        out_spec = ctx.allocate_value_for_var(
            out_var, name_hint=ctx.fresh_name("sin_out")
        )
        x_val = promote_float32_operand(
            ctx, x_val, target_dtype=out_spec.dtype, name_hint="sin_input_double"
        )

        result = ctx.builder.op("Sin", [x_val], output=out_spec)
        ctx.bind_value_for_var(out_var, result)
