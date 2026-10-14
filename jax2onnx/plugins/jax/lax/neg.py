# jax2onnx/plugins/jax/lax/neg.py


from jax2onnx._compat.jax import JaxprEqn
import jax

from jax2onnx.converter.typing_support import LoweringContextProtocol

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


@register_primitive(
    jaxpr_primitive=jax.lax.neg_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.neg.html",
    onnx=[{"component": "Neg", "doc": "https://onnx.ai/onnx/operators/onnx__Neg.html"}],
    since="0.2.0",
    context="primitives.lax",
    component="neg",
    testcases=[
        {
            "testcase": "neg",
            "callable": lambda x: -x,
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                ["Neg:3"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class NegPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.require_value_for_var(x_var)
        out_spec = ctx.allocate_value_for_var(
            out_var, name_hint=ctx.fresh_name("neg_out")
        )

        result = ctx.builder.op("Neg", [x_val], output=out_spec)
        ctx.bind_value_for_var(out_var, result)
