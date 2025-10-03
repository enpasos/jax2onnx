from typing import TYPE_CHECKING
import jax
import onnx_ir as ir
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    pass  # hints


@register_primitive(
    jaxpr_primitive=jax.lax.neg_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.neg.html",
    onnx=[{"component": "Neg", "doc": "https://onnx.ai/onnx/operators/onnx__Neg.html"}],
    since="v0.2.0",
    context="primitives.lax",
    component="neg",
    testcases=[
        {
            "testcase": "neg",
            "callable": lambda x: -x,
            "input_shapes": [(3,)],
        }
    ],
)
class NegPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx, eqn):
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("neg_in"))
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("neg_out"))

        node = ir.Node(
            op_type="Neg",
            domain="",
            inputs=[x_val],
            outputs=[out_val],
            name=ctx.fresh_name("Neg"),
        )
        ctx.add_node(node)
