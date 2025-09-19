from typing import TYPE_CHECKING

import jax
import onnx_ir as ir

from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    pass


@register_primitive(
    jaxpr_primitive=jax.lax.logistic_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.logistic.html",
    onnx=[
        {
            "component": "Sigmoid",
            "doc": "https://onnx.ai/onnx/operators/onnx__Sigmoid.html",
        }
    ],
    since="v0.7.2",
    context="primitives2.lax",
    component="logistic",
    testcases=[
        {
            "testcase": "lax_logistic_basic",
            "callable": jax.lax.logistic,
            "input_shapes": [(3, 4)],
            "use_onnx_ir": True,
        },
    ],
)
class LogisticPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx, eqn):
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("logistic_in"))
        out_val = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("logistic_out")
        )

        node = ir.Node(
            op_type="Sigmoid",
            domain="",
            inputs=[x_val],
            outputs=[out_val],
            name=ctx.fresh_name("Sigmoid"),
        )
        ctx.add_node(node)
