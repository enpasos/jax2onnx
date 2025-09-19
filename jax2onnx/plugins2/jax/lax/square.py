from typing import TYPE_CHECKING

import jax
import onnx_ir as ir

from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    pass


@register_primitive(
    jaxpr_primitive=jax.lax.square_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.square.html",
    onnx=[
        {
            "component": "Mul",
            "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html",
        }
    ],
    since="v0.2.0",
    context="primitives2.lax",
    component="square",
    testcases=[
        {
            "testcase": "square",
            "callable": lambda x: jax.lax.square(x),
            "input_shapes": [(3,)],
            "use_onnx_ir": True,
        }
    ],
)
class SquarePlugin(PrimitiveLeafPlugin):
    def lower(self, ctx, eqn):
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("square_in"))
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("square_out"))

        node = ir.Node(
            op_type="Mul",
            domain="",
            inputs=[x_val, x_val],
            outputs=[out_val],
            name=ctx.fresh_name("Mul"),
        )
        ctx.add_node(node)
