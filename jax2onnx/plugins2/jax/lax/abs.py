from typing import TYPE_CHECKING

import jax
import onnx_ir as ir

from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    pass  # type hints only


@register_primitive(
    jaxpr_primitive=jax.lax.abs_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.abs.html",
    onnx=[
        {
            "component": "Abs",
            "doc": "https://onnx.ai/onnx/operators/onnx__Abs.html",
        }
    ],
    since="v0.5.0",
    context="primitives2.lax",
    component="abs",
    testcases=[
        {
            "testcase": "abs",
            "callable": lambda x: jax.lax.abs(x),
            "input_shapes": [(3,)],
            "use_onnx_ir": True,
        }
    ],
)
class AbsPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx, eqn):
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("abs_in"))
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("abs_out"))

        node = ir.Node(
            op_type="Abs",
            domain="",
            inputs=[x_val],
            outputs=[out_val],
            name=ctx.fresh_name("abs"),
        )
        ctx.add_node(node)
