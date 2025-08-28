from typing import TYPE_CHECKING, Optional
import jax
import numpy as np
import onnx_ir as ir
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    pass  # hints


@register_primitive(
    jaxpr_primitive=jax.lax.mul_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.mul.html",
    onnx=[{"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"}],
    since="v0.2.0",
    context="primitives2.lax",
    component="mul",
    testcases=[
        {
            "testcase": "mul",
            "callable": lambda x1, x2: x1 * x2,
            "input_shapes": [(3,), (3,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "mul_const",
            "callable": lambda x: x * 0.5,
            "input_shapes": [(3,)],
            "use_onnx_ir": True,
        },
    ],
)
class MulPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx, eqn):
        x_var, y_var = eqn.invars
        out_var = eqn.outvars[0]

        prefer_dt: Optional[np.dtype] = np.dtype(
            getattr(x_var.aval, "dtype", np.float32)
        )
        a_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("mul_lhs"))
        b_val = ctx.get_value_for_var(
            y_var, name_hint=ctx.fresh_name("mul_rhs"), prefer_np_dtype=prefer_dt
        )
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("mul_out"))

        node = ir.Node(
            op_type="Mul",
            domain="",
            inputs=[a_val, b_val],
            outputs=[out_val],
            name=ctx.fresh_name("Mul"),
        )
        ctx.add_node(node)
