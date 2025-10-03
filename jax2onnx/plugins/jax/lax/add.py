# file: jax2onnx/plugins/jax/lax/add.py

from typing import TYPE_CHECKING, Optional
import jax
import numpy as np
import onnx_ir as ir  # IR builder (converter lane)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    pass  # for type hints


def lower_add(ctx, eqn) -> None:
    """Shared lowering routine for lax/jnp Add plugins."""

    x_var, y_var = eqn.invars
    out_var = eqn.outvars[0]

    prefer_dt: Optional[np.dtype] = np.dtype(getattr(x_var.aval, "dtype", np.float32))

    a_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("add_lhs"))
    b_val = ctx.get_value_for_var(
        y_var, name_hint=ctx.fresh_name("add_rhs"), prefer_np_dtype=prefer_dt
    )
    out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("add_out"))

    node = ir.Node(
        op_type="Add",
        domain="",
        inputs=[a_val, b_val],
        outputs=[out_val],
        name=ctx.fresh_name("add"),
    )
    ctx.add_node(node)


@register_primitive(
    jaxpr_primitive=jax.lax.add_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.add.html",
    onnx=[
        {
            "component": "Add",
            "doc": "https://onnx.ai/onnx/operators/onnx__Add.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="add",
    testcases=[
        {
            "testcase": "add",
            "callable": lambda x1, x2: x1 + x2,
            "input_shapes": [(3,), (3,)],
        },
        {
            "testcase": "add_const",
            "callable": lambda x: x + 1.0,
            "input_shapes": [(3,)],
        },
    ],
)
class AddPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx, eqn):
        lower_add(ctx, eqn)
