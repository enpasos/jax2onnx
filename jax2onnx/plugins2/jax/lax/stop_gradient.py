from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import onnx_ir as ir

from jax2onnx.plugins2._ir_shapes import _stamp_type_and_shape, _ensure_value_info
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.stop_gradient_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.stop_gradient.html",
    onnx=[
        {
            "component": "Identity",
            "doc": "https://onnx.ai/onnx/operators/onnx__Identity.html",
        }
    ],
    since="v0.2.0",
    context="primitives2.lax",
    component="stop_gradient",
    testcases=[
        {
            "testcase": "stop_gradient_basic",
            "callable": lambda x: jax.lax.stop_gradient(x),
            "input_shapes": [(4,)],
            "use_onnx_ir": True,
        }
    ],
)
class StopGradientPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.stop_gradient`` to an ONNX Identity node."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        inp_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        inp_val = ctx.get_value_for_var(inp_var, name_hint=ctx.fresh_name("stop_in"))
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("stop_out"))

        node = ir.Node(
            op_type="Identity",
            domain="",
            inputs=[inp_val],
            outputs=[out_val],
            name=ctx.fresh_name("Identity"),
        )
        ctx.add_node(node)

        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        _stamp_type_and_shape(out_val, out_shape)
        _ensure_value_info(ctx, out_val)
