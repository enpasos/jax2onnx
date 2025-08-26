# file: jax2onnx/plugins2/jax/lax/tanh.py

from typing import TYPE_CHECKING

import jax
from onnx import helper

from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.tanh_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.tanh.html",
    onnx=[
        {
            "component": "Tanh",
            "doc": "https://onnx.ai/onnx/operators/onnx__Tanh.html",
        }
    ],
    since="v0.2.0",
    context="primitives2.lax",
    component="tanh",
    testcases=[
        {
            "testcase": "tanh",
            "callable": lambda x: jax.lax.tanh(x),
            "input_shapes": [(3,)],
            "use_onnx_ir": True
        }
    ],
)
class TanhPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.tanh to ONNX Tanh."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX tanh primitive."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])
        node = helper.make_node(
            "Tanh",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("tanh"),
        )
        s.add_node(node)

    # ─────────────────────────────────────────────────────────────────────────
    # IR path (converter2): optional hook the new converter can call.
    # This keeps the old decorator & metadata (test discovery) unchanged.
    # The converter2 should look for `lower(ctx, eqn)` on the plugin.
    # `ctx` is expected to offer:
    #   - get_value_for_var(var, name_hint: str | None = None) -> ir.Value
    #   - add_node(op_type: str, inputs: list[ir.Value], outputs: list[ir.Value], **attrs) -> None
    #   - fresh_name(prefix: str) -> str
    # `eqn` is a JAX jaxpr equation with `.invars` and `.outvars`.
    # ─────────────────────────────────────────────────────────────────────────
    def lower(self, ctx, eqn):
        """
        Lower a single jaxpr equation for tanh to onnx_ir:
            y = Tanh(x)
        """
        x_var = eqn.invars[0]
        y_var = eqn.outvars[0]

        # Map jaxpr vars to IR values (the ctx owns the value table)
        x_val = ctx.get_value_for_var(x_var)
        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("tanh_out"))

        # Emit the ONNX op into the IR graph
        ctx.add_node("Tanh", [x_val], [y_val])
