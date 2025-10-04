# jax2onnx/plugins/jax/lax/tanh.py


import jax

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
import onnx_ir as ir


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
    context="primitives.lax",
    component="tanh",
    testcases=[
        {
            "testcase": "tanh",
            "callable": lambda x: jax.lax.tanh(x),
            "input_shapes": [(3,)],
        }
    ],
)
class TanhPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.tanh to ONNX Tanh."""

    # ─────────────────────────────────────────────────────────────────────────
    # IR path (converter): optional hook the new converter can call.
    # This keeps the old decorator & metadata (test discovery) unchanged.
    # The converter should look for `lower(ctx, eqn)` on the plugin.
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
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("tanh_in"))
        y_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("tanh_out"))

        node = ir.Node(
            op_type="Tanh",
            domain="",  # default ONNX domain
            inputs=[x_val],
            outputs=[y_val],
            name=ctx.fresh_name("tanh"),
        )
        ctx.add_node(node)
