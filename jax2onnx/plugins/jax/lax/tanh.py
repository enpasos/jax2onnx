# jax2onnx/plugins/jax/lax/tanh.py

from typing import Any

from jax import core
import jax

from jax2onnx.converter.typing_support import LoweringContextProtocol

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

JaxprEqn = getattr(core, "JaxprEqn", Any)


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
            "post_check_onnx_graph": EG(
                ["Tanh:3"],
                no_unused_inputs=True,
            ),
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
    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        """
        Lower a single jaxpr equation for tanh to onnx_ir:
            y = Tanh(x)
        """
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("tanh_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("tanh_out"))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("tanh_out")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("tanh_out")

        result = ctx.builder.Tanh(x_val, _outputs=[desired_name])
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
