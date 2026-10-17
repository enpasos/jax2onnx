# jax2onnx/plugins/jax/lax/tanh.py


from jax2onnx._compat.jax import JaxprEqn
import jax

from jax2onnx.converter.typing_support import LoweringContextProtocol

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._precision_utils import promote_float32_operand
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)


@register_native_primitive(
    jaxpr_primitive=jax.lax.tanh_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.tanh.html",
    onnx=[
        {
            "component": "Tanh",
            "doc": "https://onnx.ai/onnx/operators/onnx__Tanh.html",
        }
    ],
    since="0.2.0",
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

        x_val = ctx.require_value_for_var(x_var)
        out_spec = ctx.allocate_value_for_var(
            out_var, name_hint=ctx.fresh_name("tanh_out")
        )
        x_val = promote_float32_operand(
            ctx, x_val, target_dtype=out_spec.dtype, name_hint="tanh_input_double"
        )

        result = ctx.builder.op("Tanh", [x_val], output=out_spec)
        ctx.bind_value_for_var(out_var, result)
