# jax2onnx/plugins/jax/lax/stop_gradient.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _stamp_type_and_shape, _ensure_value_metadata
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


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
    context="primitives.lax",
    component="stop_gradient",
    testcases=[
        {
            "testcase": "stop_gradient",
            "callable": lambda x: jax.lax.stop_gradient(x),
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                ["Identity:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "stop_gradient_basic",
            "callable": lambda x: jax.lax.stop_gradient(x),
            "input_shapes": [(4,)],
            "post_check_onnx_graph": EG(
                ["Identity:4"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class StopGradientPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.stop_gradient`` to an ONNX Identity node."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        inp_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        inp_val = ctx.get_value_for_var(inp_var, name_hint=ctx.fresh_name("stop_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("stop_out"))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("Identity")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("Identity")

        result = ctx.builder.Identity(inp_val, _outputs=[desired_name])
        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        else:
            result.type = getattr(inp_val, "type", None)
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)
