# jax2onnx/plugins/jax/core/name.py

from __future__ import annotations

from typing import TYPE_CHECKING

from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive="name",
    jax_doc="https://github.com/jax-ml/jax/blob/main/jax/_src/ad_checkpoint.py",
    onnx=[
        {
            "component": "Identity",
            "doc": "https://onnx.ai/onnx/operators/onnx__Identity.html",
        }
    ],
    since="0.5.0",
    context="primitives.core",
    component="name",
    testcases=[],
)
class NamePlugin(PrimitiveLeafPlugin):
    """Lower the ad_checkpoint name primitive to an ONNX Identity node."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        inp_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        inp_val = ctx.get_value_for_var(inp_var, name_hint=ctx.fresh_name("name_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("name_out"))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("Identity")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("Identity")

        result = ctx.builder.Identity(inp_val, _outputs=[desired_name])
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        else:
            result.type = getattr(inp_val, "type", None)
        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)
