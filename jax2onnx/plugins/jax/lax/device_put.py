from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import onnx_ir as ir

from jax2onnx.plugins._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.device_put_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.device_put.html#jax.device_put",
    onnx=[
        {
            "component": "Identity",
            "doc": "https://onnx.ai/onnx/operators/onnx__Identity.html",
        }
    ],
    since="v0.4.0",
    context="primitives.lax",
    component="device_put",
    testcases=[
        {
            "testcase": "device_put_array",
            "callable": lambda x: jax.device_put(x),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "device_put_scalar",
            "callable": lambda: jax.device_put(42),
            "input_shapes": [],
        },
    ],
)
class DevicePutPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.device_put`` to an IR Identity node."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        in_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        in_val = ctx.get_value_for_var(
            in_var, name_hint=ctx.fresh_name("device_put_in")
        )
        out_val = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("device_put_out")
        )

        ctx.add_node(
            ir.Node(
                op_type="Identity",
                domain="",
                inputs=[in_val],
                outputs=[out_val],
                name=ctx.fresh_name("Identity"),
            )
        )

        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        _stamp_type_and_shape(out_val, out_shape)
        _ensure_value_info(ctx, out_val)
