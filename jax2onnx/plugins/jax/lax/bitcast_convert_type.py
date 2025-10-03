# file: jax2onnx/plugins/jax/lax/bitcast_convert_type.py

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive="bitcast_convert_type",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.bitcast_convert_type.html",
    onnx=[
        {
            "component": "Bitcast",
            "doc": "https://onnx.ai/onnx/operators/onnx__Bitcast.html",
        }
    ],
    since="v0.7.2",
    context="primitives.lax",
    component="bitcast_convert_type",
    testcases=[],
)
class BitcastConvertTypePlugin(PrimitiveLeafPlugin):
    """Lower ``lax.bitcast_convert_type`` via ONNX Bitcast."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        operand_var = eqn.invars[0]
        out_var = eqn.outvars[0]
        target_dtype = _dtype_to_ir(
            np.dtype(eqn.params.get("new_dtype", out_var.aval.dtype)),
            ctx.builder.enable_double_precision,
        )

        operand_val = ctx.get_value_for_var(
            operand_var, name_hint=ctx.fresh_name("bitcast_in")
        )
        out_val = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("bitcast_out")
        )

        ctx.add_node(
            ir.Node(
                op_type="Bitcast",
                domain="",
                inputs=[operand_val],
                outputs=[out_val],
                name=ctx.fresh_name("Bitcast"),
                attributes=[IRAttr("to", IRAttrType.INT, int(target_dtype.value))],
            )
        )
