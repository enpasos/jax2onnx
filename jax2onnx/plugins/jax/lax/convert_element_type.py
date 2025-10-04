# jax2onnx/plugins/jax/lax/convert_element_type.py

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive="convert_element_type",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.convert_element_type.html",
    onnx=[
        {"component": "Cast", "doc": "https://onnx.ai/onnx/operators/onnx__Cast.html"}
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="convert_element_type",
    testcases=[
        {
            "testcase": "convert_element_type",
            "callable": lambda x: x.astype(np.int16),
            "input_shapes": [(3,)],
        }
    ],
)
class ConvertElementTypePlugin(PrimitiveLeafPlugin):
    """Lower ``lax.convert_element_type`` to a single ONNX Cast."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        operand_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        operand_val = ctx.get_value_for_var(
            operand_var, name_hint=ctx.fresh_name("convert_in")
        )
        out_val = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("convert_out")
        )

        target_dtype = _dtype_to_ir(
            np.dtype(out_var.aval.dtype), ctx.builder.enable_double_precision
        )

        ctx.add_node(
            ir.Node(
                op_type="Cast",
                domain="",
                inputs=[operand_val],
                outputs=[out_val],
                name=ctx.fresh_name("Cast"),
                attributes=[IRAttr("to", IRAttrType.INT, int(target_dtype.value))],
            )
        )

        out_val.type = ir.TensorType(target_dtype)
        out_val.dtype = target_dtype
        _stamp_type_and_shape(out_val, tuple(getattr(out_var.aval, "shape", ())))
        _ensure_value_info(ctx, out_val)
