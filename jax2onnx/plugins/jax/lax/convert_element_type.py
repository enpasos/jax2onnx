# jax2onnx/plugins/jax/lax/convert_element_type.py

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
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
            "post_check_onnx_graph": EG(
                ["Cast:3"],
                no_unused_inputs=True,
            ),
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
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("convert_out")
        )

        target_dtype = _dtype_to_ir(
            np.dtype(out_var.aval.dtype), ctx.builder.enable_double_precision
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("convert_out")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("convert_out")

        result = ctx.builder.Cast(
            operand_val,
            _outputs=[desired_name],
            to=int(target_dtype.value),
        )
        result.type = ir.TensorType(target_dtype)
        result.shape = operand_val.shape
        _stamp_type_and_shape(result, tuple(getattr(out_var.aval, "shape", ())))
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)
