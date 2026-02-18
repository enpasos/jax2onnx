# jax2onnx/plugins/jax/lax/cumprod.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.cumprod_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.cumprod.html",
    onnx=[
        {
            "component": "CumProd",
            "doc": "https://onnx.ai/onnx/operators/onnx__CumProd.html",
        }
    ],
    since="0.12.1",
    context="primitives.lax",
    component="cumprod",
    testcases=[
        {
            "testcase": "cumprod_i32_axis2",
            "callable": lambda x: jax.lax.cumprod(x, axis=2),
            "input_shapes": [(2, 3, 4)],
            "input_dtypes": [np.int32],
            "skip_numeric_validation": True,
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "CumProd:2x3x4",
                        "inputs": {1: {"const": 2.0}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "cumprod_f32_axism1_reverse",
            "callable": lambda x: jax.lax.cumprod(x, axis=x.ndim - 1, reverse=True),
            "input_shapes": [(1, 2, 3, 4)],
            "input_dtypes": [np.float32],
            "skip_numeric_validation": True,
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "CumProd:1x2x3x4",
                        "inputs": {1: {"const": 3.0}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
    ],
)
class CumProdPlugin(PrimitiveLeafPlugin):
    """IR-only lowering of ``lax.cumprod`` via ONNX ``CumProd``."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        operand_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        params = getattr(eqn, "params", {})
        axis_param = int(params.get("axis", 0))
        reverse = bool(params.get("reverse", False))

        operand_val = ctx.get_value_for_var(
            operand_var, name_hint=ctx.fresh_name("cumprod_in")
        )
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("cumprod_out")
        )

        operand_shape = tuple(getattr(operand_var.aval, "shape", ()))
        rank = len(operand_shape)
        axis = axis_param % rank if rank > 0 and axis_param < 0 else axis_param

        axis_const = _const_i64(ctx, np.asarray(axis, dtype=np.int64), "cumprod_axis")

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("CumProd")
        result = ctx.builder.CumProd(
            operand_val,
            axis_const,
            exclusive=0,
            reverse=1 if reverse else 0,
            _outputs=[desired_name],
        )

        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        out_dtype_enum = _dtype_to_ir(
            np.dtype(getattr(out_var.aval, "dtype", operand_var.aval.dtype)),
            ctx.builder.enable_double_precision,
        )
        result.type = ir.TensorType(out_dtype_enum)
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)
