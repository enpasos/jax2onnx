# jax2onnx/plugins/jax/lax/cumlogsumexp.py

from __future__ import annotations

from typing import Any, TYPE_CHECKING

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
    jaxpr_primitive=jax.lax.cumlogsumexp_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.cumlogsumexp.html",
    onnx=[
        {"component": "Exp", "doc": "https://onnx.ai/onnx/operators/onnx__Exp.html"},
        {
            "component": "CumSum",
            "doc": "https://onnx.ai/onnx/operators/onnx__CumSum.html",
        },
        {"component": "Log", "doc": "https://onnx.ai/onnx/operators/onnx__Log.html"},
    ],
    since="0.12.1",
    context="primitives.lax",
    component="cumlogsumexp",
    testcases=[
        {
            "testcase": "cumlogsumexp_axis1",
            "callable": lambda x: jax.lax.cumlogsumexp(x, axis=1),
            "input_values": [
                np.asarray(
                    [[-2.0, -0.5, 0.0, 0.25], [0.1, -0.3, 0.7, -0.2]],
                    dtype=np.float32,
                )
            ],
            "post_check_onnx_graph": EG(
                ["Exp:2x4 -> CumSum:2x4 -> Log:2x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "cumlogsumexp_reverse_last_axis",
            "callable": lambda x: jax.lax.cumlogsumexp(
                x, axis=x.ndim - 1, reverse=True
            ),
            "input_values": [
                np.asarray([[0.2, -0.1, 0.4, -0.2, 0.0]], dtype=np.float32)
            ],
            "post_check_onnx_graph": EG(
                ["Exp:1x5 -> CumSum:1x5 -> Log:1x5"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class CumLogSumExpPlugin(PrimitiveLeafPlugin):
    """IR-only lowering of ``lax.cumlogsumexp`` via ``Exp -> CumSum -> Log``."""

    def lower(self, ctx: "IRContext", eqn: Any) -> None:
        operand_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        params = getattr(eqn, "params", {})
        axis_param = int(params.get("axis", 0))
        reverse = bool(params.get("reverse", False))

        operand_val = ctx.get_value_for_var(
            operand_var, name_hint=ctx.fresh_name("cumlogsumexp_in")
        )
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("cumlogsumexp_out")
        )

        operand_shape = tuple(getattr(operand_var.aval, "shape", ()))
        rank = len(operand_shape)
        axis = axis_param % rank if rank > 0 and axis_param < 0 else axis_param
        axis_const = _const_i64(
            ctx, np.asarray(axis, dtype=np.int64), "cumlogsumexp_axis"
        )

        exp_val = ctx.builder.Exp(
            operand_val,
            _outputs=[ctx.fresh_name("cumlogsumexp_exp")],
        )
        if getattr(operand_val, "type", None) is not None:
            exp_val.type = operand_val.type
        if getattr(operand_val, "shape", None) is not None:
            exp_val.shape = operand_val.shape

        cumsum_val = ctx.builder.CumSum(
            exp_val,
            axis_const,
            exclusive=0,
            reverse=1 if reverse else 0,
            _outputs=[ctx.fresh_name("cumlogsumexp_cumsum")],
        )
        if getattr(operand_val, "type", None) is not None:
            cumsum_val.type = operand_val.type
        if getattr(operand_val, "shape", None) is not None:
            cumsum_val.shape = operand_val.shape

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
            "cumlogsumexp_out"
        )
        result = ctx.builder.Log(cumsum_val, _outputs=[desired_name])

        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        out_dtype_enum = _dtype_to_ir(
            np.dtype(getattr(out_var.aval, "dtype", operand_var.aval.dtype)),
            ctx.builder.enable_double_precision,
        )
        result.type = ir.TensorType(out_dtype_enum)
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)
