# jax2onnx/plugins/jax/lax/pow.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def lower_pow(ctx: "IRContext", eqn) -> None:  # type: ignore[name-defined]
    base_var, exponent_var = eqn.invars
    out_var = eqn.outvars[0]

    base_val = ctx.get_value_for_var(base_var, name_hint=ctx.fresh_name("pow_base"))
    exp_val = ctx.get_value_for_var(exponent_var, name_hint=ctx.fresh_name("pow_exp"))
    out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("pow_out"))

    target_dtype = np.dtype(getattr(base_var.aval, "dtype", np.float32))
    exp_dtype = np.dtype(getattr(exponent_var.aval, "dtype", target_dtype))

    if exp_dtype != target_dtype:
        cast_name = ctx.fresh_name("pow_exp_cast")
        cast_val = ctx.builder.Cast(
            exp_val,
            _outputs=[cast_name],
            to=int(
                _dtype_to_ir(target_dtype, ctx.builder.enable_double_precision).value
            ),
        )
        cast_val.shape = exp_val.shape
        cast_val.type = ir.TensorType(
            _dtype_to_ir(target_dtype, ctx.builder.enable_double_precision)
        )
        _stamp_type_and_shape(cast_val, tuple(getattr(exponent_var.aval, "shape", ())))
        _ensure_value_metadata(ctx, cast_val)
        exp_input = cast_val
    else:
        exp_input = exp_val

    desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("pow_out")
    producer = getattr(out_spec, "producer", lambda: None)
    if callable(producer) and producer() is not None:
        desired_name = ctx.fresh_name("pow_out")

    result = ctx.builder.Pow(base_val, exp_input, _outputs=[desired_name])
    if getattr(out_spec, "type", None) is not None:
        result.type = out_spec.type
    else:
        out_dtype_enum = _dtype_to_ir(target_dtype, ctx.builder.enable_double_precision)
        result.type = ir.TensorType(out_dtype_enum)
    if getattr(out_spec, "shape", None) is not None:
        result.shape = out_spec.shape
    else:
        _stamp_type_and_shape(result, tuple(getattr(out_var.aval, "shape", ())))
    ctx.bind_value_for_var(out_var, result)


@register_primitive(
    jaxpr_primitive=jax.lax.pow_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.pow.html",
    onnx=[
        {
            "component": "Pow",
            "doc": "https://onnx.ai/onnx/operators/onnx__Pow.html",
        }
    ],
    since="v0.8.2",
    context="primitives.lax",
    component="pow",
    testcases=[
        {
            "testcase": "pow_basic",
            "callable": lambda x, y: jax.lax.pow(x, y),
            "input_shapes": [(3,), (3,)],
            "post_check_onnx_graph": EG(
                ["Pow:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "pow_lax",
            "callable": lambda x, y: jax.lax.pow(x, y),
            "input_shapes": [(3,), (3,)],
            "post_check_onnx_graph": EG(
                ["Pow:3"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class PowPlugin(PrimitiveLeafPlugin):
    """Lower elementwise ``lax.pow`` to ONNX ``Pow`` with dtype harmonisation."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        lower_pow(ctx, eqn)
