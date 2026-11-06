# jax2onnx/plugins/jax/lax/pow.py

from __future__ import annotations

from typing import Any, cast

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.ir_utils import numpy_dtype_to_ir
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _stamp_type_and_shape
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._utils import cast_value_to_dtype
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)


def lower_pow(ctx: LoweringContextProtocol, eqn: Any) -> None:
    base_var, exponent_var = eqn.invars
    out_var = eqn.outvars[0]

    base_val = ctx.get_value_for_var(base_var, name_hint=ctx.fresh_name("pow_base"))
    exp_val = ctx.get_value_for_var(exponent_var, name_hint=ctx.fresh_name("pow_exp"))
    out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("pow_out"))

    # lax.pow keeps the base dtype; jnp.power promotes both operands.
    target_dtype: np.dtype[Any] = np.dtype(out_var.aval.dtype)
    target_enum = numpy_dtype_to_ir(target_dtype)
    base_input = cast_value_to_dtype(
        ctx, base_val, target_enum, name_hint="pow_base_cast"
    )
    exp_input = cast_value_to_dtype(ctx, exp_val, target_enum, name_hint="pow_exp_cast")

    desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("pow_out")
    producer = getattr(out_spec, "producer", lambda: None)
    if callable(producer) and producer() is not None:
        desired_name = ctx.fresh_name("pow_out")

    result = cast(
        ir.Value, ctx.builder.Pow(base_input, exp_input, _outputs=[desired_name])
    )
    if getattr(out_spec, "type", None) is not None:
        result.type = out_spec.type
    else:
        result.type = ir.TensorType(target_enum)
    if getattr(out_spec, "shape", None) is not None:
        result.shape = out_spec.shape
    else:
        _stamp_type_and_shape(result, tuple(getattr(out_var.aval, "shape", ())))
    ctx.bind_value_for_var(out_var, result)


@register_native_primitive(
    jaxpr_primitive=jax.lax.pow_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.pow.html",
    onnx=[
        {
            "component": "Pow",
            "doc": "https://onnx.ai/onnx/operators/onnx__Pow.html",
        }
    ],
    since="0.8.2",
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

    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        lower_pow(ctx, eqn)
