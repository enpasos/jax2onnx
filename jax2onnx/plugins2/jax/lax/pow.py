from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.converter2.ir_builder import _dtype_to_ir
from jax2onnx.plugins2._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


def lower_pow(ctx: "IRContext", eqn) -> None:  # type: ignore[name-defined]
    base_var, exponent_var = eqn.invars
    out_var = eqn.outvars[0]

    base_val = ctx.get_value_for_var(base_var, name_hint=ctx.fresh_name("pow_base"))
    exp_val = ctx.get_value_for_var(exponent_var, name_hint=ctx.fresh_name("pow_exp"))
    out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("pow_out"))

    target_dtype = np.dtype(getattr(base_var.aval, "dtype", np.float32))
    exp_dtype = np.dtype(getattr(exponent_var.aval, "dtype", target_dtype))
    if exp_dtype != target_dtype:
        cast_val = ir.Value(
            name=ctx.fresh_name("pow_exp_cast"),
            type=ir.TensorType(
                _dtype_to_ir(target_dtype, ctx.builder.enable_double_precision)
            ),
            shape=exp_val.shape,
        )
        ctx.add_node(
            ir.Node(
                op_type="Cast",
                domain="",
                inputs=[exp_val],
                outputs=[cast_val],
                name=ctx.fresh_name("Cast"),
                attributes=[
                    IRAttr(
                        "to",
                        IRAttrType.INT,
                        int(
                            _dtype_to_ir(
                                target_dtype, ctx.builder.enable_double_precision
                            ).value
                        ),
                    )
                ],
            )
        )
        _stamp_type_and_shape(cast_val, tuple(getattr(exponent_var.aval, "shape", ())))
        _ensure_value_info(ctx, cast_val)
        exp_input = cast_val
    else:
        exp_input = exp_val

    node = ir.Node(
        op_type="Pow",
        domain="",
        inputs=[base_val, exp_input],
        outputs=[out_val],
        name=ctx.fresh_name("Pow"),
    )
    ctx.add_node(node)

    out_shape = tuple(getattr(out_var.aval, "shape", ()))
    out_dtype_enum = _dtype_to_ir(target_dtype, ctx.builder.enable_double_precision)
    out_val.type = ir.TensorType(out_dtype_enum)
    _stamp_type_and_shape(out_val, out_shape)
    _ensure_value_info(ctx, out_val)


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
    context="primitives2.lax",
    component="pow",
    testcases=[
        {
            "testcase": "pow_basic",
            "callable": lambda x, y: jax.lax.pow(x, y),
            "input_shapes": [(3,), (3,)],
        },
        {
            "testcase": "pow_lax",
            "callable": lambda x, y: jax.lax.pow(x, y),
            "input_shapes": [(3,), (3,)],
        },
    ],
)
class PowPlugin(PrimitiveLeafPlugin):
    """Lower elementwise ``lax.pow`` to ONNX ``Pow`` with dtype harmonisation."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        lower_pow(ctx, eqn)
