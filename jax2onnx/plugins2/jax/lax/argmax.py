from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.converter2.ir_builder import _dtype_to_ir
from jax2onnx.plugins2._ir_shapes import (
    _ensure_value_info,
    _stamp_type_and_shape,
    _to_ir_dim_for_shape,
)
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.argmax_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.argmax.html",
    onnx=[
        {
            "component": "ArgMax",
            "doc": "https://onnx.ai/onnx/operators/onnx__ArgMax.html",
        }
    ],
    since="v0.2.0",
    context="primitives2.lax",
    component="argmax",
    testcases=[
        {
            "testcase": "argmax_axis0",
            "callable": lambda x: jax.lax.argmax(x, axis=0, index_dtype=np.int32),
            "input_shapes": [(3, 4)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "argmax_bool",
            "callable": lambda x: jax.lax.argmax(x, axis=1, index_dtype=np.int64),
            "input_shapes": [(2, 3)],
            "input_dtypes": [np.bool_],
            "use_onnx_ir": True,
        },
    ],
)
class ArgMaxPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.argmax`` to ONNX ``ArgMax`` with optional dtype casts."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        operand_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        params = getattr(eqn, "params", {})
        axes = params.get("axes") or (0,)
        axis = int(axes[0])
        select_last = int(params.get("select_last_index", 0))
        index_dtype = np.dtype(params.get("index_dtype", np.int64))

        operand_val = ctx.get_value_for_var(
            operand_var, name_hint=ctx.fresh_name("argmax_in")
        )
        operand_shape = tuple(getattr(operand_var.aval, "shape", ()))
        operand_dtype = np.dtype(getattr(operand_var.aval, "dtype", np.float32))
        rank = len(operand_shape)
        if rank > 0 and axis < 0:
            axis = axis % rank

        input_for_argmax = operand_val
        if operand_dtype == np.bool_:
            cast_dtype = _dtype_to_ir(
                np.dtype(np.int64), ctx.builder.enable_double_precision
            )
            cast_val = ir.Value(
                name=ctx.fresh_name("argmax_cast"),
                type=ir.TensorType(cast_dtype),
                shape=operand_val.shape,
            )
            ctx.add_node(
                ir.Node(
                    op_type="Cast",
                    domain="",
                    inputs=[operand_val],
                    outputs=[cast_val],
                    name=ctx.fresh_name("Cast"),
                    attributes=[IRAttr("to", IRAttrType.INT, int(cast_dtype.value))],
                )
            )
            _stamp_type_and_shape(cast_val, operand_shape)
            _ensure_value_info(ctx, cast_val)
            input_for_argmax = cast_val

        tmp_out = ir.Value(
            name=ctx.fresh_name("argmax_tmp"),
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape(tuple(_to_ir_dim_for_shape(d) for d in out_var.aval.shape)),
        )
        node = ir.Node(
            op_type="ArgMax",
            domain="",
            inputs=[input_for_argmax],
            outputs=[tmp_out],
            name=ctx.fresh_name("ArgMax"),
            attributes=[
                IRAttr("axis", IRAttrType.INT, axis),
                IRAttr("keepdims", IRAttrType.INT, 0),
                IRAttr("select_last_index", IRAttrType.INT, select_last),
            ],
        )
        ctx.add_node(node)
        _stamp_type_and_shape(tmp_out, tuple(getattr(out_var.aval, "shape", ())))
        _ensure_value_info(ctx, tmp_out)

        target_enum = _dtype_to_ir(index_dtype, ctx.builder.enable_double_precision)
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("argmax_out"))
        if target_enum != ir.DataType.INT64:
            ctx.add_node(
                ir.Node(
                    op_type="Cast",
                    domain="",
                    inputs=[tmp_out],
                    outputs=[out_val],
                    name=ctx.fresh_name("Cast"),
                    attributes=[IRAttr("to", IRAttrType.INT, int(target_enum.value))],
                )
            )
        else:
            ctx.add_node(
                ir.Node(
                    op_type="Identity",
                    domain="",
                    inputs=[tmp_out],
                    outputs=[out_val],
                    name=ctx.fresh_name("Identity"),
                )
            )

        out_val.type = ir.TensorType(target_enum)
        _stamp_type_and_shape(out_val, tuple(getattr(out_var.aval, "shape", ())))
        _ensure_value_info(ctx, out_val)
