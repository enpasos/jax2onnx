from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.converter2.ir_builder import _dtype_to_ir
from jax2onnx.plugins2._ir_shapes import _ensure_value_info, _stamp_type_and_shape, _to_ir_dim_for_shape
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.argmin_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.argmin.html",
    onnx=[
        {
            "component": "ArgMin",
            "doc": "https://onnx.ai/onnx/operators/onnx__ArgMin.html",
        }
    ],
    since="v0.1.0",
    context="primitives2.lax",
    component="argmin",
    testcases=[
        {
            "testcase": "argmin_axis0",
            "callable": lambda x: jax.lax.argmin(x, axis=0, index_dtype=np.int32),
            "input_shapes": [(3, 4)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "argmin_axis1",
            "callable": lambda x: jax.lax.argmin(x, axis=1, index_dtype=np.int64),
            "input_shapes": [(2, 5)],
            "use_onnx_ir": True,
        },
    ],
)
class ArgMinPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.argmin`` to ONNX ``ArgMin`` with optional index casts."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        operand_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        params = getattr(eqn, "params", {})
        axes = params.get("axes") or (0,)
        axis = int(axes[0])
        select_last = int(params.get("select_last_index", 0))
        index_dtype = np.dtype(params.get("index_dtype", np.int64))

        operand_val = ctx.get_value_for_var(
            operand_var, name_hint=ctx.fresh_name("argmin_in")
        )
        operand_shape = tuple(getattr(operand_var.aval, "shape", ()))
        rank = len(operand_shape)
        if rank > 0 and axis < 0:
            axis = axis % rank

        tmp_out = ir.Value(
            name=ctx.fresh_name("argmin_tmp"),
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape(tuple(_to_ir_dim_for_shape(d) for d in out_var.aval.shape)),
        )
        node = ir.Node(
            op_type="ArgMin",
            domain="",
            inputs=[operand_val],
            outputs=[tmp_out],
            name=ctx.fresh_name("ArgMin"),
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
        out_val = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("argmin_out")
        )
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
