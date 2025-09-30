from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.plugins2._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins2.jax.lax._index_utils import _const_i64
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.sort_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.sort.html",
    onnx=[
        {"component": "TopK", "doc": "https://onnx.ai/onnx/operators/onnx__TopK.html"}
    ],
    since="v0.2.0",
    context="primitives2.lax",
    component="sort",
    testcases=[
        {
            "testcase": "sort_1d",
            "callable": lambda x: jax.lax.sort(x),
            "input_shapes": [(3,)],
        },
        {
            "testcase": "sort_2d",
            "callable": lambda x: jax.lax.sort(x, dimension=0),
            "input_shapes": [(3, 4)],
        },
    ],
)
class SortPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        params = getattr(eqn, "params", {})
        axis = int(params.get("dimension", -1))

        (arr_var,) = eqn.invars
        (out_var,) = eqn.outvars

        arr_shape = tuple(getattr(arr_var.aval, "shape", ()))
        if not arr_shape:
            axis = 0
        else:
            if axis < 0:
                axis += len(arr_shape)
            if axis < 0 or axis >= len(arr_shape):
                raise ValueError("sort axis out of range")

        arr_val = ctx.get_value_for_var(arr_var, name_hint=ctx.fresh_name("sort_in"))
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("sort_out"))

        axis_size = arr_shape[axis] if arr_shape else 1
        if not isinstance(axis_size, (int, np.integer)):
            raise TypeError("lax.sort currently requires static axis length")

        k_val = _const_i64(ctx, np.asarray([axis_size], dtype=np.int64), "sort_k")
        values = ir.Value(
            name=ctx.fresh_name("sort_values"),
            type=arr_val.type,
            shape=arr_val.shape,
        )
        indices = ir.Value(
            name=ctx.fresh_name("sort_indices"),
            type=ir.TensorType(ir.DataType.INT64),
            shape=arr_val.shape,
        )
        ctx.add_node(
            ir.Node(
                op_type="TopK",
                domain="",
                inputs=[arr_val, k_val],
                outputs=[values, indices],
                name=ctx.fresh_name("TopK"),
                attributes=[
                    IRAttr("axis", IRAttrType.INT, int(axis)),
                    IRAttr("largest", IRAttrType.INT, 0),
                    IRAttr("sorted", IRAttrType.INT, 1),
                ],
            )
        )

        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        _stamp_type_and_shape(values, out_shape)
        _ensure_value_info(ctx, values)

        ctx.add_node(
            ir.Node(
                op_type="Identity",
                domain="",
                inputs=[values],
                outputs=[out_val],
                name=ctx.fresh_name("Identity"),
            )
        )
        _stamp_type_and_shape(out_val, out_shape)
        _ensure_value_info(ctx, out_val)
