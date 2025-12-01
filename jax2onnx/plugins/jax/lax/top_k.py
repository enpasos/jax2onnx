# jax2onnx/plugins/jax/lax/top_k.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np

import onnx_ir as ir

from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.top_k_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.top_k.html",
    onnx=[
        {
            "component": "TopK",
            "doc": "https://onnx.ai/onnx/operators/onnx__TopK.html",
        }
    ],
    since="v0.10.2",
    context="primitives.lax",
    component="top_k",
    testcases=[
        {
            "testcase": "top_k_last_axis",
            "callable": lambda x: jax.lax.top_k(x, 3),
            "input_shapes": [(5,)],
        },
        {
            "testcase": "top_k_matrix",
            "callable": lambda x: jax.lax.top_k(x, 2),
            "input_shapes": [(4, 6)],
        },
    ],
)
class TopKPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: "IRContext", eqn):
        (arr_var,) = eqn.invars
        values_var, indices_var = eqn.outvars

        arr_val = ctx.get_value_for_var(arr_var, name_hint=ctx.fresh_name("topk_in"))
        [
            ctx.get_value_for_var(values_var, name_hint=ctx.fresh_name("topk_values")),
            ctx.get_value_for_var(
                indices_var, name_hint=ctx.fresh_name("topk_indices")
            ),
        ]

        arr_shape = tuple(getattr(getattr(arr_var, "aval", None), "shape", ()))
        arr_dtype = getattr(getattr(arr_val, "type", None), "dtype", None)

        k = int(eqn.params.get("k", 1))
        axis = int(eqn.params.get("dimension", -1))
        if axis < 0 and arr_shape:
            axis += len(arr_shape)

        k_val = _const_i64(ctx, np.asarray([k], dtype=np.int64), "topk_k")
        values, indices = ctx.builder.TopK(
            arr_val,
            k_val,
            axis=axis,
            largest=1,
            sorted=1,
            _outputs=[
                ctx.fresh_name("TopK_Values"),
                ctx.fresh_name("TopK_Indices"),
            ],
        )

        if arr_dtype is not None:
            values.type = ir.TensorType(arr_dtype)
        indices.type = ir.TensorType(ir.DataType.INT64)

        result_shape = list(arr_shape)
        if arr_shape:
            result_shape[axis] = k
        result_shape = tuple(result_shape)

        _stamp_type_and_shape(values, result_shape)
        _stamp_type_and_shape(indices, result_shape)
        _ensure_value_metadata(ctx, values)
        _ensure_value_metadata(ctx, indices)

        ctx.bind_value_for_var(values_var, values)
        ctx.bind_value_for_var(indices_var, indices)
