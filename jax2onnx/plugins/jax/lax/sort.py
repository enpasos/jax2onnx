# jax2onnx/plugins/jax/lax/sort.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.sort_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.sort.html",
    onnx=[
        {"component": "TopK", "doc": "https://onnx.ai/onnx/operators/onnx__TopK.html"}
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="sort",
    testcases=[
        {
            "testcase": "sort_1d",
            "callable": lambda x: jax.lax.sort(x),
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                ["TopK:3 -> Identity:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "sort_2d",
            "callable": lambda x: jax.lax.sort(x, dimension=0),
            "input_shapes": [(3, 4)],
            "post_check_onnx_graph": EG(
                ["TopK:3x4 -> Identity:3x4"],
                no_unused_inputs=True,
            ),
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
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("sort_out"))

        axis_size = arr_shape[axis] if arr_shape else 1
        if not isinstance(axis_size, (int, np.integer)):
            raise TypeError("lax.sort currently requires static axis length")

        k_val = _const_i64(ctx, np.asarray([axis_size], dtype=np.int64), "sort_k")
        values, _indices = ctx.builder.TopK(
            arr_val,
            k_val,
            _outputs=[
                ctx.fresh_name("sort_values"),
                ctx.fresh_name("sort_indices"),
            ],
            axis=int(axis),
            largest=0,
            sorted=1,
        )
        arr_dtype = getattr(getattr(arr_val, "type", None), "dtype", None)
        if arr_dtype is not None:
            values.type = ir.TensorType(arr_dtype)

        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        _stamp_type_and_shape(values, out_shape)
        _ensure_value_metadata(ctx, values)

        result_name = getattr(out_spec, "name", None) or ctx.fresh_name("sort_out")
        result = ctx.builder.Identity(
            values,
            _outputs=[result_name],
        )
        if arr_dtype is not None:
            result.type = ir.TensorType(arr_dtype)
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)
