# jax2onnx/plugins/jax/lax/top_k.py

from __future__ import annotations

from typing import Any

import jax
import numpy as np

import onnx_ir as ir

from jax2onnx._compat.jax import JaxprEqn
from jax2onnx.converter.typing_support import NativeLoweringContext
from jax2onnx.ir_utils import numpy_dtype_to_ir
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)


@register_native_primitive(
    jaxpr_primitive=jax.lax.top_k_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.top_k.html",
    onnx=[
        {
            "component": "TopK",
            "doc": "https://onnx.ai/onnx/operators/onnx__TopK.html",
        }
    ],
    since="0.10.2",
    context="primitives.lax",
    component="top_k",
    testcases=[
        {
            "testcase": "top_k_last_axis",
            "callable": lambda x: jax.lax.top_k(x, 3),
            "input_shapes": [(5,)],
            "expected_output_dtypes": [np.float32, np.int32],
        },
        {
            "testcase": "top_k_matrix",
            "callable": lambda x: jax.lax.top_k(x, 2),
            "input_shapes": [(4, 6)],
            "expected_output_dtypes": [np.float32, np.int32],
        },
    ],
)
class TopKPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: NativeLoweringContext, eqn: JaxprEqn) -> None:
        (arr_var,) = eqn.invars
        values_var, indices_var = eqn.outvars

        arr_val = ctx.require_value_for_var(arr_var)
        values_spec = ctx.allocate_value_for_var(
            values_var, name_hint=ctx.fresh_name("topk_values")
        )
        indices_spec = ctx.allocate_value_for_var(
            indices_var, name_hint=ctx.fresh_name("topk_indices")
        )

        arr_shape = tuple(arr_var.aval.shape)
        arr_dtype = arr_val.dtype

        k = int(eqn.params.get("k", 1))
        axis = int(eqn.params.get("axis", eqn.params.get("dimension", -1)))
        if axis < 0 and arr_shape:
            axis += len(arr_shape)

        k_val = _const_i64(ctx, np.asarray([k], dtype=np.int64), "topk_k")
        values_spec.name = ctx.fresh_name("TopK_Values")
        values, indices = ctx.builder.op_multi_out(
            "TopK",
            [arr_val, k_val],
            {"axis": axis, "largest": 1, "sorted": 1},
            outputs=[values_spec, ir.Value(name=ctx.fresh_name("TopK_Indices"))],
        )

        if arr_dtype is not None:
            values.type = ir.TensorType(arr_dtype)
        indices.type = ir.TensorType(ir.DataType.INT64)

        result_shape_list = list(arr_shape)
        if arr_shape:
            result_shape_list[axis] = k
        result_shape = tuple(result_shape_list)

        _stamp_type_and_shape(values, result_shape)
        _stamp_type_and_shape(indices, result_shape)
        _ensure_value_metadata(ctx, values)
        _ensure_value_metadata(ctx, indices)

        target_idx_dtype: np.dtype[Any] = np.dtype(indices_var.aval.dtype)
        idx_dtype_enum = numpy_dtype_to_ir(target_idx_dtype)
        result_indices = indices
        if idx_dtype_enum != ir.DataType.INT64:
            result_indices = ctx.builder.op(
                "Cast",
                [indices],
                {"to": int(idx_dtype_enum.value)},
                output=indices_spec,
            )
            result_indices.type = ir.TensorType(idx_dtype_enum)
            _stamp_type_and_shape(result_indices, result_shape)
            _ensure_value_metadata(ctx, result_indices)

        ctx.bind_value_for_var(values_var, values)
        ctx.bind_value_for_var(indices_var, result_indices)
