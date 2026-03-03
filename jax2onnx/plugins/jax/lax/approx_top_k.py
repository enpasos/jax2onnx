# jax2onnx/plugins/jax/lax/approx_top_k.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
    jaxpr_primitive=jax.lax.approx_top_k_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.approx_max_k.html",
    onnx=[
        {
            "component": "TopK",
            "doc": "https://onnx.ai/onnx/operators/onnx__TopK.html",
        }
    ],
    since="0.12.1",
    context="primitives.lax",
    component="approx_top_k",
    testcases=[
        {
            "testcase": "approx_max_k_matrix",
            "callable": lambda x: jax.lax.approx_max_k(x, 2),
            "input_values": [
                np.asarray(
                    [
                        [1.0, -1.0, 3.0, 2.0, 0.5],
                        [5.0, 4.0, 6.0, -2.0, 1.0],
                        [0.1, 0.2, 0.3, 0.4, 0.5],
                    ],
                    dtype=np.float32,
                )
            ],
            "post_check_onnx_graph": EG(
                ["TopK"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "approx_min_k_axis0",
            "callable": lambda x: jax.lax.approx_min_k(x, 2, reduction_dimension=0),
            "input_values": [
                np.asarray(
                    [
                        [4.0, 2.0, -1.0],
                        [1.0, 5.0, 0.0],
                        [3.0, -2.0, 8.0],
                        [-4.0, 3.0, 6.0],
                    ],
                    dtype=np.float32,
                )
            ],
            "post_check_onnx_graph": EG(
                ["TopK"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class ApproxTopKPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.approx_top_k`` to ONNX ``TopK``."""

    def lower(self, ctx: "IRContext", eqn: Any) -> None:
        (arr_var,) = eqn.invars
        values_var, indices_var = eqn.outvars
        params = dict(getattr(eqn, "params", {}) or {})

        arr_val = ctx.get_value_for_var(
            arr_var, name_hint=ctx.fresh_name("approx_topk_in")
        )
        values_spec = ctx.get_value_for_var(
            values_var, name_hint=ctx.fresh_name("approx_topk_values")
        )
        indices_spec = ctx.get_value_for_var(
            indices_var, name_hint=ctx.fresh_name("approx_topk_indices")
        )

        arr_shape = tuple(getattr(getattr(arr_var, "aval", None), "shape", ()) or ())
        rank = len(arr_shape)

        k = int(params.get("k", 1))
        axis = int(params.get("reduction_dimension", -1))
        if axis < 0 and rank:
            axis += rank
        if rank and (axis < 0 or axis >= rank):
            raise ValueError(
                f"approx_top_k reduction_dimension {axis} out of range for rank {rank}"
            )
        largest = 1 if bool(params.get("is_max_k", True)) else 0

        k_val = _const_i64(ctx, np.asarray([k], dtype=np.int64), "approx_topk_k")
        values, indices = ctx.builder.TopK(
            arr_val,
            k_val,
            axis=axis,
            largest=largest,
            sorted=1,
            _outputs=[
                getattr(values_spec, "name", None) or ctx.fresh_name("approx_topk_v"),
                ctx.fresh_name("approx_topk_i64"),
            ],
        )

        values_dtype = getattr(getattr(arr_val, "type", None), "dtype", None)
        if values_dtype is not None:
            values.type = ir.TensorType(values_dtype)

        result_shape = list(arr_shape)
        if rank:
            result_shape[axis] = k
        result_shape_t = tuple(result_shape)
        _stamp_type_and_shape(values, result_shape_t)
        _ensure_value_metadata(ctx, values)

        target_idx_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(indices_var, "aval", None), "dtype", np.int32)
        )
        idx_dtype_enum = _dtype_to_ir(
            target_idx_dtype, ctx.builder.enable_double_precision
        )
        if idx_dtype_enum is None:
            raise TypeError(
                f"Unsupported approx_top_k index dtype '{target_idx_dtype}'"
            )
        result_indices = indices
        result_indices.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(result_indices, result_shape_t)
        _ensure_value_metadata(ctx, result_indices)
        if idx_dtype_enum != ir.DataType.INT64:
            result_indices = ctx.builder.Cast(
                indices,
                to=int(idx_dtype_enum.value),
                _outputs=[
                    getattr(indices_spec, "name", None)
                    or ctx.fresh_name("approx_topk_indices")
                ],
            )
            result_indices.type = ir.TensorType(idx_dtype_enum)
            _stamp_type_and_shape(result_indices, result_shape_t)
            _ensure_value_metadata(ctx, result_indices)

        ctx.bind_value_for_var(values_var, values)
        ctx.bind_value_for_var(indices_var, result_indices)
