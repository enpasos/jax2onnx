# jax2onnx/plugins/jax/lax/iota.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.plugins._ir_shapes import _stamp_type_and_shape, _ensure_value_metadata
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import _const_i64, _scalar_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


_DTYPE_TO_IR: Final[dict[np.dtype[Any], ir.DataType]] = {
    np.dtype(np.int32): ir.DataType.INT32,
    np.dtype(np.int64): ir.DataType.INT64,
    np.dtype(np.float32): ir.DataType.FLOAT,
    np.dtype(np.float64): ir.DataType.DOUBLE,
    np.dtype(np.bool_): ir.DataType.BOOL,
}


@register_primitive(
    jaxpr_primitive=jax.lax.iota_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.iota.html",
    onnx=[
        {
            "component": "Range",
            "doc": "https://onnx.ai/onnx/operators/onnx__Range.html",
        }
    ],
    since="v0.5.0",
    context="primitives.lax",
    component="iota",
    testcases=[
        {
            "testcase": "iota_int32",
            "callable": lambda: jax.lax.iota(np.int32, 5),
            "input_shapes": [],
            "post_check_onnx_graph": EG(
                ["Range:5 -> Cast:5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "iota_float32",
            "callable": lambda: jax.lax.iota(np.float32, 10),
            "input_shapes": [],
            "post_check_onnx_graph": EG(
                ["Range:10 -> Cast:10"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "broadcasted_iota",
            "callable": lambda: jax.lax.broadcasted_iota(np.int32, (3, 4), 1),
            "input_shapes": [],
            "post_check_onnx_graph": EG(
                ["Range:4 -> Unsqueeze:1x4 -> Expand:3x4 -> Cast:3x4"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class IotaPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.iota`` (and broadcasted variants) with pure IR ops."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError(
                "IR build context missing builder for lax.iota lowering"
            )

        params = eqn.params
        dtype = np.dtype(params["dtype"])
        shape_param = params.get("shape", ())
        dimension = int(params.get("dimension", 0))

        out_var = eqn.outvars[0]

        if not shape_param:
            # scalar iota is treated as vector of length size param (when provided as int)
            shape_param = (int(params.get("size", 0)),)

        try:
            shape = tuple(int(d) for d in shape_param)
        except TypeError as exc:  # dynamic dims not yet supported in IR path
            raise NotImplementedError(
                "Dynamic shapes for lax.iota are not supported yet"
            ) from exc

        rank = len(shape)
        if dimension < 0 or dimension >= rank:
            raise ValueError(
                f"iota dimension {dimension} out of range for shape {shape}"
            )

        # Build the 1-D range along the chosen axis.
        start = _scalar_i64(ctx, 0, "iota_start")
        limit = _scalar_i64(ctx, shape[dimension], "iota_limit")
        delta = _scalar_i64(ctx, 1, "iota_delta")

        range_out = builder.Range(
            start,
            limit,
            delta,
            _outputs=[ctx.fresh_name("iota_range")],
        )
        range_out.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(range_out, (shape[dimension],))
        _ensure_value_metadata(ctx, range_out)

        current = range_out
        if rank > 1:
            axes = [ax for ax in range(rank) if ax != dimension]
            axes_tensor = (
                _const_i64(ctx, np.asarray(axes, dtype=np.int64), "iota_unsq_axes")
                if axes
                else None
            )
            if axes_tensor is not None:
                unsq_shape = [1] * rank
                unsq_shape[dimension] = shape[dimension]
                current_unsq = builder.Unsqueeze(
                    current,
                    axes_tensor,
                    _outputs=[ctx.fresh_name("iota_unsq")],
                )
                current_unsq.type = ir.TensorType(ir.DataType.INT64)
                _stamp_type_and_shape(current_unsq, tuple(unsq_shape))
                _ensure_value_metadata(ctx, current_unsq)
                current = current_unsq

            expand_shape = _const_i64(
                ctx, np.asarray(shape, dtype=np.int64), "iota_expand_shape"
            )
            expanded = builder.Expand(
                current,
                expand_shape,
                _outputs=[ctx.fresh_name("iota_expanded")],
            )
            expanded.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(expanded, tuple(shape))
            _ensure_value_metadata(ctx, expanded)
            current = expanded

        target_dtype = _DTYPE_TO_IR.get(dtype)
        if target_dtype is None:
            raise TypeError(f"Unsupported dtype for lax.iota: {dtype}")

        if target_dtype != ir.DataType.INT64:
            cast_out = builder.Cast(
                current,
                _outputs=[ctx.fresh_name("iota_cast")],
                to=int(target_dtype.value),
            )
            cast_out.type = ir.TensorType(target_dtype)
            _stamp_type_and_shape(cast_out, tuple(shape))
            _ensure_value_metadata(ctx, cast_out)
            ctx.bind_value_for_var(out_var, cast_out)
        else:
            identity_out = builder.Identity(
                current,
                _outputs=[ctx.fresh_name("iota_out")],
            )
            identity_out.type = ir.TensorType(target_dtype)
            _stamp_type_and_shape(identity_out, tuple(shape))
            _ensure_value_metadata(ctx, identity_out)
            ctx.bind_value_for_var(out_var, identity_out)
