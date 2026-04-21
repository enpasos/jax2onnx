# jax2onnx/plugins/jax/lax/iota.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.ir_utils import numpy_dtype_to_ir
from jax2onnx.plugins._ir_shapes import _stamp_type_and_shape, _ensure_value_metadata
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import (
    _const_i64,
    _lower_i64_vector,
    _scalar_i64,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


_SUPPORTED_IOTA_DTYPES: Final[frozenset[np.dtype[Any]]] = frozenset(
    {
        np.dtype(np.int32),
        np.dtype(np.int64),
        np.dtype(np.float32),
        np.dtype(np.float64),
        np.dtype(np.bool_),
    }
)


@register_primitive(
    jaxpr_primitive=jax.lax.iota_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.iota.html",
    onnx=[
        {
            "component": "Range",
            "doc": "https://onnx.ai/onnx/operators/onnx__Range.html",
        }
    ],
    since="0.5.0",
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

    def lower(self, ctx: "IRContext", eqn: Any) -> None:
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
            shape_param = (params.get("size", 0),)

        shape = tuple(shape_param)

        rank = len(shape)
        if dimension < 0 or dimension >= rank:
            raise ValueError(
                f"iota dimension {dimension} out of range for shape {shape}"
            )

        dim_extent = shape[dimension]
        limit_vec = _lower_i64_vector(ctx, [dim_extent], "iota_limit_vec")
        squeeze_axes = _const_i64(
            ctx, np.asarray([0], dtype=np.int64), "iota_limit_squeeze_axes"
        )
        limit = builder.Squeeze(
            limit_vec,
            squeeze_axes,
            _outputs=[ctx.fresh_name("iota_limit")],
        )
        limit.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(limit, ())
        _ensure_value_metadata(ctx, limit)

        # Build the 1-D range along the chosen axis.
        start = _scalar_i64(ctx, 0, "iota_start")
        delta = _scalar_i64(ctx, 1, "iota_delta")

        range_out = builder.Range(
            start,
            limit,
            delta,
            _outputs=[ctx.fresh_name("iota_range")],
        )
        range_out.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(range_out, (dim_extent,))
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
                unsq_shape[dimension] = dim_extent
                current_unsq = builder.Unsqueeze(
                    current,
                    axes_tensor,
                    _outputs=[ctx.fresh_name("iota_unsq")],
                )
                current_unsq.type = ir.TensorType(ir.DataType.INT64)
                _stamp_type_and_shape(current_unsq, tuple(unsq_shape))
                _ensure_value_metadata(ctx, current_unsq)
                current = current_unsq

            expand_shape = _lower_i64_vector(ctx, shape, "iota_expand_shape")
            expanded = builder.Expand(
                current,
                expand_shape,
                _outputs=[ctx.fresh_name("iota_expanded")],
            )
            expanded.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(expanded, tuple(shape))
            _ensure_value_metadata(ctx, expanded)
            current = expanded

        if dtype not in _SUPPORTED_IOTA_DTYPES:
            raise TypeError(f"Unsupported dtype for lax.iota: {dtype}")
        target_dtype = numpy_dtype_to_ir(dtype)

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
