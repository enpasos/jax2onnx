from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.plugins2._ir_shapes import _stamp_type_and_shape, _ensure_value_info
from jax2onnx.plugins2.jax.lax._index_utils import _const_i64, _scalar_i64
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


_DTYPE_TO_IR = {
    np.dtype(np.int32): ir.DataType.INT32,
    np.dtype(np.int64): ir.DataType.INT64,
    np.dtype(np.float32): ir.DataType.FLOAT,
    np.dtype(np.float64): getattr(ir.DataType, "DOUBLE", ir.DataType.FLOAT),
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
    context="primitives2.lax",
    component="iota",
    testcases=[
        {
            "testcase": "iota_int32",
            "callable": lambda: jax.lax.iota(np.int32, 5),
            "input_shapes": [],
            "use_onnx_ir": True,
        },
        {
            "testcase": "iota_float32",
            "callable": lambda: jax.lax.iota(np.float32, 10),
            "input_shapes": [],
            "use_onnx_ir": True,
        },
        {
            "testcase": "broadcasted_iota",
            "callable": lambda: jax.lax.broadcasted_iota(np.int32, (3, 4), 1),
            "input_shapes": [],
            "use_onnx_ir": True,
        },
    ],
)
class IotaPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.iota`` (and broadcasted variants) with pure IR ops."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        params = eqn.params
        dtype = np.dtype(params["dtype"])
        shape_param = params.get("shape", ())
        dimension = int(params.get("dimension", 0))

        out_var = eqn.outvars[0]
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("iota_out"))

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

        range_out = ir.Value(
            name=ctx.fresh_name("iota_range"),
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape((shape[dimension],)),
        )
        ctx.add_node(
            ir.Node(
                op_type="Range",
                domain="",
                inputs=[start, limit, delta],
                outputs=[range_out],
                name=ctx.fresh_name("Range"),
            )
        )
        _stamp_type_and_shape(range_out, (shape[dimension],))
        _ensure_value_info(ctx, range_out)

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
                current_unsq = ir.Value(
                    name=ctx.fresh_name("iota_unsq"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape(tuple(unsq_shape)),
                )
                ctx.add_node(
                    ir.Node(
                        op_type="Unsqueeze",
                        domain="",
                        inputs=[current, axes_tensor],
                        outputs=[current_unsq],
                        name=ctx.fresh_name("Unsqueeze"),
                    )
                )
                _stamp_type_and_shape(current_unsq, tuple(unsq_shape))
                _ensure_value_info(ctx, current_unsq)
                current = current_unsq

            expand_shape = _const_i64(
                ctx, np.asarray(shape, dtype=np.int64), "iota_expand_shape"
            )
            expanded = ir.Value(
                name=ctx.fresh_name("iota_expanded"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape(tuple(shape)),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Expand",
                    domain="",
                    inputs=[current, expand_shape],
                    outputs=[expanded],
                    name=ctx.fresh_name("Expand"),
                )
            )
            _stamp_type_and_shape(expanded, tuple(shape))
            _ensure_value_info(ctx, expanded)
            current = expanded

        target_dtype = _DTYPE_TO_IR.get(dtype)
        if target_dtype is None:
            raise TypeError(f"Unsupported dtype for lax.iota: {dtype}")

        if target_dtype != ir.DataType.INT64:
            ctx.add_node(
                ir.Node(
                    op_type="Cast",
                    domain="",
                    inputs=[current],
                    outputs=[out_val],
                    name=ctx.fresh_name("Cast"),
                    attributes=[IRAttr("to", IRAttrType.INT, int(target_dtype.value))],
                )
            )
            out_val.type = ir.TensorType(target_dtype)
            out_val.dtype = target_dtype
            _stamp_type_and_shape(out_val, tuple(shape))
            _ensure_value_info(ctx, out_val)
        else:
            # No cast needed, but ensure SSA wiring via Identity to reuse out_val.
            ctx.add_node(
                ir.Node(
                    op_type="Identity",
                    domain="",
                    inputs=[current],
                    outputs=[out_val],
                    name=ctx.fresh_name("Identity"),
                )
            )
            out_val.type = ir.TensorType(target_dtype)
            out_val.dtype = target_dtype
            _stamp_type_and_shape(out_val, tuple(shape))
            _ensure_value_info(ctx, out_val)
