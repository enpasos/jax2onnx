# jax2onnx/plugins/jax/core/dim_as_value.py

from __future__ import annotations

import numpy as np
import onnx_ir as ir

from jax2onnx.converter.typing_support import (
    LoweringContextProtocol,
    SymbolicDimOrigin,
)
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _stamp_type_and_shape
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


def _dynamic_or_constant(specs):
    dynamic_checker = EG(specs, no_unused_inputs=True)
    constant_checker = EG([])

    def _check(model):
        return dynamic_checker(model) or constant_checker(model)

    return _check


def _infer_rank(value: ir.Value, axis: int) -> int:
    """Best-effort rank extraction with a safe fallback."""
    rank = None
    shape_obj = getattr(value, "shape", None)
    if shape_obj is not None:
        dims = getattr(shape_obj, "dims", None)
        if dims is not None:
            rank = len(dims)
        else:
            try:
                rank = len(tuple(shape_obj))
            except TypeError:
                rank = None
    if rank is None:
        type_obj = getattr(value, "type", None)
        if isinstance(type_obj, ir.TensorType):
            type_shape = getattr(type_obj, "shape", None)
            if type_shape is not None:
                dims = getattr(type_shape, "dims", None)
                if dims is not None:
                    rank = len(dims)
                else:
                    try:
                        rank = len(tuple(type_shape))
                    except TypeError:
                        rank = None
    if rank is None:
        rank = int(axis) + 1
    return rank


@register_primitive(
    jaxpr_primitive="dim_as_value",
    jax_doc="https://github.com/jax-ml/jax/blob/main/jax/_src/export/shape_poly.py",
    onnx=[
        {
            "component": "Shape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Shape.html",
        },
        {
            "component": "Gather",
            "doc": "https://onnx.ai/onnx/operators/onnx__Gather.html",
        },
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        },
        {
            "component": "Cast",
            "doc": "https://onnx.ai/onnx/operators/onnx__Cast.html",
        },
    ],
    since="v0.5.0",
    context="primitives.core",
    component="dim_as_value",
    testcases=[
        {
            "testcase": "dim_as_value",
            "callable": lambda x: x.shape[0],
            "input_shapes": [("B", 8)],
            "post_check_onnx_graph": _dynamic_or_constant(
                ["Shape:2 -> Gather:1 -> Reshape -> Cast"]
            ),
        }
    ],
)
class DimAsValuePlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn):
        out_var = eqn.outvars[0]
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("dim_as_value_out")
        )

        dim_expr = eqn.params.get("dim")
        origin_getter = getattr(ctx, "get_symbolic_dim_origin", None)
        origin = SymbolicDimOrigin.resolve(origin_getter, dim_expr)
        if origin is None:
            raise ValueError(
                f"Symbolic dimension '{dim_expr}' has no registered input origin."
            )

        axis = int(origin.axis)
        src_rank = _infer_rank(origin.value, axis)

        shape_vec = ctx.builder.Shape(
            origin.value,
            _outputs=[ctx.fresh_name("dim_as_value_shape")],
        )
        shape_vec.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(shape_vec, (src_rank,))

        gather_idx = _const_i64(
            ctx, np.asarray([axis], dtype=np.int64), "dim_as_value_axis"
        )
        gathered_dim = ctx.builder.Gather(
            shape_vec,
            gather_idx,
            axis=0,
            _outputs=[ctx.fresh_name("dim_as_value_gather")],
        )
        gathered_dim.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(gathered_dim, (1,))

        reshape_shape = _const_i64(
            ctx, np.asarray([], dtype=np.int64), "dim_as_value_scalar_shape"
        )

        target_dtype = getattr(getattr(out_spec, "type", None), "dtype", None)
        needs_cast = target_dtype is not None and target_dtype != ir.DataType.INT64

        reshape_name = (
            getattr(out_spec, "name", None)
            if not needs_cast
            else ctx.fresh_name("dim_as_value_scalar")
        )

        reshape_result = ctx.builder.Reshape(
            gathered_dim,
            reshape_shape,
            _outputs=[reshape_name or ctx.fresh_name("dim_as_value_scalar")],
        )
        reshape_result.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(reshape_result, ())

        final_val = reshape_result
        if needs_cast and target_dtype is not None:
            cast_result = ctx.builder.Cast(
                reshape_result,
                _outputs=[
                    getattr(out_spec, "name", None)
                    or ctx.fresh_name("dim_as_value_cast")
                ],
                to=int(target_dtype.value),
            )
            cast_result.type = ir.TensorType(target_dtype)
            _stamp_type_and_shape(cast_result, ())
            final_val = cast_result
        else:
            if target_dtype is not None:
                reshape_result.type = ir.TensorType(target_dtype)

        _stamp_type_and_shape(final_val, ())
        ctx.bind_value_for_var(out_var, final_val)
