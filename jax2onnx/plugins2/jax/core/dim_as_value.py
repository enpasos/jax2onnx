from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.plugins2._ir_shapes import _stamp_type_and_shape
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover - import is only for type hints
    from jax2onnx.converter2.ir_context import IRContext


def _const_i64(ctx, values, name_hint):
    arr = np.asarray(values, dtype=np.int64)
    shape = () if arr.ndim == 0 else (arr.size,)
    val = ir.Value(
        name=ctx.fresh_name(name_hint),
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape(shape),
        const_value=ir.tensor(arr),
    )
    ctx._initializers.append(val)
    return val


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
    context="primitives2.core",
    component="dim_as_value",
    testcases=[
        {
            "testcase": "dim_as_value",
            "callable": lambda x: x.shape[0],
            "input_shapes": [("B", 8)],
            "use_onnx_ir": True,
        }
    ],
)
class DimAsValuePlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: IRContext, eqn):  # type: ignore[name-defined]
        out_var = eqn.outvars[0]
        out_val = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("dim_as_value_out")
        )

        dim_expr = eqn.params.get("dim")
        origin_getter = getattr(ctx, "get_symbolic_dim_origin", None)
        origin = origin_getter(dim_expr) if callable(origin_getter) else None
        if origin is None and callable(origin_getter):
            origin = origin_getter(str(dim_expr))
        if origin is None:
            raise ValueError(
                f"Symbolic dimension '{dim_expr}' has no registered input origin."
            )

        src_val, axis = origin
        axis = int(axis)
        src_rank = _infer_rank(src_val, axis)

        shape_vec = ir.Value(
            name=ctx.fresh_name("dim_as_value_shape"),
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape((src_rank,)),
        )
        ctx.add_node(
            ir.Node(
                op_type="Shape",
                domain="",
                inputs=[src_val],
                outputs=[shape_vec],
                name=ctx.fresh_name("Shape"),
            )
        )

        gather_idx = _const_i64(ctx, [axis], "dim_as_value_axis")
        gathered_dim = ir.Value(
            name=ctx.fresh_name("dim_as_value_gather"),
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape((1,)),
        )
        ctx.add_node(
            ir.Node(
                op_type="Gather",
                domain="",
                inputs=[shape_vec, gather_idx],
                outputs=[gathered_dim],
                name=ctx.fresh_name("Gather"),
                attributes=[IRAttr("axis", IRAttrType.INT, 0)],
            )
        )

        reshape_shape = _const_i64(ctx, [], "dim_as_value_scalar_shape")

        target_dtype = getattr(getattr(out_val, "type", None), "dtype", None)
        needs_cast = target_dtype is not None and target_dtype != ir.DataType.INT64

        reshape_output = (
            out_val
            if not needs_cast
            else ir.Value(
                name=ctx.fresh_name("dim_as_value_scalar"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape(()),
            )
        )

        ctx.add_node(
            ir.Node(
                op_type="Reshape",
                domain="",
                inputs=[gathered_dim, reshape_shape],
                outputs=[reshape_output],
                name=ctx.fresh_name("Reshape"),
            )
        )

        if needs_cast and target_dtype is not None:
            ctx.add_node(
                ir.Node(
                    op_type="Cast",
                    domain="",
                    inputs=[reshape_output],
                    outputs=[out_val],
                    name=ctx.fresh_name("Cast"),
                    attributes=[IRAttr("to", IRAttrType.INT, int(target_dtype.value))],
                )
            )
            scalar_shape = ir.Shape(())
            out_val.dtype = target_dtype
            try:
                out_val.type = ir.TensorType(target_dtype, scalar_shape)
            except TypeError:
                # onnx_ir older builds expect shape optional; retry without.
                out_val.type = ir.TensorType(target_dtype)

        _stamp_type_and_shape(out_val, ())
