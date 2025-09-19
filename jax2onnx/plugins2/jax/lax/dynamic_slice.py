from typing import TYPE_CHECKING, Any, Dict

import jax
import numpy as np
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.plugins2._ir_shapes import _stamp_type_and_shape, _ensure_value_info
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    pass


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


def _scalar_i64(ctx, value, name_hint):
    return _const_i64(ctx, np.asarray(value, dtype=np.int64), name_hint)


def _cast_to_i64(ctx, tensor_val, name_hint):
    exemplar = _scalar_i64(ctx, 0, "dyn_slice_cast_like_exemplar")
    out = ctx.cast_like(tensor_val, exemplar, name_hint=name_hint)
    out.dtype = getattr(exemplar, "dtype", ir.DataType.INT64)
    shape_obj = getattr(tensor_val, "shape", None)
    if shape_obj is None:
        dims = ()
    else:
        dims = getattr(shape_obj, "dims", None)
        if dims is None:
            try:
                dims = tuple(shape_obj)
            except Exception:
                dims = ()
    _stamp_type_and_shape(out, dims)
    _ensure_value_info(ctx, out)
    return out


def _infer_rank(value: ir.Value, axis: int) -> int:
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
        aval = getattr(value, "aval", None)
        if aval is not None:
            rank = len(getattr(aval, "shape", ()) or ())
    if rank is None:
        rank = int(axis) + 1
    return rank


@register_primitive(
    jaxpr_primitive=jax.lax.dynamic_slice_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_slice.html",
    onnx=[
        {
            "component": "Slice",
            "doc": "https://onnx.ai/onnx/operators/onnx__Slice.html",
        }
    ],
    since="v0.1.0",
    context="primitives2.lax",
    component="dynamic_slice",
    testcases=[
        {
            "testcase": "dynamic_slice_test1",
            "callable": lambda x: jax.lax.dynamic_slice(x, [1], [2]),
            "input_shapes": [(5,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "dynamic_slice_2d",
            "callable": lambda x: jax.lax.dynamic_slice(x, (1, 2), (2, 3)),
            "input_shapes": [(4, 6)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "dynamic_slice_3d",
            "callable": lambda x: jax.lax.dynamic_slice(x, (1, 0, 2), (2, 3, 1)),
            "input_shapes": [(3, 4, 5)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "dynamic_slice_vit_like",
            "context": "jax.lax.dynamic_slice",
            "callable": lambda x: jax.lax.dynamic_slice(
                x, (0, 0, 0), (x.shape[0], 1, 256)
            ),
            "input_shapes": [("B", 50, 256)],
            "expected_output_shapes": [("B", 1, 256)],
            "use_onnx_ir": True,
        },
    ],
)
class DynamicSlicePlugin(PrimitiveLeafPlugin):
    def lower(self, ctx, eqn):
        operand_var = eqn.invars[0]
        start_vars = eqn.invars[1:]
        out_var = eqn.outvars[0]

        operand_val = ctx.get_value_for_var(
            operand_var, name_hint=ctx.fresh_name("dyn_slice_in")
        )
        out_val = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("dyn_slice_out")
        )

        origin_getter = getattr(ctx, "get_symbolic_dim_origin", None)
        shape_cache: Dict[ir.Value, ir.Value] = {}

        def _shape_vec_for(src: ir.Value, axis: int) -> ir.Value:
            shape_val = shape_cache.get(src)
            if shape_val is not None:
                return shape_val
            rank = _infer_rank(src, axis)
            shape_val = ir.Value(
                name=ctx.fresh_name("dyn_slice_shape"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape((rank,)),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Shape",
                    domain="",
                    inputs=[src],
                    outputs=[shape_val],
                    name=ctx.fresh_name("Shape"),
                )
            )
            shape_cache[src] = shape_val
            return shape_val

        def _symbolic_dim_vector(dim_expr: Any, idx: int) -> ir.Value:
            if origin_getter is None:
                raise ValueError(
                    f"Symbolic dimension '{dim_expr}' encountered without origin resolver"
                )
            origin = origin_getter(dim_expr)
            if origin is None:
                origin = origin_getter(str(dim_expr))
            if origin is None:
                raise ValueError(
                    f"Symbolic dimension '{dim_expr}' has no registered origin"
                )
            src_val, axis = origin
            axis = int(axis)
            shape_vec = _shape_vec_for(src_val, axis)
            gather_idx = _const_i64(ctx, [axis], f"dyn_slice_size_axis_{idx}")
            gathered = ir.Value(
                name=ctx.fresh_name("dyn_slice_size"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape((1,)),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Gather",
                    domain="",
                    inputs=[shape_vec, gather_idx],
                    outputs=[gathered],
                    name=ctx.fresh_name("Gather"),
                    attributes=[IRAttr("axis", IRAttrType.INT, 0)],
                )
            )
            return gathered

        rank = len(getattr(operand_var.aval, "shape", ()))
        if rank != len(start_vars):
            raise ValueError(
                f"dynamic_slice expected {rank} start indices but got {len(start_vars)}"
            )

        starts_vec_parts = []
        axes_const = _const_i64(ctx, [0], "dyn_slice_unsq_axis")

        for start_var in start_vars:
            start_val = ctx.get_value_for_var(
                start_var, name_hint=ctx.fresh_name("dyn_slice_start")
            )
            cast_scalar = _cast_to_i64(ctx, start_val, "dyn_slice_start_i64")

            unsqueezed = ir.Value(
                name=ctx.fresh_name("dyn_slice_unsq"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape((1,)),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Unsqueeze",
                    domain="",
                    inputs=[cast_scalar, axes_const],
                    outputs=[unsqueezed],
                    name=ctx.fresh_name("Unsqueeze"),
                )
            )
            starts_vec_parts.append(unsqueezed)

        starts_concat = ir.Value(
            name=ctx.fresh_name("dyn_slice_starts"),
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape((rank,)),
        )
        ctx.add_node(
            ir.Node(
                op_type="Concat",
                domain="",
                inputs=starts_vec_parts,
                outputs=[starts_concat],
                name=ctx.fresh_name("Concat"),
                attributes=[IRAttr("axis", IRAttrType.INT, 0)],
            )
        )

        slice_sizes = eqn.params.get("slice_sizes", ())
        try:
            const_sizes = [int(v) for v in slice_sizes]
            slice_sizes_val = _const_i64(ctx, const_sizes, "dyn_slice_sizes")
        except Exception:
            size_vectors = []
            all_const = True
            for idx, val in enumerate(slice_sizes):
                try:
                    const_val = int(val)
                except Exception:
                    const_val = None
                if const_val is not None:
                    size_vectors.append(
                        _const_i64(ctx, [const_val], f"dyn_slice_size_const_{idx}")
                    )
                else:
                    all_const = False
                    size_vectors.append(_symbolic_dim_vector(val, idx))
            if all_const:
                slice_sizes_val = _const_i64(
                    ctx,
                    [int(v) for v in slice_sizes],
                    "dyn_slice_sizes",
                )
            else:
                if len(size_vectors) == 1 and len(slice_sizes) == 1:
                    slice_sizes_val = size_vectors[0]
                else:
                    slice_sizes_val = ir.Value(
                        name=ctx.fresh_name("dyn_slice_sizes"),
                        type=ir.TensorType(ir.DataType.INT64),
                        shape=ir.Shape((len(slice_sizes),)),
                    )
                    ctx.add_node(
                        ir.Node(
                            op_type="Concat",
                            domain="",
                            inputs=size_vectors,
                            outputs=[slice_sizes_val],
                            name=ctx.fresh_name("Concat"),
                            attributes=[IRAttr("axis", IRAttrType.INT, 0)],
                        )
                    )

        ends_val = ir.Value(
            name=ctx.fresh_name("dyn_slice_ends"),
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape((rank,)),
        )
        ctx.add_node(
            ir.Node(
                op_type="Add",
                domain="",
                inputs=[starts_concat, slice_sizes_val],
                outputs=[ends_val],
                name=ctx.fresh_name("Add"),
            )
        )

        axes_val = _const_i64(ctx, list(range(rank)), "dyn_slice_axes")

        inputs = [operand_val, starts_concat, ends_val, axes_val]
        strides = eqn.params.get("strides")
        if strides:
            inputs.append(_const_i64(ctx, strides, "dyn_slice_strides"))

        ctx.add_node(
            ir.Node(
                op_type="Slice",
                domain="",
                inputs=inputs,
                outputs=[out_val],
                name=ctx.fresh_name("Slice"),
            )
        )

        _stamp_type_and_shape(out_val, getattr(out_var.aval, "shape", ()))
