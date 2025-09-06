# file: jax2onnx/plugins/jax/lax/dynamic_update_slice.py

from __future__ import annotations
from typing import TYPE_CHECKING, Sequence, Any, List
import numpy as np
from jax import core as jcore, lax
from onnx import helper, TensorProto
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=lax.dynamic_update_slice_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dynamic_update_slice.html",
    onnx=[
        {
            "component": "ScatterND",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterND.html",
        },
        {
            "component": "Range",
            "doc": "https://onnx.ai/onnx/operators/onnx__Range.html",
        },
        {
            "component": "Shape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Shape.html",
        },
        {
            "component": "Gather",
            "doc": "https://onnx.ai/onnx/operators/onnx__Gather.html",
        },
        {
            "component": "Expand",
            "doc": "https://onnx.ai/onnx/operators/onnx__Expand.html",
        },
        {
            "component": "ReduceProd",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceProd.html",
        },
        {
            "component": "Unsqueeze",
            "doc": "https://onnx.ai/onnx/operators/onnx__Unsqueeze.html",
        },
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        },
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        },
    ],
    since="v0.8.1",
    context="primitives.lax",
    component="dynamic_update_slice",
    testcases=[
        {
            "testcase": "dus_1d_scalar_update",
            "callable": lambda ref, val, idx: lax.dynamic_update_slice(
                ref, val, (idx,)
            ),
            "input_shapes": [(10,), (1,), ()],
            # keep arrays float; index must be integer
            "input_dtypes": [np.float32, np.float32, np.int32],
            "expected_output_shapes": [(10,)],
        },
        {
            "testcase": "dus_1d_block_update",
            "callable": lambda ref, upd, idx: lax.dynamic_update_slice(
                ref, upd, (idx,)
            ),
            "input_shapes": [(10,), (3,), ()],
            "input_dtypes": [np.float32, np.float32, np.int32],
            "expected_output_shapes": [(10,)],
        },
        {
            "testcase": "dus_2d_block_update",
            "callable": lambda ref, upd, i, j: lax.dynamic_update_slice(ref, upd, (i, j)),
            "input_shapes": [(4, 4), (2, 2), (), ()],
            "input_dtypes": [np.float32, np.float32, np.int32, np.int32],
            "expected_output_shapes": [(4, 4)],
        },
        {
            "testcase": "dus_3d_block_update",
            "callable": lambda ref, upd, i, j, k: lax.dynamic_update_slice(ref, upd, (i, j, k)),
            "input_shapes": [(3, 4, 4), (1, 2, 2), (), (), ()],
            "input_dtypes": [np.float32, np.float32, np.int32, np.int32, np.int32],
            "expected_output_shapes": [(3, 4, 4)],
        },
        {
            "testcase": "dus_4d_block_update",
            "callable": lambda ref, upd, a, b, c, d: lax.dynamic_update_slice(ref, upd, (a, b, c, d)),
            "input_shapes": [(5, 10, 10, 1), (1, 5, 5, 1), (), (), (), ()],
            "input_dtypes": [np.float32, np.float32, np.int32, np.int32, np.int32, np.int32],
            "expected_output_shapes": [(5, 10, 10, 1)],
        },
    ],
)
class DynamicUpdateSlice1D(PrimitiveLeafPlugin):
    """N-D lowering: lax.dynamic_update_slice -> ONNX ScatterND (element-wise scatter over update block)."""

    @staticmethod
    def abstract_eval(ref, update, *start_indices):
        # Output matches ref in shape/dtype
        return jcore.ShapedArray(ref.shape, ref.dtype)

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[Any],
        node_outputs: Sequence[Any],
        params: dict[str, Any],
    ) -> None:
        if len(node_inputs) < 3:
            raise ValueError("dynamic_update_slice expects operand, update, and at least one start index.")

        ref_v = node_inputs[0]
        upd_v = node_inputs[1]
        start_vs: List[Any] = list(node_inputs[2:])
        out_v = node_outputs[0]

        ref_name = s.get_name(ref_v)
        upd_name = s.get_name(upd_v)
        out_name = s.get_name(out_v)
        rank = len(ref_v.aval.shape)

        if len(start_vs) != rank:
            raise ValueError(f"dynamic_update_slice: got {len(start_vs)} start indices, but operand rank is {rank}.")

        # Constants 0, 1 as INT64 initializers (for index ops)
        zero_i64 = s.get_unique_name("dus_zero_i64")
        one_i64 = s.get_unique_name("dus_one_i64")
        s.builder.add_initializer_from_scalar(zero_i64, 0)  # INT64
        s.builder.add_initializer_from_scalar(one_i64, 1)  # INT64

        # Shapes
        shape_node_ref = s.get_unique_name("dus_shape_ref")
        ref_shape = s.get_unique_name("dus_ref_shape")
        s.add_node(helper.make_node("Shape", [ref_name], [ref_shape], name=shape_node_ref))
        s.add_shape_info(ref_shape, (rank,), np.int64)

        shape_node_upd = s.get_unique_name("dus_shape_upd")
        upd_shape = s.get_unique_name("dus_upd_shape")
        s.add_node(helper.make_node("Shape", [upd_name], [upd_shape], name=shape_node_upd))
        s.add_shape_info(upd_shape, (rank,), np.int64)

        # Build per-axis start_i64 and length_i64 (len = upd_dim[i])
        start_i64s: List[str] = []              # raw (possibly negative / unclamped) starts
        start_clamped_i64s: List[str] = []      # normalized & clamped starts (used below)
        upd_dims: List[str] = []
        for i, sv in enumerate(start_vs):
            start_name = s.get_name(sv)
            cast_node = s.get_unique_name(f"dus_cast_start_{i}")
            start_i64 = s.get_unique_name(f"dus_start_i64_{i}")
            s.add_node(
                helper.make_node(
                    "Cast", [start_name], [start_i64], name=cast_node, to=TensorProto.INT64
                )
            )
            s.add_shape_info(start_i64, (), np.int64)
            start_i64s.append(start_i64)

            gather_node = s.get_unique_name(f"dus_gather_upd_dim_{i}")
            upd_dim_i = s.get_unique_name(f"dus_upd_dim_{i}")
            axis_i = s.get_unique_name(f"dus_axis_{i}")
            s.builder.add_initializer_from_scalar(axis_i, i)
            s.add_node(
                helper.make_node(
                    "Gather", [upd_shape, axis_i], [upd_dim_i], name=gather_node, axis=0
                )
            )
            s.add_shape_info(upd_dim_i, (), np.int64)
            upd_dims.append(upd_dim_i)

            # ref_dim[i] (gather from ref_shape)
            gather_ref_node = s.get_unique_name(f"dus_gather_ref_dim_{i}")
            ref_dim_i = s.get_unique_name(f"dus_ref_dim_{i}")
            # reuse axis_i initializer
            s.add_node(
                helper.make_node(
                    "Gather", [ref_shape, axis_i], [ref_dim_i], name=gather_ref_node, axis=0
                )
            )
            s.add_shape_info(ref_dim_i, (), np.int64)

            # Normalize negatives if any: start_norm = Where(start < 0, start + ref_dim, start)
            less_node = s.get_unique_name(f"dus_start_is_neg_{i}")
            cond_neg = s.get_unique_name(f"dus_cond_neg_{i}")
            s.add_node(helper.make_node("Less", [start_i64, zero_i64], [cond_neg], name=less_node))
            s.add_shape_info(cond_neg, (), np.bool_)

            add_ref_node = s.get_unique_name(f"dus_add_ref_{i}")
            start_plus_ref = s.get_unique_name(f"dus_start_plus_ref_{i}")
            s.add_node(helper.make_node("Add", [start_i64, ref_dim_i], [start_plus_ref], name=add_ref_node))
            s.add_shape_info(start_plus_ref, (), np.int64)

            where_node = s.get_unique_name(f"dus_where_norm_{i}")
            start_norm = s.get_unique_name(f"dus_start_norm_{i}")
            s.add_node(helper.make_node("Where", [cond_neg, start_plus_ref, start_i64], [start_norm], name=where_node))
            s.add_shape_info(start_norm, (), np.int64)

            # max_start = ref_dim - upd_dim
            sub_node = s.get_unique_name(f"dus_sub_maxstart_{i}")
            max_start = s.get_unique_name(f"dus_max_start_{i}")
            s.add_node(helper.make_node("Sub", [ref_dim_i, upd_dim_i], [max_start], name=sub_node))
            s.add_shape_info(max_start, (), np.int64)

            # clamp lower bound: start_ge0 = Max(start_norm, 0)
            max_node = s.get_unique_name(f"dus_clamp_lo_{i}")
            start_ge0 = s.get_unique_name(f"dus_start_ge0_{i}")
            s.add_node(helper.make_node("Max", [start_norm, zero_i64], [start_ge0], name=max_node))
            s.add_shape_info(start_ge0, (), np.int64)

            # clamp upper bound: start_clamped = Min(start_ge0, max_start)
            min_node = s.get_unique_name(f"dus_clamp_hi_{i}")
            start_clamped = s.get_unique_name(f"dus_start_clamped_{i}")
            s.add_node(helper.make_node("Min", [start_ge0, max_start], [start_clamped], name=min_node))
            s.add_shape_info(start_clamped, (), np.int64)

            start_clamped_i64s.append(start_clamped)

        # numel(update) = ReduceProd(upd_shape)  (scalar i64)
        rprod_node = s.get_unique_name("dus_reduceprod_numel")
        numel_upd = s.get_unique_name("dus_numel_upd")
        s.add_node(
            helper.make_node("ReduceProd", [upd_shape], [numel_upd], name=rprod_node, keepdims=0)
        )
        s.add_shape_info(numel_upd, (), np.int64)

        # Build expanded per-axis offset grids:
        #   range_i = Range(0, upd_dim[i], 1)                    -> [upd_dim_i]
        #   unsq_i  = Unsqueeze(range_i, axes=all axes != i)     -> [1,..,upd_dim_i,..,1] (rank dims)
        #   exp_i   = Expand(unsq_i, upd_shape)                  -> update_shape (int64)
        #   start_b = Expand(start_i64, upd_shape)               -> update_shape (int64)
        #   idx_i   = Add(exp_i, start_b)                        -> update_shape (int64)
        idx_components: List[str] = []
        for i in range(rank):
            # Constants for Range bounds: 0 and 1 are available (zero_i64, one_i64)
            range_node = s.get_unique_name(f"dus_range_{i}")
            range_i = s.get_unique_name(f"dus_range_out_{i}")
            s.add_node(
                helper.make_node(
                    "Range", [zero_i64, upd_dims[i], one_i64], [range_i], name=range_node
                )
            )
            s.add_shape_info(range_i, (-1,), np.int64)  # 1D dynamic

            # Unsqueeze range to place along axis i.
            # NOTE: opset>=13 requires axes as a TENSOR input (not attribute).
            axes_unsq = [ax for ax in range(rank) if ax != i]
            if axes_unsq:
                axes_name = s.get_unique_name(f"dus_axes_unsq_{i}")
                axes_tensor = helper.make_tensor(
                    axes_name + "_val", TensorProto.INT64, [len(axes_unsq)],
                    np.asarray(axes_unsq, dtype=np.int64).tolist()
                )
                s.add_node(helper.make_node("Constant", [], [axes_name],
                                            name=s.get_unique_name(f"dus_const_axes_unsq_{i}"),
                                            value=axes_tensor))
                s.add_shape_info(axes_name, (len(axes_unsq),), np.int64)

                unsq_node = s.get_unique_name(f"dus_unsqueeze_{i}")
                range_unsq = s.get_unique_name(f"dus_range_unsq_{i}")
                s.add_node(helper.make_node("Unsqueeze", [range_i, axes_name], [range_unsq],
                                            name=unsq_node))
                s.add_shape_info(range_unsq, tuple([1] * i + [-1] + [1] * (rank - i - 1)), np.int64)
            else:
                # rank==1 case: no unsqueeze needed.
                range_unsq = range_i
                s.add_shape_info(range_unsq, (-1,), np.int64)

            # Expand to update shape
            exp_node = s.get_unique_name(f"dus_expand_{i}")
            range_exp = s.get_unique_name(f"dus_range_exp_{i}")
            s.add_node(
                helper.make_node("Expand", [range_unsq, upd_shape], [range_exp], name=exp_node)
            )
            s.add_shape_info(range_exp, tuple([-1] * rank), np.int64)

            # Broadcast start to update shape
            start_b_node = s.get_unique_name(f"dus_expand_start_{i}")
            start_b = s.get_unique_name(f"dus_start_b_{i}")
            s.add_node(
                helper.make_node("Expand", [start_clamped_i64s[i], upd_shape], [start_b], name=start_b_node)
            )
            s.add_shape_info(start_b, tuple([-1] * rank), np.int64)

            # idx_i = start_b + range_exp
            add_node = s.get_unique_name(f"dus_add_idx_{i}")
            idx_i = s.get_unique_name(f"dus_idx_{i}")
            s.add_node(helper.make_node("Add", [start_b, range_exp], [idx_i], name=add_node))
            s.add_shape_info(idx_i, tuple([-1] * rank), np.int64)

            # Unsqueeze to append as last dim for stacking (axes=[rank])
            axes_last = s.get_unique_name(f"dus_axes_last_{i}")
            axes_last_tensor = helper.make_tensor(
                axes_last + "_val", TensorProto.INT64, [1],
                np.asarray([rank], dtype=np.int64).tolist()
            )
            s.add_node(helper.make_node("Constant", [], [axes_last],
                                        name=s.get_unique_name(f"dus_const_axes_last_{i}"),
                                        value=axes_last_tensor))
            s.add_shape_info(axes_last, (1,), np.int64)

            unsq2_node = s.get_unique_name(f"dus_unsqueeze_last_{i}")
            idx_i_unsq = s.get_unique_name(f"dus_idx_unsq_{i}")
            s.add_node(helper.make_node("Unsqueeze", [idx_i, axes_last], [idx_i_unsq],
                                        name=unsq2_node))
            s.add_shape_info(idx_i_unsq, tuple([-1] * rank + [1]), np.int64)
            idx_components.append(idx_i_unsq)

        # Concat all idx_i_unsq along the last axis -> indices_nd: shape upd_shape + [rank]
        concat_node = s.get_unique_name("dus_concat_indices")
        indices_nd = s.get_unique_name("dus_indices_nd")
        s.add_node(
            helper.make_node(
                "Concat", idx_components, [indices_nd], name=concat_node, axis=rank
            )
        )
        s.add_shape_info(indices_nd, tuple([-1] * rank + [rank]), np.int64)

        # Reshape indices to [numel, rank]
        # Build shape vector: [numel_upd, rank]  (the Unsqueeze w/ axes tensor lives below)
        rank_scalar = s.get_unique_name("dus_rank_scalar")
        s.builder.add_initializer_from_scalar(rank_scalar, rank)
        # (the rest of this block now appears below using axes-as-input Unsqueeze)

        # Flatten updates to [numel]
        # Unsqueeze(numel_upd, axes=[0])  (axes provided as a tensor input)
        axes0_a = s.get_unique_name("dus_axes0_numel")
        axes0_a_tensor = helper.make_tensor(axes0_a + "_val", TensorProto.INT64, [1],
                                            np.asarray([0], dtype=np.int64).tolist())
        s.add_node(helper.make_node("Constant", [], [axes0_a],
                                    name=s.get_unique_name("dus_const_axes0_numel"),
                                    value=axes0_a_tensor))
        s.add_shape_info(axes0_a, (1,), np.int64)

        # Unsqueeze(rank_scalar, axes=[0])  (axes provided as a tensor input)
        axes0_b = s.get_unique_name("dus_axes0_rank")
        axes0_b_tensor = helper.make_tensor(axes0_b + "_val", TensorProto.INT64, [1],
                                            np.asarray([0], dtype=np.int64).tolist())
        s.add_node(helper.make_node("Constant", [], [axes0_b],
                                    name=s.get_unique_name("dus_const_axes0_rank"),
                                    value=axes0_b_tensor))
        s.add_shape_info(axes0_b, (1,), np.int64)

        # Unsqueeze(numel_upd, axes=[0]) for 1D shape param (axes as tensor)
        axes0_c = s.get_unique_name("dus_axes0_shape1d")
        axes0_c_tensor = helper.make_tensor(axes0_c + "_val", TensorProto.INT64, [1],
                                            np.asarray([0], dtype=np.int64).tolist())
        s.add_node(helper.make_node("Constant", [], [axes0_c],
                                    name=s.get_unique_name("dus_const_axes0_shape1d"),
                                    value=axes0_c_tensor))
        s.add_shape_info(axes0_c, (1,), np.int64)
        shape1d_node = s.get_unique_name("dus_shape1d")
        shape1d = s.get_unique_name("dus_shape1d_out")
        s.add_node(helper.make_node("Unsqueeze", [numel_upd, axes0_c], [shape1d],
                                    name=shape1d_node))
        s.add_shape_info(shape1d, (1,), np.int64)

        # Build [numel_upd, rank] vector and reshape indices
        usq_numel_node = s.get_unique_name("dus_unsq_numel_axes")
        numel_1d = s.get_unique_name("dus_numel_1d")
        s.add_node(helper.make_node("Unsqueeze", [numel_upd, axes0_a], [numel_1d],
                                    name=usq_numel_node))
        s.add_shape_info(numel_1d, (1,), np.int64)

        usq_rank_node = s.get_unique_name("dus_unsq_rank_axes")
        rank_1d = s.get_unique_name("dus_rank_1d")
        s.add_node(helper.make_node("Unsqueeze", [rank_scalar, axes0_b], [rank_1d],
                                    name=usq_rank_node))
        s.add_shape_info(rank_1d, (1,), np.int64)

        shape2d_node = s.get_unique_name("dus_shape2d")
        shape2d = s.get_unique_name("dus_shape2d_out")
        s.add_node(helper.make_node("Concat", [numel_1d, rank_1d], [shape2d],
                                    name=shape2d_node, axis=0))
        s.add_shape_info(shape2d, (2,), np.int64)

        reshape_idx_node = s.get_unique_name("dus_reshape_idx")
        indices_2d = s.get_unique_name("dus_indices_2d")
        s.add_node(
            helper.make_node("Reshape", [indices_nd, shape2d], [indices_2d], name=reshape_idx_node)
        )
        s.add_shape_info(indices_2d, (-1, rank), np.int64)

        reshape_upd_node = s.get_unique_name("dus_reshape_upd")
        upd_flat = s.get_unique_name("dus_upd_flat")
        s.add_node(
            helper.make_node("Reshape", [upd_name, shape1d], [upd_flat], name=reshape_upd_node)
        )
        # length is dynamic; dtype matches ref/update dtype
        s.add_shape_info(upd_flat, (-1,), ref_v.aval.dtype)

        # ScatterND(data=ref, indices=[numel, rank], updates=[numel]) -> out
        scatter_node = s.get_unique_name("dus_scatternd")
        s.add_node(
            helper.make_node(
                "ScatterND",
                inputs=[ref_name, indices_2d, upd_flat],
                outputs=[out_name],
                name=scatter_node,
            )
        )
        s.add_shape_info(out_name, ref_v.aval.shape, ref_v.aval.dtype)
