# file: jax2onnx/plugins/jax/lax/dynamic_update_slice.py

from __future__ import annotations
from typing import TYPE_CHECKING, Sequence, Any
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
            "component": "ScatterElements",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterElements.html",
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
    ],
)
class DynamicUpdateSlice1D(PrimitiveLeafPlugin):
    """1-D lowering: lax.dynamic_update_slice -> ONNX ScatterElements(axis=0)."""

    @staticmethod
    def abstract_eval(ref, update, start_index):
        # Output matches ref in shape/dtype
        return jcore.ShapedArray(ref.shape, ref.dtype)

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[Any],
        node_outputs: Sequence[Any],
        params: dict[str, Any],
    ) -> None:
        ref_v, upd_v, start_v = node_inputs
        out_v = node_outputs[0]

        # Only 1-D for this first cut (covers grad(cumsum(x)[-1]) pattern)
        if len(ref_v.aval.shape) != 1:
            raise NotImplementedError(
                "dynamic_update_slice: only 1-D supported currently."
            )

        ref_name = s.get_name(ref_v)
        upd_name = s.get_name(upd_v)
        start_name = s.get_name(start_v)
        out_name = s.get_name(out_v)

        # Constants 0, 1 as INT64 initializers (for Gather/Range)
        zero_i64 = s.get_unique_name("dus_zero_i64")
        one_i64 = s.get_unique_name("dus_one_i64")
        s.builder.add_initializer_from_scalar(zero_i64, 0)  # INT64
        s.builder.add_initializer_from_scalar(one_i64, 1)  # INT64

        # Cast start -> INT64 (Range expects INT64)
        cast_node = s.get_unique_name("dus_cast_start")
        start_i64 = s.get_unique_name("dus_start_i64")
        s.add_node(
            helper.make_node(
                "Cast", [start_name], [start_i64], name=cast_node, to=TensorProto.INT64
            )
        )
        s.add_shape_info(start_i64, (), np.int64)

        shape_node = s.get_unique_name("dus_shape_upd")
        upd_shape = s.get_unique_name("dus_upd_shape")
        s.add_node(helper.make_node("Shape", [upd_name], [upd_shape], name=shape_node))
        s.add_shape_info(upd_shape, (1,), np.int64)  # 1D i64

        gather_node = s.get_unique_name("dus_gather_len")
        upd_len = s.get_unique_name("dus_upd_len")
        s.add_node(
            helper.make_node(
                "Gather", [upd_shape, zero_i64], [upd_len], name=gather_node, axis=0
            )
        )
        s.add_shape_info(upd_len, (), np.int64)  # scalar i64

        add_node = s.get_unique_name("dus_add_stop")
        stop_i64 = s.get_unique_name("dus_stop_i64")
        s.add_node(
            helper.make_node("Add", [start_i64, upd_len], [stop_i64], name=add_node)
        )
        s.add_shape_info(stop_i64, (), np.int64)

        range_node = s.get_unique_name("dus_range_idx")
        indices = s.get_unique_name("dus_indices")
        s.add_node(
            helper.make_node(
                "Range", [start_i64, stop_i64, one_i64], [indices], name=range_node
            )
        )
        s.add_shape_info(indices, (-1,), np.int64)  # dynamic 1D

        scatter_node = s.get_unique_name("dus_scatter")
        s.add_node(
            helper.make_node(
                "ScatterElements",
                inputs=[ref_name, indices, upd_name],
                outputs=[out_name],
                name=scatter_node,
                axis=0,
            )
        )
        s.add_shape_info(out_name, ref_v.aval.shape, ref_v.aval.dtype)
