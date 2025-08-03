# jax2onnx/plugins/jax/lax/scatter.py

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Any

import numpy as np
import jax.numpy as jnp  # Keep for potential use in test cases or future needs
from jax import ShapeDtypeStruct, lax, core
from jax.lax import (
    ScatterDimensionNumbers,
    GatherScatterMode,
)
from onnx import helper
import onnx
from onnx import numpy_helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive
from .scatter_utils import _prepare_scatter_inputs_for_onnx

import logging

logger = logging.getLogger("jax2onnx.plugins.jax.lax.scatter")

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


def _count_reshape_to_shape_in_model(
    model: onnx.ModelProto, target_shape: Sequence[int]
) -> int:
    """Count Reshape nodes whose 2nd input is a constant equal to target_shape."""
    # Gather constant tensors by name from initializers and Constant nodes
    const_map = {}
    for init in model.graph.initializer:
        const_map[init.name] = numpy_helper.to_array(init)
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if (
                    attr.name == "value"
                    and getattr(attr, "t", None) is not None
                    and attr.t.name
                ):
                    const_map[attr.t.name] = numpy_helper.to_array(attr.t)

    def as_tuple(a):
        try:
            return tuple(int(x) for x in a.tolist())
        except Exception:
            return None

    count = 0
    tgt = tuple(target_shape)
    for node in model.graph.node:
        if node.op_type != "Reshape" or len(node.input) < 2:
            continue
        shp_name = node.input[1]
        if shp_name in const_map and as_tuple(const_map[shp_name]) == tgt:
            count += 1
    return count


@register_primitive(
    jaxpr_primitive=lax.scatter_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scatter.html",
    onnx=[
        {
            "component": "ScatterND",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterND.html",
        },
    ],
    since="v0.4.4",
    context="primitives.lax",
    component="scatter",
    testcases=[
        {
            "testcase": "scatter_set_axis0",
            "callable": lambda x: x.at[0].set(-100.0),
            "input_shapes": [(1, 1)],
        },
        {
            "testcase": "scatter_set_middle",
            "callable": lambda x: x.at[1].set(42.0),
            "input_shapes": [(3,)],
        },
        {
            "testcase": "scatter_correct_axis_determination",
            "callable": lambda op, idx, upd_scalar_batch: lax.scatter(
                op,
                idx,
                jnp.reshape(upd_scalar_batch, idx.shape[:-1]),
                ScatterDimensionNumbers(
                    update_window_dims=(),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                ),
            ),
            "input_shapes": [(5,), (1, 1, 1, 1), (1,)],
            "input_dtypes": [np.float32, np.int32, np.float32],
        },
        {
            "testcase": "scatter_updates_slice_needed_axis0",
            "callable": lambda op, idx, upd_scalar_batch: lax.scatter(
                op,
                idx,
                jnp.reshape(upd_scalar_batch, idx.shape[:-1]),
                ScatterDimensionNumbers(
                    update_window_dims=(),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                ),
            ),
            "input_shapes": [(5,), (1, 1, 1, 1), (1,)],
            "input_dtypes": [np.float32, np.int32, np.float32],
        },
        {
            "testcase": "scatter_from_user_warning_shapes_valid_jax",
            "callable": lambda operand, indices, updates_sliced_scalar_batch: lax.scatter(
                operand,
                indices,
                jnp.reshape(updates_sliced_scalar_batch, indices.shape[:-1]),
                ScatterDimensionNumbers(
                    update_window_dims=(),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                ),
            ),
            "input_shapes": [(5,), (1, 1, 1, 1), (1,)],
            "input_dtypes": [np.float32, np.int32, np.float32],
        },
        {
            "testcase": "scatter_user_error_scenario_precise",
            "callable": lambda operand, indices, updates: lax.scatter(
                operand,
                indices,
                updates,
                ScatterDimensionNumbers(
                    update_window_dims=(1, 2, 3),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                    operand_batching_dims=(),
                    scatter_indices_batching_dims=(),
                ),
                mode=GatherScatterMode.FILL_OR_DROP,
                unique_indices=False,
                indices_are_sorted=False,
            ),
            "input_shapes": [(5, 201, 1, 1), (2, 1), (2, 201, 1, 1)],
            "input_dtypes": [np.float32, np.int32, np.float32],
        },
        # ────────────────────────────────────────────────────────────────
        #  Window‑scatter (moved from examples/lax/scatter_window.py)
        # ────────────────────────────────────────────────────────────────
        {
            "testcase": "scatter_window_update_f64",
            # identical to the old `scatter_window_function`
            "callable": lambda operand, indices, updates: lax.scatter(
                operand=operand,
                scatter_indices=indices,
                updates=updates,
                dimension_numbers=ScatterDimensionNumbers(
                    update_window_dims=(1, 2, 3, 4),
                    inserted_window_dims=(),
                    scatter_dims_to_operand_dims=(1, 2),
                ),
                indices_are_sorted=True,
                unique_indices=True,
                mode=GatherScatterMode.FILL_OR_DROP,
            ),
            "input_values": [
                np.zeros((5, 266, 266, 1), dtype=np.float64),
                np.array([[10, 10]], dtype=np.int32),
                np.ones((1, 5, 256, 256, 1), dtype=np.float64),
            ],
            # keep the original flag so we only run the double‑precision variant
            "run_only_f64_variant": True,
        },
        {
            "testcase": "scatter_window_update_depth3_shapes_ok",
            "callable": lambda operand, indices, updates: lax.scatter(
                operand=operand,
                scatter_indices=indices,
                updates=updates,
                dimension_numbers=ScatterDimensionNumbers(
                    update_window_dims=(1, 2, 3, 4),
                    inserted_window_dims=(),
                    scatter_dims_to_operand_dims=(1, 2),
                ),
                indices_are_sorted=True,
                unique_indices=True,
                mode=GatherScatterMode.FILL_OR_DROP,
            ),
            "input_values": [
                np.zeros((5, 266, 266, 1), dtype=np.float64),
                np.array([[10, 10]], dtype=np.int32),
                np.ones((1, 5, 256, 256, 1), dtype=np.float64),
            ],
            "run_only_f64_variant": True,
            # ONNX graph sanity checks: exactly one [-1,3] and one [-1,1] Reshape.
            "post_check_onnx_graph": lambda model: (
                (
                    lambda n_idx, n_upd: (
                        True
                        if (n_idx == 1 and n_upd == 1)
                        else (_ for _ in ()).throw(
                            AssertionError(
                                f"Expected exactly one Reshape to [-1,3] and [-1,1]; got n_idx={n_idx}, n_upd={n_upd}"
                            )
                        )
                    )
                )(
                    _count_reshape_to_shape_in_model(model, [-1, 3]),
                    _count_reshape_to_shape_in_model(model, [-1, 1]),
                )
            ),
        },
    ],
)
class ScatterPlugin(PrimitiveLeafPlugin):
    @staticmethod
    def abstract_eval(
        operand: core.ShapedArray,
        indices: core.ShapedArray,
        updates: core.ShapedArray,
        update_jaxpr,
        *,
        dimension_numbers: ScatterDimensionNumbers,
        indices_are_sorted: bool,
        unique_indices: bool,
        mode: GatherScatterMode | str | None,
        **params,
    ):
        return core.ShapedArray(operand.shape, operand.dtype)

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[Any],
        node_outputs: Sequence[Any],
        params: dict[str, Any],
    ):
        operand_v, indices_v, updates_v = node_inputs
        out_v = node_outputs[0]
        out_name = s.get_name(out_v)

        # original operand info
        aval = operand_v.aval
        op_shape = tuple(aval.shape)
        op_dtype = np.dtype(aval.dtype)

        # prepare inputs
        logger.info(
            f"Preparing inputs for ONNX ScatterND with {params['dimension_numbers']}"
        )
        in_name, idx_name, upd_name = _prepare_scatter_inputs_for_onnx(
            s, operand_v, indices_v, updates_v, params["dimension_numbers"]
        )

        # emit ScatterND
        attrs: dict[str, Any] = {}
        if s.builder.opset >= 16:
            attrs["reduction"] = "none"
        s.add_node(
            helper.make_node(
                "ScatterND",
                [in_name, idx_name, upd_name],
                [out_name],
                name=s.get_unique_name(f"scatter_nd_{out_name}"),
                **attrs,
            )
        )

        # register output
        s.shape_env[out_name] = ShapeDtypeStruct(op_shape, op_dtype)
        s.add_shape_info(out_name, op_shape, op_dtype)
        logger.debug(f"[ScatterPlugin] '{out_name}' -> {op_shape}/{op_dtype}")
