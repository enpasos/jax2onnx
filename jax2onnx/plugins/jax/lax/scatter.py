# file: jax2onnx/plugins/jax/lax/scatter.py

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

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive
from .scatter_utils import _prepare_scatter_inputs_for_onnx

import logging

logger = logging.getLogger("jax2onnx.plugins.jax.lax.scatter")

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


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
