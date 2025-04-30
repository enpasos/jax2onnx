# jax2onnx/plugins/jax/lax/scatter.py
from __future__ import annotations
from typing import TYPE_CHECKING, Sequence, Any
import numpy as np
from jax import lax, core
from onnx import helper, TensorProto

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # only for static type checkers
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


# ---------------------------------------------------------------------
# 1. primitive alias
# ---------------------------------------------------------------------
scatter_p = lax.scatter_p
# ---------------------------------------------------------------------


@register_primitive(
    jaxpr_primitive=scatter_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.scatter.html",
    onnx=[
        {
            "component": "ScatterElements",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterElements.html",
        }
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
    ],
)
class ScatterPlugin(PrimitiveLeafPlugin):
    # -----------------------------------------------------------------
    # abstract-eval
    # -----------------------------------------------------------------
    @staticmethod
    def abstract_eval(
        operand: core.ShapedArray,
        indices: core.ShapedArray,
        updates: core.ShapedArray,
        update_jaxpr,
        *,
        dimension_numbers,
        **__,  # scatter has a lot of params we do not need here
    ):
        # result has same shape/dtype as operand
        return core.ShapedArray(operand.shape, operand.dtype)

    # -----------------------------------------------------------------
    # lowering to ONNX
    # -----------------------------------------------------------------
    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[Any],
        node_outputs: Sequence[Any],
        params: dict[str, Any],
    ):
        operand_v, indices_v, updates_v = node_inputs
        out_v = node_outputs[0]

        operand_name = s.get_name(operand_v)
        indices_name = s.get_name(indices_v)
        updates_name = s.get_name(updates_v)
        out_name = s.get_name(out_v)

        # --- Get shapes and dtypes ---
        operand_shape = tuple(operand_v.aval.shape)
        operand_rank = len(operand_shape)
        operand_dtype = operand_v.aval.dtype

        indices_shape = tuple(indices_v.aval.shape)
        indices_rank = len(indices_shape)
        indices_dtype = indices_v.aval.dtype  # Original dtype

        updates_shape = tuple(updates_v.aval.shape)
        updates_rank = len(updates_shape)
        updates_dtype = updates_v.aval.dtype

        # ---- Cast indices to INT64 if needed ----
        indices_target_dtype = np.int64
        if indices_dtype != indices_target_dtype:
            cast_indices_out = s.get_unique_name("indices_int64")
            s.add_node(
                helper.make_node(
                    "Cast",
                    inputs=[indices_name],
                    outputs=[cast_indices_out],
                    name=s.get_unique_name("cast_indices"),
                    to=int(TensorProto.INT64),
                )
            )
            # Use original indices_shape but target dtype for shape info
            s.add_shape_info(cast_indices_out, indices_shape, np.dtype("int64"))
            indices_name = cast_indices_out
            indices_dtype = indices_target_dtype  # Update dtype after cast

        # --- Ensure Indices Rank Matches Operand Rank ---
        if indices_rank < operand_rank:
            # Convert to list for manipulation, then back to tuple for final assignment
            target_indices_shape_list = list(indices_shape)
            # Add leading singleton dimensions to match rank
            for _ in range(operand_rank - indices_rank):
                target_indices_shape_list.insert(0, 1)
            target_indices_shape = tuple(target_indices_shape_list)

            reshape_indices_out = s.get_unique_name("indices_reshaped")
            reshape_indices_shape_const = np.array(target_indices_shape, dtype=np.int64)
            reshape_indices_shape_name = s.get_constant_name(
                reshape_indices_shape_const
            )

            s.add_node(
                helper.make_node(
                    "Reshape",
                    inputs=[indices_name, reshape_indices_shape_name],
                    outputs=[reshape_indices_out],
                    name=s.get_unique_name("reshape_indices_rank"),
                )
            )
            # Use INT64 dtype for ONNX indices
            s.add_shape_info(
                reshape_indices_out, target_indices_shape, np.dtype("int64")
            )
            indices_name = reshape_indices_out
            indices_shape = target_indices_shape  # Now it's a proper tuple
        elif indices_rank > operand_rank:
            raise ValueError(
                f"Scatter indices rank ({indices_rank}) cannot be greater than operand rank ({operand_rank})"
            )

        # --- Ensure Updates Rank Matches Operand Rank ---
        if updates_rank < operand_rank:
            # Convert to list for manipulation, then back to tuple for final assignment
            target_updates_shape_list = list(updates_shape)
            for _ in range(operand_rank - updates_rank):
                target_updates_shape_list.insert(0, 1)
            target_updates_shape = tuple(target_updates_shape_list)

            reshape_updates_out = s.get_unique_name("updates_reshaped")
            reshape_updates_shape_const = np.array(target_updates_shape, dtype=np.int64)
            reshape_updates_shape_name = s.get_constant_name(
                reshape_updates_shape_const
            )

            s.add_node(
                helper.make_node(
                    "Reshape",
                    inputs=[updates_name, reshape_updates_shape_name],
                    outputs=[reshape_updates_out],
                    name=s.get_unique_name("reshape_updates_rank"),
                )
            )
            s.add_shape_info(reshape_updates_out, target_updates_shape, updates_dtype)
            updates_name = reshape_updates_out
            updates_shape = target_updates_shape  # Now it's a proper tuple
        elif updates_rank > operand_rank:
            raise ValueError(
                f"Scatter updates rank ({updates_rank}) cannot be greater than operand rank ({operand_rank})"
            )

        # ---- (Optional) Broadcast 'updates' to match 'indices' shape ----
        if indices_shape != updates_shape:
            shape_const = np.array(indices_shape, dtype=np.int64)
            shape_const_name = s.get_constant_name(shape_const)
            expanded_updates = s.get_unique_name("updates_broadcast")
            s.add_node(
                helper.make_node(
                    "Expand",
                    inputs=[updates_name, shape_const_name],
                    outputs=[expanded_updates],
                    name=s.get_unique_name("expand_updates"),
                )
            )
            s.add_shape_info(
                expanded_updates,
                indices_shape,
                updates_dtype,
            )
            updates_name = expanded_updates

        # ---- ScatterElements Node ----
        scatter_node = helper.make_node(
            "ScatterElements",
            inputs=[operand_name, indices_name, updates_name],
            outputs=[out_name],
            name=s.get_unique_name("scatter"),
            axis=0,
        )
        s.add_node(scatter_node)
        s.add_shape_info(out_name, operand_shape, operand_dtype)


# --- Register abstract eval ---
# This should remain the same as it defines the JAX-level behavior
scatter_p.def_abstract_eval(ScatterPlugin.abstract_eval)
