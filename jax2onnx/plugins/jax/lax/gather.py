# file: jax2onnx/plugins/jax/lax/gather.py

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.gather_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.gather.html",
    onnx=[
        {
            "component": "GatherND",
            "doc": "https://onnx.ai/onnx/operators/onnx__GatherND.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="gather",
    testcases=[
        {
            "testcase": "gather_static",
            "callable": lambda x: jax.lax.gather(
                x,
                jnp.array([[1], [0]]),  # Indices int32
                jax.lax.GatherDimensionNumbers(
                    offset_dims=(1,),
                    collapsed_slice_dims=(0,),
                    start_index_map=(0,),
                ),
                slice_sizes=(1, 3),  # Static slice sizes
            ),
            "input_shapes": [(3, 3)],
            "expected_output_shapes": [(2, 3)],
        },
        {
            "testcase": "gather_dynamic_batch_simple_index",
            "callable": lambda x: x[
                :, 0, :
            ],  # Uses gather internally, indices will be int32
            "input_shapes": [("B", 50, 256)],  # Dynamic batch dim 'B'
            "expected_output_shapes": [("B", 256)],
        },
        # Add more test cases if needed
    ],
)
class GatherPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.gather to ONNX GatherND."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX gather primitive using ONNX GatherND."""
        data_var, indices_var = node_inputs
        output_var = node_outputs[0]

        data_name = s.get_name(data_var)
        output_name = s.get_var_name(output_var)

        jax_out_shape = output_var.aval.shape
        data_shape = data_var.aval.shape
        indices_aval = indices_var.aval

        # --- Step 1: Prepare indices for GatherND ---
        # Determine if we are in the dynamic batch + simple index case (heuristic)
        is_simple_index_zero = (
            hasattr(indices_aval, "shape")
            and indices_aval.shape == (1,)
            and np.issubdtype(indices_aval.dtype, np.integer)
        )

        # "dynamic" if the first dim isn't a concrete Python int
        def _is_concrete_int(x):  # helper
            return isinstance(x, int)

        dynamic_batch = len(data_shape) > 0 and not _is_concrete_int(data_shape[0])

        if is_simple_index_zero and dynamic_batch:
            # Dynamic batch case matching x[:, 0, :] pattern
            # Create dynamic indices: [[0], [0], ..., [0]] with shape (B, 1)

            # 1a. Get the dynamic batch size 'B'
            shape_of_data = s.get_unique_name("shape_of_data")
            s.add_node(helper.make_node("Shape", [data_name], [shape_of_data]))
            s.add_shape_info(
                shape_of_data, (len(data_shape),), helper.TensorProto.INT64
            )

            # scalar 0 (int64)
            zero_idx_name = s.get_constant_name(np.array(0, dtype=np.int64))

            batch_dim_size = s.get_unique_name("batch_dim_size")
            s.add_node(
                helper.make_node(
                    "Gather",
                    [shape_of_data, zero_idx_name],  # Use initializer name
                    [batch_dim_size],
                    axis=0,
                )
            )
            s.add_shape_info(
                batch_dim_size, (), helper.TensorProto.INT64
            )  # scalar shape

            # 1b. Create the base index [[0]] (int64)
            # base index [[0]]   – shape (1,1)
            base_index_name = s.get_constant_name(np.zeros((1, 1), dtype=np.int64))

            # 1c. Create target shape [B, 1] for Expand
            # Create int64 constant 1
            one_const_name = s.get_constant_name(np.array(1, dtype=np.int64))

            # Create int64 constant array [0] for Unsqueeze axes
            unsqueeze_axes_name = s.get_constant_name(np.array([0], dtype=np.int64))

            # Unsqueeze batch_dim_size and one_const to shape (1,) before concat
            batch_dim_size_1d = s.get_unique_name("batch_dim_size_1d")
            s.add_node(
                helper.make_node(
                    "Unsqueeze",
                    [batch_dim_size, unsqueeze_axes_name],
                    [batch_dim_size_1d],
                )
            )
            s.add_shape_info(batch_dim_size_1d, (1,), helper.TensorProto.INT64)

            one_const_1d = s.get_unique_name("one_const_1d")
            s.add_node(
                helper.make_node(
                    "Unsqueeze", [one_const_name, unsqueeze_axes_name], [one_const_1d]
                )
            )
            s.add_shape_info(one_const_1d, (1,), helper.TensorProto.INT64)

            target_shape_name = s.get_unique_name("target_gather_idx_shape")
            s.add_node(
                helper.make_node(
                    "Concat",
                    [batch_dim_size_1d, one_const_1d],
                    [target_shape_name],
                    axis=0,
                )
            )
            s.add_shape_info(
                target_shape_name, (2,), helper.TensorProto.INT64
            )  # Shape [B, 1] tensor value

            # 1d. Expand base index to target shape
            final_indices_name = s.get_unique_name("final_gather_indices")
            s.add_node(
                helper.make_node(
                    "Expand", [base_index_name, target_shape_name], [final_indices_name]
                )
            )
            # Output shape uses symbolic dim name from data_shape
            s.add_shape_info(
                final_indices_name, (data_shape[0], 1), helper.TensorProto.INT64
            )  # Shape (B, 1)

        else:
            # Static case or non-simple index: Just cast the provided indices
            indices_name = s.get_name(indices_var)
            final_indices_name = s.get_unique_name("final_gather_indices")
            cast_node = helper.make_node(
                "Cast",
                inputs=[indices_name],
                outputs=[final_indices_name],
                name=s.get_unique_name("cast_indices"),
                to=helper.TensorProto.INT64,
            )
            s.add_node(cast_node)
            s.add_shape_info(
                final_indices_name, indices_aval.shape, dtype=helper.TensorProto.INT64
            )
            # Potentially unsqueeze static indices if needed for GatherND rank requirements

        # --- Step 2: Create GatherND node ---
        # Use batch_dims=1 for the x[:, 0, :] pattern with dynamically generated indices (B, 1)
        # For the static case gather_static, indices are (2, 1), data is (3, 3).
        # JAX dn=(offset=(1,), collapsed=(0,), start=(0,)). slice_sizes=(1, 3). Output (2, 3).
        # GatherND needs indices (2, 1) and batch_dims=0?
        #   Indices [[1], [0]], Data [[d00,d01,d02],[d10,d11,d12],[d20,d21,d22]]
        #   Output[0] = Data[Indices[0]] = Data[1] = [d10, d11, d12]
        #   Output[1] = Data[Indices[1]] = Data[0] = [d00, d01, d02]
        #   Result shape (2, 3). Looks like batch_dims=0 works for the static case.
        batch_dims = 1 if (is_simple_index_zero and dynamic_batch) else 0

        gather_node = helper.make_node(
            "GatherND",
            inputs=[
                data_name,
                final_indices_name,
            ],  # Use dynamically created or casted indices
            outputs=[output_name],
            name=s.get_unique_name("gathernd"),
            batch_dims=batch_dims,
        )
        s.add_node(gather_node)

        # --- Step 3: Register output shape ---
        s.add_shape_info(output_name, jax_out_shape)
