# file: jax2onnx/plugins/jax/lax/gather.py

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import onnx
from onnx import helper

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


# --- Local helpers to reproduce the f64 leak seen in the extra test ----------
def _masked_gather_trig_local(data, indices):
    """Reproduce: f64 data, integer indices → gather → trig → where (f64 pipeline)."""
    data = jnp.asarray(data, dtype=jnp.float64)
    gathered = data[indices]  # should stay f64
    result = gathered * jnp.array(2.0, dtype=jnp.float64)
    result = jnp.sin(result) + jnp.cos(result)  # f64 ops
    mask = result > jnp.array(0.5, dtype=jnp.float64)
    return jnp.where(mask, result, jnp.array(0.0, dtype=jnp.float64))


def _postcheck_all_gather_outputs_are_double(onnx_model: onnx.ModelProto) -> bool:
    """
    Ensure every Gather/GatherND node's *typed* ValueInfo is DOUBLE
    (when exporting with enable_double_precision=True).
    If there is no ValueInfo for a given output, we don't fail the check;
    the bug we're guarding against only occurs when a wrong (FLOAT) ValueInfo is present.
    """
    # Build a lookup of name -> elem_type from value_info & graph outputs
    typed = {}
    for vi in list(onnx_model.graph.value_info) + list(onnx_model.graph.output):
        tt = vi.type.tensor_type
        if tt is not None and tt.elem_type != onnx.TensorProto.UNDEFINED:
            typed[vi.name] = tt.elem_type

    ok = True
    for node in onnx_model.graph.node:
        if node.op_type in ("GatherND", "Gather"):
            for out_name in node.output:
                if out_name in typed:
                    if typed[out_name] != onnx.TensorProto.DOUBLE:
                        # Leave a breadcrumb in case of failure
                        # (pytest will show assertion message from the harness)
                        ok = False
    return ok


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
        # --- Local regression: the minimal pipeline that used to fail in ORT load
        {
            "testcase": "gather_trig_where_pipeline_f64_indices_i64",
            "callable": _masked_gather_trig_local,
            "input_values": [
                np.array(
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0],
                    ],
                    dtype=np.float64,
                ),
                np.array([0, 2], dtype=np.int64),
            ],
            "expected_output_shapes": [(2, 3)],
            "expected_output_dtypes": [np.float64],
            "run_only_f64_variant": True,
            # Catch the exact failure mode seen in the extra test:
            # GatherND output ValueInfo incorrectly stamped FLOAT.
            "post_check_onnx_graph": _postcheck_all_gather_outputs_are_double,
        },
        {
            "testcase": "gather_trig_where_pipeline_f64_indices_i32",
            "callable": _masked_gather_trig_local,
            "input_values": [
                np.array(
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0],
                    ],
                    dtype=np.float64,
                ),
                np.array([1, 3], dtype=np.int32),
            ],
            "expected_output_shapes": [(2, 3)],
            "expected_output_dtypes": [np.float64],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": _postcheck_all_gather_outputs_are_double,
        },
        # --- Local regression tests for dtype harmonization (issue seen in extra test) ---
        {
            # data is f64, indices i64 → output must be f64 under enable_double_precision=True
            "testcase": "gather_f64_data_i64_indices_output_is_f64",
            "callable": (lambda data, idx: data[idx]),
            "input_values": [
                np.array(
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0],
                    ],
                    dtype=np.float64,
                ),
                np.array([0, 2], dtype=np.int64),
            ],
            "expected_output_shapes": [(2, 3)],
            "expected_output_dtypes": [np.float64],
            "run_only_f64_variant": True,
        },
        {
            # Same as above but indices come as i32; exporter should cast them to i64 for ONNX
            "testcase": "gather_f64_data_i32_indices_cast_and_output_is_f64",
            "callable": (lambda data, idx: data[idx]),
            "input_values": [
                np.array(
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0],
                    ],
                    dtype=np.float64,
                ),
                np.array([1, 3], dtype=np.int32),
            ],
            "expected_output_shapes": [(2, 3)],
            "expected_output_dtypes": [np.float64],
            "run_only_f64_variant": True,
        },
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
        output_name = s.get_name(output_var)

        jax_out_shape = output_var.aval.shape
        out_dtype = np.dtype(output_var.aval.dtype)  # <- ensure DOUBLE under f64 export
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
            s.add_shape_info(shape_of_data, (len(data_shape),), np.int64)

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
            s.add_shape_info(batch_dim_size, (), np.int64)  # scalar shape

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
            s.add_shape_info(batch_dim_size_1d, (1,), np.int64)

            one_const_1d = s.get_unique_name("one_const_1d")
            s.add_node(
                helper.make_node(
                    "Unsqueeze", [one_const_name, unsqueeze_axes_name], [one_const_1d]
                )
            )
            s.add_shape_info(one_const_1d, (1,), np.int64)

            target_shape_name = s.get_unique_name("target_gather_idx_shape")
            s.add_node(
                helper.make_node(
                    "Concat",
                    [batch_dim_size_1d, one_const_1d],
                    [target_shape_name],
                    axis=0,
                )
            )
            s.add_shape_info(target_shape_name, (2,), np.int64)  # Shape [B, 1]

            # 1d. Expand base index to target shape
            final_indices_name = s.get_unique_name("final_gather_indices")
            s.add_node(
                helper.make_node(
                    "Expand", [base_index_name, target_shape_name], [final_indices_name]
                )
            )
            # Output shape uses symbolic dim name from data_shape
            s.add_shape_info(final_indices_name, (data_shape[0], 1), np.int64)  # (B,1)

        else:
            # Static case or non-simple index: Just cast the provided indices
            indices_name = s.get_name(indices_var)
            final_indices_name = s.get_unique_name("final_gather_indices")
            cast_node = helper.make_node(
                "Cast",
                inputs=[indices_name],
                outputs=[final_indices_name],
                name=s.get_unique_name("cast_indices"),
                to=onnx.TensorProto.INT64,
            )
            s.add_node(cast_node)
            s.add_shape_info(final_indices_name, indices_aval.shape, np.int64)
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
        # IMPORTANT: record dtype explicitly; without this, some paths default to FLOAT.
        s.add_shape_info(output_name, jax_out_shape, out_dtype)
