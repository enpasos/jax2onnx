import jax.numpy as jnp


def cond_scatter_repro_f64(operand, scatter_indices, updates_for_set, updates_for_add):
    """
    This function reproduces the pattern that causes 'Missing value_info' errors.
    It contains two different scatter operations, each in a separate branch of a
    `jnp.where` clause, mimicking the logic flow derived from the logs.

    Args:
        operand: The main data array.
        scatter_indices: 1D array of indices.
        updates_for_set: Updates for the 'true' branch (scatter set).
        updates_for_add: Updates for the 'false' branch (scatter add).
    """
    # Branch 1: A scatter 'set' operation.
    # This corresponds to ScatterDimensionNumbers(update_window_dims=(1, 2, 3),
    # inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,))
    branch_if_true = operand.at[scatter_indices, :, :, :].set(updates_for_set)

    # Branch 2: A scatter 'add' operation.
    # This corresponds to ScatterDimensionNumbers(update_window_dims=(0, 1, 2, 3),
    # inserted_window_dims=(), scatter_dims_to_operand_dims=(1,))
    branch_if_false = operand.at[:, scatter_indices, :, :].add(updates_for_add)

    # Conditional logic that will be lowered to `lax.select_n`. This is the
    # trigger for the bug, as the scatter ops are in its branches.
    condition = jnp.sum(operand) > 0.0
    final_output = jnp.where(condition, branch_if_true, branch_if_false)

    # Return a PyTree to match the complexity of the original model.
    return {"final_output": final_output, "condition_used": condition}


# register_example(
#     component="cond_scatter_repro",
#     description="""
#     Tests a scenario derived from production logs where different scatter
#     operations are placed within the branches of a conditional (jnp.where).
#     This specific pattern has been shown to cause `Missing value_info`
#     errors during ONNX export. The bug occurs because the scatter plugins
#     generate intermediate tensors (e.g., 'updates_reshaped_...') without
#     correctly registering their metadata in the ONNX graph when they are
#     part of a conditional branch.
#     """,
#     since="v0.7.0",
#     context="examples.lax",
#     children=[],
#     testcases=[
#         {
#             "testcase": "cond_scatter_repro_f64",
#             "callable": cond_scatter_repro_f64,
#             "input_shapes": [
#                 (5, 208, 1, 1),  # operand
#                 (2,),  # scatter_indices
#                 (2, 208, 1, 1),  # updates_for_set
#                 (5, 2, 1, 1),  # updates_for_add
#             ],
#             "input_dtypes": [jnp.float64, jnp.int64, jnp.float64, jnp.float64],
#             # Expected outputs from the PyTree: {"final_output": ..., "condition_used": ...}
#             "expected_output_shapes": [
#                 (5, 208, 1, 1),  # final_output
#                 (),  # condition_used
#             ],
#             "expected_output_dtypes": [jnp.float64, jnp.bool_],
#             "run_only_f64_variant": True,
#         },
#     ],
# )
