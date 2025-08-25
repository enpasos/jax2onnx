import jax
import jax.numpy as jnp
import numpy as np

from jax2onnx.plugins.plugin_system import register_example

# This test isolates a pattern where `scatter_add` and `scatter_mul` are used
# within the branches of a `jnp.where` conditional. This pattern was identified
# as a potential source of numerical discrepancies during the conversion of


def cond_scatter_add_mul_f64(
    operand, scatter_indices, updates_for_add, updates_for_mul
):
    """
    This function reproduces a pattern where different scatter operations are
    placed in separate branches of a conditional clause. This is intended to
    stress-test the conversion plugins for scatter ops.

    Args:
        operand: The main data array.
        scatter_indices: Indices for the scatter operations.
        updates_for_add: Updates for the 'true' branch (scatter_add).
        updates_for_mul: Updates for the 'false' branch (scatter_mul).
    """
    dimension_numbers = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1, 2, 3),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )

    # Branch 1: A scatter 'add' operation.
    branch_if_true = jax.lax.scatter_add(
        operand, scatter_indices, updates_for_add, dimension_numbers
    )

    # Branch 2: A scatter 'mul' operation.
    branch_if_false = jax.lax.scatter_mul(
        operand, scatter_indices, updates_for_mul, dimension_numbers
    )

    # Conditional logic that will be lowered to `lax.select_n`.
    condition = jnp.sum(operand) > 0.0
    final_output = jnp.where(condition, branch_if_true, branch_if_false)

    # FIX: Return a tuple instead of a dictionary to match the expected flat output.
    # The order must match the ONNX graph's output order, which is (condition, final_tensor).
    return (condition, final_output)


register_example(
    component="cond_scatter_add_mul",
    description="Tests scatter_add/mul inside jnp.where branches",
    since="v0.6.4",
    context="examples.lax",
    children=[],
    testcases=[
        {
            "testcase": "cond_scatter_add_mul_f64_a",
            "callable": cond_scatter_add_mul_f64,
            "input_values": [
                # operand: shape (1, 5, 4, 4)
                np.ones((1, 5, 4, 4), dtype=np.float64),
                # scatter_indices: shape (2, 1) -> must contain only valid indices (i.e., 0)
                np.array([[0], [0]], dtype=np.int64),
                # updates_add: shape (2, 5, 4, 4)
                np.full((2, 5, 4, 4), 2.0, dtype=np.float64),
                # updates_mul: shape (2, 5, 4, 4)
                np.full((2, 5, 4, 4), 3.0, dtype=np.float64),
            ],
            "expected_output_shapes": [
                (),  # condition
                (1, 5, 4, 4),  # final_output
            ],
            "expected_output_dtypes": [jnp.bool_, jnp.float64],
            "run_only_f64_variant": True,
        },
        {
            "testcase": "cond_scatter_add_mul_f64_b",
            "callable": cond_scatter_add_mul_f64,
            "input_values": [
                # operand: shape (4, 5, 4, 4) to allow for more indices
                np.ones((4, 5, 4, 4), dtype=np.float64),
                # scatter_indices: shape (2, 1) -> using indices 1 and 3
                np.array([[1], [3]], dtype=np.int64),
                # updates_add: shape (2, 5, 4, 4) with a different value
                np.full((2, 5, 4, 4), 5.0, dtype=np.float64),
                # updates_mul: shape (2, 5, 4, 4) with a different value
                np.full((2, 5, 4, 4), 7.0, dtype=np.float64),
            ],
            "expected_output_shapes": [
                (),  # condition
                (4, 5, 4, 4),  # final_output
            ],
            "expected_output_dtypes": [jnp.bool_, jnp.float64],
            "run_only_f64_variant": True,
        },
    ],
)
