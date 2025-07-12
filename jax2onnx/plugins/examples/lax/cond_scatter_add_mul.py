import jax
import jax.numpy as jnp

from jax2onnx.plugin_system import register_example

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
        # TODO: enable testcases
        # {
        #     "testcase": "cond_scatter_add_mul_f64",
        #     "callable": cond_scatter_add_mul_f64,
        #     "input_shapes": [
        #         (1, 5, 4, 4),  # operand
        #         (2, 1),  # scatter_indices
        #         (2, 5, 4, 4),  # updates_for_add
        #         (2, 5, 4, 4),  # updates_for_mul
        #     ],
        #     "input_dtypes": [jnp.float64, jnp.int64, jnp.float64, jnp.float64],
        #     # FIX: Reorder expected outputs to match the function's tuple return order.
        #     # Expected outputs from the tuple: (condition, final_output)
        #     "expected_output_shapes": [
        #         (),  # condition
        #         (1, 5, 4, 4),  # final_output
        #     ],
        #     "expected_output_dtypes": [jnp.bool_, jnp.float64],
        #     "run_only_f64_variant": True,
        #     #"skip_numeric_validation": True,  # TODO: Enable numeric validation
        # },
    ],
)
