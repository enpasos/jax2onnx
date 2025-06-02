# test_jax_scatter_add.py
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


def scatter_add_simplified():
    print("\n--- Testing: scatter_add_mismatched_window_dims_simplified ---")
    try:
        operand = jnp.array(np.zeros((2, 5), dtype=np.float64))
        indices = jnp.array(np.array([[4]], dtype=np.int32))
        updates = jnp.array(np.ones((2, 2), dtype=np.float64))

        dimension_numbers = lax.ScatterDimensionNumbers(
            update_window_dims=(0, 1),
            inserted_window_dims=(),
            scatter_dims_to_operand_dims=(1,),
            operand_batching_dims=(),
            scatter_indices_batching_dims=(),
        )

        # Direct call
        # result = lax.scatter_add(operand, indices, updates, dimension_numbers)

        # Call with jax.jit for closer behavior to tracing
        jitted_scatter_add = jax.jit(lax.scatter_add, static_argnums=(3,))
        result = jitted_scatter_add(operand, indices, updates, dimension_numbers)

        print("JAX direct call PASSED")
        print("Result shape:", result.shape)
        # print("Result:\n", result) # Optional: print result
    except Exception as e:
        print("JAX direct call FAILED")
        print(f"Error type: {type(e)}")
        print(f"Error message: {e}")


def test_scatter_add_user_report():
    print("\n--- Testing: scatter_add_mismatched_window_dims_from_user_report ---")
    try:
        operand = jnp.array(np.zeros((5, 208, 1, 1), dtype=np.float64))
        indices = jnp.array(np.array([4], dtype=np.int32))  # Original shape (1,)
        updates = jnp.array(np.ones((5, 200, 1, 1), dtype=np.float64))

        dimension_numbers = lax.ScatterDimensionNumbers(
            update_window_dims=(0, 1, 2, 3),
            inserted_window_dims=(),
            scatter_dims_to_operand_dims=(
                1,
            ),  # scatter_dims_to_operand_dims must have len == indices.shape[-1]
            operand_batching_dims=(),
            scatter_indices_batching_dims=(),
        )

        # JAX scatter expects indices to be of shape updates_batch_dims + (index_vector_dim,)
        # Here, update_window_dims=(0,1,2,3) for updates(5,200,1,1) => updates_batch_dims = ()
        # scatter_dims_to_operand_dims=(1,) => index_vector_dim = 1
        # Expected indices shape: () + (1,) = (1,)
        # Provided indices shape is indeed (1,)

        # Direct call
        # result = lax.scatter_add(operand, indices, updates, dimension_numbers)

        # Call with jax.jit
        jitted_scatter_add = jax.jit(lax.scatter_add, static_argnums=(3,))
        result = jitted_scatter_add(operand, indices, updates, dimension_numbers)

        print("JAX direct call PASSED")
        print("Result shape:", result.shape)
        # print("Result:\n", result) # Optional: print result
    except Exception as e:
        print("JAX direct call FAILED")
        print(f"Error type: {type(e)}")
        print(f"Error message: {e}")


def test_scatter_add_toy():
    print("\n--- Testing: scatter_add_mismatched_window_dims_toy ---")
    try:
        operand = jnp.array(np.zeros((2, 2), dtype=np.float64))
        indices = jnp.array(np.array([[1]], dtype=np.int32))
        updates = jnp.array(np.ones((2, 1), dtype=np.float64))

        dimension_numbers = lax.ScatterDimensionNumbers(
            update_window_dims=(0,),
            inserted_window_dims=(),
            scatter_dims_to_operand_dims=(1,),
            operand_batching_dims=(),
            scatter_indices_batching_dims=(),
        )

        # Direct call
        # result = lax.scatter_add(operand, indices, updates, dimension_numbers)

        # Call with jax.jit
        jitted_scatter_add = jax.jit(lax.scatter_add, static_argnums=(3,))
        result = jitted_scatter_add(operand, indices, updates, dimension_numbers)

        print("JAX direct call PASSED")
        print("Result shape:", result.shape)
        # print("Result:\n", result) # Optional: print result
    except Exception as e:
        print("JAX direct call FAILED")
        print(f"Error type: {type(e)}")
        print(f"Error message: {e}")


if __name__ == "__main__":
    print(f"JAX version: {jax.__version__}")
    scatter_add_simplified()
    test_scatter_add_user_report()
    test_scatter_add_toy()
