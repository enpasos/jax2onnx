import jax
import jax.numpy as jnp
from jax2onnx import to_onnx
import jax
jax.config.update("jax_enable_x64", True)

def minimal_scatter_error():
    def scatter_function(operand, indices, updates):
        dimension_numbers = jax.lax.ScatterDimensionNumbers(
            update_window_dims=(1, 2, 3, 4),
            inserted_window_dims=(),
            scatter_dims_to_operand_dims=(1, 2) 
        )
        
        return jax.lax.scatter(
            operand,
            indices,
            updates,
            dimension_numbers=dimension_numbers,
            indices_are_sorted=True,
            unique_indices=True,
            mode=jax.lax.GatherScatterMode.FILL_OR_DROP
        )

    operand = jnp.zeros((5, 266, 266, 1), dtype=jnp.float64)
    indices = jnp.array([[10, 10]], dtype=jnp.int32)  # Shape (1, 2)
    updates = jnp.ones((1, 5, 256, 256, 1), dtype=jnp.float64)

    result = scatter_function(operand, indices, updates)

    try:
        model = to_onnx(
            scatter_function,
            inputs=[operand, indices, updates],
            enable_double_precision=True
        )
        print("ONNX conversion successful!")
        return None
    except Exception as e:
        print(f"ONNX conversion failed with error: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        return e

if __name__ == "__main__":
    error1 = minimal_scatter_error()