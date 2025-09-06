import jax
import jax.numpy as jnp
from jax import lax
from jax2onnx import to_onnx

jax.config.update("jax_enable_x64", True) 

def example_2d():
    """2D example of dynamic_update_slice"""
    original = jnp.arange(16).reshape(4, 4)
    update = jnp.array([[99, 98], 
                       [97, 96]])
    start_indices = [1, 2]
    
    result = lax.dynamic_update_slice(original, update, start_indices)
    print(f"\n2D: Original:\n{original}")
    print(f"2D: Update:\n{update}")
    print(f"2D: Start indices: {start_indices}")
    print(f"2D: Result:\n{result}")
    return result

def example_3d():
    """3D example of dynamic_update_slice"""
    original = jnp.arange(48).reshape(3, 4, 4)
    update = jnp.array([[[99, 98], 
                        [97, 96]]])
    start_indices = [1, 1, 1]
    
    result = lax.dynamic_update_slice(original, update, start_indices)
    print(f"\n3D: Original shape: {original.shape}")
    print(f"3D: Update shape: {update.shape}")
    print(f"3D: Start indices: {start_indices}")
    print(f"3D: Result shape: {result.shape}")
    print(f"3D: Updated slice at [1, 1:3, 1:3]:\n{result[1, 1:3, 1:3]}")
    return result

def example_4d():
    """4D example similar to your case: shape [5,266,266,1]"""
    original = jnp.ones((5, 10, 10, 1), dtype=jnp.float64)
    update = jnp.full((1, 5, 5, 1), 99, dtype=jnp.float64)
    start_indices = [2, 3, 3, 0]

    result = lax.dynamic_update_slice(original, update, start_indices)
    print(f"\n4D: Original shape: {original.shape}, dtype: {original.dtype}")
    print(f"4D: Update shape: {update.shape}, dtype: {update.dtype}")
    print(f"4D: Start indices: {start_indices}")
    print(f"4D: Updated region sum: {result[2, 3:8, 3:8, 0].sum()}")
    return result


def jit_compiled_example():
    """JIT compiled version"""
    @jax.jit
    def update_slice(original, update, start_indices):
        return lax.dynamic_update_slice(original, update, start_indices)
    
    original = jnp.arange(20).reshape(4, 5)
    update = jnp.array([[99, 98]])
    start_indices = [2, 1]
    
    result = update_slice(original, update, start_indices)
    print(f"\nJIT: Original:\n{original}")
    print(f"JIT: Result:\n{result}")
    return result

def export(fn):
    fn()
    model = to_onnx(
        fn=fn,
        inputs=[],
        enable_double_precision=True,
    )
    print("ONNX model exported.")

if __name__ == "__main__":
    print("=== JAX dynamic_update_slice Examples ===")
    try:
        export(example_2d)
        export(example_3d)
        export(example_4d)
        export(jit_compiled_example)
        print("All examples completed.")
    except Exception as e:
        print(f"An error occurred: {e}")