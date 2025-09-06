import pytest
import jax
import jax.numpy as jnp
from jax import lax
from jax2onnx import to_onnx

# Keep x64 so shapes/dtypes match the examples used in discussion
jax.config.update("jax_enable_x64", True)


# --- Example callables with no inputs (to_onnx will trace with empty input list) ---

def example_2d():
    """2D example of dynamic_update_slice"""
    original = jnp.arange(16).reshape(4, 4)
    update = jnp.array([[99, 98],
                        [97, 96]])
    start_indices = [1, 2]  # per-dimension scalar indices
    return lax.dynamic_update_slice(original, update, start_indices)


def example_3d():
    """3D example of dynamic_update_slice"""
    original = jnp.arange(48).reshape(3, 4, 4)
    update = jnp.array([[[99, 98],
                         [97, 96]]])
    start_indices = [1, 1, 1]
    return lax.dynamic_update_slice(original, update, start_indices)


def example_4d():
    """4D example similar to typical CNN shapes"""
    original = jnp.ones((5, 10, 10, 1), dtype=jnp.float64)
    update = jnp.full((1, 5, 5, 1), 99, dtype=jnp.float64)
    start_indices = [2, 3, 3, 0]
    return lax.dynamic_update_slice(original, update, start_indices)


def jit_compiled_example():
    """JIT-compiled wrapper to ensure call-primitive paths are covered."""
    @jax.jit
    def update_slice(original, update, start_indices):
        return lax.dynamic_update_slice(original, update, start_indices)

    original = jnp.arange(20).reshape(4, 5)
    update = jnp.array([[99, 98]])
    start_indices = [2, 1]
    return update_slice(original, update, start_indices)

 
def test_export_dynamic_update_slice_2d():
    # Function has no args; we pass an empty inputs list to to_onnx.
    to_onnx(fn=example_2d, inputs=[], enable_double_precision=True)  # noqa: F841


def test_export_dynamic_update_slice_3d():
    to_onnx(fn=example_3d, inputs=[], enable_double_precision=True)  # noqa: F841


def test_export_dynamic_update_slice_4d():
    to_onnx(fn=example_4d, inputs=[], enable_double_precision=True)  # noqa: F841


def test_export_dynamic_update_slice_jit():
    to_onnx(fn=jit_compiled_example, inputs=[], enable_double_precision=True)  # noqa: F841
