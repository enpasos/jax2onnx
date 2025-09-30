from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
import pytest

from jax2onnx.user_interface import to_onnx


def example_2d(original):
    update = jnp.array([[99, 98], [97, 96]], dtype=original.dtype)
    start_indices = [1, 2]
    return lax.dynamic_update_slice(original, update, start_indices)


def example_3d(original):
    update = jnp.array([[[99, 98], [97, 96]]], dtype=original.dtype)
    start_indices = [1, 1, 1]
    return lax.dynamic_update_slice(original, update, start_indices)


def example_4d(original):
    update = jnp.full((1, 5, 5, 1), 99, dtype=original.dtype)
    start_indices = [2, 3, 3, 0]
    return lax.dynamic_update_slice(original, update, start_indices)


def jit_compiled_example(original):
    update = jnp.array([[99, 98]], dtype=original.dtype)
    start_indices = [2, 1]

    @jax.jit
    def update_slice(x):
        return lax.dynamic_update_slice(x, update, start_indices)

    return update_slice(original)


@pytest.mark.parametrize(
    "fn, original",
    [
        (example_2d, jnp.arange(16, dtype=jnp.int32).reshape(4, 4)),
        (example_3d, jnp.arange(48, dtype=jnp.int32).reshape(3, 4, 4)),
        (example_4d, jnp.ones((5, 10, 10, 1), dtype=jnp.float64)),
        (jit_compiled_example, jnp.arange(20, dtype=jnp.float32).reshape(4, 5)),
    ],
)
def test_dynamic_update_slice_exports_with_ir_pipeline(fn, original):
    to_onnx(
        fn=fn,
        inputs=[original],
        enable_double_precision=True,
    )
