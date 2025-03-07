# jax2onnx/tests/test_save_onnx.py

import pytest
import jax
import jax.numpy as jnp
from jax2onnx import save_onnx, allclose
import flax.nnx as nnx
import os


def test_example():
    seed = 1001

    fn = nnx.LinearGeneral(
        in_features=(8, 32), out_features=(256,), axis=(-2, -1), rngs=nnx.Rngs(seed)
    )

    dir = "docs/onnx"
    os.makedirs(dir, exist_ok=True)

    model_path = dir + "/example5.onnx"

    save_onnx(fn, [("B", 4, 8, 32)], model_path, include_intermediate_shapes=True)

    rng = jax.random.PRNGKey(seed)
    example_batch_size = 2
    x = jax.random.normal(rng, (example_batch_size, 4, 8, 32))

    # Verify outputs match
    assert allclose(fn, model_path, x)
