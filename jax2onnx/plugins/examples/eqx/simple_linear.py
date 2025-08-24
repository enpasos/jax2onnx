# file: jax2onnx/plugins/examples/eqx/simple_linear.py

from __future__ import annotations
import jax
import jax.numpy as jnp
import equinox as eqx

from jax2onnx.plugin_system import register_example


class Linear(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, in_size: int, out_size: int, *, key):
        wkey, bkey = jax.random.split(key)
        # JAX 0.7 prefers explicit dtype; default is fine but being explicit is safest
        self.weight = jax.random.normal(wkey, (out_size, in_size), dtype=jnp.float32)
        self.bias = jax.random.normal(bkey, (out_size,), dtype=jnp.float32)

    def __call__(self, x):
        return x @ self.weight.T + self.bias


# --- Test Case Definition ---
# ✅ GOOD: build the model when the testcase actually runs
def _run_simple_linear(x):
    # Create params lazily, not at import
    model = jax.vmap(Linear(x.shape[-1], 3, key=jax.random.PRNGKey(0)))
    return model(x)

# Example using eqx.nn.Linear
def _run_nn_linear(x):
    # Create params lazily, not at import
    model_nn = jax.vmap(
        eqx.nn.Linear(in_features=x.shape[-1], out_features=3, key=jax.random.PRNGKey(1))
    )
    return model_nn(x)


register_example(
    component="SimpleLinearExample",
    description="A simple linear layer example using Equinox.",
    source="https://github.com/patrick-kidger/equinox",
    since="v0.7.1",
    context="examples.eqx",
    children=["eqx.nn.Linear"],
    testcases=[
        {
            "testcase": "simple_linear",
            "callable": _run_simple_linear,
            "input_shapes": [("B", 30)],
        },
        {
            "testcase": "nn_linear",
            "callable": _run_nn_linear,
            "input_shapes": [("B", 30)],
        },
    ],
)
