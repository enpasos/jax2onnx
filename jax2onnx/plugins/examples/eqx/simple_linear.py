# file: jax2onnx/plugins/examples/eqx/simple_linear.py

import equinox as eqx
import jax

from jax2onnx.plugin_system import register_example


class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_size, out_size, key):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (out_size, in_size))
        self.bias = jax.random.normal(bkey, (out_size,))

    def __call__(self, x):
        return self.weight @ x + self.bias


# --- Test Case Definition ---
# 1. Create the model instance once, outside the testcase's callable.
#    This ensures that the random weight initialization is not part of the
#    function that gets traced for ONNX conversion.
# 2. We also apply jax.vmap here to create a batched version of the model.
model = jax.vmap(Linear(30, 3, key=jax.random.PRNGKey(0)))

# Example using eqx.nn.Linear
model_nn = jax.vmap(
    eqx.nn.Linear(in_features=30, out_features=3, key=jax.random.PRNGKey(1))
)


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
            "callable": model,
            "input_shapes": [("B", 30)],
        },
        {
            "testcase": "nn_linear",
            "callable": model_nn,
            "input_shapes": [("B", 30)],
        },
    ],
)
