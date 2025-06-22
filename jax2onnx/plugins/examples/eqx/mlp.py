# file: jax2onnx/plugins/examples/eqx/mlp.py

import equinox as eqx
import jax

from jax2onnx.plugin_system import register_example


# WARNING: this is temporary until I add other layers
class Mlp(eqx.Module):
    linear1: eqx.nn.Linear
    identity: eqx.nn.Identity

    def __init__(self, in_size: int, out_size: int, key: jax.Array):
        self.linear1 = eqx.nn.Linear(
            in_features=in_size, out_features=out_size, key=key
        )
        self.identity = eqx.nn.Identity()

    def __call__(self, x):
        return self.linear1(x + self.identity(x))


# --- Test Case Definition ---
# 1. Create the model instance once, outside the testcase's callable.
#    This ensures that the random weight initialization is not part of the
#    function that gets traced for ONNX conversion.
# 2. We also apply jax.vmap here to create a batched version of the model.
model = jax.vmap(Mlp(30, 3, key=jax.random.PRNGKey(0)))


register_example(
    component="MlpExample",
    description="A simple MLP example using Equinox.",
    source="https://github.com/patrick-kidger/equinox",
    since="v0.7.0",
    context="examples.eqx",
    children=["eqx.nn.Linear", "eqx.nn.Identity"],
    testcases=[
        {
            "testcase": "simple_mlp",
            "callable": model,
            "input_shapes": [("B", 30)],
        },
    ],
)
