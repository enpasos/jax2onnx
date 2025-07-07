# file: jax2onnx/sandbox/linen_bridge.py

import flax.linen as nn
from flax import nnx
from jax import numpy as jnp
import jax2onnx


class LinenModule(nn.Module):
    def setup(self):
        self.dense = nn.Dense(128)

    def __call__(self, x):
        x = self.dense(x)
        return nn.relu(x)


model = LinenModule()
model = nnx.bridge.ToNNX(model, rngs=nnx.Rngs(0))
inputs = jnp.ones((1, 10))
model = nnx.bridge.lazy_init(model, inputs)

# avoid flax.errors.TraceContextError: Cannot mutate RngStream from a different trace level
model.rngs = False

jax2onnx.to_onnx(
    model,
    [inputs],
    record_primitive_calls_file="logs/jax2onnx_calls.txt",
)
