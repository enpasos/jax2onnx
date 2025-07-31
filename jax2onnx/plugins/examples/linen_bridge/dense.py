# file: jax2onnx/plugins/examples/linen_bridge/dense.py

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import nnx

from jax2onnx.plugin_system import register_example

# 1. Define the Linen module using the standard `setup` method.
class LinenDense(nn.Module):
    """A simple Linen module with a Dense layer and a ReLU activation."""
    features: int

    def setup(self):
        # Layers are defined once here, which is the canonical and trace-safe
        # pattern for Flax Linen. This avoids the ScopeParamShapeError.
        self.dense = nn.Dense(features=self.features)

    def __call__(self, x):
        x = self.dense(x)
        return nn.relu(x)

# 2. Factory function to handle the bridge initialization.
def create_bridged_model(input_shape, features):
    """Handles the full initialization process for the bridged Linen module."""
    linen_model = LinenDense(features=features)
    model = nnx.bridge.ToNNX(linen_model, rngs=nnx.Rngs(0))
    dummy_inputs = jnp.ones(input_shape)
    initialized_model = nnx.bridge.lazy_init(model, dummy_inputs)
    return initialized_model

# 3. Register the example with the jax2onnx test system.
register_example(
    component="LinenDense",
    description="A simple Flax Linen Dense layer wrapped with the NNX bridge.",
    source="This example tests the jax2onnx Linen bridge handler.",
    since="v0.2.0",
    context="examples.linen_bridge",
    children=["nnx.bridge.ToNNX", "flax.linen.Dense", "flax.linen.relu"],
    testcases=[
        {
            "testcase": "bridged_dense_dynamic_batch",
            "callable": create_bridged_model(input_shape=(1, 10), features=128),
            "input_shapes": [("B", 10)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "bridged_dense_static_batch",
            "callable": create_bridged_model(input_shape=(4, 10), features=64),
            "input_shapes": [(4, 10)],
            "run_only_f32_variant": True,
        },
    ],
)