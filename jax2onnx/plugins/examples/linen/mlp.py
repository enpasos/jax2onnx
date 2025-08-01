# jax2onnx/plugins/examples/linen_bridge/mlp.py
#
# A simple MLP (Dense → ReLU → Dense) example without any pre-applied
# nnx.bridge wrapper. The bridge is created later by jax2onnx itself
# during export.

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

import jax2onnx
from jax2onnx.plugin_system import register_example


# -----------------------------------------------------------------------------#
# 1. Pure Linen module
# -----------------------------------------------------------------------------#
class LinenMLP(nn.Module):
    """A 2-layer MLP with ReLU activations."""
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return nn.relu(x)


# -----------------------------------------------------------------------------#
# 2. Helper that returns an *initialised, pure* JAX callable
# -----------------------------------------------------------------------------#
def create_model(*, input_shape: tuple[int, ...], hidden_dim: int, output_dim: int):
    """
    Returns a plain Python function `f(x)` that runs the `LinenMLP`.
    No nnx.bridge wrappers are created here – the jax2onnx converter will
    add them internally when it needs to trace to ONNX.
    """
    model = LinenMLP(hidden_dim=hidden_dim, output_dim=output_dim)
    variables = model.init(jax.random.PRNGKey(0), jnp.ones(input_shape, jnp.float32))

    # Capture `model` & its variables in a closure so the function is
    # stateless from the caller’s perspective.
    def fn(x):
        return model.apply(variables, x)

    return fn


# -----------------------------------------------------------------------------#
# 3. Register with jax2onnx test-suite
# -----------------------------------------------------------------------------#
register_example(
    component="LinenMLP",
    description="Two-layer MLP (Dense+ReLU) – bridge wrapper added by converter.",
    since="v0.7.4",
    context="examples.linen",
    children=[
        "flax.linen.Dense",
        "flax.linen.relu",
    ],
    testcases=[ 
        {
            "testcase": "bridged_mlp",
            "callable": create_model(
                input_shape=(1, 10),  # concrete batch=1 for init
                hidden_dim=64,
                output_dim=32,
            ),
            "input_shapes": [("B", 10)],  # symbolic batch dim
            "run_only_f32_variant": True,
        }
    ],
)