# jax2onnx/plugins/examples/linen_bridge/conv.py
#
# A 1-layer Conv → ReLU example registered for the jax2onnx test-suite.
# NOTE:
#   The testcases return a *pure* Linen function (initialized once, then apply).
#   We do NOT wrap with nnx.bridge.ToNNX here; the export path is owned by
#   jax2onnx.converter.user_interface (which may add a bridge when appropriate).

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

import jax2onnx  # keep isort happy
from jax2onnx.plugin_system import register_example


# ──────────────────────────────────────────────────────────────────────────────
# 1. Linen module
# ──────────────────────────────────────────────────────────────────────────────
class LinenConv(nn.Module):
    """Single 2-D convolution followed by ReLU."""

    features: int = 32
    kernel_size: tuple[int, int] = (3, 3)

    def setup(self) -> None:
        self.conv = nn.Conv(self.features, self.kernel_size, padding="SAME")

    def __call__(self, x):
        return nn.relu(self.conv(x))


# ──────────────────────────────────────────────────────────────────────────────
# 2. Helper that returns a *pure* initialized Linen function
# ──────────────────────────────────────────────────────────────────────────────
def create_model(*, input_shape: tuple[int, ...], features: int):
    """
    Builds a LinenConv, initialises its variables with a deterministic key,
    and returns a pure function `fn(x)` that applies the fixed variables.

    Parameters
    ----------
    input_shape : tuple of int
        Concrete shape for the dummy tensor used during init.
    features : int
        Number of output channels in the Conv.
    """
    model = LinenConv(features=features)
    dummy = jnp.ones(input_shape, jnp.float32)
    variables = model.init(jax.random.PRNGKey(0), dummy)

    def fn(x):
        return model.apply(variables, x)

    return fn


# ──────────────────────────────────────────────────────────────────────────────
# 3. Register with jax2onnx test-suite
# ──────────────────────────────────────────────────────────────────────────────
register_example(
    component="LinenConv",
    description="Single Conv + ReLU as a pure initialised Linen function.",
    since="v0.7.4",
    context="examples.linen",
    children=[
        "flax.linen.Conv",
        "flax.linen.relu",
    ],
    testcases=[
        # {
        #     "testcase": "conv",
        #     "callable": create_model(
        #         input_shape=(1, 28, 28, 3),
        #         features=16,
        #     ),
        #     "input_shapes": [("B", 28, 28, 3)],
        #     "run_only_f32_variant": True,
        # } 
    ],
)
