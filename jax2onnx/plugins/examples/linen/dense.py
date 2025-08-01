# jax2onnx/plugins/examples/linen_bridge/dense.py
#
# Minimal Flax-Linen Dense → ReLU example *without* any pre-applied
# nnx.bridge wrapper.  The bridge is created later by jax2onnx itself
# during export; the pure function returned here is the numeric
# reference the tests compare the ONNX output against.

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

import jax2onnx                                   # keep isort happy
from jax2onnx.plugin_system import register_example


# -----------------------------------------------------------------------------#
# 1.  Pure Linen module (creates parameters once in `setup`)
# -----------------------------------------------------------------------------#
class LinenDense(nn.Module):
    """Dense → ReLU block used for the bridge tests."""
    features: int

    def setup(self) -> None:
        self.dense = nn.Dense(self.features)

    def __call__(self, x):
        return self.dense(x) # nn.relu(self.dense(x))


# -----------------------------------------------------------------------------#
# 2.  Helper that returns an *initialised, pure* JAX callable
# -----------------------------------------------------------------------------#
def create_model(*, input_shape: tuple[int, ...], features: int):
    """
    Returns a **plain Python function** `f(x)` that runs the original
    `LinenDense`.  No nnx.bridge wrappers are created here – the jax2onnx
    converter will add them internally when it needs to trace to ONNX.

    Parameters
    ----------
    input_shape
        Shape of the dummy tensor used for parameter initialisation.
    features
        Number of output units for the underlying `LinenDense`.
    """
    model = LinenDense(features=features)
    variables = model.init(jax.random.PRNGKey(0),
                           jnp.ones(input_shape, jnp.float32))

    # Capture `model` & its variables in a closure so the function is
    # stateless from the caller’s perspective.
    def fn(x):
        return model.apply(variables, x)

    return fn


# -----------------------------------------------------------------------------#
# 3.  Register the example for automated test discovery
# -----------------------------------------------------------------------------#
register_example(
    component="LinenDense",
    description="Flax Linen Dense layer – bridge wrapper added by converter.",
    source="Validates the jax2onnx Linen-bridge handler (Flax ≥ 0.11).",
    since="v0.7.4",
    context="examples.linen",
    children=[
        "flax.linen.Dense",
        "flax.linen.relu",
    ],
    testcases=[
        { 
            "testcase": "dense_a",
            "callable": create_model(input_shape=(1, 10), features=128),
            "input_shapes": [("B", 10)],
            "run_only_f32_variant": True,
        },
        { 
            "testcase": "dense_b",
            "callable": create_model(input_shape=(1, 10), features=64),
            "input_shapes": [("B", 10)],
            "run_only_f32_variant": True,
        },
    ],
)
