# jax2onnx/examples/nnx/gru_cell.py

import jax
from flax import nnx
from flax.nnx.nn.activations import tanh
from jax2onnx.plugin_system import register_example, construct_and_call
import numpy as np


def _safe_kernel_init(key, shape, dtype=jax.numpy.float32):
    return jax.random.normal(key, shape, dtype) * 0.02

# Helper to create a GRUCell instance.
def _gru(in_feat=3, hid_feat=4):
    return nnx.GRUCell(
        in_features=in_feat, 
        hidden_features=hid_feat,
        # gate_fn is sigmoid by default, which maps to lax.logistic
        activation_fn=tanh,
        kernel_init=_safe_kernel_init,
        rngs=nnx.Rngs(0),
    )


# Wrapper to ensure JAX tracer sees two distinct outputs.
# The nnx.GRUCell returns the same object twice, which gets optimized
# to a single output in the jaxpr. Adding zero creates a new `add`
# primitive and thus a distinct output variable.
# Do not construct at import time; the generator only needs metadata.


def _gru_wrapper(carry, inputs):
    new_h, y = _gru()(carry, inputs)
    return new_h, y + 0.0


register_example(
    component="GRUCell",
    context="examples.nnx",
    description=(
        "Vanilla gated-recurrent-unit cell from **Flax/nnx**. "
        "There is no 1-to-1 ONNX operator, so the converter decomposes it "
        "into MatMul, Add, Sigmoid, Tanh, etc."
    ),
    since="v0.7.2",
    source="https://flax.readthedocs.io/en/latest/",
    children=[
        "nnx.Linear",
        "jax.lax.split",
        "jax.lax.logistic",
        "jax.lax.dot_general",
    ],
    testcases=[
        {
            "testcase": "gru_cell_basic",
            # Strict late-construction pattern
            "callable": construct_and_call(
                nnx.GRUCell,
                in_features=3,
                hidden_features=4,
                activation_fn=tanh,           # default gate_fn is sigmoid; keep tanh for activation
                kernel_init=_safe_kernel_init,
                rngs=nnx.Rngs(0),
            ),
            "input_values": [
                np.zeros((2, 4), np.float32),  # carry   h₀
                np.ones((2, 3), np.float32),  # inputs  x₀
            ],
            "expected_output_shapes": [(2, 4), (2, 4)],  # (new_h, y)
            "run_only_f32_variant": True,
        },
    ],
)
