# jax2onnx/examples/nnx/gru_cell.py

from flax import nnx
from flax.nnx.nn.activations import tanh
from jax2onnx.plugin_system import register_example
import numpy as np


# Helper to create a GRUCell instance.
def _gru(in_feat=3, hid_feat=4):
    return nnx.GRUCell(
        in_features=in_feat,
        hidden_features=hid_feat,
        # gate_fn is sigmoid by default, which maps to lax.logistic
        activation_fn=tanh,
        rngs=nnx.Rngs(0),
    )

# Wrapper to ensure JAX tracer sees two distinct outputs.
# The nnx.GRUCell returns the same object twice, which gets optimized
# to a single output in the jaxpr. Adding zero creates a new `add`
# primitive and thus a distinct output variable.
_gru_instance = _gru()
def _gru_wrapper(carry, inputs):
    new_h, y = _gru_instance(carry, inputs)
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
            "callable": _gru_wrapper,   
            "input_values": [
                np.zeros((2, 4), np.float32),  # carry   h₀
                np.ones((2, 3), np.float32),   # inputs  x₀
            ],
            "expected_output_shapes": [(2, 4), (2, 4)],  # (new_h, y)
            "run_only_f32_variant": True 
        },
    ],
)