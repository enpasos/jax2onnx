# file: jax2onnx/plugins/examples/eqx/mlp.py

import equinox as eqx
import jax
import numpy as np
from onnx import numpy_helper

from jax2onnx.plugin_system import register_example


# WARNING: this is temporary until I add other layers
class Mlp(eqx.Module):
    linear1: eqx.nn.Linear
    dropout1: eqx.nn.Dropout

    def __init__(self, in_size: int, out_size: int, key: jax.Array):
        self.linear1 = eqx.nn.Linear(
            in_features=in_size, out_features=out_size, key=key
        )
        self.dropout1 = eqx.nn.Dropout(p=0.2, inference=False)

    def __call__(self, x, key=None):
        return jax.nn.gelu(self.dropout1(self.linear1(x), key=key))


# --- Test Case Definition ---
# 1. Create the model instance once, outside the testcase's callable.
#    This ensures that the random weight initialization is not part of the
#    function that gets traced for ONNX conversion.
# 2. Create variations for inference and batching.
model = Mlp(30, 3, key=jax.random.PRNGKey(0))
inference_model = eqx.nn.inference_mode(model, value=True)
batched_model = jax.vmap(model, in_axes=(0, None))


def _check_dropout_training_mode(m, expected_mode: bool) -> bool:
    """Helper to check the training_mode input of the Dropout node."""
    try:
        dropout_node = next(n for n in m.graph.node if n.op_type == "Dropout")
        training_mode_input_name = dropout_node.input[2]
        training_mode_init = next(
            i for i in m.graph.initializer if i.name == training_mode_input_name
        )
        return np.isclose(
            numpy_helper.to_array(training_mode_init), expected_mode
        ).all()
    except StopIteration:
        return False


register_example(
    component="MlpExample",
    description="A simple MLP example using Equinox.",
    source="https://github.com/patrick-kidger/equinox",
    since="v0.7.0",
    context="examples.eqx",
    children=["eqx.nn.Linear", "eqx.nn.Dropout", "jax.nn.gelu"],
    testcases=[
        {
            "testcase": "mlp_training_mode",
            "callable": lambda x, key, model=model: model(x, key),
            "input_shapes": [(30,), ()],
            "post_check_onnx_graph": lambda m: (
                _check_dropout_training_mode(m, expected_mode=True)
            ),
        },
        {
            "testcase": "mlp_inference_mode",
            "callable": lambda x, key, model=inference_model: model(x, key),
            "input_shapes": [(30,), ()],
            "post_check_onnx_graph": lambda m: (
                _check_dropout_training_mode(m, expected_mode=False)
            ),
        },
        {
            "testcase": "mlp_batched_training_mode",
            "callable": lambda x, key, model=batched_model: model(x, key),
            "input_shapes": [("B", 30), ()],
        },
    ],
)
