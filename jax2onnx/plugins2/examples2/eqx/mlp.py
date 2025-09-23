from __future__ import annotations

import equinox as eqx
import jax
import jax.random as jr
import numpy as np
from onnx import numpy_helper

from jax2onnx.plugins2.plugin_system import register_example


class Mlp(eqx.Module):
    linear1: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    norm: eqx.nn.LayerNorm
    linear2: eqx.nn.Linear

    def __init__(
        self, in_features: int, hidden_features: int, out_features: int, *, key: jax.Array
    ):
        key_1, key_2 = jr.split(key, 2)
        self.linear1 = eqx.nn.Linear(in_features, hidden_features, key=key_1)
        self.dropout = eqx.nn.Dropout(p=0.2, inference=False)
        self.norm = eqx.nn.LayerNorm(hidden_features)
        self.linear2 = eqx.nn.Linear(hidden_features, out_features, key=key_2)

    def __call__(self, x: jax.Array, key: jax.Array | None = None) -> jax.Array:
        x = jax.nn.gelu(self.dropout(self.norm(self.linear1(x)), key=key))
        return self.linear2(x)


_model = Mlp(30, 20, 10, key=jax.random.PRNGKey(0))
_inference_model = eqx.nn.inference_mode(_model, value=True)
_batched_model = jax.vmap(_model, in_axes=(0, None))


def _check_dropout_training_mode(model, expected_mode: bool) -> bool:
    try:
        dropout_node = next(n for n in model.graph.node if n.op_type == "Dropout")
        training_mode_input_name = dropout_node.input[2]
        training_mode_init = next(
            i for i in model.graph.initializer if i.name == training_mode_input_name
        )
        return np.isclose(
            numpy_helper.to_array(training_mode_init), expected_mode
        ).all()
    except StopIteration:
        return False


register_example(
    component="MlpExample",
    description="A simple Equinox MLP (converter2 pipeline).",
    source="https://github.com/patrick-kidger/equinox",
    since="v0.8.0",
    context="examples2.eqx",
    children=["eqx.nn.Linear", "eqx.nn.Dropout", "jax.nn.gelu"],
    testcases=[
        {
            "testcase": "mlp_training_mode",
            "callable": (
                lambda x, *, model=_model, _k=jax.random.PRNGKey(0): model(x, _k)
            ),
            "input_shapes": [(30,)],
            "use_onnx_ir": True,
            "post_check_onnx_graph": lambda m: (
                _check_dropout_training_mode(m, expected_mode=True)
            ),
            "skip_numeric_validation": True,
        },
        {
            "testcase": "mlp_inference_mode",
            "callable": (lambda x, *, model=_inference_model: model(x, key=None)),
            "input_shapes": [(30,)],
            "use_onnx_ir": True,
            "post_check_onnx_graph": lambda m: (
                _check_dropout_training_mode(m, expected_mode=False)
            ),
        },
        {
            "testcase": "mlp_batched_training_mode",
            "callable": (
                lambda x, *, model=_batched_model, _k=jax.random.PRNGKey(0): model(x, _k)
            ),
            "input_shapes": [(8, 30)],
            "use_onnx_ir": True,
            "skip_numeric_validation": True,
        },
    ],
)
