from __future__ import annotations

import numpy as np
import jax
from flax import nnx
from flax.nnx.nn.activations import tanh

from jax2onnx.plugins2.plugin_system import (
    construct_and_call,
    register_example,
    with_rng_seed,
)


class GRUCellWrapper(nnx.Module):
    def __init__(
        self,
        *,
        in_feat: int = 3,
        hid_feat: int = 4,
        rngs: nnx.Rngs,
    ):
        self.cell = nnx.GRUCell(
            in_features=in_feat,
            hidden_features=hid_feat,
            activation_fn=tanh,
            rngs=rngs,
        )

    def __call__(
        self,
        carry: jax.Array,
        inputs: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        new_hidden, output = self.cell(carry, inputs)
        return new_hidden, output + 0.0


register_example(
    component="GRUCell",
    description="Flax/nnx GRUCell lowered through converter2 primitives.",
    source="https://flax.readthedocs.io/en/latest/",
    since="v0.7.2",
    context="examples2.nnx",
    children=[
        "nnx.Linear",
        "jax.lax.split",
        "jax.lax.logistic",
        "jax.lax.dot_general",
    ],
    testcases=[
        {
            "testcase": "gru_cell_basic",
            "callable": construct_and_call(
                GRUCellWrapper,
                in_feat=3,
                hid_feat=4,
                rngs=with_rng_seed(0),
            ),
            "input_values": [
                np.zeros((2, 4), np.float32),
                np.ones((2, 3), np.float32),
            ],
            "expected_output_shapes": [(2, 4), (2, 4)],
            "run_only_f32_variant": True,
        },
    ],
)
