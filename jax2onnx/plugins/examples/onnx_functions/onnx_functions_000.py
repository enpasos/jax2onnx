# jax2onnx/plugins/examples/onnx_functions/onnx_functions_000.py

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    onnx_function,
    register_example,
    with_rng_seed,
)


class MLPBlock(nnx.Module):
    """Tiny MLP block used by SuperBlock."""

    def __init__(self, num_hiddens: int, mlp_dim: int, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(num_hiddens, mlp_dim, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=0.1, rngs=rngs)
        self.linear2 = nnx.Linear(mlp_dim, num_hiddens, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=0.1, rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, deterministic: bool = False) -> jnp.ndarray:
        y = self.linear1(x)
        y = nnx.gelu(y, approximate=False)
        y = self.dropout1(y, deterministic=deterministic)
        y = self.linear2(y)
        return self.dropout2(y, deterministic=deterministic)


@onnx_function
class SuperBlock(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.layer_norm2 = nnx.LayerNorm(3, rngs=rngs)
        self.mlp = MLPBlock(num_hiddens=3, mlp_dim=6, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # explicit deterministic flag for reproducible export
        x_norm = self.layer_norm2(x)
        # Keep LayerNorm in the path to exercise function bodies; numeric
        # tolerance needs to account for small LN drift vs JAX.
        return self.mlp(x_norm, deterministic=True)


register_example(
    component="onnx_functions_000",
    description="One function boundary on an outer NNX module (new-world).",
    since="v0.4.0",
    context="examples.onnx_functions",
    children=["MLPBlock"],
    testcases=[
        {
            "testcase": "000_one_function_on_outer_layer",
            "callable": construct_and_call(SuperBlock, rngs=with_rng_seed(0)),
            "input_shapes": [("B", 10, 3)],
            "expected_number_of_function_instances": 1,
            "run_only_f32_variant": True,
            # LayerNorm in the path introduces small, known ORT↔JAX drift.
            # Match the tolerances used by LN unit tests.
            "rtol": 1e-3,
            "atol": 1e-5,
            "post_check_onnx_graph": EG(
                ["SuperBlock_1:Bx10x3"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        }
    ],
)
