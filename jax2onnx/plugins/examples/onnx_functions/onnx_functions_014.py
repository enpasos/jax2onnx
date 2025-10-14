# jax2onnx/plugins/examples/onnx_functions/onnx_functions_014.py

from __future__ import annotations


import jax.numpy as jnp
from flax import nnx

from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    onnx_function,
    register_example,
    with_rng_seed,
)


class MLPBlock(nnx.Module):
    """MLP block for Transformer layers."""

    def __init__(self, num_hiddens, mlp_dim, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(num_hiddens, mlp_dim, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=0.1, rngs=rngs)
        self.linear2 = nnx.Linear(mlp_dim, num_hiddens, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=0.1, rngs=rngs)

    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        x = self.linear1(x)
        x = nnx.gelu(x, approximate=False)
        x = self.dropout1(x, deterministic=deterministic)
        x = self.linear2(x)
        return self.dropout2(x, deterministic=deterministic)


@onnx_function
class SuperBlock(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.layer_norm2 = nnx.LayerNorm(3, rngs=rngs)
        self.mlp = MLPBlock(num_hiddens=3, mlp_dim=6, rngs=rngs)

    def __call__(self, x, deterministic: bool = True):
        # Explicitly pass the deterministic parameter to the MLPBlock
        x_normalized = self.layer_norm2(x)
        return self.mlp(x_normalized, deterministic=deterministic)


register_example(
    component="onnx_functions_014",
    description="one function on an outer layer.",
    since="v0.4.0",
    context="examples.onnx_functions",
    children=["MLPBlock"],
    testcases=[
        {
            "testcase": "014_one_function_with_input_param_with_default_value",
            "callable": construct_and_call(SuperBlock, rngs=with_rng_seed(0)),
            "input_shapes": [(5, 10, 3)],
            "expected_number_of_function_instances": 1,
            "input_params": {
                "deterministic": True,
            },
            "run_only_f32_variant": True,
            "rtol": 3e-4,
            "atol": 2e-6,
            "post_check_onnx_graph": expect_graph(
                ["SuperBlock_1:5x10x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "014_one_function_without_input_param_with_default_value",
            "callable": construct_and_call(SuperBlock, rngs=with_rng_seed(0)),
            "input_shapes": [("B", 10, 3)],
            "expected_number_of_function_instances": 1,
            "run_only_f32_variant": True,
            "rtol": 3e-4,
            "atol": 2e-6,
            "post_check_onnx_graph": expect_graph(
                ["SuperBlock_1:Bx10x3"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)
