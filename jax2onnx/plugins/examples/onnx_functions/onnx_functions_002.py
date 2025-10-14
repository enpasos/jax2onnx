# jax2onnx/plugins/examples/onnx_functions/onnx_functions_002.py

from __future__ import annotations


import jax.numpy as jnp
from flax import nnx

from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    onnx_function,
    register_example,
)


@onnx_function
class MLPBlock(nnx.Module):
    """MLP block for Transformer layers."""

    def __init__(self, num_hiddens, mlp_dim, rngs: nnx.Rngs):
        self.layers = nnx.List(
            [
                nnx.Linear(num_hiddens, mlp_dim, rngs=rngs),
                lambda x: nnx.gelu(x, approximate=False),
                nnx.Dropout(rate=0.1, rngs=rngs),
                nnx.Linear(mlp_dim, num_hiddens, rngs=rngs),
                nnx.Dropout(rate=0.1, rngs=rngs),
            ]
        )

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        for layer in self.layers:
            if isinstance(layer, nnx.Dropout):
                x = layer(x, deterministic=deterministic)
            else:
                x = layer(x)
        return x


@onnx_function
class SuperBlock(nnx.Module):
    def __init__(self):
        rngs = nnx.Rngs(0)
        self.mlp = MLPBlock(num_hiddens=256, mlp_dim=512, rngs=rngs)

    def __call__(self, x):
        return self.mlp(x)


register_example(
    component="onnx_functions_002",
    description="two nested functions.",
    since="v0.4.0",
    context="examples.onnx_functions",
    children=["MLPBlock"],
    testcases=[
        {
            "testcase": "002_two_nested_functions",
            "callable": construct_and_call(SuperBlock),
            "input_shapes": [("B", 10, 256)],
            "expected_number_of_function_instances": 2,
            "run_only_f32_variant": True,
            "post_check_onnx_graph": expect_graph(
                ["SuperBlock_1:Bx10x256"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)
