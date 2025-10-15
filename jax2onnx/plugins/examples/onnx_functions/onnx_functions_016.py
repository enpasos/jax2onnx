# jax2onnx/plugins/examples/onnx_functions/onnx_functions_016.py

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


@onnx_function
class NestedBlock(nnx.Module):
    def __init__(self, num_hiddens, mlp_dim, dropout_rate=0.1, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(num_hiddens, mlp_dim, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.linear2 = nnx.Linear(mlp_dim, num_hiddens, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        x = self.linear1(x)
        x = nnx.gelu(x, approximate=False)
        x = self.dropout1(x, deterministic=deterministic)
        x = self.linear2(x)
        return self.dropout2(x, deterministic=deterministic)


@onnx_function
class SuperBlock(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        num_hiddens = 256
        self.layer_norm2 = nnx.LayerNorm(num_hiddens, rngs=rngs)
        self.mlp = NestedBlock(num_hiddens, mlp_dim=512, rngs=rngs)

    def __call__(self, x, deterministic: bool = True):
        return self.mlp(self.layer_norm2(x), deterministic=deterministic)


register_example(
    component="onnx_functions_016",
    description="nested function plus more components",
    since="v0.4.0",
    context="examples.onnx_functions",
    children=["NestedBlock"],
    testcases=[
        {
            "testcase": "016_internal_function_with_input_param_with_default_value",
            "callable": construct_and_call(SuperBlock, rngs=with_rng_seed(0)),
            "input_shapes": [("B", 10, 256)],
            "expected_number_of_function_instances": 2,
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["SuperBlock_1:Bx10x256"], symbols={"B": None}, no_unused_inputs=True
            ),
        },
    ],
)
