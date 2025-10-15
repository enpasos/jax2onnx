# jax2onnx/plugins/examples/onnx_functions/onnx_functions_015.py

from __future__ import annotations


import jax.numpy as jnp
from flax import nnx

from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    onnx_function,
    register_example,
)


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

    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
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
        self.layer_norm2 = nnx.LayerNorm(3, rngs=rngs)
        self.mlp = MLPBlock(num_hiddens=3, mlp_dim=6, rngs=rngs)

    def __call__(self, x, deterministic: bool):
        # Explicitly pass the deterministic parameter to the MLPBlock
        x_normalized = self.layer_norm2(x)
        return self.mlp(x_normalized, deterministic=deterministic)


register_example(
    component="onnx_functions_015",
    description="one function on an outer layer.",
    since="v0.4.0",
    context="examples.onnx_functions",
    children=["MLPBlock"],
    testcases=[
        {
            "testcase": "015_one_function_with_input_param_without_default_value",
            "callable": construct_and_call(SuperBlock),
            "input_shapes": [("B", 10, 3)],
            "expected_number_of_function_instances": 1,
            "input_params": {
                "deterministic": True,
            },
            "run_only_f32_variant": True,
            # GeLU inside the function leverages erf which accumulates ~1e-3 FP32 noise
            # across the nested calls. Relax the numeric tolerance accordingly.
            "rtol": 2e-3,
            "post_check_onnx_graph": expect_graph(
                ["SuperBlock_1:Bx10x3"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        }
    ],
)
