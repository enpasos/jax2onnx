# jax2onnx/plugins/examples/onnx_functions/onnx_functions_008.py

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


@onnx_function
class FeedForward(nnx.Module):
    def __init__(self, num_hiddens, mlp_dim, dropout_rate=0.1, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(num_hiddens, mlp_dim, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.linear2 = nnx.Linear(mlp_dim, num_hiddens, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, deterministic: bool = True) -> jnp.ndarray:
        x = self.linear1(x)
        x = nnx.gelu(x, approximate=False)
        x = self.dropout1(x, deterministic=deterministic)
        x = self.linear2(x)
        return self.dropout2(x, deterministic=deterministic)


@onnx_function
def attention(*args, **kwargs):
    return nnx.dot_product_attention(*args, **kwargs)


@onnx_function
class MultiHeadAttention(nnx.Module):
    def __init__(
        self,
        num_hiddens: int,
        num_heads: int,
        attention_dropout_rate: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            qkv_features=num_hiddens,
            out_features=num_hiddens,
            in_features=num_hiddens,
            attention_fn=attention,
            rngs=rngs,
            decode=False,
        )
        self.dropout = nnx.Dropout(rate=attention_dropout_rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, deterministic: bool = True) -> jnp.ndarray:
        x = self.attention(x, deterministic=deterministic)
        x = self.dropout(x, deterministic=deterministic)
        return x


@onnx_function
class TransformerBlock(nnx.Module):
    def __init__(
        self,
        num_hiddens: int,
        num_heads: int,
        mlp_dim: int,
        attention_dropout_rate: float = 0.1,
        mlp_dropout_rate: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        self.layer_norm1 = nnx.LayerNorm(num_hiddens, rngs=rngs)
        self.attention = MultiHeadAttention(
            num_hiddens=num_hiddens,
            num_heads=num_heads,
            attention_dropout_rate=attention_dropout_rate,
            rngs=rngs,
        )
        self.layer_norm2 = nnx.LayerNorm(num_hiddens, rngs=rngs)
        self.mlp_block = FeedForward(num_hiddens, mlp_dim, mlp_dropout_rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, deterministic: bool = True) -> jnp.ndarray:
        r = self.layer_norm1(x)
        r = self.attention(r, deterministic=deterministic)
        x = x + r
        r = self.layer_norm2(x)
        return x + self.mlp_block(r, deterministic=deterministic)


register_example(
    component="onnx_functions_008",
    description="transformer block with nested mlp block no call parameter",
    since="v0.4.0",
    context="examples.onnx_functions",
    children=[
        "FeedForward",
        "MultiHeadAttention",
    ],
    testcases=[
        {
            "testcase": "008_transformer_block",
            "callable": construct_and_call(
                TransformerBlock,
                num_hiddens=256,
                num_heads=8,
                mlp_dim=512,
                attention_dropout_rate=0.1,
                mlp_dropout_rate=0.1,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 10, 256)],
            "expected_number_of_function_instances": 3,
            "run_only_f32_variant": True,
            "post_check_onnx_graph": expect_graph(
                ["TransformerBlock_1:Bx10x256"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)
