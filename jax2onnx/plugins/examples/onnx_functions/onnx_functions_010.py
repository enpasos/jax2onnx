# file: jax2onnx/plugins/examples/onnx_functions/onnx_functions_010.py


import jax.numpy as jnp
from flax import nnx

from jax2onnx.plugin_system import onnx_function, register_example


@onnx_function
class FeedForward(nnx.Module):
    def __init__(self, num_hiddens, mlp_dim, dropout_rate=0.1, *, rngs: nnx.Rngs):
        self.layers = [
            nnx.Linear(num_hiddens, mlp_dim, rngs=rngs),
            lambda x: nnx.gelu(x, approximate=False),
            nnx.Dropout(rate=0.1, rngs=rngs),
            nnx.Linear(mlp_dim, num_hiddens, rngs=rngs),
            nnx.Dropout(rate=0.1, rngs=rngs),
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        deterministic = True
        for layer in self.layers:
            if isinstance(layer, nnx.Dropout):
                x = layer(x, deterministic=deterministic)
            else:
                x = layer(x)
        return x


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
            attention_fn=lambda *args, **kwargs: attention(*args),
            rngs=rngs,
            decode=False,
        )
        self.dropout = nnx.Dropout(rate=attention_dropout_rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.attention(x, deterministic=True)
        x = self.dropout(x, deterministic=True)
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

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        r = self.layer_norm1(x)
        r = self.attention(r)
        x = x + r
        r = self.layer_norm2(x)
        return x + self.mlp_block(r)


@onnx_function
class TransformerStack(nnx.Module):
    def __init__(
        self,
        num_hiddens: int,
        num_heads: int,
        mlp_dim: int,
        num_layers: int,
        attention_dropout_rate: float = 0.1,
        mlp_dropout_rate: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        self.blocks = [
            TransformerBlock(
                num_hiddens,
                num_heads,
                mlp_dim,
                attention_dropout_rate,
                mlp_dropout_rate,
                rngs=rngs,
            )
            for _ in range(num_layers)
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for block in self.blocks:
            x = block(x)
        return x


register_example(
    component="onnx_functions_010",
    description="transformer stack",
    since="v0.4.0",
    context="examples.onnx_functions",
    children=["TransformerBlock"],
    testcases=[
        {
            "testcase": "010_transformer_stack",
            "callable": TransformerStack(
                num_hiddens=256,
                num_heads=8,
                mlp_dim=512,
                num_layers=6,
                attention_dropout_rate=0.1,
                mlp_dropout_rate=0.1,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [("B", 10, 256)],
            "expected_number_of_function_instances": 25,
            "run_only_f32_variant": True,
        },
    ],
)
