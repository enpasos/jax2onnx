# file: jax2onnx/sandbox/onnx_functions_example.py


import os

import jax.numpy as jnp
import onnx
from flax import nnx

from jax2onnx import to_onnx
from jax2onnx.plugin_system import onnx_function


@onnx_function
class MLPBlock(nnx.Module):
    """MLP block for Transformer layers."""

    def __init__(self, num_hiddens, mlp_dim, dropout_rate=0.1, *, rngs: nnx.Rngs):
        self.layers = [
            nnx.Linear(num_hiddens, mlp_dim, rngs=rngs),
            # lambda x: nnx.gelu(x, approximate=False),
            # nnx.Dropout(rate=0.1, rngs=rngs),
            # nnx.Linear(mlp_dim, num_hiddens, rngs=rngs),
            # nnx.Dropout(rate=0.1, rngs=rngs),
        ]

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        for layer in self.layers:
            if isinstance(layer, nnx.Dropout):
                x = layer(x, deterministic=deterministic)
            else:
                x = layer(x)
        return x


@onnx_function
class TransformerBlock(nnx.Module):
    """Transformer block with multi-head attention and MLP."""

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
        self.rng_collection = rngs
        self.layer_norm1 = nnx.LayerNorm(num_hiddens, rngs=rngs)
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            qkv_features=num_hiddens,
            out_features=num_hiddens,
            in_features=num_hiddens,
            rngs=rngs,
            decode=False,
        )
        self.layer_norm2 = nnx.LayerNorm(num_hiddens, rngs=rngs)
        self.mlp_block = MLPBlock(num_hiddens, mlp_dim, mlp_dropout_rate, rngs=rngs)
        self.dropout = nnx.Dropout(rate=attention_dropout_rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        y = self.attention(self.layer_norm1(x))
        y = x
        y = self.dropout(y, deterministic=deterministic)
        x = x + y
        return x + self.mlp_block(self.layer_norm2(x), deterministic)


top_model = TransformerBlock(
    num_hiddens=256,
    num_heads=8,
    mlp_dim=512,
    attention_dropout_rate=0.1,
    mlp_dropout_rate=0.1,
    rngs=nnx.Rngs(0),
)
onnx_model = to_onnx(top_model, [("B", 10, 256)])
output_path = "./docs/onnx/sandbox/transformer_block.onnx"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
onnx.save(onnx_model, output_path)
