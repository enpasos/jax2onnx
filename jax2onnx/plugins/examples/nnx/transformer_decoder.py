# file: jax2onnx/examples/transformer_decoder.py

import jax
import jax.numpy as jnp
from flax import nnx
from jax2onnx.plugin_system import onnx_function, register_example

# ------------------------------------------------------------------------------
# A tiny Transformer‐style decoder built from nnx primitives
# ------------------------------------------------------------------------------
class TransformerDecoderLayer(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        *,
        rngs: nnx.Rngs,
        rate: float = 0.1,
        attention_dropout: float = 0.0,
        encoder_attention_dropout: float = 0.0,
        allow_residue: bool = True,
    ):
        # masked self‐attention
        self.self_attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            dropout_rate=attention_dropout,
            decode=False,
            rngs=rngs,
        )
        # cross‐attention
        self.cross_attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            dropout_rate=encoder_attention_dropout,
            decode=False,
            rngs=rngs,
        )
        # feed‐forward
        self.ffn = nnx.Sequential(
            nnx.Linear(in_features=embed_dim, out_features=ff_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(in_features=ff_dim, out_features=embed_dim, rngs=rngs),
        )
        # norms & dropouts
        self.layernorm1 = nnx.LayerNorm(num_features=embed_dim, rngs=rngs)
        self.layernorm2 = nnx.LayerNorm(num_features=embed_dim, rngs=rngs)
        self.layernorm3 = nnx.LayerNorm(num_features=embed_dim, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=rate, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=rate, rngs=rngs)
        self.dropout3 = nnx.Dropout(rate=rate, rngs=rngs)

        self.allow_residue = allow_residue

    def __call__(
        self,
        x: jax.Array,
        encoder_output: jax.Array,
        mask: jax.Array | None = None,
        cross_attn_mask: jax.Array | None = None,
        *,
        deterministic: bool = False,
        decode=None,
    ) -> jax.Array:
        # self‐attention block
        attn1 = self.self_attn(
            inputs_q=x, mask=mask, deterministic=deterministic, decode=decode
        )
        attn1 = self.dropout1(attn1, deterministic=deterministic)
        x = (x + attn1) if self.allow_residue else attn1
        x = self.layernorm1(x)

        # cross‐attention block
        attn2 = self.cross_attn(
            inputs_q=x,
            inputs_k=encoder_output,
            mask=cross_attn_mask,
            deterministic=deterministic,
        )
        attn2 = self.dropout2(attn2, deterministic=deterministic)
        x = self.layernorm2(x + attn2)

        # feed‐forward block
        ffn_out = self.ffn(x)
        ffn_out = self.dropout3(ffn_out, deterministic=deterministic)
        x = self.layernorm3(x + ffn_out)

        return x


class TransformerDecoder(nnx.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        *,
        rngs: nnx.Rngs,
        rate: float = 0.1,
        attention_dropout: float = 0.0,
        encoder_attention_dropout: float = 0.0,
        allow_residue: bool = True,
    ):
        self.layers = [
            TransformerDecoderLayer(
                embed_dim,
                num_heads,
                ff_dim,
                rngs=rngs,
                rate=rate,
                attention_dropout=attention_dropout,
                encoder_attention_dropout=encoder_attention_dropout,
                allow_residue=allow_residue,
            )
            for _ in range(num_layers)
        ]

    def __call__(
        self,
        x: jax.Array,
        encoder_output: jax.Array,
        mask: jax.Array | None = None,
        cross_attn_mask: jax.Array | None = None,
        *,
        deterministic: bool = False,
        decode=None,
    ) -> jax.Array:
        for layer in self.layers:
            x = layer(
                x,
                encoder_output,
                mask,
                cross_attn_mask,
                deterministic=deterministic,
                decode=decode,
            )
        return x


register_example(
    component="TransformerDecoder",
    description="A single-layer Transformer decoder built with nnx primitives (MHA, LayerNorm, Feed-Forward, Dropout).",
    source="https://github.com/google/flax/tree/main/flax/nnx",
    since="v0.7.1",
    context="examples.nnx",
    children=[
        "nnx.MultiHeadAttention",
        "nnx.LayerNorm",
        "nnx.Linear",
        "nnx.Dropout",
        "nnx.relu",
    ],
    testcases=[
        {
            "testcase": "tiny_decoder",
            "callable": TransformerDecoder(
                num_layers=1,
                embed_dim=16,
                num_heads=4,
                ff_dim=32,
                rngs=nnx.Rngs(0, params=42, dropout=1),
                attention_dropout=0.0,
                encoder_attention_dropout=0.0,
            ),
            "input_shapes": [("B", 8, 16), ("B", 4, 16)],
        }
    ],
)
