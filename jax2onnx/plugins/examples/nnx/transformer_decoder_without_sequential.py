# jax2onnx/plugins/examples/nnx/transformer_decoder_without_sequential.py

from __future__ import annotations

import jax
from flax import nnx

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_example,
    with_rng_seed,
)


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
        self.self_attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            dropout_rate=attention_dropout,
            decode=False,
            rngs=rngs,
        )
        self.cross_attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            dropout_rate=encoder_attention_dropout,
            decode=False,
            rngs=rngs,
        )
        self.lin1 = nnx.Linear(in_features=embed_dim, out_features=ff_dim, rngs=rngs)
        self.lin2 = nnx.Linear(in_features=ff_dim, out_features=embed_dim, rngs=rngs)
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
        deterministic: bool = True,
        decode=None,
    ) -> jax.Array:
        attn_output = self.self_attn(
            inputs_q=x, mask=mask, deterministic=deterministic, decode=decode
        )
        attn_output = self.dropout1(attn_output, deterministic=deterministic)
        x_resid = (x + attn_output) if self.allow_residue else attn_output
        x = self.layernorm1(x_resid)

        cross_attn_output = self.cross_attn(
            inputs_q=x,
            inputs_k=encoder_output,
            mask=cross_attn_mask,
            deterministic=deterministic,
        )
        x = self.layernorm2(
            x + self.dropout2(cross_attn_output, deterministic=deterministic)
        )

        ffn_output = self.lin2(nnx.relu(self.lin1(x)))
        x = self.layernorm3(x + self.dropout3(ffn_output, deterministic=deterministic))
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
        self.layers = nnx.List(
            [
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
        )

    def __call__(
        self,
        x: jax.Array,
        encoder_output: jax.Array,
        mask: jax.Array | None = None,
        cross_attn_mask: jax.Array | None = None,
        *,
        deterministic: bool = True,
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
    component="TransformerDecoderWithoutSequential",
    description="Tiny nnx Transformer decoder with explicit FFN layers (no Sequential).",
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
            "testcase": "tiny_decoder_without_sequential",
            "callable": construct_and_call(
                TransformerDecoder,
                num_layers=1,
                embed_dim=16,
                num_heads=4,
                ff_dim=32,
                attention_dropout=0.5,
                encoder_attention_dropout=0.5,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 8, 16), (2, 4, 16)],
            "expected_output_shapes": [(2, 8, 16)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    "Add:Bx8x16 -> LayerNormalization:Bx8x16 -> Add:Bx8x16 -> "
                    "LayerNormalization:Bx8x16 -> Add:Bx8x16 -> "
                    "LayerNormalization:Bx8x16"
                ],
                symbols={"B": 2},
                no_unused_inputs=True,
            ),
        }
    ],
)
