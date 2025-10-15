# jax2onnx/plugins/examples/onnx_functions/onnx_functions_012.py

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    onnx_function,
    register_example,
    with_rng_seed,
)


def create_sinusoidal_embeddings(num_patches: int, num_hiddens: int) -> jnp.ndarray:
    position = jnp.arange(num_patches + 1)[:, jnp.newaxis]
    div_term = jnp.exp(
        jnp.arange(0, num_hiddens, 2) * -(jnp.log(10000.0) / num_hiddens)
    )
    pos_embedding = jnp.zeros((num_patches + 1, num_hiddens))
    pos_embedding = pos_embedding.at[:, 0::2].set(jnp.sin(position * div_term))
    pos_embedding = pos_embedding.at[:, 1::2].set(jnp.cos(position * div_term))
    return pos_embedding[jnp.newaxis, :, :]


@onnx_function
class ConvEmbedding(nnx.Module):
    """Convolutional embedding for MNIST."""

    def __init__(
        self,
        W: int = 28,
        H: int = 28,
        embed_dims: Sequence[int] = (32, 64, 128),
        kernel_size: int = 3,
        strides: Sequence[int] = (1, 2, 2),
        dropout_rate: float = 0.5,
        *,
        rngs: nnx.Rngs,
    ):
        embed_dims = tuple(embed_dims)
        strides = tuple(strides)
        padding = "SAME"
        layernormfeatures = embed_dims[-1] * W // 4 * H // 4
        self.conv1 = nnx.Conv(
            in_features=1,
            out_features=embed_dims[0],
            kernel_size=(kernel_size, kernel_size),
            strides=(strides[0], strides[0]),
            padding=padding,
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            in_features=embed_dims[0],
            out_features=embed_dims[1],
            kernel_size=(kernel_size, kernel_size),
            strides=(strides[1], strides[1]),
            padding=padding,
            rngs=rngs,
        )
        self.conv3 = nnx.Conv(
            in_features=embed_dims[1],
            out_features=embed_dims[2],
            kernel_size=(kernel_size, kernel_size),
            strides=(strides[2], strides[2]),
            padding=padding,
            rngs=rngs,
        )
        self.layer_norm = nnx.LayerNorm(
            num_features=layernormfeatures,
            reduction_axes=(1, 2, 3),
            feature_axes=(1, 2, 3),
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, deterministic: bool = True) -> jnp.ndarray:
        x = self.conv1(x)
        x = nnx.gelu(x, approximate=False)
        x = self.conv2(x)
        x = nnx.gelu(x, approximate=False)
        x = self.conv3(x)
        x = nnx.gelu(x, approximate=False)
        x = self.layer_norm(x)
        x = self.dropout(x, deterministic=deterministic)
        batch_size, height, width, channels = x.shape
        return jnp.reshape(x, (batch_size, height * width, channels))


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
        self.blocks = nnx.List(
            [
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
        )

    def __call__(self, x: jnp.ndarray, *, deterministic: bool = True) -> jnp.ndarray:
        for block in self.blocks:
            x = block(x, deterministic=deterministic)
        return x


@onnx_function
class ClassificationHead(nnx.Module):
    def __init__(
        self,
        num_hiddens: int,
        num_classes: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.layer_norm = nnx.LayerNorm(num_features=num_hiddens, rngs=rngs)
        self.dense = nnx.Linear(num_hiddens, num_classes, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.layer_norm(x)
        x = x[:, 0, :]
        return nnx.log_softmax(self.dense(x))


@onnx_function
class ConcatClsToken(nnx.Module):
    def __init__(
        self,
        num_hiddens: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.cls_token = nnx.Param(
            jax.random.normal(rngs.params(), (1, 1, num_hiddens))
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch_size = x.shape[0]
        cls_tokens = jnp.tile(self.cls_token.value, (batch_size, 1, 1))
        x = jnp.concatenate([cls_tokens, x], axis=1)
        return x


@onnx_function
class PositionalEmbedding(nnx.Module):
    def __init__(
        self, num_patches: int, num_hiddens: int, *, rngs: nnx.Rngs | None = None
    ):
        del rngs
        self.positional_embedding = nnx.Param(
            create_sinusoidal_embeddings(num_patches, num_hiddens)
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x + self.positional_embedding.value
        return x


@onnx_function
class VisionTransformer(nnx.Module):
    """Vision Transformer model for MNIST with configurable embedding type."""

    def __init__(
        self,
        height: int,
        width: int,
        num_hiddens: int,
        num_layers: int,
        num_heads: int,
        mlp_dim: int,
        num_classes: int,
        embed_dims: Sequence[int] = (32, 128, 256),
        kernel_size: int = 3,
        strides: Sequence[int] = (1, 2, 2),
        embedding_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        mlp_dropout_rate: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        embed_dims = tuple(embed_dims)
        if len(embed_dims) != 3 or embed_dims[2] != num_hiddens:
            raise ValueError(
                "embed_dims should be a list of size 3 with embed_dims[2] == num_hiddens"
            )
        self.embedding = ConvEmbedding(
            embed_dims=embed_dims,
            kernel_size=kernel_size,
            strides=strides,
            dropout_rate=embedding_dropout_rate,
            rngs=rngs,
        )

        self.concat_cls_token = ConcatClsToken(num_hiddens=num_hiddens, rngs=rngs)

        num_patches = (height // 4) * (width // 4)

        self.cls_token = nnx.Param(
            jax.random.normal(rngs.params(), (1, 1, num_hiddens))
        )
        self.positional_embedding = PositionalEmbedding(
            num_hiddens=num_hiddens, num_patches=num_patches
        )

        self.transformer_stack = TransformerStack(
            num_hiddens,
            num_heads,
            mlp_dim,
            num_layers,
            attention_dropout_rate,
            mlp_dropout_rate,
            rngs=rngs,
        )
        self.classification_head = ClassificationHead(
            num_hiddens=num_hiddens,
            num_classes=num_classes,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray, *, deterministic: bool = True) -> jnp.ndarray:
        if x is None or x.shape[0] == 0:
            raise ValueError("Input tensor 'x' must not be None or empty.")

        # Ensure input tensor has the expected shape
        if len(x.shape) != 4 or x.shape[-1] != 1:
            raise ValueError("Input tensor 'x' must have shape (B, H, W, 1).")

        x = self.embedding(x, deterministic=deterministic)
        x = self.concat_cls_token(x)

        x = self.positional_embedding(x)

        x = self.transformer_stack(x, deterministic=deterministic)
        x = self.classification_head(x)
        return x


register_example(
    component="onnx_functions_012",
    description="Vision Transformer (ViT)",
    since="v0.4.0",
    context="examples.onnx_functions",
    children=[
        "PatchEmbedding",
        "ConvEmbedding",
        "MLPBlock",
        "TransformerBlock",
        "ClassificationHead",
        "nnx.MultiHeadAttention",
        "nnx.LayerNorm",
        "nnx.Linear",
        "nnx.gelu",
        "nnx.Dropout",
        "nnx.Param",
    ],
    testcases=[
        {
            "testcase": "012_vit_conv_embedding",
            "callable": construct_and_call(
                VisionTransformer,
                height=28,
                width=28,
                num_hiddens=256,
                num_layers=6,
                num_heads=8,
                mlp_dim=512,
                num_classes=10,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 28, 28, 1)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": expect_graph(
                ["VisionTransformer_1:Bx10"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        }
    ],
)
