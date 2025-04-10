# file: jax2onnx/sandbox/onnx_functions_example2.py


import os

import jax
import jax.numpy as jnp
import onnx
from flax import nnx

from jax2onnx import to_onnx
from jax2onnx.plugin_system import onnx_function


# ---------------------------------------------------------------------------
# Helper for positional embeddings
# ---------------------------------------------------------------------------
def create_sinusoidal_embeddings(num_patches: int, num_hiddens: int) -> jnp.ndarray:
    position = jnp.arange(num_patches + 1)[:, jnp.newaxis]
    div_term = jnp.exp(
        jnp.arange(0, num_hiddens, 2) * -(jnp.log(10000.0) / num_hiddens)
    )
    pos_embedding = jnp.zeros((num_patches + 1, num_hiddens))
    pos_embedding = pos_embedding.at[:, 0::2].set(jnp.sin(position * div_term))
    pos_embedding = pos_embedding.at[:, 1::2].set(jnp.cos(position * div_term))
    return pos_embedding[jnp.newaxis, :, :]


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------


@onnx_function
class PatchEmbedding(nnx.Module):
    """Patch embedding for Vision Transformers."""

    def __init__(
        self, height, width, patch_size, num_hiddens, in_features, *, rngs: nnx.Rngs
    ):
        num_patches_h, num_patches_w = height // patch_size, width // patch_size
        num_patches = num_patches_h * num_patches_w
        self.layers = [
            lambda x: x.reshape(
                x.shape[0],
                height // patch_size,
                patch_size,
                width // patch_size,
                patch_size,
                in_features,
            ),
            lambda x: x.transpose((0, 1, 3, 2, 4, 5)),
            lambda x: x.reshape(-1, num_patches, patch_size * patch_size * x.shape[-1]),
            nnx.Linear(
                in_features=patch_size * patch_size * in_features,
                out_features=num_hiddens,
                rngs=rngs,
            ),
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x


@onnx_function
class ConvEmbedding(nnx.Module):
    """Convolutional embedding for MNIST."""

    def __init__(
        self,
        W: int = 28,
        H: int = 28,
        embed_dims: list[int] = [32, 64, 128],
        kernel_size: int = 3,
        strides: list[int] = [1, 2, 2],
        dropout_rate: float = 0.5,
        *,
        rngs=nnx.Rngs(0),
    ):
        padding = "SAME"
        layernormfeatures = embed_dims[-1] * W // 4 * H // 4
        self.layers = [
            nnx.Conv(
                in_features=1,
                out_features=embed_dims[0],
                kernel_size=(kernel_size, kernel_size),
                strides=(strides[0], strides[0]),
                padding=padding,
                rngs=rngs,
            ),
            lambda x: nnx.gelu(x, approximate=False),
            nnx.Conv(
                in_features=embed_dims[0],
                out_features=embed_dims[1],
                kernel_size=(kernel_size, kernel_size),
                strides=(strides[1], strides[1]),
                padding=padding,
                rngs=rngs,
            ),
            lambda x: nnx.gelu(x, approximate=False),
            nnx.Conv(
                in_features=embed_dims[1],
                out_features=embed_dims[2],
                kernel_size=(kernel_size, kernel_size),
                strides=(strides[2], strides[2]),
                padding=padding,
                rngs=rngs,
            ),
            lambda x: nnx.gelu(x, approximate=False),
            nnx.LayerNorm(
                num_features=layernormfeatures,
                reduction_axes=(1, 2, 3),
                feature_axes=(1, 2, 3),
                rngs=rngs,
            ),
            nnx.Dropout(rate=dropout_rate, rngs=rngs),
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        return x


@onnx_function
class MLPBlock(nnx.Module):
    """MLP block for Transformer layers."""

    def __init__(self, num_hiddens, mlp_dim, dropout_rate=0.1, *, rngs: nnx.Rngs):
        self.layers = [
            nnx.Linear(num_hiddens, mlp_dim, rngs=rngs),
            lambda x: nnx.gelu(x, approximate=False),
            nnx.Dropout(rate=dropout_rate, rngs=rngs),
            nnx.Linear(mlp_dim, num_hiddens, rngs=rngs),
            nnx.Dropout(rate=dropout_rate, rngs=rngs),
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
        y = self.dropout(y, deterministic=deterministic)
        x = x + y
        return x + self.mlp_block(self.layer_norm2(x), deterministic)


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
        embed_dims: list[int] = [32, 128, 256],
        kernel_size: int = 3,
        strides: list[int] = [1, 2, 2],
        patch_size: int = 4,
        embedding_type: str = "conv",  # "conv" or "patch"
        embedding_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        mlp_dropout_rate: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        if embedding_type not in ["conv", "patch"]:
            raise ValueError("embedding_type must be either 'conv' or 'patch'")
        if embedding_type == "conv":
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
            num_patches = (height // 4) * (width // 4)
        else:
            self.embedding = PatchEmbedding(
                height=height,
                width=width,
                patch_size=patch_size,
                num_hiddens=num_hiddens,
                in_features=1,
                rngs=rngs,
            )
            num_patches = (height // patch_size) * (width // patch_size)
        self.cls_token = nnx.Param(
            jax.random.normal(rngs.params(), (1, 1, num_hiddens))
        )
        self.positional_embedding = nnx.Param(
            create_sinusoidal_embeddings(num_patches, num_hiddens)
        )
        self.transformer_blocks = [
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
        self.layer_norm = nnx.LayerNorm(num_features=num_hiddens, rngs=rngs)
        self.dense = nnx.Linear(num_hiddens, num_classes, rngs=rngs)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        x = self.embedding(x)
        batch_size = x.shape[0]
        cls_tokens = jnp.tile(self.cls_token.value, (batch_size, 1, 1))
        x = jnp.concatenate([cls_tokens, x], axis=1)
        pos_emb_expanded = jax.lax.dynamic_slice(
            self.positional_embedding.value, (0, 0, 0), (1, x.shape[1], x.shape[2])
        )
        pos_emb_expanded = jnp.asarray(pos_emb_expanded)
        x = x + pos_emb_expanded
        for block in self.transformer_blocks:
            x = block(x, deterministic)
        x = self.layer_norm(x)
        x = x[:, 0, :]
        return nnx.log_softmax(self.dense(x))


top_model = VisionTransformer(
    height=28,
    width=28,
    num_hiddens=256,
    num_layers=6,
    num_heads=8,
    mlp_dim=512,
    num_classes=10,
    embedding_type="conv",
    rngs=nnx.Rngs(0),
)

onnx_model = to_onnx(top_model, [(2, 28, 28, 1)])
output_path = "./docs/onnx/sandbox/vit.onnx"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
onnx.save(onnx_model, output_path)
