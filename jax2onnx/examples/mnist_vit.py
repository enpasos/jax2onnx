# file: tests/examples/mnist_vit.py

import jax.numpy as jnp
from flax import nnx
import jax
import onnx
import onnx.helper as oh

from typing import List, Any

import jax2onnx.plugins  # noqa: F401

from jax2onnx.convert import Z

from jax2onnx.typing_helpers import PartialWithOnnx, Supports2Onnx


class ReshapeWithOnnx:
    """Wrapper for reshape function with ONNX support."""

    def __init__(self, shape_fn):
        self.shape_fn = shape_fn

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Reshapes input using the provided shape function."""
        new_shape = self.shape_fn(x.shape)
        return x.reshape(new_shape)  # use the shape_fn

    def to_onnx(self, z, **params):
        new_shape = list(self.shape_fn(z.shapes[0]))
        # Ensure the shape is valid for ONNX
        # new_shape = [dim if dim != -1 else z.shapes[0][i] for i, dim in enumerate(new_shape)]
        # Convert dynamic dimensions to a valid ONNX representation
        # new_shape = [dim if isinstance(dim, int) else z.shapes[0][i] for i, dim in enumerate(new_shape)]
        return jax.numpy.reshape.to_onnx(z, shape=new_shape)
    

class TransposeWithOnnx:
    """Wrapper for transpose function with ONNX support."""

    def __init__(self, axes):
        self.axes = axes

    def __call__(self, x):
        return x.transpose(*self.axes)

    def to_onnx(self, z, **params):
        return jax.numpy.transpose.to_onnx(z, axes=self.axes)


def create_sinusoidal_embeddings(num_patches: int, num_hiddens: int) -> jnp.ndarray:
    """Generate sinusoidal positional embeddings."""
    position = jnp.arange(num_patches + 1)[:, jnp.newaxis]
    div_term = jnp.exp(
        jnp.arange(0, num_hiddens, 2) * -(jnp.log(10000.0) / num_hiddens)
    )
    pos_embedding = jnp.zeros((num_patches + 1, num_hiddens))
    pos_embedding = pos_embedding.at[:, 0::2].set(jnp.sin(position * div_term))
    pos_embedding = pos_embedding.at[:, 1::2].set(jnp.cos(position * div_term))
    return pos_embedding[jnp.newaxis, :, :]


class PatchEmbedding(nnx.Module):
    """Apply patch embedding for Vision Transformers."""

    def __init__(
        self, height, width, patch_size, num_hiddens, in_features, *, rngs: nnx.Rngs
    ):
        """Initializes the patch embedding module."""
        num_patches_h, num_patches_w = height // patch_size, width // patch_size
        num_patches = num_patches_h * num_patches_w

        self.layers = [
            ReshapeWithOnnx(
                lambda shape: (
                    -1,
                    num_patches_h,
                    patch_size,
                    num_patches_w,
                    patch_size,
                    in_features,  # Corrected: Use in_features
                )
            ),
            TransposeWithOnnx([0, 1, 3, 2, 4, 5]),
            ReshapeWithOnnx(
                lambda shape: (
                    -1,
                    num_patches,
                    patch_size * patch_size * shape[-1],
                )
            ),
            nnx.Linear(
                in_features=patch_size * patch_size * in_features,
                out_features=num_hiddens,
                rngs=rngs,
            ),
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass for patch embedding."""
        for layer in self.layers:
            x = layer(x)
        return x

    def to_onnx(self, z, **params):
        """ONNX conversion for patch embedding."""
        for layer in self.layers:
            z = layer.to_onnx(z, **params)
        return z

class MNISTConvolutionalTokenEmbedding(nnx.Module):
    """Convolutional Token Embedding for MNIST with hierarchical downsampling."""

    def __init__(
        self,
        W: int = 28,
        H: int = 28,
        embed_dims: List[int] = [32, 64, 128],
        kernel_size: int = 3,
        strides: List[int] = [1, 2, 2],
        dropout_rate=0.5,
        *,
        rngs=nnx.Rngs(0),
    ):
        """Initializes the convolutional embedding layers with hierarchical downsampling."""
        padding = "SAME"  # Explicit padding to match ONNX behavior

        layernormfeatures = embed_dims[-1] * W // 4 * H // 4

        self.layers: List[Supports2Onnx] = [
            nnx.Conv(
                in_features=1,
                out_features=embed_dims[0],
                kernel_size=(kernel_size, kernel_size),
                strides=(strides[0], strides[0]),
                padding=padding,
                rngs=rngs,
            ),
            PartialWithOnnx(nnx.gelu, approximate=False),
            nnx.Conv(
                in_features=embed_dims[0],
                out_features=embed_dims[1],
                kernel_size=(kernel_size, kernel_size),
                strides=(strides[1], strides[1]),
                padding=padding,
                rngs=rngs,
            ),
            PartialWithOnnx(nnx.gelu, approximate=False),
            nnx.Conv(
                in_features=embed_dims[1],
                out_features=embed_dims[2],
                kernel_size=(kernel_size, kernel_size),
                strides=(strides[2], strides[2]),
                padding=padding,
                rngs=rngs,
            ),
            PartialWithOnnx(nnx.gelu, approximate=False),
            nnx.LayerNorm(
                num_features=layernormfeatures,
                reduction_axes=(1, 2, 3),
                feature_axes=(1, 2, 3),
                rngs=rngs,
            ),
            nnx.Dropout(rate=dropout_rate, rngs=rngs),
        ]

        # LayerNorm will be initialized dynamically based on the last feature dimension
        self.layer_norm = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass: Applies convolutional layers and normalizes token embeddings."""
        for layer in self.layers:
            x = layer(x)

        B, H, W, C = x.shape

        # Flatten into token sequence
        x = x.reshape(B, H * W, C)  # [B, N, C]
        return x

    def to_onnx(self, z, **params):
        """Defines the ONNX export logic for convolutional token embedding."""
        for layer in self.layers:
            z = layer.to_onnx(z, **params)

        B, C, H, W = z.shapes[0]
        reshape_params = {
            "pre_transpose": [(0, 2, 3, 1)],  # Ensure correct ordering if needed
            "shape": (B, H * W, C),  # Flatten the feature map
        }
        # Handle dynamic batch dimension
        if isinstance(B, str):
            reshape_params["shape"] = [-1, H * W, C]
        return jax.numpy.reshape.to_onnx(z, **reshape_params)


class MLPBlock(nnx.Module):
    """MLP block for Transformer layers."""

    def __init__(self, num_hiddens, mlp_dim, dropout_rate=0.1, *, rngs: nnx.Rngs):
        """Initializes the MLP block."""
        self.layers: list[Supports2Onnx] = [
            nnx.Linear(num_hiddens, mlp_dim, rngs=rngs),
            PartialWithOnnx(nnx.gelu, approximate=False),
            nnx.Dropout(rate=dropout_rate, rngs=rngs),
            nnx.Linear(mlp_dim, num_hiddens, rngs=rngs),
            nnx.Dropout(rate=dropout_rate, rngs=rngs),
        ]

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """Forward pass for MLP block."""
        for layer in self.layers:
            if isinstance(layer, nnx.Dropout):
                x = layer(x, deterministic=deterministic)
            else:
                x = layer(x)
        return x

    def to_onnx(self, z, **params):
        for layer in self.layers:
            z = layer.to_onnx(z, **params)
        return z


class TransformerBlock(nnx.Module):
    """Transformer block with multi-head attention and MLP."""

    def __init__(
        self,
        num_hiddens: int,
        num_heads: int,
        mlp_dim: int,
        dropout_rate: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        self.rng_collection = rngs
        self.layer_norm1 = nnx.LayerNorm(num_hiddens, rngs=rngs)
        self.dropout_rate = dropout_rate
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            qkv_features=num_hiddens,
            out_features=num_hiddens,
            in_features=num_hiddens,
            rngs=rngs,
            decode=False,
        )
        self.layer_norm2 = nnx.LayerNorm(num_hiddens, rngs=rngs)
        self.mlp_block = MLPBlock(num_hiddens, mlp_dim, dropout_rate, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        y = self.attention(self.layer_norm1(x))
        y = self.dropout(y, deterministic=deterministic)
        x = x + y
        return x + self.mlp_block(self.layer_norm2(x), deterministic)

    def to_onnx(self, z, parameters=None):
        z_orig = z.clone()
        z = self.layer_norm1.to_onnx(z)
        z = self.attention.to_onnx(z)

        z = self.dropout.to_onnx(z, parameters={"deterministic": True})

        z = jnp.add.to_onnx(z_orig + z)

        z_orig = z.clone()
        z = self.layer_norm2.to_onnx(z)
        z = self.mlp_block.to_onnx(z)
        z = jnp.add.to_onnx(z_orig + z)

        return z


class VisionTransformer(nnx.Module):
    """Vision Transformer model for MNIST."""

    def __init__(
        self,
        height: int,
        width: int,
        # patch_size: int,
        num_hiddens: int,
        num_layers: int,
        num_heads: int,
        mlp_dim: int,
        num_classes: int,
        embed_dims: List[int] = [32, 128, 256],
        kernel_size: int = 3,
        strides: List[int] = [1, 2, 2],
        dropout_rate: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):

        # raise exception if embed_dims size is not 3 and embed_dims[2] != num_hiddens
        if len(embed_dims) != 3 or embed_dims[2] != num_hiddens:
            raise ValueError(
                "embed_dims should be a list of size 3 with embed_dims[2] == num_hiddens"
            )

        self.embedding = MNISTConvolutionalTokenEmbedding(
            embed_dims=embed_dims,
            kernel_size=kernel_size,
            strides=strides,
            dropout_rate=dropout_rate,
            rngs=rngs,
        )
        self.cls_token = nnx.Param(
            jax.random.normal(rngs.params(), (1, 1, num_hiddens))
        )

        self.positional_embedding = nnx.Param(
            create_sinusoidal_embeddings((height // 4) * (width // 4), num_hiddens)
        )
        self.transformer_blocks = [
            TransformerBlock(num_hiddens, num_heads, mlp_dim, dropout_rate, rngs=rngs)
            for _ in range(num_layers)
        ]
        self.layer_norm = nnx.LayerNorm(num_features=num_hiddens, rngs=rngs)
        self.dense = nnx.Linear(num_hiddens, num_classes, rngs=rngs)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        x = self.embedding(x)
        batch_size = x.shape[0]

        cls_tokens = jnp.tile(
            self.cls_token.value, (batch_size, 1, 1)
        )  # ✅ Use `.value`
        x = jnp.concatenate([cls_tokens, x], axis=1)

        pos_emb_expanded = jax.lax.dynamic_slice(
            self.positional_embedding.value, (0, 0, 0), (1, x.shape[1], x.shape[2])
        )  # ✅ Ensure `.value` usage
        pos_emb_expanded = jnp.asarray(pos_emb_expanded)

        x = x + pos_emb_expanded

        for block in self.transformer_blocks:
            x = block(x, deterministic)
        x = self.layer_norm(x)
        x = x[:, 0, :]
        return nnx.log_softmax(self.dense(x))

    def to_onnx(self, z, parameters=None):
        z = self.embedding.to_onnx(z)
        batch_size = z.shapes[0][0]

        cls_token_z = Z(
            shapes=[(1, 1, self.cls_token.value.shape[-1])],  # ✅ Ensure `.value`
            names=["cls_token"],
            onnx_graph=z.onnx_graph,
        )
        z.onnx_graph.add_initializer(
            oh.make_tensor(
                "cls_token",
                onnx.TensorProto.FLOAT,
                [1, 1, self.cls_token.value.shape[-1]],
                jnp.array(self.cls_token.value).flatten(),  # ✅ Use `.value`
            )
        )

        cls_tokens = jax.numpy.tile.to_onnx(cls_token_z, repeats=(batch_size, 1, 1))

        z = jax.numpy.concatenate.to_onnx(cls_tokens + z, axis=1)

        pos_emb_z = Z(
            shapes=[self.positional_embedding.value.shape],  # ✅ Use `.value`
            names=["positional_embedding"],
            onnx_graph=z.onnx_graph,
        )
        z.onnx_graph.add_initializer(
            oh.make_tensor(
                "positional_embedding",
                onnx.TensorProto.FLOAT,
                self.positional_embedding.value.shape,
                jnp.array(self.positional_embedding.value).flatten(),  # ✅ Use `.value`
            )
        )

        z = jnp.add.to_onnx(z + pos_emb_z)

        for block in self.transformer_blocks:
            z = block.to_onnx(z)
        z = self.layer_norm.to_onnx(z)

        z = jax.lax.gather.to_onnx(z, indices=0, axis=1)

        z = self.dense.to_onnx(z)
        return nnx.log_softmax.to_onnx(z, parameters={"axis": -1})


def get_test_params() -> list:
    """Return test parameters for Vision Transformer."""

    vit_params = {
        "height": 28,
        "width": 28,
        "num_hiddens": 256,
        "num_layers": 6,
        "num_heads": 8,
        "mlp_dim": 512,
        "num_classes": 10,
        "dropout_rate": 0.5,
        "rngs": nnx.Rngs(0),
    }

    return [
        {
            "component": "ViT",
            "description": "A MNIST Vision Transformer (ViT) model",
            "children": [
                "MNISTConvolutionalTokenEmbedding",
                "TransformerBlock",
                "flax.nnx.Linear",
                "flax.nnx.LayerNorm",
                "jax.lax.gather",
            ],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "mnist_vit",
                    "component": VisionTransformer(**vit_params),
                    "input_shapes": [(1, 28, 28, 1)],
                    "params": {
                        "pre_transpose": [(0, 3, 1, 2)],  # Convert JAX → ONNX
                    },
                }
            ],
        },
        {
            "component": "TransformerBlock",
            "description": "Transformer from 'Attention Is All You Need.'",
            "children": [
                "flax.nnx.MultiHeadAttention",
                "flax.nnx.LayerNorm",
                "MLPBlock",
                "flax.nnx.Dropout",
            ],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "transformer_block",
                    "component": TransformerBlock(
                        num_hiddens=256,
                        num_heads=8,
                        mlp_dim=512,
                        dropout_rate=0.1,
                        rngs=nnx.Rngs(0),
                    ),
                    "input_shapes": [(1, 10, 256)],
                },
            ],
        },
        {
            "component": "PatchEmbedding",
            "description": "Cutting the image into patches and linearly embedding them.",
            "children": ["flax.nnx.Linear", "jax.numpy.Transpose", "jax.numpy.Reshape"],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "patch_embedding",
                    "component": PatchEmbedding(
                        height=28,
                        width=28,
                        patch_size=4,
                        num_hiddens=256,
                        in_features=1,
                        rngs=nnx.Rngs(0),
                    ),
                    "input_shapes": [(1, 28, 28, 1)],
                }
            ],
        },
        {
            "component": "MLP Block",
            "description": "MLP in Transformer",
            "children": ["flax.nnx.Linear", "flax.nnx.Dropout", "flax.nnx.gelu"],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "mlp_block",
                    "component": MLPBlock(
                        num_hiddens=256, mlp_dim=512, dropout_rate=0.1, rngs=nnx.Rngs(0)
                    ),
                    "input_shapes": [(1, 10, 256)],
                },
            ],
        },
        {
            "component": "MNISTConvolutionalTokenEmbedding",
            "description": "Convolutional Token Embedding for MNIST with hierarchical downsampling.",
            "children": [
                "flax.nnx.Conv",
                "flax.nnx.LayerNorm",
                "jax.numpy.Reshape",
                "jax.nn.relu",
            ],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "mnist_conv_embedding",
                    "component": MNISTConvolutionalTokenEmbedding(
                        embed_dims=[32, 64, 128],
                        kernel_size=3,
                        strides=[1, 2, 2],
                        rngs=nnx.Rngs(0),
                    ),
                    "input_shapes": [(1, 28, 28, 1)],
                    "params": {
                        "pre_transpose": [(0, 3, 1, 2)],  # Convert JAX → ONNX
                    },
                }
            ],
        },
    ]
