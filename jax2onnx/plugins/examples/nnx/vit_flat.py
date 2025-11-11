# jax2onnx/plugins/examples/nnx/vit_flat.py


import jax
import jax.numpy as jnp
from flax import nnx

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_example,
    with_rng_seed,
)


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


class PatchEmbedding(nnx.Module):
    """Patch embedding for Vision Transformers."""

    def __init__(
        self, height, width, patch_size, num_hiddens, in_features, *, rngs: nnx.Rngs
    ):
        num_patches_h, num_patches_w = height // patch_size, width // patch_size
        num_patches = num_patches_h * num_patches_w
        self.layers = nnx.List(
            [
                lambda x: x.reshape(
                    x.shape[0],
                    height // patch_size,
                    patch_size,
                    width // patch_size,
                    patch_size,
                    in_features,
                ),
                lambda x: x.transpose((0, 1, 3, 2, 4, 5)),
                lambda x: x.reshape(
                    -1, num_patches, patch_size * patch_size * x.shape[-1]
                ),
                nnx.Linear(
                    in_features=patch_size * patch_size * in_features,
                    out_features=num_hiddens,
                    rngs=rngs,
                ),
            ]
        )

    def __call__(self, x: jnp.ndarray, deterministic=True) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x


register_example(
    component="PatchEmbeddingFlatten",
    description="Cutting the image into patches and linearly embedding them.",
    source="https://github.com/google/flax/blob/main/README.md",
    since="v0.1.0",
    context="examples.vit",
    children=["flax.nnx.Linear", "jax.numpy.Transpose", "jax.numpy.Reshape"],
    testcases=[
        {
            "testcase": "vit_patch_embedding_flat",
            "callable": construct_and_call(
                PatchEmbedding,
                height=28,
                width=28,
                patch_size=4,
                num_hiddens=256,
                in_features=1,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 28, 28, 1)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    "Reshape:Bx7x4x7x4x1 -> Transpose:Bx7x7x4x4x1 -> "
                    "Reshape:Bx49x16 -> Reshape:?x16 -> Gemm:?x256 -> "
                    "Reshape:Bx49x256"
                ],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        }
    ],
)


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
        rngs=with_rng_seed(0),
    ):
        padding = "SAME"
        layernormfeatures = embed_dims[-1] * W // 4 * H // 4
        self.layers = nnx.List(
            [
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
        )

    def __call__(self, x: jnp.ndarray, deterministic=True) -> jnp.ndarray:
        for layer in self.layers:
            if isinstance(layer, nnx.Dropout):
                x = layer(x, deterministic=deterministic)
            else:
                x = layer(x)
        B, H, W, C = x.shape
        x = jnp.reshape(x, (B, H * W, C))
        return x


register_example(
    component="ConvEmbeddingFlatten",
    description="Convolutional Token Embedding for MNIST with hierarchical downsampling.",
    source="https://github.com/google/flax/blob/main/README.md",
    since="v0.1.0",
    context="examples.vit",
    children=[
        "flax.nnx.Conv",
        "flax.nnx.LayerNorm",
        "jax.numpy.Reshape",
        "jax.nn.relu",
    ],
    testcases=[
        {
            "testcase": "vit_mnist_conv_embedding_flat",
            "callable": construct_and_call(
                ConvEmbedding,
                embed_dims=[32, 64, 128],
                kernel_size=3,
                strides=[1, 2, 2],
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 28, 28, 1)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    "Transpose:Bx1x28x28 -> Conv:Bx32x28x28 -> Gelu:Bx32x28x28 -> "
                    "Conv:Bx64x14x14 -> Gelu:Bx64x14x14 -> Conv:Bx128x7x7 -> "
                    "Transpose:Bx7x7x128 -> Gelu:Bx7x7x128 -> Reshape:Bx6272 -> "
                    "LayerNormalization:Bx6272 -> Reshape:Bx7x7x128 -> "
                    "Dropout:Bx7x7x128 -> Reshape:Bx49x128"
                ],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        }
    ],
)


class FeedForward(nnx.Module):
    """MLP block for Transformer layers."""

    def __init__(self, num_hiddens, mlp_dim, dropout_rate=0.1, *, rngs: nnx.Rngs):
        self.layers = nnx.List(
            [
                nnx.Linear(num_hiddens, mlp_dim, rngs=rngs),
                lambda x: nnx.gelu(x, approximate=False),
                nnx.Dropout(rate=0.1, rngs=rngs),
                nnx.Linear(mlp_dim, num_hiddens, rngs=rngs),
                nnx.Dropout(rate=0.1, rngs=rngs),
            ]
        )

    def __call__(self, x: jnp.ndarray, deterministic=True) -> jnp.ndarray:
        for layer in self.layers:
            if isinstance(layer, nnx.Dropout):
                x = layer(x, deterministic=deterministic)
            else:
                x = layer(x)
        return x


register_example(
    component="FeedForwardFlatten",
    description="MLP in Transformer",
    source="https://github.com/google/flax/blob/main/README.md",
    since="v0.1.0",
    context="examples.vit",
    children=["flax.nnx.Linear", "flax.nnx.Dropout", "flax.nnx.gelu"],
    testcases=[
        {
            "testcase": "vit_feed_forward_flat",
            "callable": construct_and_call(
                FeedForward,
                num_hiddens=256,
                mlp_dim=512,
                dropout_rate=0.1,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 10, 256)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {
                            1: {"const": 0.1},
                            2: {"const_bool": False},
                        },
                        "path": "Reshape:?x256 -> Gemm:?x512 -> Reshape:Bx10x512 -> "
                        "Gelu:Bx10x512 -> Dropout:Bx10x512 -> Reshape:?x512 -> "
                        "Gemm:?x256 -> Reshape:Bx10x256 -> Dropout:Bx10x256",
                    }
                ],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)


def attention(*args, **kwargs):
    return nnx.dot_product_attention(*args, **kwargs)


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
            attention_fn=lambda *args, **kwargs: attention(*args, **kwargs),
            rngs=rngs,
            decode=False,
        )
        self.dropout = nnx.Dropout(rate=attention_dropout_rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray, deterministic=True) -> jnp.ndarray:
        x = self.attention(x, deterministic=deterministic)
        x = self.dropout(x, deterministic=deterministic)
        return x


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
        self.layer_norm1 = nnx.LayerNorm(num_hiddens, rngs=rngs)
        self.attention = MultiHeadAttention(
            num_hiddens=num_hiddens,
            num_heads=num_heads,
            attention_dropout_rate=attention_dropout_rate,
            rngs=rngs,
        )
        self.layer_norm2 = nnx.LayerNorm(num_hiddens, rngs=rngs)
        self.mlp_block = FeedForward(num_hiddens, mlp_dim, mlp_dropout_rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray, deterministic=True) -> jnp.ndarray:
        r = self.layer_norm1(x)
        r = self.attention(r, deterministic=deterministic)
        x = x + r
        r = self.layer_norm2(x)
        return x + self.mlp_block(r, deterministic=deterministic)


register_example(
    component="TransformerBlockFlatten",
    description="Transformer from 'Attention Is All You Need.'",
    source="https://github.com/google/flax/blob/main/README.md",
    since="v0.1.0",
    context="examples.vit",
    children=[
        "flax.nnx.MultiHeadAttention",
        "flax.nnx.LayerNorm",
        "MLPBlock",
        "flax.nnx.Dropout",
    ],
    testcases=[
        {
            "testcase": "vit_transformer_block_flat",
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
            "input_params": {
                "deterministic": True,
            },
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Add:Bx10x256 -> Add:Bx10x256"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)


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

    def __call__(self, x: jnp.ndarray, deterministic=True) -> jnp.ndarray:
        for block in self.blocks:
            x = block(x, deterministic=deterministic)
        return x


register_example(
    component="TransformerStackFlatten",
    description="Stack of Transformer blocks",
    source="https://github.com/google/flax/blob/main/README.md",
    since="v0.1.0",
    context="examples.vit",
    children=["TransformerBlock"],
    testcases=[
        {
            "testcase": "vit_transformer_stack_flat",
            "callable": construct_and_call(
                TransformerStack,
                num_hiddens=256,
                num_heads=8,
                mlp_dim=512,
                num_layers=6,
                attention_dropout_rate=0.1,
                mlp_dropout_rate=0.1,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 10, 256)],
            "input_params": {
                "deterministic": True,
            },
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    "Add:Bx10x256 -> Add:Bx10x256 -> Add:Bx10x256 -> Add:Bx10x256 -> "
                    "Add:Bx10x256 -> Add:Bx10x256 -> Add:Bx10x256 -> Add:Bx10x256 -> "
                    "Add:Bx10x256 -> Add:Bx10x256 -> Add:Bx10x256 -> Add:Bx10x256"
                ],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)


def get_token(x, index) -> jnp.ndarray:
    return x[:, index, :]


register_example(
    component="GetTokenFlatten",
    description="Get the CLS token from the input embedding",
    since="v0.4.0",
    context="examples.vit",
    children=[],
    testcases=[
        {
            "testcase": "vit_get_token_flat",
            "callable": lambda x: get_token(x, 0),
            "input_shapes": [("B", 50, 256)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    "Slice -> Squeeze",
                    {
                        "path": "Transpose:50xBx256 -> Gather:Bx256",
                        "inputs": {1: {"const": 0.0}},
                    },
                ],
                mode="any",
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)


class ClassificationHead(nnx.Module):
    """Classification head for Vision Transformer."""

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
        x = get_token(x, 0)
        x = self.layer_norm(x)
        return nnx.log_softmax(self.dense(x))


register_example(
    component="ClassificationHeadFlatten",
    description="Classification head for Vision Transformer",
    since="v0.4.0",
    context="examples.vit",
    children=["flax.nnx.LayerNorm", "flax.nnx.Linear", "flax.nnx.log_softmax"],
    testcases=[
        {
            "testcase": "vit_classification_head_flat",
            "callable": construct_and_call(
                ClassificationHead,
                num_hiddens=256,
                num_classes=10,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 50, 256)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["LayerNormalization -> Gemm -> LogSoftmax"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)


class ConcatClsToken(nnx.Module):
    """Concatenate CLS token to the input embedding."""

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


register_example(
    component="ConcatClsTokenFlatten",
    description="Concatenate CLS token to the input embedding",
    since="v0.4.0",
    context="examples.vit",
    children=["flax.nnx.Param", "jax.numpy.tile", "jax.numpy.concatenate"],
    testcases=[
        {
            "testcase": "vit_concat_cls_token_flat",
            "callable": construct_and_call(
                ConcatClsToken,
                num_hiddens=256,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 49, 256)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Tile -> Concat:Bx50x256"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)


class PositionalEmbedding(nnx.Module):
    """Add positional embedding to the input embedding."""

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


register_example(
    component="PositionalEmbeddingFlatten",
    description="Add positional embedding to the input embedding",
    since="v0.4.0",
    context="examples.vit",
    children=["flax.nnx.Param"],
    testcases=[
        {
            "testcase": "vit_positional_embedding_flat",
            "callable": construct_and_call(
                PositionalEmbedding,
                num_patches=49,
                num_hiddens=256,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 50, 256)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Add:Bx50x256"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)


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
        embedding_type: str = "conv",
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

        self.concat_cls_token = ConcatClsToken(num_hiddens=num_hiddens, rngs=rngs)
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

    def __call__(self, x: jnp.ndarray, deterministic=True) -> jnp.ndarray:
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
    component="VisionTransformerFlatten",
    description="A Vision Transformer (ViT) model for MNIST with configurable embedding type.",
    source="https://github.com/google/flax/blob/main/README.md",
    since="v0.2.0",
    context="examples.vit",
    children=[
        "PatchEmbedding",
        "ConvEmbedding",
        "MLPBlock",
        "TransformerBlock",
        "nnx.MultiHeadAttention",
        "nnx.LayerNorm",
        "nnx.Linear",
        "nnx.gelu",
        "nnx.Dropout",
        "nnx.Param",
    ],
    testcases=[
        {
            "testcase": "vit_model_conv_embedding_flat",
            "callable": construct_and_call(
                VisionTransformer,
                height=28,
                width=28,
                num_hiddens=256,
                num_layers=6,
                num_heads=8,
                mlp_dim=512,
                num_classes=10,
                embedding_type="conv",
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 28, 28, 1)],
            "input_params": {
                "deterministic": True,
            },
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["LayerNormalization -> Gemm -> LogSoftmax"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "vit_model_patch_embedding_flat",
            "callable": construct_and_call(
                VisionTransformer,
                height=28,
                width=28,
                num_hiddens=256,
                num_layers=6,
                num_heads=8,
                mlp_dim=512,
                num_classes=10,
                embedding_type="patch",
                patch_size=4,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 28, 28, 1)],
            "input_params": {
                "deterministic": True,
            },
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["LayerNormalization -> Gemm -> LogSoftmax"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)
