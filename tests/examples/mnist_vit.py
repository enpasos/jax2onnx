# file: tests/examples/mnist_vit.py


import jax.numpy as jnp
from flax import nnx
import jax
from functools import partial
import onnx
import onnx.helper as oh
from jax2onnx.to_onnx import Z


class ReshapeWithOnnx:
    """Wrapper for reshape function with ONNX support."""

    def __init__(self, shape_fn):
        self.shape_fn = shape_fn

    def __call__(self, x):
        return x.reshape(self.shape_fn(x.shape))

    def to_onnx(self, z, _):
        return jax.numpy.reshape.to_onnx(z, shape=self.shape_fn(z.shapes[0]))


class TransposeWithOnnx:
    """Wrapper for transpose function with ONNX support."""

    def __init__(self, axes):
        self.axes = axes

    def __call__(self, x):
        return x.transpose(*self.axes)

    def to_onnx(self, z, _):
        return jax.numpy.transpose.to_onnx(z, axes=self.axes)


# Function for sinusoidal embeddings
def create_sinusoidal_embeddings(num_patches: int, num_hiddens: int) -> jnp.ndarray:
    position = jnp.arange(num_patches + 1)[:, jnp.newaxis]
    div_term = jnp.exp(
        jnp.arange(0, num_hiddens, 2) * -(jnp.log(10000.0) / num_hiddens)
    )
    pos_embedding = jnp.zeros((num_patches + 1, num_hiddens))
    pos_embedding = pos_embedding.at[:, 0::2].set(jnp.sin(position * div_term))
    pos_embedding = pos_embedding.at[:, 1::2].set(jnp.cos(position * div_term))
    return pos_embedding[jnp.newaxis, :, :]


class PatchEmbedding(nnx.Module):
    def __init__(
        self,
        height: int,
        width: int,
        patch_size: int,
        num_hiddens: int,
        in_features: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.num_patches_h = height // patch_size
        self.num_patches_w = width // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.linear = nnx.Linear(
            in_features=patch_size * patch_size * in_features,
            out_features=num_hiddens,
            rngs=rngs,
        )
        self.reshape1 = ReshapeWithOnnx(
            lambda shape: (
                shape[0],
                self.num_patches_h,
                self.patch_size,
                self.num_patches_w,
                self.patch_size,
                shape[-1],
            )
        )
        self.transpose = TransposeWithOnnx([0, 1, 3, 2, 4, 5])
        self.reshape2 = ReshapeWithOnnx(
            lambda shape: (
                shape[0],
                self.num_patches,
                self.patch_size * self.patch_size * shape[-1],
            )
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.reshape1(x)
        x = self.transpose(x)
        x = self.reshape2(x)
        return self.linear(x)

    def to_onnx(self, z, parameters=None):
        z = self.reshape1.to_onnx(z, parameters)
        z = self.transpose.to_onnx(z, parameters)
        z = self.reshape2.to_onnx(z, parameters)
        return self.linear.to_onnx(z, parameters)


class MLPBlock(nnx.Module):
    def __init__(
        self,
        num_hiddens: int,
        mlp_dim: int,
        dropout_rate: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        self.linear1 = nnx.Linear(
            in_features=num_hiddens, out_features=mlp_dim, rngs=rngs
        )
        self.linear2 = nnx.Linear(
            in_features=mlp_dim, out_features=num_hiddens, rngs=rngs
        )
        self.activation = partial(jax.nn.gelu, approximate=False)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        x = self.activation(self.linear1(x))
        x = self.dropout(x, deterministic=deterministic)
        x = self.linear2(x)
        x = self.dropout(x, deterministic=deterministic)
        return x

    def to_onnx(self, z, parameters=None):
        self.activation.to_onnx = jax.nn.gelu.to_onnx
        z = self.linear1.to_onnx(z)
        z = self.activation.to_onnx(z)
        z = self.dropout.to_onnx(z, parameters={"deterministic": True})
        z = self.linear2.to_onnx(z)
        z = self.dropout.to_onnx(z, parameters={"deterministic": True})
        return z


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        num_hiddens: int,
        num_heads: int,
        mlp_dim: int,
        dropout_rate: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        self.rng_collection = rngs  # <- Add this line
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
    def __init__(
        self,
        height: int,
        width: int,
        patch_size: int,
        num_hiddens: int,
        num_layers: int,
        num_heads: int,
        mlp_dim: int,
        num_classes: int,
        in_features: int,
        dropout_rate: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        self.patch_embedding = PatchEmbedding(
            height, width, patch_size, num_hiddens, in_features, rngs=rngs
        )
        self.cls_token = nnx.Param(jnp.zeros((1, 1, num_hiddens)))
        self.positional_embedding = nnx.Param(
            create_sinusoidal_embeddings(
                (height // patch_size) * (width // patch_size), num_hiddens
            )
        )
        self.transformer_blocks = [
            TransformerBlock(num_hiddens, num_heads, mlp_dim, dropout_rate, rngs=rngs)
            for _ in range(num_layers)
        ]
        self.layer_norm = nnx.LayerNorm(num_features=num_hiddens, rngs=rngs)
        self.dense = nnx.Linear(num_hiddens, num_classes, rngs=rngs)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        x = self.patch_embedding(x)
        batch_size = x.shape[0]

        cls_tokens = jnp.tile(self.cls_token, (batch_size, 1, 1))
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

    def to_onnx(self, z, parameters=None):
        z = self.patch_embedding.to_onnx(z)
        batch_size = z.shapes[0][0]

        cls_token_z = Z(
            shapes=[(1, 1, self.cls_token.value.shape[-1])],
            names=["cls_token"],
            onnx_graph=z.onnx_graph,
        )
        z.onnx_graph.add_initializer(
            oh.make_tensor(
                "cls_token",
                onnx.TensorProto.FLOAT,
                [1, 1, self.cls_token.value.shape[-1]],
                jnp.array(self.cls_token.value).flatten(),
            )
        )

        cls_tokens = jax.numpy.tile.to_onnx(cls_token_z, repeats=(batch_size, 1, 1))

        z = jax.numpy.concatenate.to_onnx(cls_tokens + z, axis=1)

        pos_emb_z = Z(
            shapes=[self.positional_embedding.value.shape],
            names=["positional_embedding"],
            onnx_graph=z.onnx_graph,
        )
        z.onnx_graph.add_initializer(
            oh.make_tensor(
                "positional_embedding",
                onnx.TensorProto.FLOAT,
                self.positional_embedding.value.shape,
                jnp.array(self.positional_embedding.value).flatten(),
            )
        )

        z = jnp.add.to_onnx(z + pos_emb_z)

        for block in self.transformer_blocks:
            z = block.to_onnx(z)
        z = self.layer_norm.to_onnx(z)
        z = jax.lax.slice.to_onnx(
            z, start=[0, 0, 0], end=[1, 1, self.dense.in_features]
        )

        z = jax.numpy.squeeze.to_onnx(z, axes=[1])

        z = self.dense.to_onnx(z)
        return jax.nn.log_softmax.to_onnx(z, parameters={"axis": -1})


def get_test_params():
    """Return test parameters for Vision Transformer."""
    return [
        {
            "testcase": "patch_embedding",
            "model": PatchEmbedding(
                height=28,
                width=28,
                patch_size=4,
                num_hiddens=256,
                in_features=1,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(1, 28, 28, 1)],
        },
        {
            "testcase": "mnist_vit",
            "model": VisionTransformer(
                height=28,
                width=28,
                patch_size=4,
                num_hiddens=256,
                num_layers=6,
                num_heads=8,
                mlp_dim=512,
                num_classes=10,
                in_features=1,
                dropout_rate=0.1,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(1, 28, 28, 1)],
        },
        {
            "testcase": "mlp_block",
            "model": MLPBlock(
                num_hiddens=256, mlp_dim=512, dropout_rate=0.1, rngs=nnx.Rngs(0)
            ),
            "input_shapes": [(1, 10, 256)],
        },
        {
            "testcase": "transformer_block",
            "model": TransformerBlock(
                num_hiddens=256,
                num_heads=8,
                mlp_dim=512,
                dropout_rate=0.1,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(1, 10, 256)],
        },
    ]
