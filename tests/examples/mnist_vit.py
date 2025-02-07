# file: tests/examples/mnist_vit.py

import jax.numpy as jnp
from flax import nnx
import jax
from functools import partial
import onnx
import onnx.helper as oh
from jax2onnx.to_onnx import Z

# todo: instead of num_patches (and adding 1 for the cls internally) use sequence length.
# this function should not know details about how it is used.
def create_sinusoidal_embeddings(num_patches: int, num_hiddens: int) -> jnp.ndarray:
    position = jnp.arange(num_patches + 1)[:, jnp.newaxis]  # Include CLS token
    div_term = jnp.exp(jnp.arange(0, num_hiddens, 2) * -(jnp.log(10000.0) / num_hiddens))
    pos_embedding = jnp.zeros((num_patches + 1, num_hiddens))
    pos_embedding = pos_embedding.at[:, 0::2].set(jnp.sin(position * div_term))
    pos_embedding = pos_embedding.at[:, 1::2].set(jnp.cos(position * div_term))
    return pos_embedding[jnp.newaxis, :, :]

class PatchEmbedding(nnx.Module):
    def __init__(self, height: int, width: int, patch_size: int, num_hiddens: int, in_features: int, *, rngs: nnx.Rngs):
        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.num_patches_h = height // patch_size
        self.num_patches_w = width // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.linear = nnx.Linear(in_features=patch_size * patch_size * in_features, out_features=num_hiddens, rngs=rngs)

        # First reshape: Extract patches in a structured manner
        def reshape1_fn(x):
            return x.reshape(x.shape[0], self.num_patches_h, patch_size, self.num_patches_w, patch_size, x.shape[-1])

        self.reshape1 = reshape1_fn
        self.reshape1.to_onnx = lambda z, _: jax.numpy.reshape.to_onnx(
            z,
            parameters={"shape": (z.shapes[0][0], self.num_patches_h, patch_size, self.num_patches_w, patch_size, z.shapes[0][-1])}
        )

        # Transpose: Reorder dimensions for proper patch extraction
        def transpose_fn(x):
            return x.transpose(0, 1, 3, 2, 4, 5)

        self.transpose = transpose_fn
        self.transpose.to_onnx = lambda z, _: jax.numpy.transpose.to_onnx(
            z, parameters={"axes": [0, 1, 3, 2, 4, 5]}
        )

        # Second reshape: Flatten patches before feeding into the linear layer
        def reshape2_fn(x):
            return x.reshape(x.shape[0], self.num_patches, patch_size * patch_size * x.shape[-1])

        self.reshape2 = reshape2_fn
        self.reshape2.to_onnx = lambda z, _: jax.numpy.reshape.to_onnx(
            z,
            parameters={"shape": (z.shapes[0][0], self.num_patches, patch_size * patch_size * z.shapes[0][-1])}
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply transformations in sequence."""
        x = self.reshape1(x)
        x = self.transpose(x)
        x = self.reshape2(x)
        return self.linear(x)

    def to_onnx(self, z, parameters=None):
        """Apply ONNX transformations in sequence."""
        z = self.reshape1.to_onnx(z, parameters)
        z = self.transpose.to_onnx(z, parameters)
        z = self.reshape2.to_onnx(z, parameters)
        return self.linear.to_onnx(z, parameters)


class MLPBlock(nnx.Module):
    def __init__(self, num_hiddens: int, mlp_dim: int, dropout_rate: float = 0.1, *, rngs: nnx.Rngs):
        self.num_hiddens = num_hiddens
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.rng_collection = rngs
        self.linear1 = nnx.Linear(in_features=num_hiddens, out_features=mlp_dim, rngs=rngs)
        self.linear2 = nnx.Linear(in_features=mlp_dim, out_features=num_hiddens, rngs=rngs)
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
    def __init__(self, num_hiddens: int, num_heads: int, mlp_dim: int, dropout_rate: float = 0.1, *, rngs: nnx.Rngs):
        self.rng_collection = rngs  # <- Add this line
        self.layer_norm1 = nnx.LayerNorm(num_hiddens, rngs=rngs)
        self.dropout_rate = dropout_rate
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            qkv_features=num_hiddens,
            out_features=num_hiddens,
            in_features=num_hiddens,
            rngs=rngs,
            decode=False
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
    def __init__(self, height: int, width: int, patch_size: int, num_hiddens: int, num_layers: int,
                 num_heads: int, mlp_dim: int, num_classes: int, in_features: int, dropout_rate: float = 0.1, *, rngs: nnx.Rngs):
        self.patch_embedding = PatchEmbedding(height, width, patch_size, num_hiddens, in_features, rngs=rngs)
        self.cls_token = nnx.Param(jnp.zeros((1, 1, num_hiddens)))
        self.positional_embedding = nnx.Param(create_sinusoidal_embeddings((height // patch_size) * (width // patch_size), num_hiddens))
        self.transformer_blocks = [
            TransformerBlock(num_hiddens, num_heads, mlp_dim, dropout_rate, rngs=rngs) for _ in range(num_layers)
        ]
        self.layer_norm = nnx.LayerNorm(num_features=num_hiddens, rngs=rngs)
        self.dense = nnx.Linear(num_hiddens, num_classes, rngs=rngs)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        x = self.patch_embedding(x)
        batch_size = x.shape[0]

        cls_tokens = jnp.tile(self.cls_token, (batch_size, 1, 1))
        x = jnp.concatenate([cls_tokens, x], axis=1)

        # pos_emb_slice = jax.lax.dynamic_slice(self.positional_embedding.value, (0, 0, 0), (1, x.shape[1] - 1, x.shape[2]))
        # pos_emb_slice = jnp.asarray(pos_emb_slice)  # Ensure it's a JAX array
        # x = x + jnp.concatenate([self.cls_token, pos_emb_slice], axis=1)

        # Ensure positional encoding applies to CLS token as well
        pos_emb_expanded = jax.lax.dynamic_slice(self.positional_embedding.value, (0, 0, 0), (1, x.shape[1], x.shape[2]))
        pos_emb_expanded = jnp.asarray(pos_emb_expanded)  # Ensure it's a JAX array

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
            onnx_graph=z.onnx_graph
        )
        z.onnx_graph.add_initializer(
            oh.make_tensor(
                "cls_token",
                onnx.TensorProto.FLOAT,
                [1, 1, self.cls_token.value.shape[-1]],
                jnp.array(self.cls_token.value).flatten()
            )
        )

        cls_tokens = jax.numpy.tile.to_onnx(cls_token_z, parameters={"repeats": (batch_size, 1, 1)})
        z = jax.numpy.concatenate.to_onnx(cls_tokens + z, parameters={"axis": 1})

        pos_emb_z = Z(
            shapes=self.positional_embedding.value.shape,
            names=["positional_embedding"],
            onnx_graph=z.onnx_graph
        )
        z.onnx_graph.add_initializer(
            oh.make_tensor(
                "positional_embedding",
                onnx.TensorProto.FLOAT,
                self.positional_embedding.value.shape,
                jnp.array(self.positional_embedding.value).flatten()
            )
        )

        pos_emb_expanded = jax.lax.slice.to_onnx(
            pos_emb_z, parameters={"start": [0, 0, 0], "end": [1, z.shapes[0][1], z.shapes[0][2]]}
        )
        z = jax.numpy.add.to_onnx(z + pos_emb_expanded)

        for block in self.transformer_blocks:
            z = block.to_onnx(z)
        z = self.layer_norm.to_onnx(z)
        z = jax.lax.slice.to_onnx(z, parameters={"start": [0, 0, 0], "end": [1, 1, self.dense.in_features]})
        z = jax.numpy.squeeze.to_onnx(z, parameters={"axes": [1]})
        z = self.dense.to_onnx(z)
        return jax.nn.log_softmax.to_onnx(z, parameters={"axis": -1})


def get_test_params():
    return [
        {
            "model_name": "patch_embedding",
            "model": PatchEmbedding(height=28, width=28, patch_size=4, num_hiddens=256, in_features=1, rngs=nnx.Rngs(0)),
            "input_shapes": [(1, 28, 28, 1)]
        },
        {
            "model_name": "mnist_vit",
            "model": VisionTransformer(
                height=28, width=28, patch_size=4, num_hiddens=256, num_layers=6, num_heads=8,
                mlp_dim=512, num_classes=10, in_features=1, dropout_rate=0.1, rngs=nnx.Rngs(0)
            ),
            "input_shapes": [(1, 28, 28, 1)]
        },
        {
            "model_name": "mlp_block",
            "model": MLPBlock(num_hiddens=256, mlp_dim=512, dropout_rate=0.1, rngs=nnx.Rngs(0)),
            "input_shapes": [(1, 10, 256)]
        },
        {
            "model_name": "transformer_block",
            "model": TransformerBlock(num_hiddens=256, num_heads=8, mlp_dim=512, dropout_rate=0.1, rngs=nnx.Rngs(0)),
            "input_shapes": [(1, 10, 256)]
        }
    ]
