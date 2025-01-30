# file: examples/mnist_vit.py

import jax.numpy as jnp
from flax import nnx

# Hyperparameters
BATCH_SIZE: int = 64



def create_sinusoidal_embeddings(num_patches: int, num_hiddens: int) -> jnp.ndarray:
    """
    Create sinusoidal positional embeddings.

    Args:
        num_patches: Number of patches in the input sequence.
        num_hiddens: Dimensionality of the hidden embeddings.

    Returns:
        Sinusoidal positional embeddings with shape (1, num_patches, num_hiddens).
    """
    position = jnp.arange(num_patches)[:, jnp.newaxis]
    div_term = jnp.exp(jnp.arange(0, num_hiddens, 2) * -(jnp.log(10000.0) / num_hiddens))

    # Alternate sine and cosine for even and odd indices
    pos_embedding = jnp.zeros((num_patches, num_hiddens))
    pos_embedding = pos_embedding.at[:, 0::2].set(jnp.sin(position * div_term))
    pos_embedding = pos_embedding.at[:, 1::2].set(jnp.cos(position * div_term))

    return pos_embedding[jnp.newaxis, :, :]  # Add batch dimension


class PatchEmbedding(nnx.Module):
    """Patch Embedding layer for Vision Transformer."""

    def __init__(self, patch_size: int, num_hiddens: int, in_features: int, *, rngs: nnx.Rngs):
        self.patch_size = patch_size
        self.num_hiddens = num_hiddens
        self.linear = nnx.Linear(in_features=patch_size * patch_size * in_features, out_features=num_hiddens, rngs=rngs)


    # x: (B, H, W, C) -> (B, num_patches, patch_size * patch_size * channels) -> (B, num_patches, num_hiddens)
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch_size, height, width, channels = x.shape
        x = x.reshape(batch_size, -1, self.patch_size * self.patch_size * channels)
        x = self.linear(x)
        return x



# Dropout implementation
def dropout(rng_key, x, rate=0.5):
    """
    Implements dropout by zeroing out elements with a given probability (rate).

    Args:
        rng_key: JAX PRNG key for randomness.
        x: Input array.
        rate: Dropout rate, probability of setting an element to 0.

    Returns:
        The array with dropout applied.
    """
    keep_prob = 1.0 - rate
    mask = random.bernoulli(rng_key, keep_prob, shape=x.shape)
    return x * mask / keep_prob


# MLPBlock class
class MLPBlock(nnx.Module):
    """MLP Block for Vision Transformer."""

    def __init__(self, num_hiddens: int, mlp_dim: int, dropout_rate: float, *, rngs: nnx.Rngs):
        self.num_hiddens = num_hiddens
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.linear1 = nnx.Linear(in_features=num_hiddens, out_features=mlp_dim, rngs=rngs)
        self.linear2 = nnx.Linear(in_features=mlp_dim, out_features=num_hiddens, rngs=rngs)
        self.rng_collection = rngs

    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        x = nnx.gelu(self.linear1(x))

        if not deterministic:
            key = self.rng_collection()
            x = dropout(key, x, rate=self.dropout_rate)

        x = self.linear2(x)

        if not deterministic:
            key = self.rng_collection()
            x = dropout(key, x, rate=self.dropout_rate)

        return x

# TransformerBlock class
class TransformerBlock(nnx.Module):
    """Transformer Block for Vision Transformer."""

    def __init__(self, num_hiddens: int, num_heads: int, mlp_dim: int, dropout_rate: float, *, rngs: nnx.Rngs):
        self.num_hiddens = num_hiddens
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.layer_norm1 = nnx.LayerNorm(num_hiddens, rngs=rngs)
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
        self.rng_collection = rngs

    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        y = self.layer_norm1(x)
        y = self.attention(y)

        if not deterministic:
            key = self.rng_collection()
            y = dropout(key, y, rate=self.dropout_rate)

        x = x + y

        y = self.layer_norm2(x)
        y = self.mlp_block(y, deterministic)
        x = x + y

        return x


class VisionTransformer(nnx.Module):
    """Vision Transformer model with [CLS] Token and 2D sinusoidal positional embeddings."""

    def __init__(self, patch_size: int, num_hiddens: int, num_layers: int, num_heads: int, mlp_dim: int, num_classes: int, dropout_rate: float, in_features: int, *, rngs: nnx.Rngs):
        self.patch_size = patch_size
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        self.patch_embedding = PatchEmbedding(patch_size, num_hiddens, in_features, rngs=rngs)
        self.transformer_blocks = [
            TransformerBlock(num_hiddens, num_heads, mlp_dim, dropout_rate, rngs=rngs)
            for _ in range(num_layers)
        ]
        self.layer_norm = nnx.LayerNorm(num_features=num_hiddens, rngs=rngs)
        self.dense = nnx.Linear(num_hiddens, num_classes, rngs=rngs)

        # Fixed 2D sinusoidal positional embeddings
        self.positional_embedding = nnx.Param(
            create_sinusoidal_embeddings(28 // patch_size * 28 // patch_size, num_hiddens)
        )

        # Learnable [CLS] token
        self.cls_token = nnx.Param(jnp.zeros((1, 1, num_hiddens)))

        self.rng_collection = rngs # Make rngs accessible

    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        # Patch embeddings
        x = self.patch_embedding(x)  # Shape: (B, num_patches, num_hiddens)

        # Add [CLS] token
        batch_size = x.shape[0]
        cls_tokens = jnp.tile(self.cls_token, (batch_size, 1, 1))  # Shape: (B, 1, num_hiddens)
        x = jnp.concatenate([cls_tokens, x], axis=1)  # Shape: (B, 1 + num_patches, num_hiddens)

        # Add positional embeddings
        x = x + jnp.concatenate([self.cls_token, self.positional_embedding], axis=1)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, deterministic)

        # Final layer normalization
        x = self.layer_norm(x)

        # Use the [CLS] token's representation for classification
        cls_representation = x[:, 0, :]  # Extract the [CLS] token (B, num_hiddens)

        # Classification head
        x = self.dense(cls_representation)  # Shape: (B, num_classes)
        return nnx.log_softmax(x)

def get_test_params():
    """
    Test parameters for the Vision Transformer model.
    """
    return [
        # {
        #     "model_name": "mnist_vit",
        #     "model": lambda: VisionTransformer(
        #         patch_size=4,
        #         num_hiddens=256,
        #         num_layers=6,
        #         num_heads=8,
        #         mlp_dim=512,
        #         num_classes=10,
        #         dropout_rate=0.1,
        #         in_features=1,
        #         rngs=nnx.Rngs(0)
        #     ),
        #     "input_shapes": [(1, 28, 28, 1)],
        #     "build_onnx_node": VisionTransformer.build_onnx_node
        # }
    ]
