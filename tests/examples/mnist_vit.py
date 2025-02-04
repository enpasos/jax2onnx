import jax.numpy as jnp
from flax import nnx
import jax
from jax2onnx.plugins.reshape import build_reshape_onnx_node

def create_sinusoidal_embeddings(num_patches: int, num_hiddens: int) -> jnp.ndarray:
    position = jnp.arange(num_patches)[:, jnp.newaxis]
    div_term = jnp.exp(jnp.arange(0, num_hiddens, 2) * -(jnp.log(10000.0) / num_hiddens))
    pos_embedding = jnp.zeros((num_patches, num_hiddens))
    pos_embedding = pos_embedding.at[:, 0::2].set(jnp.sin(position * div_term))
    pos_embedding = pos_embedding.at[:, 1::2].set(jnp.cos(position * div_term))
    return pos_embedding[jnp.newaxis, :, :]

class PatchEmbedding(nnx.Module):
    def __init__(self, patch_size: int, num_hiddens: int, in_features: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(in_features=patch_size * patch_size * in_features, out_features=num_hiddens, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch_size, _, _, channels = x.shape
        x = x.reshape(batch_size, -1, self.linear.in_features)
        return self.linear(x)

    def build_onnx_node(self, xs, names, onnx_graph, parameters=None):
        return self.linear.build_onnx_node(xs, names, onnx_graph)

class MLPBlock(nnx.Module):
    def __init__(self, num_hiddens: int, mlp_dim: int, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(in_features=num_hiddens, out_features=mlp_dim, rngs=rngs)
        self.linear2 = nnx.Linear(in_features=mlp_dim, out_features=num_hiddens, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.linear2(jax.nn.gelu(self.linear1(x)))

    def build_onnx_node(self, xs, names, onnx_graph, parameters=None):
        xs, names = self.linear1.build_onnx_node(xs, names, onnx_graph)
        xs, names = jax.nn.gelu.build_onnx_node(jax.nn.gelu, xs, names, onnx_graph)
        return self.linear2.build_onnx_node(xs, names, onnx_graph)

class TransformerBlock(nnx.Module):
    def __init__(self, num_hiddens: int, num_heads: int, mlp_dim: int, *, rngs: nnx.Rngs):
        self.layer_norm1 = nnx.LayerNorm(num_hiddens, rngs=rngs)
        self.attention = nnx.MultiHeadAttention(num_heads=num_heads, qkv_features=num_hiddens, out_features=num_hiddens, in_features=num_hiddens, rngs=rngs, decode=False)
        self.layer_norm2 = nnx.LayerNorm(num_hiddens, rngs=rngs)
        self.mlp_block = MLPBlock(num_hiddens, mlp_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x + self.attention(self.layer_norm1(x))
        return x + self.mlp_block(self.layer_norm2(x))

    def build_onnx_node(self, xs, names, onnx_graph, parameters=None):
        xs, names = self.layer_norm1.build_onnx_node(xs, names, onnx_graph)
        xs, names = self.attention.build_onnx_node(xs, names, onnx_graph)
        xs, names = self.layer_norm2.build_onnx_node(xs, names, onnx_graph)
        return self.mlp_block.build_onnx_node(xs, names, onnx_graph)

class VisionTransformer(nnx.Module):
    def __init__(self, patch_size: int, num_hiddens: int, num_layers: int, num_heads: int, mlp_dim: int, num_classes: int, in_features: int, *, rngs: nnx.Rngs):
        self.patch_embedding = PatchEmbedding(patch_size, num_hiddens, in_features, rngs=rngs)
        self.transformer_blocks = [TransformerBlock(num_hiddens, num_heads, mlp_dim, rngs=rngs) for _ in range(num_layers)]
        self.layer_norm = nnx.LayerNorm(num_features=num_hiddens, rngs=rngs)
        self.dense = nnx.Linear(num_hiddens, num_classes, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.patch_embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        return nnx.log_softmax(self.dense(self.layer_norm(x)[:, 0, :]))

    def build_onnx_node(self, xs, names, onnx_graph, parameters=None):
        xs, names = self.patch_embedding.build_onnx_node(xs, names, onnx_graph)
        for block in self.transformer_blocks:
            xs, names = block.build_onnx_node(xs, names, onnx_graph)
        xs, names = self.layer_norm.build_onnx_node(xs, names, onnx_graph)
        xs, names = self.dense.build_onnx_node(xs, names, onnx_graph)
        return jnp.log_softmax.build_onnx_node(nnx.log_softmax, xs, names, onnx_graph)

def get_test_params():
    return [
        {
            "model_name": "mnist_vit",
            "model": lambda: VisionTransformer(patch_size=4, num_hiddens=256, num_layers=6, num_heads=8, mlp_dim=512, num_classes=10, in_features=1, rngs=nnx.Rngs(0)),
            "input_shapes": [(1, 28, 28, 1)],
            "build_onnx_node": VisionTransformer.build_onnx_node
        }
    ]
