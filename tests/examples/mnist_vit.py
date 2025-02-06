# file: tests/examples/mnist_vit.py
import jax.numpy as jnp
from flax import nnx
import jax
from jax2onnx.to_onnx import Z
from functools import partial
import numpy as np
import onnx
import onnx.helper as oh

def create_sinusoidal_embeddings(num_patches: int, num_hiddens: int) -> jnp.ndarray:
    position = jnp.arange(num_patches)[:, jnp.newaxis]
    div_term = jnp.exp(jnp.arange(0, num_hiddens, 2) * -(jnp.log(10000.0) / num_hiddens))
    pos_embedding = jnp.zeros((num_patches, num_hiddens))
    pos_embedding = pos_embedding.at[:, 0::2].set(jnp.sin(position * div_term))
    pos_embedding = pos_embedding.at[:, 1::2].set(jnp.cos(position * div_term))
    return pos_embedding[jnp.newaxis, :, :]

class PatchEmbedding(nnx.Module):
    def __init__(self, patch_size: int, num_hiddens: int, in_features: int, *, rngs: nnx.Rngs):
        self.patch_size = patch_size
        self.linear = nnx.Linear(in_features=patch_size * patch_size * in_features, out_features=num_hiddens, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch_size, _, _, channels = x.shape
        x = x.reshape(batch_size, -1, self.linear.in_features)
        return self.linear(x)

    def to_onnx(self, z, parameters=None):
        onnx_graph = z.onnx_graph
        input_shape = z.shapes[0]
        input_name = z.names[0]

        batch_size, height, width, channels = input_shape
        num_patches = (height // self.patch_size) * (width // self.patch_size)
        in_features = self.patch_size * self.patch_size * channels
        out_features = self.linear.out_features

        flattened_shape = (batch_size, num_patches, in_features)
        reshape_input_name = f"{input_name}_reshaped"

        onnx_graph.add_node(
            oh.make_node(
                "Reshape",
                inputs=[input_name, f"{reshape_input_name}_shape"],
                outputs=[reshape_input_name],
                name=f"reshape_before_{input_name}",
            )
        )

        onnx_graph.add_initializer(
            oh.make_tensor(
                f"{reshape_input_name}_shape",
                onnx.TensorProto.INT64,
                [3],
                np.array(flattened_shape, dtype=np.int64),
            )
        )
        onnx_graph.add_local_outputs([list(flattened_shape)], [reshape_input_name])

        z.shapes = [list(flattened_shape)]
        z.names = [reshape_input_name]
        z = self.linear.to_onnx(z)

        return z

class MLPBlock(nnx.Module):
    def __init__(self, num_hiddens: int, mlp_dim: int, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(in_features=num_hiddens, out_features=mlp_dim, rngs=rngs)
        self.linear2 = nnx.Linear(in_features=mlp_dim, out_features=num_hiddens, rngs=rngs)
        self.activation = partial(jax.nn.gelu, approximate=False)
        self.activation.to_onnx = jax.nn.gelu.to_onnx

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.linear2(self.activation(self.linear1(x)))

    def to_onnx(self, z, parameters=None):
        z = self.linear1.to_onnx(z)
        z = self.activation.to_onnx(z)
        z = self.linear2.to_onnx(z)
        return z

class TransformerBlock(nnx.Module):
    def __init__(self, num_hiddens: int, num_heads: int, mlp_dim: int, *, rngs: nnx.Rngs):
        self.layer_norm1 = nnx.LayerNorm(num_hiddens, rngs=rngs)
        self.attention = nnx.MultiHeadAttention(num_heads=num_heads, qkv_features=num_hiddens, out_features=num_hiddens, in_features=num_hiddens, rngs=rngs, decode=False)
        self.layer_norm2 = nnx.LayerNorm(num_hiddens, rngs=rngs)
        self.mlp_block = MLPBlock(num_hiddens, mlp_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x + self.attention(self.layer_norm1(x))
        return x + self.mlp_block(self.layer_norm2(x))

    def to_onnx(self, z, parameters=None):
        z_orig = z.clone()
        z = self.layer_norm1.to_onnx(z)
        z = self.attention.to_onnx(z)
        z = jnp.add.to_onnx(z_orig + z)

        z_orig = z.clone()
        z = self.layer_norm2.to_onnx(z)
        z = self.mlp_block.to_onnx(z)
        z = jnp.add.to_onnx(z_orig + z)

        return z


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
        x = self.layer_norm(x)
        x = x[:, 0, :]
        return nnx.log_softmax(self.dense(x))

    def to_onnx(self, z, parameters=None):
        z = self.patch_embedding.to_onnx(z)

        for block in self.transformer_blocks:
            z = block.to_onnx(z)

        z = self.layer_norm.to_onnx(z)
        z = jax.lax.slice.to_onnx(z, parameters={"start": [0, 0, 0], "end": [1, 1, 256]})
        z = jax.numpy.squeeze.to_onnx(z, parameters={"axes": [1]})  # Remove singleton dimension
 
        z = self.dense.to_onnx(z)

        return jax.nn.log_softmax.to_onnx(z, parameters={"axis": -1})

def get_test_params():
    return [
        {
            "model_name": "mnist_vit",
            "model":  VisionTransformer(patch_size=4, num_hiddens=256, num_layers=6, num_heads=8, mlp_dim=512, num_classes=10, in_features=1, rngs=nnx.Rngs(0)),
            "input_shapes": [(1, 28, 28, 1)]
        },
        {
            "model_name": "patch_embedding",
            "model":  PatchEmbedding(patch_size=4, num_hiddens=256, in_features=1, rngs=nnx.Rngs(0)),
            "input_shapes": [(1, 28, 28, 1)]
        },
        {
            "model_name": "mlp_block",
            "model":  MLPBlock(num_hiddens=256, mlp_dim=512, rngs=nnx.Rngs(0)),
            "input_shapes": [(1, 10, 256)]
        },
        {
            "model_name": "transformer_block",
            "model":  TransformerBlock(num_hiddens=256, num_heads=8, mlp_dim=512, rngs=nnx.Rngs(0)),
            "input_shapes": [(1, 10, 256)]
        }
    ]
