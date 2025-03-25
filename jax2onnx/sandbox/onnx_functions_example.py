# file: jax2onnx/sandbox/onnx_functions_example.py


from jax2onnx import to_onnx
from flax import nnx
import os
import onnx
import jax.numpy as jnp

from jax2onnx.plugin_system import onnx_function


@onnx_function
class MLPBlock(nnx.Module):
    """MLP block for Transformer layers."""

    def __init__(self, num_hiddens, mlp_dim, rngs: nnx.Rngs):
        self.layers = [
            nnx.Linear(num_hiddens, mlp_dim, rngs=rngs),
            # lambda x: nnx.gelu(x, approximate=False),
            # nnx.Dropout(rate=dropout_rate, rngs=rngs),
            # nnx.Linear(mlp_dim, num_hiddens, rngs=rngs),
            # nnx.Dropout(rate=dropout_rate, rngs=rngs),
        ]

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        for layer in self.layers:
            # if isinstance(layer, nnx.Dropout):
            #     x = layer(x, deterministic=deterministic)
            # else:
            x = layer(x)
        return x


# @onnx_function
class TransformerBlock(nnx.Module):
    def __init__(self):
        rngs = nnx.Rngs(0)  # Example RNGs initialization
        self.mlp = MLPBlock(num_hiddens=256, mlp_dim=512, rngs=rngs)

    def __call__(self, x):
        return self.mlp(x)


top_model = TransformerBlock()
onnx_model = to_onnx(top_model, [("B", 10, 256)])
output_path = "./docs/onnx/sandbox/transformer_block.onnx"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
onnx.save(onnx_model, output_path)
