import flax.nnx as nnx
import jax.numpy as jnp

from jax2onnx.plugins.examples.onnx_functions.onnx_functions_012 import (
    VisionTransformer,
)

vit = VisionTransformer(
    height=28,
    width=28,
    num_hiddens=256,
    num_layers=6,
    num_heads=8,
    mlp_dim=512,
    num_classes=10,
    rngs=nnx.Rngs(0),
)
x = jnp.ones((3, 28, 28, 1))
out = vit(x)
print(out.shape)
