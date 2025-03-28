# file: jax2onnx/sandbox/onnx_functions_example.py


from flax import nnx
import jax.numpy as jnp

from jax2onnx.plugin_system import onnx_function, register_example


@onnx_function
class MLPBlock001(nnx.Module):
    """MLP block for Transformer layers."""

    def __init__(self, num_hiddens, mlp_dim, rngs: nnx.Rngs):
        self.layers = [
            nnx.Linear(num_hiddens, mlp_dim, rngs=rngs),
            lambda x: nnx.gelu(x, approximate=False),
            nnx.Dropout(rate=0.1, rngs=rngs),
            nnx.Linear(mlp_dim, num_hiddens, rngs=rngs),
            nnx.Dropout(rate=0.1, rngs=rngs),
        ]

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        for layer in self.layers:
            if isinstance(layer, nnx.Dropout):
                x = layer(x, deterministic=deterministic)
            else:
                x = layer(x)
        return x


class SuperBlock001(nnx.Module):
    def __init__(self):
        rngs = nnx.Rngs(0)  # Example RNGs initialization
        self.layer_norm2 = nnx.LayerNorm(256, rngs=rngs)
        self.mlp = MLPBlock001(num_hiddens=256, mlp_dim=512, rngs=rngs)

    def __call__(self, x):
        return self.mlp(x)


register_example(
    component="onnx_functions_001",
    description="one function on an inner layer.",
    # source="https:/",
    since="v0.4.0",
    context="examples.onnx_functions",
    children=["MLPBlock001"],
    testcases=[
        {
            "testcase": "001_one_function_inner",
            "callable": SuperBlock001(),
            "input_shapes": [("B", 10, 256)],
        },
    ],
)
