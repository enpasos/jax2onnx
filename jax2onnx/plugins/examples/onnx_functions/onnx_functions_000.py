# file: jax2onnx/plugins/examples/onnx_functions/onnx_functions_000.py


from flax import nnx
import jax.numpy as jnp

from jax2onnx.plugin_system import onnx_function, register_example


class MLPBlock000(nnx.Module):
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


@onnx_function
class SuperBlock000(nnx.Module):
    def __init__(self):
        rngs = nnx.Rngs(0)  # Example RNGs initialization
        self.layer_norm2 = nnx.LayerNorm(256, rngs=rngs)
        self.mlp = MLPBlock000(num_hiddens=256, mlp_dim=512, rngs=rngs)

    def __call__(self, x):
        return self.mlp(x)


register_example(
    component="onnx_functions_000",
    description="one function on an outer layer.",
    # source="https:/",
    since="v0.4.0",
    context="examples.onnx_functions",
    children=["MLPBlock000"],
    testcases=[
        {
            "testcase": "one_function_outer",
            "callable": SuperBlock000(),
            "input_shapes": [("B", 10, 256)],
        },
    ],
)
