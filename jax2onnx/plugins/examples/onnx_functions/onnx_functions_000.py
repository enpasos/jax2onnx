# file: jax2onnx/plugins/examples/onnx_functions/onnx_functions_000.py


import jax.numpy as jnp
from flax import nnx

from jax2onnx.plugin_system import onnx_function, register_example


class MLPBlock(nnx.Module):
    """MLP block for Transformer layers."""

    def __init__(self, num_hiddens, mlp_dim, rngs: nnx.Rngs):
        self.layers = [
            nnx.Linear(num_hiddens, mlp_dim, rngs=rngs),
            lambda x: nnx.gelu(x, approximate=False),
            nnx.Dropout(rate=0.1, rngs=rngs),
            nnx.Linear(mlp_dim, num_hiddens, rngs=rngs),
            nnx.Dropout(rate=0.1, rngs=rngs),
        ]

    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        for layer in self.layers:
            if isinstance(layer, nnx.Dropout):
                x = layer(x, deterministic=deterministic)
            else:
                x = layer(x)
        return x


@onnx_function
class SuperBlock(nnx.Module):
    def __init__(self):
        rngs = nnx.Rngs(0)
        self.layer_norm2 = nnx.LayerNorm(3, rngs=rngs)
        self.mlp = MLPBlock(num_hiddens=3, mlp_dim=6, rngs=rngs)

    def __call__(self, x):
        # Explicitly pass the deterministic parameter to the MLPBlock
        x_normalized = self.layer_norm2(x)
        return self.mlp(x_normalized, deterministic=True)


register_example(
    component="onnx_functions_000",
    description="one function on an outer layer.",
    since="v0.4.0",
    context="examples.onnx_functions",
    children=["MLPBlock"],
    testcases=[
        {
            "testcase": "000_one_function_on_outer_layer",
            "callable": SuperBlock(),
            "input_shapes": [("B", 10, 3)],
            "expected_number_of_function_instances": 1,
            "run_only_f32_variant": True,
        },
    ],
)
