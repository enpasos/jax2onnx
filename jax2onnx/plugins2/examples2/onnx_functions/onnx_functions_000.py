# jax2onnx/plugins2/examples2/onnx_functions/onnx_functions_000.py

from __future__ import annotations
import jax.numpy as jnp
from flax import nnx

from jax2onnx.plugins2.plugin_system import onnx_function, register_example


class MLPBlock(nnx.Module):
    """Tiny MLP block used by SuperBlock."""

    def __init__(self, num_hiddens: int, mlp_dim: int, rngs: nnx.Rngs):
        self.layers = [
            nnx.Linear(num_hiddens, mlp_dim, rngs=rngs),
            lambda x: nnx.gelu(x, approximate=False),
            nnx.Dropout(rate=0.1, rngs=rngs),
            nnx.Linear(mlp_dim, num_hiddens, rngs=rngs),
            nnx.Dropout(rate=0.1, rngs=rngs),
        ]

    def __call__(self, x: jnp.ndarray, *, deterministic: bool = False) -> jnp.ndarray:
        y = x
        for layer in self.layers:
            if isinstance(layer, nnx.Dropout):
                y = layer(y, deterministic=deterministic)
            else:
                y = layer(y)
        return y


@onnx_function
class SuperBlock(nnx.Module):
    def __init__(self):
        rngs = nnx.Rngs(0)
        self.layer_norm2 = nnx.LayerNorm(3, rngs=rngs)
        self.mlp = MLPBlock(num_hiddens=3, mlp_dim=6, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # explicit deterministic flag for reproducible export
        x_norm = self.layer_norm2(x)
        return self.mlp(x_norm, deterministic=True)


register_example(
    component="onnx_functions_000",
    description="One function boundary on an outer NNX module (new-world).",
    since="v0.4.0",
    context="examples2.onnx_functions",
    children=["MLPBlock"],
    testcases=[
        {
            "testcase": "000_one_function_on_outer_layer",
            "callable": SuperBlock(),
            "input_shapes": [("B", 10, 3)],
            "expected_number_of_function_instances": 1,
            "run_only_f32_variant": True,
            "rtol": 3e-5,
            "atol": 2e-5,
            "use_onnx_ir": True,
        }
    ],
)
