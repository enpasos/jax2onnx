# file: jax2onnx/plugins/examples/onnx_functions/onnx_functions_001.py
# Note: Original header comment mentioned sandbox, corrected path


import jax.numpy as jnp
from flax import nnx

from jax2onnx.plugin_system import onnx_function, register_example


# === Renamed ===
@onnx_function
class MLPBlock(nnx.Module):  # Renamed from MLPBlock001
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


class SuperBlock(nnx.Module):  # Keep outer block name as it's not decorated
    def __init__(self):
        rngs = nnx.Rngs(0)
        self.layer_norm2 = nnx.LayerNorm(256, rngs=rngs)
        # === Updated internal reference ===
        self.mlp = MLPBlock(
            num_hiddens=256, mlp_dim=512, rngs=rngs
        )  # Use renamed class

    def __call__(self, x):
        # Apply LayerNorm (not part of ONNX function), then call the @onnx_function MLPBlock
        # Note: LayerNorm would be part of the main graph, MLP call becomes a function call node.
        return self.mlp(self.layer_norm2(x))


register_example(
    component="onnx_functions_001",  # Keep component name matching file
    description="one function on an inner layer.",
    since="v0.4.0",
    context="examples.onnx_functions",
    # === Updated children name ===
    children=["MLPBlock"],
    testcases=[
        {
            "testcase": "001_one_function_inner",
            "callable": SuperBlock(),  # Callable is the outer block
            "input_shapes": [("B", 10, 256)],
            "expected_number_of_function_instances": 1,
            "run_only_f32_variant": True,
        },
    ],
)
