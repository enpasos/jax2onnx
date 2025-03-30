# file: jax2onnx/plugins/examples/onnx_functions/onnx_functions_002.py


from flax import nnx
import jax.numpy as jnp

from jax2onnx.plugin_system import onnx_function, register_example


# === Renamed ===
@onnx_function
class MLPBlock(nnx.Module):  # Renamed from MLPBlock002
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


# === Renamed ===
@onnx_function
class SuperBlock(nnx.Module):  # Renamed from SuperBlock002
    def __init__(self):
        rngs = nnx.Rngs(0)
        # === Updated internal reference ===
        self.mlp = MLPBlock(
            num_hiddens=256, mlp_dim=512, rngs=rngs
        )  # Use renamed class

    def __call__(self, x):
        return self.mlp(x)


register_example(
    component="onnx_functions_002",  # Keep component name matching file
    description="two nested functions.",
    since="v0.4.0",
    context="examples.onnx_functions",
    # === Updated children name ===
    children=["MLPBlock"],
    testcases=[
        {
            "testcase": "002_two_nested_functions",
            # === Updated callable name ===
            "callable": SuperBlock(),
            "input_shapes": [("B", 10, 256)],
        },
    ],
)
