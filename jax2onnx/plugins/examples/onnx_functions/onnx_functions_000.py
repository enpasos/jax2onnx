# file: jax2onnx/plugins/examples/onnx_functions/onnx_functions_000.py


from flax import nnx
import jax.numpy as jnp

from jax2onnx.plugin_system import onnx_function, register_example


# === Renamed ===
# Note: This block is NOT decorated with @onnx_function in this example
class MLPBlock(nnx.Module):  # Renamed from MLPBlock000
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
@onnx_function  # This outer block IS decorated
class SuperBlock(nnx.Module):  # Renamed from SuperBlock000
    def __init__(self):
        rngs = nnx.Rngs(0)
        self.layer_norm2 = nnx.LayerNorm(256, rngs=rngs)
        # === Updated internal reference ===
        self.mlp = MLPBlock(
            num_hiddens=256, mlp_dim=512, rngs=rngs
        )  # Use renamed class

    def __call__(self, x):
        # MLPBlock is called *within* the decorated SuperBlock
        # So MLPBlock's layers will become nodes inside the FunctionProto for SuperBlock
        return self.mlp(self.layer_norm2(x))


register_example(
    component="onnx_functions_000",  # Keep component name matching file
    description="one function on an outer layer.",
    since="v0.4.0",
    context="examples.onnx_functions",
    # === Updated children name ===
    children=["MLPBlock"],
    testcases=[
        {
            "testcase": "000_one_function_outer",
            # === Updated callable name ===
            "callable": SuperBlock(),
            "input_shapes": [("B", 10, 256)],
        },
    ],
)
