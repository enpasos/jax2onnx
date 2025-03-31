# file: jax2onnx/plugins/examples/onnx_functions/onnx_functions_005.py
# Note: Original header comment mentioned 004, corrected to 005

from flax import nnx
import jax.numpy as jnp

from jax2onnx.plugin_system import onnx_function, register_example


# === Renamed ===
@onnx_function
class NestedBlock(nnx.Module):  # Renamed from NestedBlock005

    def __init__(self, num_hiddens, mlp_dim, dropout_rate=0.1, *, rngs: nnx.Rngs):
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
class SuperBlock(nnx.Module):  # Renamed from SuperBlock005
    def __init__(self):
        rngs = nnx.Rngs(0)
        num_hiddens = 256
        self.layer_norm2 = nnx.LayerNorm(num_hiddens, rngs=rngs)
        # === Updated internal reference ===
        self.mlp = NestedBlock(num_hiddens, mlp_dim=512, rngs=rngs)  # Use renamed class

    def __call__(self, x):
        return self.mlp(self.layer_norm2(x))


register_example(
    component="onnx_functions_005",  # Keep component name matching file
    description="nested function plus more components",
    since="v0.4.0",
    context="examples.onnx_functions",
    # === Updated children name ===
    children=["NestedBlock"],
    testcases=[
        {
            "testcase": "005_nested_function_plus_component",
            # === Updated callable name ===
            "callable": SuperBlock(),
            "input_shapes": [("B", 10, 256)],
            "expected_number_of_function_instances": 2,
        },
    ],
)
