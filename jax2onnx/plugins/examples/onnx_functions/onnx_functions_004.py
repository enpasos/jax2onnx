# file: jax2onnx/plugins/examples/onnx_functions/onnx_functions_004.py


import jax.numpy as jnp
from flax import nnx

from jax2onnx.plugin_system import onnx_function, register_example


@onnx_function
class NestedBlock(nnx.Module):

    def __init__(self, num_hiddens, mlp_dim, rngs: nnx.Rngs):
        self.linear = nnx.Linear(num_hiddens, mlp_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        return self.linear(x)


@onnx_function
class SuperBlock(nnx.Module):
    def __init__(self):
        rngs = nnx.Rngs(0)
        num_hiddens = 256
        self.layer_norm2 = nnx.LayerNorm(num_hiddens, rngs=rngs)
        self.mlp = NestedBlock(num_hiddens, mlp_dim=512, rngs=rngs)

    def __call__(self, x):
        return self.mlp(self.layer_norm2(x))


register_example(
    component="onnx_functions_004",
    description="nested function plus component",
    since="v0.4.0",
    context="examples.onnx_functions",
    children=["NestedBlock"],
    testcases=[
        {
            "testcase": "004_nested_function_plus_component",
            "callable": SuperBlock(),
            "input_shapes": [("B", 10, 256)],
            "expected_number_of_function_instances": 2,
            "run_only_f32_variant": True,
        },
    ],
)
