# file: jax2onnx/plugins/examples/onnx_functions/onnx_functions_003.py


from flax import nnx
import jax.numpy as jnp

from jax2onnx.plugin_system import onnx_function, register_example


@onnx_function
class NestedBlock003(nnx.Module):

    def __init__(self, num_hiddens, mlp_dim, rngs: nnx.Rngs):
        self.linear = nnx.Linear(num_hiddens, mlp_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        return self.linear(x)


@onnx_function
class SuperBlock003(nnx.Module):
    def __init__(self):
        rngs = nnx.Rngs(0)
        self.mlp = NestedBlock003(num_hiddens=256, mlp_dim=512, rngs=rngs)

    def __call__(self, x):
        return self.mlp(x)


register_example(
    component="onnx_functions_003",
    description="two nested functions.",
    # source="https:/",
    since="v0.4.0",
    context="examples.onnx_functions",
    children=["NestedBlock003"],
    testcases=[
        {
            "testcase": "003_two_simple_nested_functions",
            "callable": SuperBlock003(),
            "input_shapes": [("B", 10, 256)],
        },
    ],
)
