import jax
import jax.numpy as jnp
from jax2onnx import to_onnx
from jax2onnx.plugin_system import register_example


def model_fn(x):
    steps = 5

    def body_func(index, args):
        x, counter = args
        x += 0.1 * x**2
        counter += 1
        return (x, counter)

    args = (x, 0)
    args = jax.lax.fori_loop(0, steps, body_func, args)

    return args


register_example(
    component="ForiLoop",
    description="fori_loop example",
    since="v0.5.1",
    context="examples.nnx",
    children=[],
    testcases=[
        {
            "testcase": "fori_loop_counter",
            "callable": lambda x: model_fn(x),
            "input_shapes": [(2,)],
        },
    ],
)
