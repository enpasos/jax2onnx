# file: jax2onnx/plugins/examples/jnp/fori_loop.py
import jax
import jax.numpy as jnp
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
    component="fori_loop_test",
    description="fori_loop_test: Demonstrates jax.lax.fori_loop with a simple loop.",
    since="v0.6.3",
    context="examples.jnp",
    children=[],
    testcases=[
        {
            "testcase": "fori_loop_test",
            "callable": lambda x: model_fn(x),
            "input_shapes": [(2,)],
            "input_dtypes": [jnp.float32],
            "expected_output_shapes": [(2,), ()],  # Output shapes for x and counter
            "run_only_f32_variant": True,
        },
        {
            "testcase": "fori_loop_test_f64",
            "callable": lambda x: model_fn(x),
            "input_shapes": [(3,)],
            "input_dtypes": [jnp.float64],
            "expected_output_shapes": [(3,), ()],  # Output shapes for x and counter
            "run_only_f64_variant": True,
        },
    ],
)
