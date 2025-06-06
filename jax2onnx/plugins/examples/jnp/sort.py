# file: jax2onnx/plugins/examples/jnp/sort.py

import jax.numpy as jnp
from jax2onnx.plugin_system import register_example


def sort_test(x):
    """
    Sorts the first and second halves of an array independently and returns
    the element-wise product of the sorted halves.
    """
    y1 = jnp.sort(x[5:])
    y2 = jnp.sort(x[:5])
    return y1 * y2


register_example(
    component="sort_test",
    description="sort_test: Demonstrates jnp.sort on slices of an input array.",
    since="v0.6.1",
    context="examples.jnp",
    children=[],
    testcases=[
        {
            "testcase": "sort_test_basic",
            "callable": lambda x_input: sort_test(x_input),
            "input_shapes": [(10,)],  # Shape for 'x_input'
            "input_dtypes": [jnp.float32],
            "run_only_f32_variant": True,
        },
    ],
)
