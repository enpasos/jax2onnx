# file: jax2onnx/plugins/examples/jnp/select.py

import jax.numpy as jnp
from jax2onnx.plugin_system import register_example


def select_test(
    x, k, phi_0, select_conditions
):  # Renamed 'select' to 'select_conditions' to avoid conflict with jnp.select
    base = jnp.sin(x * k + phi_0)
    return jnp.select(
        [
            select_conditions == i for i in range(3)
        ],  # select_conditions is expected to be an integer array
        [base ** (i + 1) for i in range(3)],
        jnp.zeros_like(x),  # Default value
    )


register_example(
    component="select_test",
    description="select_test: Demonstrates jnp.select with a dynamic condition based on an input array.",
    since="v0.6.1",
    context="examples.jnp",
    children=[],
    testcases=[
        {
            "testcase": "select_test_all_options",
            "callable": lambda x_input: select_test(
                x_input,
                jnp.array(2.0, dtype=jnp.float32),
                jnp.array(0.5, dtype=jnp.float32),
                jnp.array(
                    [0, 1, 2], dtype=jnp.int32
                ),  # select_conditions matches shape of x_input if x_input.shape == (3,)
            ),
            "input_shapes": [(3,)],  # Shape for 'x_input'
            "input_dtypes": [jnp.float32],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "select_test_scalar_select_option_0",
            "callable": lambda x_input: select_test(
                x_input,
                jnp.array(1.5, dtype=jnp.float32),
                jnp.array(0.3, dtype=jnp.float32),
                jnp.array(0, dtype=jnp.int32),  # scalar select_conditions, broadcasts
            ),
            "input_shapes": [(4,)],  # Shape for 'x_input'
            "input_dtypes": [jnp.float32],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "select_test_scalar_select_option_1",
            "callable": lambda x_input: select_test(
                x_input,
                jnp.array(2.5, dtype=jnp.float32),
                jnp.array(0.8, dtype=jnp.float32),
                jnp.array(1, dtype=jnp.int32),  # scalar select_conditions
            ),
            "input_shapes": [(2,)],
            "input_dtypes": [jnp.float32],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "select_test_scalar_select_option_2",
            "callable": lambda x_input: select_test(
                x_input,
                jnp.array(0.7, dtype=jnp.float32),
                jnp.array(1.2, dtype=jnp.float32),
                jnp.array(2, dtype=jnp.int32),  # scalar select_conditions
            ),
            "input_shapes": [(5,)],
            "input_dtypes": [jnp.float32],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "select_test_default_case",
            "callable": lambda x_input: select_test(
                x_input,
                jnp.array(1.0, dtype=jnp.float32),
                jnp.array(1.0, dtype=jnp.float32),
                jnp.array(
                    3, dtype=jnp.int32
                ),  # scalar select_conditions, leads to default
            ),
            "input_shapes": [(3,)],
            "input_dtypes": [jnp.float32],
            "run_only_f32_variant": True,
        },
    ],
)
