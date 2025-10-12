# jax2onnx/plugins/examples/jnp/select.py

from __future__ import annotations

import jax.numpy as jnp

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import register_example


def _select(
    x: jnp.ndarray,
    k: jnp.ndarray,
    phi_0: jnp.ndarray,
    select_conditions: jnp.ndarray,
) -> jnp.ndarray:
    base = jnp.sin(x * k + phi_0)
    branches = [base ** (i + 1) for i in range(3)]
    masks = [select_conditions == i for i in range(3)]
    return jnp.select(masks, branches, default=jnp.zeros_like(base))


register_example(
    component="select_test",
    description="Demonstrates jnp.select with scalar and tensor predicates.",
    since="v0.9.0",
    context="examples.jnp",
    children=[],
    testcases=[
        {
            "testcase": "select_test_all_options",
            "callable": lambda x_input: _select(
                x_input,
                jnp.array(2.0, dtype=jnp.float32),
                jnp.array(0.5, dtype=jnp.float32),
                jnp.array([0, 1, 2], dtype=jnp.int32),
            ),
            "input_shapes": [(3,)],
            "input_dtypes": [jnp.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Equal:3 -> Where:3 -> Identity:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "select_test_scalar_select_option_0",
            "callable": lambda x_input: _select(
                x_input,
                jnp.array(1.5, dtype=jnp.float32),
                jnp.array(0.3, dtype=jnp.float32),
                jnp.array(0, dtype=jnp.int32),
            ),
            "input_shapes": [(4,)],
            "input_dtypes": [jnp.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Equal -> Where:4 -> Identity:4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "select_test_scalar_select_option_1",
            "callable": lambda x_input: _select(
                x_input,
                jnp.array(2.5, dtype=jnp.float32),
                jnp.array(0.8, dtype=jnp.float32),
                jnp.array(1, dtype=jnp.int32),
            ),
            "input_shapes": [(2,)],
            "input_dtypes": [jnp.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Equal -> Where:2 -> Identity:2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "select_test_scalar_select_option_2",
            "callable": lambda x_input: _select(
                x_input,
                jnp.array(0.7, dtype=jnp.float32),
                jnp.array(1.2, dtype=jnp.float32),
                jnp.array(2, dtype=jnp.int32),
            ),
            "input_shapes": [(5,)],
            "input_dtypes": [jnp.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Equal -> Where:5 -> Identity:5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "select_test_default_case",
            "callable": lambda x_input: _select(
                x_input,
                jnp.array(1.0, dtype=jnp.float32),
                jnp.array(1.0, dtype=jnp.float32),
                jnp.array(3, dtype=jnp.int32),
            ),
            "input_shapes": [(3,)],
            "input_dtypes": [jnp.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Equal -> Where:3 -> Identity:3"],
                no_unused_inputs=True,
            ),
        },
    ],
)
