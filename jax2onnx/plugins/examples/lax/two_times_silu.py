# jax2onnx/plugins/examples/lax/two_times_silu.py

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from jax2onnx.plugins.plugin_system import register_example


def two_times_silu(x: jax.Array) -> jax.Array:
    """Apply SiLU twice (issue #139 reproduction)."""
    first = jax.nn.silu(x)
    return jax.nn.silu(first)


register_example(
    component="two_times_silu",
    description="Regression for calling jax.nn.silu twice (issue #139).",
    since="v0.10.2",
    context="examples.lax",
    children=["jax.nn.silu"],
    testcases=[
        {
            "testcase": "two_times_silu_scalar",
            "callable": two_times_silu,
            "input_values": [np.array([0.75], dtype=np.float32)],
            "expected_output_shapes": [(1,)],
            "expected_output_dtypes": [jnp.float32],
        },
    ],
)
