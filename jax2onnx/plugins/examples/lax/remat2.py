# jax2onnx/plugins/examples/lax/remat2.py

from __future__ import annotations

import jax
import jax.numpy as jnp

# Ensure the lowering plugin is imported so converter can handle remat2.
from jax2onnx.plugins.jax.lax import remat2 as _remat2_plugin  # noqa: F401

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import register_example


@jax.checkpoint
def checkpoint_scalar_f32(x: jax.Array) -> jax.Array:
    """Simple checkpointed function exercising lax.remat2."""
    y = jnp.sin(x)
    z = jnp.sin(y)
    return z


register_example(
    component="remat2",
    description="Tests a simple case of `jax.checkpoint` (also known as `jax.remat2`).",
    since="v0.6.5",
    context="examples.lax",
    children=[],
    testcases=[
        {
            "testcase": "checkpoint_scalar_f32",
            "callable": checkpoint_scalar_f32,
            "input_shapes": [()],
            "input_dtypes": [jnp.float32],
            "expected_output_shapes": [()],
            "expected_output_dtypes": [jnp.float32],
            "post_check_onnx_graph": EG(
                ["Sin -> Sin"],
                no_unused_inputs=True,
            ),
        },
    ],
)
