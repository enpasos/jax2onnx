import jax
import jax.numpy as jnp

from jax2onnx.plugin_system import register_example

# This test isolates the behavior of `jax.checkpoint`, also known as `jax.remat`.
# The `@jax.checkpoint` decorator wraps the function's computation in a `remat2`
# primitive. This test ensures that the converter can handle this primitive
# by correctly inlining its content, as the concept of re-materialization
# is not relevant for ONNX inference graphs.


@jax.checkpoint
def checkpoint_scalar_f32(x):
    """
    A simple function decorated with `@jax.checkpoint`. The converter should
    be able to "see through" the checkpoint and convert the inner operations.
    """
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
            "callable": lambda x: checkpoint_scalar_f32(x),
            "input_shapes": [
                (),  # Input is a scalar
            ],
            "input_dtypes": [jnp.float32],
            "expected_output_shapes": [
                (),  # Output is a scalar
            ],
            "expected_output_dtypes": [jnp.float32],
        },
    ],
)
