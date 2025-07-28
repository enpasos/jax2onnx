"""
ONNX‑friendly “window scatter” using dynamic_update_slice + vmap
================================================================

We update the same 256×256 patch in every image of a batch.
Only one patch coordinate is provided (N = 1).  We therefore vmap over
the batch dimension only and treat indices / updates as constants.
"""

import jax
import jax.numpy as jnp
from jax2onnx.plugin_system import register_example


def scatter_window_function_onnx_friendly(operand, indices, updates):
    """
    operand : (B, H, W, C)   – e.g. (5, 266, 266, 1)
    indices : (1, 2)         – (y, x) upper‑left corner of the patch
    updates : (1, h, w, C)   – (1, 256, 256, 1) patch to write
    """
    # unwrap the single patch / index
    (y, x) = indices[0]
    patch = updates[0]  # (256, 256, 1)

    def insert_one(img):
        # all three index scalars are int32
        return jax.lax.dynamic_update_slice(img, patch, (y, x, jnp.int32(0)))

    # Map over the batch axis only.
    return jax.vmap(insert_one, in_axes=0)(operand)


# ---------------------------------------------------------------------
# Register as an example (for docs & automatic tests)
# ---------------------------------------------------------------------
register_example(
    component="scatter_window",
    context="examples.lax",
    description="Windowed update via `dynamic_update_slice` (ONNX‑friendly).",
    since="v0.7.2",
    children=["jax.lax.dynamic_update_slice", "jax.vmap"],
    testcases=[
        # TODO: enable testcases
        # {
        #     "testcase": "scatter_window_update_f64",
        #     "callable": scatter_window_function_onnx_friendly,
        #     "input_values": [
        #         np.zeros((5, 266, 266, 1), dtype=np.float64),      # operand  (B,H,W,C)
        #         np.array([[10, 10]], dtype=np.int32),              # indices  (1,2)
        #         np.ones((1, 256, 256, 1), dtype=np.float64),       # updates  (1,h,w,C)
        #     ],
        #     "run_only_f64_variant": True,
        # }
    ],
)
