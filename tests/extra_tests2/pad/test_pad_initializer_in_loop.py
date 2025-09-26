from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx.shape_inference as shape_inference
import pytest

from jax2onnx.user_interface import to_onnx


def _pad_in_loop_fn(x: jax.Array) -> jax.Array:
    """Pad last dim by (1, 1) inside a loop and slice it away again."""

    def body(_, carry):
        padded = jax.lax.pad(
            carry,
            jnp.array(0, dtype=carry.dtype),
            ((0, 0, 0), (1, 1, 0)),
        )
        return padded[:, 1:-1]

    return jax.lax.fori_loop(0, 1, body, x)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pad_inside_loop_ir_pipeline(dtype):
    """converter2 should emit a Loop that shape-infers successfully."""

    spec = jnp.ones((2, 3), dtype=dtype)
    model = to_onnx(
        _pad_in_loop_fn,
        inputs=[spec],
        enable_double_precision=(dtype == jnp.float64),
        loosen_internal_shapes=True,
        opset=21,
        model_name=f"pad_in_loop_{np.dtype(dtype).name}",
        use_onnx_ir=True,
    )

    # structural sanity
    onnx.checker.check_model(model)

    # regression: ONNX strict shape inference should also succeed
    inferred = shape_inference.infer_shapes(model, strict_mode=True)
    assert inferred is not None
