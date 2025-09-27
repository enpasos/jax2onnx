import onnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax2onnx.user_interface import to_onnx

HAS_ORT = True
try:
    import onnxruntime as ort  # noqa: F401
except Exception:
    HAS_ORT = False


def pad_in_loop_fn(x):
    # pad last dim by (1,1) with constant 0 and slice it back (Pad→Slice pattern)
    def body(i, carry):
        y = jax.lax.pad(carry, jnp.array(0, dtype=carry.dtype), ((0, 0, 0), (1, 1, 0)))
        return y[:, 1:-1]

    return jax.lax.fori_loop(0, 1, body, x)


@pytest.mark.skipif(not HAS_ORT, reason="onnxruntime not installed")
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pad_inside_loop_builds_and_infers(dtype):
    x = jnp.ones((2, 3), dtype=dtype)
    m = to_onnx(
        pad_in_loop_fn,
        inputs=[x],
        enable_double_precision=(dtype == jnp.float64),
        opset=21,
        model_name=f"pad_in_loop_{np.dtype(dtype).name}",
    )
    onnx.checker.check_model(m)
    # This used to fail with: Graph has 4 inputs but 3 were provided
    onnx.shape_inference.infer_shapes(m, strict_mode=True)
