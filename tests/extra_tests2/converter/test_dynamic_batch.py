from __future__ import annotations

import numpy as np
import pytest

import jax

from jax2onnx.user_interface import to_onnx


@pytest.mark.parametrize("batch_sizes", [(3, 17)])
def test_dynamic_batch_ir(batch_sizes):
    batch_axes = ("B", 50, 256)

    @jax.jit
    def get_token(x):
        return x[:, 0, :]

    model = to_onnx(
        get_token,
        [batch_axes],
        model_name="dynamic_batch_ir",
        use_onnx_ir=True,
    )

    try:
        import onnxruntime as ort
    except ImportError:  # pragma: no cover
        pytest.skip("onnxruntime not available")

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    for b in batch_sizes:
        x = np.random.randn(b, 50, 256).astype(np.float32)
        (out,) = sess.run(None, {input_name: x})
        assert out.shape == (b, 256)
