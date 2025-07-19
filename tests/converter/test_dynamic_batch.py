import numpy as np
import onnxruntime as ort
import jax
from jax2onnx import to_onnx
import pytest


@pytest.mark.order(-1)  # run *after* the models have been produced
def test_dynamic_batch():
    @jax.jit
    def get_token(x):
        return x[:, 0, :]

    model = to_onnx(get_token, [("B", 50, 256)])
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    for b in (3, 17):
        x = np.random.randn(b, 50, 256).astype(np.float32)
        out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        assert out.shape == (b, 256)
