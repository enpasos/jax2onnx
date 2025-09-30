from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from jax2onnx.user_interface import to_onnx


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.timeout(60)
def test_scan_loop_no_xs_subset_outputs_loads_and_matches(tmp_path: pathlib.Path):
    ort = pytest.importorskip("onnxruntime")

    def ff_subset():
        def body(c, _):
            y1 = c
            y2 = c + 1.0
            y3 = c * 2.0
            y4 = c - 3.0
            y5 = c / 4.0
            y6 = c**2
            return c + 1.0, (y1, y2, y3, y4, y5, y6)

        c0 = jnp.array(0.0, dtype=jnp.float64)
        _, ys = lax.scan(body, c0, xs=None, length=4)
        return ys[1], ys[4]

    y2_ref, y5_ref = ff_subset()
    y2_ref = np.asarray(jax.device_get(y2_ref))
    y5_ref = np.asarray(jax.device_get(y5_ref))

    model = to_onnx(
        ff_subset,
        inputs=[],
        opset=21,
        enable_double_precision=True,
        model_name="scan_loop_no_xs_subset",
    )

    path = tmp_path / "scan_loop_no_xs_subset.onnx"
    path.write_bytes(model.SerializeToString())

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    y2_onnx, y5_onnx = sess.run(None, {})

    assert np.allclose(np.asarray(y2_onnx), y2_ref, rtol=1e-7, atol=1e-12)
    assert np.allclose(np.asarray(y5_onnx), y5_ref, rtol=1e-7, atol=1e-12)
