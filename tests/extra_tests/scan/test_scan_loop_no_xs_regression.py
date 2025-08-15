# tests/permanent_examples/test_scan_loop_no_xs_regression.py
from __future__ import annotations

import pathlib
import numpy as np
import jax
import jax.numpy as jnp
import pytest
import onnx
import onnxruntime as ort

from jax import lax
from jax2onnx import to_onnx


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.timeout(60)
@pytest.mark.skipif(ort.get_device() is None, reason="onnxruntime not available")
def test_scan_loop_no_xs_subset_outputs_loads_and_matches(tmp_path: pathlib.Path):
    """
    Regression for: Loop shape inference must not fail when converting scan(xs=None)
    and only a subset of scan outputs are used by the outer function.

    The scan body returns:
        carry' (scalar) and 6 per-iter outputs (all scalar-derived),
    but the outer function returns only (y2, y5).
    """

    def ff_subset():
        def body(c, _):
            # produce a bunch of per-iter outputs so the Loop has multiple scan outs
            y1 = c
            y2 = c + 1.0
            y3 = c * 2.0
            y4 = c - 3.0
            y5 = c / 4.0
            y6 = c**2
            return c + 1.0, (y1, y2, y3, y4, y5, y6)

        c0 = jnp.array(0.0, dtype=jnp.float64)
        # no xs → hits the Loop special-case in the Scan plugin
        _, ys = lax.scan(
            body, c0, xs=None, length=4
        )  # ys is a tuple of 6 arrays, each (4,)
        return ys[1], ys[4]  # return only a subset (y2, y5)

    # Reference results from JAX
    y2_ref, y5_ref = ff_subset()
    y2_ref = np.asarray(jax.device_get(y2_ref))
    y5_ref = np.asarray(jax.device_get(y5_ref))

    # Convert to ONNX (runs shape inference internally)
    model = to_onnx(
        ff_subset,
        inputs=[],  # no runtime inputs
        opset=21,
        enable_double_precision=True,  # match feed-forward use (f64)
        model_name="scan_loop_no_xs_subset",
    )
    onnx_path = tmp_path / "scan_loop_no_xs_subset.onnx"
    onnx.save_model(model, onnx_path)

    # Ensure ORT can load it (this is where the old bug would explode)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # Run ONNX and compare numerically
    y2_onnx, y5_onnx = sess.run(None, {})
    y2_onnx = np.asarray(y2_onnx)
    y5_onnx = np.asarray(y5_onnx)

    assert np.allclose(y2_onnx, y2_ref, rtol=1e-7, atol=1e-12)
    assert np.allclose(y5_onnx, y5_ref, rtol=1e-7, atol=1e-12)
