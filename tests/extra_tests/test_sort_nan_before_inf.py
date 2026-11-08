# tests/extra_tests/test_sort_nan_before_inf.py

import jax
import jax.numpy as jnp
import numpy as np
import pytest

onnxruntime = pytest.importorskip("onnxruntime")

from jax2onnx import to_onnx  # noqa: E402


@pytest.mark.parametrize(
    "data",
    [
        [np.nan, np.inf],
        [np.inf, np.nan],
        [np.nan, np.inf, 1.0, np.inf, np.nan],
    ],
)
def test_jnp_sort_keeps_nan_after_positive_inf(data):
    x = np.asarray(data, dtype=np.float32)
    expected = np.asarray(jnp.sort(jnp.asarray(x)))

    model = to_onnx(
        lambda v: jnp.sort(v),
        [jax.ShapeDtypeStruct(x.shape, x.dtype)],
        input_names=["x"],
    )
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    session = onnxruntime.InferenceSession(
        model.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    actual = session.run(None, {"x": x})[0]

    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_allclose(actual, expected, equal_nan=True)
