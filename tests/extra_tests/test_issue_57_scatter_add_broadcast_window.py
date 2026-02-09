# tests/extra_tests/test_issue_57_scatter_add_broadcast_window.py

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax2onnx.user_interface import to_onnx


def _issue57_scatter_add_broadcast_window():
    x = jnp.ones((10, 10), dtype=jnp.float32)
    indices = jnp.array([1, 2, 3, 4], dtype=jnp.int32)[:, None]
    updates = jnp.array([10, 20, 30, 40], dtype=jnp.float32)[:, None]
    dnums = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    return jax.lax.scatter_add(x, indices, updates, dnums)


def test_issue57_scatter_add_broadcast_window_roundtrip():
    ort = pytest.importorskip("onnxruntime")

    expected = np.asarray(_issue57_scatter_add_broadcast_window())
    model = to_onnx(
        _issue57_scatter_add_broadcast_window,
        inputs=[],
        model_name="issue57_scatter_add_broadcast_window",
        opset=21,
    )

    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    observed = session.run(None, {})[0]

    np.testing.assert_allclose(observed, expected, rtol=1e-5, atol=1e-6)
