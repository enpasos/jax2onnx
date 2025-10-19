# tests/extra_tests/test_gather_modes.py

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax import lax

from jax2onnx.user_interface import to_onnx


def _gather_one_hot(x: jnp.ndarray) -> jnp.ndarray:
    indices = jnp.array([[0]], dtype=jnp.int32)
    return lax.gather(
        x,
        indices,
        dimension_numbers=lax.GatherDimensionNumbers(
            offset_dims=(1, 2),
            collapsed_slice_dims=(0,),
            start_index_map=(0,),
        ),
        slice_sizes=(1, x.shape[1], x.shape[2]),
        mode=lax.GatherScatterMode.ONE_HOT,
    )[0]


def test_gather_clip_mode_raises_not_implemented():
    x = np.ones((4, 5, 6), dtype=np.float32)
    with pytest.raises(NotImplementedError):
        to_onnx(
            _gather_one_hot,
            [jax.ShapeDtypeStruct(x.shape, x.dtype)],
            model_name="gather_clip",
            enable_double_precision=False,
        )


def _take_window(x: jnp.ndarray, start: jnp.ndarray) -> jnp.ndarray:
    idx = jnp.arange(start, start + 3, dtype=jnp.int32)
    return jnp.take(x, idx, axis=1)


def test_gather_dynamic_window_matches_onnx_ir():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(2, 8)).astype(np.float32)
    start = np.array(2, dtype=np.int32)

    expected = np.asarray(_take_window(data, start))

    onnx_model = to_onnx(
        _take_window,
        [
            jax.ShapeDtypeStruct(data.shape, jnp.float32),
            jax.ShapeDtypeStruct((), jnp.int32),
        ],
        model_name="gather_dynamic_window",
        enable_double_precision=False,
    )

    ort = pytest.importorskip("onnxruntime")
    session = ort.InferenceSession(onnx_model.SerializeToString())
    feeds = {
        session.get_inputs()[0].name: data,
        session.get_inputs()[1].name: np.asarray(start, dtype=np.int32),
    }
    (got,) = session.run(None, feeds)

    np.testing.assert_allclose(expected, got, rtol=1e-6, atol=1e-6)
