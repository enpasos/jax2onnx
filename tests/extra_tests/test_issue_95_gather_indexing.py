# tests/extra_tests/test_issue_95_gather_indexing.py

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from jax2onnx.user_interface import to_onnx


def _slice_middle_axis(x: jnp.ndarray) -> jnp.ndarray:
    return x[:, 1]


def _slice_last_axis(x: jnp.ndarray) -> jnp.ndarray:
    return x[:, :, 0]


@pytest.mark.parametrize(
    ("fn", "input_shape"),
    [
        (_slice_middle_axis, (5, 6)),
        (_slice_last_axis, (4, 5, 6)),
    ],
    ids=["x_colon_1", "x_colon_colon_0"],
)
def test_issue_95_gather_slice_indexing_matches_onnx_ir(fn, input_shape) -> None:
    model = to_onnx(
        fn,
        [jax.ShapeDtypeStruct(input_shape, jnp.float32)],
        model_name=f"issue95_{fn.__name__}",
    )

    ort = pytest.importorskip(
        "onnxruntime", reason="onnxruntime is required for issue #95 regression test"
    )
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    x = np.random.default_rng(0).normal(size=input_shape).astype(np.float32)
    expected = np.asarray(fn(x))
    (got,) = session.run(None, {input_name: x})

    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
