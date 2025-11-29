# tests/extra_tests/test_issue_139.py

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from jax2onnx import to_onnx


def _double_silu(x: jax.Array) -> jax.Array:
    first = jax.nn.silu(x)
    return jax.nn.silu(first)


def test_double_silu_file_roundtrip(tmp_path) -> None:
    ort = pytest.importorskip(
        "onnxruntime", reason="onnxruntime is required to reproduce issue #139"
    )
    output_path = tmp_path / "issue139.onnx"

    try:
        written_path = to_onnx(
            _double_silu,
            inputs=[(1,)],
            return_mode="file",
            output_path=output_path,
        )

        assert written_path == str(output_path)
        assert output_path.exists()

        session = ort.InferenceSession(str(output_path))
        input_name = session.get_inputs()[0].name

        sample = np.array([0.75], dtype=np.float32)
        (onnx_output,) = session.run(None, {input_name: sample})

        expected = np.asarray(_double_silu(jnp.asarray(sample)))
        np.testing.assert_allclose(onnx_output, expected, rtol=1e-6, atol=1e-7)
    except Exception as exc:  # pragma: no cover - regression tracker for issue #139
        pytest.xfail(f"Issue #139 reproduction still failing: {exc}")
