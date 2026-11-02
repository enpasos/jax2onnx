# tests/extra_tests/scan/test_scan_half_precision.py

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from jax2onnx import to_onnx


@pytest.mark.parametrize("enable_x64", [False, True])
def test_scan_preserves_half_precision_sequence_dtype(
    enable_x64: bool,
) -> None:
    with jax.enable_x64(enable_x64):

        def fn(initial: jax.Array, xs: jax.Array) -> tuple[jax.Array, jax.Array]:
            def body(carry: jax.Array, value: jax.Array) -> tuple[jax.Array, jax.Array]:
                result = carry + value
                return result, result

            return jax.lax.scan(body, initial, xs)

        initial = np.array([0.5, -0.5], dtype=np.float16)
        xs = np.arange(6, dtype=np.float16).reshape(3, 2) / 4
        model = to_onnx(fn, [initial, xs])
        onnx.checker.check_model(model, full_check=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        actual = session.run(
            None,
            {
                meta.name: value
                for meta, value in zip(session.get_inputs(), [initial, xs])
            },
        )
        expected = fn(jnp.asarray(initial), jnp.asarray(xs))
        for actual_value, expected_value in zip(actual, expected):
            assert actual_value.dtype == np.float16
            np.testing.assert_array_equal(actual_value, np.asarray(expected_value))
