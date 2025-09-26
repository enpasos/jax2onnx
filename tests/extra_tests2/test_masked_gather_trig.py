from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from jax2onnx.user_interface import to_onnx


def _masked_gather_trig(data: jnp.ndarray, indices: jnp.ndarray) -> jnp.ndarray:
    data64 = jnp.asarray(data, dtype=jnp.float64)
    gathered = data64[indices]
    trig = jnp.sin(gathered * 2.0) + jnp.cos(gathered * 2.0)
    mask = trig > 0.5
    return jnp.where(mask, trig, 0.0)


class TestMaskedGatherTrig:
    def test_masked_gather_trig_f64_pipeline_ir(self) -> None:
        data = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ],
            dtype=np.float64,
        )
        indices = np.array([0, 2], dtype=np.int64)

        original_x64 = jax.config.read("jax_enable_x64")
        jax.config.update("jax_enable_x64", True)
        try:
            jax_result = np.asarray(_masked_gather_trig(data, indices))

            onnx_model = to_onnx(
                _masked_gather_trig,
                [
                    jax.ShapeDtypeStruct(data.shape, data.dtype),
                    jax.ShapeDtypeStruct(indices.shape, indices.dtype),
                ],
                model_name="masked_gather_trig_f64_ir",
                enable_double_precision=True,
                use_onnx_ir=True,
            )

            assert len(onnx_model.graph.output) == 1

            try:
                import onnxruntime as ort

                sess = ort.InferenceSession(onnx_model.SerializeToString())
                feed = {
                    sess.get_inputs()[0].name: data,
                    sess.get_inputs()[1].name: indices,
                }
                (onnx_out,) = sess.run(None, feed)
                np.testing.assert_allclose(
                    jax_result, onnx_out, rtol=5e-8, atol=1e-12
                )
            except ImportError:
                pytest.skip("onnxruntime not available")
        finally:
            jax.config.update("jax_enable_x64", original_x64)
