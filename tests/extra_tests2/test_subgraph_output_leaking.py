from __future__ import annotations

import numpy as np
import pytest

import jax

from jax2onnx.user_interface import to_onnx


@jax.jit
def _leaky_sub_function(x, y):
    return x * y, x + y


def _leaky_main_function(input_a, input_b, passthrough_c):
    sub_res1, _ = _leaky_sub_function(input_a, input_b)
    return sub_res1, passthrough_c


class TestSubgraphOutputLeaking:
    def test_pjit_output_leaking_ir(self):
        input_a = np.array([1.0, 2.0], dtype=np.float32)
        input_b = np.array([3.0, 4.0], dtype=np.float32)
        input_c = np.array([100.0], dtype=np.float32)

        specs = [
            jax.ShapeDtypeStruct(input_a.shape, input_a.dtype),
            jax.ShapeDtypeStruct(input_b.shape, input_b.dtype),
            jax.ShapeDtypeStruct(input_c.shape, input_c.dtype),
        ]

        jax_outputs = jax.tree_util.tree_leaves(
            _leaky_main_function(input_a, input_b, input_c)
        )
        expected_num_outputs = len(jax_outputs)
        assert expected_num_outputs == 2

        onnx_model = to_onnx(
            _leaky_main_function,
            specs,
            model_name="pjit_leak_test_ir",
        )

        assert len(onnx_model.graph.output) == expected_num_outputs

        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(onnx_model.SerializeToString())
            feed = {
                sess.get_inputs()[0].name: input_a,
                sess.get_inputs()[1].name: input_b,
                sess.get_inputs()[2].name: input_c,
            }
            onnx_results = sess.run(None, feed)
            assert len(onnx_results) == expected_num_outputs
            np.testing.assert_allclose(jax_outputs[0], onnx_results[0], rtol=1e-6)
            np.testing.assert_allclose(jax_outputs[1], onnx_results[1], rtol=1e-6)
        except ImportError:
            pytest.skip("onnxruntime not available")
