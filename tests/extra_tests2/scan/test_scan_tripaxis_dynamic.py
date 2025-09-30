from __future__ import annotations

import jax.numpy as jnp
from jax import lax
import onnx
import pytest

from jax2onnx.plugins2.jax.lax.scan import _two_scans_len_mismatch_broadcast_f32
from jax2onnx.user_interface import to_onnx


def test_tripaxis_dynamic_loads():
    ort = pytest.importorskip("onnxruntime")
    model = to_onnx(
        _two_scans_len_mismatch_broadcast_f32,
        [],
        model_name="two_scans_broadcast",
        use_onnx_ir=True,
    )
    ort.InferenceSession(model.SerializeToString())


def _two_scans_len_mismatch():
    xs_small = jnp.arange(5, dtype=jnp.float32)
    xs_big = jnp.arange(100, dtype=jnp.float32)

    def body(c, x):
        return (c + x, c)

    carry, _ = lax.scan(body, 0.0, xs_small)
    carry, _ = lax.scan(body, carry, xs_big)
    return carry


@pytest.mark.order(-1)
@pytest.mark.parametrize("enable_double_precision", [False])
def test_tripaxis_is_dynamic(tmp_path, enable_double_precision):
    ort = pytest.importorskip("onnxruntime")

    path = tmp_path / "two_scans_dynamic.onnx"
    model = to_onnx(
        _two_scans_len_mismatch,
        [],
        model_name="two_scans_dynamic",
        enable_double_precision=enable_double_precision,
        use_onnx_ir=True,
    )
    onnx.save_model(model, path)

    onnx.checker.check_model(model)
    ort.InferenceSession(str(path))

    for vi in model.graph.value_info:
        if vi.name.startswith("scan_unused_output"):
            first_dim = vi.type.tensor_type.shape.dim[0]
            assert not first_dim.HasField(
                "dim_value"
            ), f"{vi.name} trip-axis is unexpectedly static"
