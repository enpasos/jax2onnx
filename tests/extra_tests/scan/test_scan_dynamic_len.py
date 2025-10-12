# tests/extra_tests/scan/test_scan_dynamic_len.py

from __future__ import annotations

import pytest

from jax2onnx.plugins.jax.lax.scan import _two_scans_diff_len_f32
from jax2onnx.user_interface import to_onnx


def test_scan_with_two_trip_counts_loads_in_ort():
    ort = pytest.importorskip("onnxruntime")

    model = to_onnx(
        _two_scans_diff_len_f32,
        [],
        model_name="two_scans_diff",
        opset=21,
    )
    ort.InferenceSession(model.SerializeToString())
