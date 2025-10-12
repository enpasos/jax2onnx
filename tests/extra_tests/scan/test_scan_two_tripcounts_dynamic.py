# tests/extra_tests/scan/test_scan_two_tripcounts_dynamic.py

from __future__ import annotations

import pytest

from jax2onnx.plugins.jax.lax.scan import _two_scans_diff_len_f32
from jax2onnx.user_interface import to_onnx


def _assert_ort_loads(model_proto):
    ort = pytest.importorskip("onnxruntime")
    ort.InferenceSession(model_proto.SerializeToString())


def test_two_scans_with_different_lengths_can_load_in_ort():
    model = to_onnx(
        _two_scans_diff_len_f32,
        [],
        model_name="two_scans_diff",
    )
    _assert_ort_loads(model)
