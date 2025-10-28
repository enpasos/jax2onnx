# tests/extra_tests/converter/test_initializer_shape_guard.py

"""Exercise the initializer guard across all metadata-driven conversions."""

from __future__ import annotations

import pytest

from tests._initializer_guard import GUARD_ENABLED, run_metadata_sweep


def test_initializers_do_not_flow_through_shape_only_ops():
    if not GUARD_ENABLED:
        pytest.skip("onnx_ir.Value lacks metadata_props support required by guard")
    run_metadata_sweep()
