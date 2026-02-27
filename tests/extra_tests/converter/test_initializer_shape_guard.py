# tests/extra_tests/converter/test_initializer_shape_guard.py

"""Exercise the initializer guard across all metadata-driven conversions."""

from __future__ import annotations

from tests._initializer_guard import run_metadata_sweep


def test_initializers_do_not_flow_through_shape_only_ops():
    run_metadata_sweep()
