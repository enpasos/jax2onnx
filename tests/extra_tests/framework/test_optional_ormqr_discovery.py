# tests/extra_tests/framework/test_optional_ormqr_discovery.py

from __future__ import annotations

import subprocess
import sys


def test_plugin_discovery_tolerates_missing_ormqr_primitive() -> None:
    code = """
import jax

if hasattr(jax.lax.linalg, "ormqr_p"):
    delattr(jax.lax.linalg, "ormqr_p")
from jax2onnx.plugins.plugin_system import import_all_plugins
import_all_plugins()
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ormqr" not in completed.stderr.lower()
