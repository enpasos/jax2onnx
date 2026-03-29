# tests/conftest.py

import os
import warnings

import pytest

# Force deterministic CPU execution so numerical parity tests are stable
# regardless of whether CUDA is installed locally.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

# JAX/XLA can emit an unraisable KeyboardInterrupt from its GC callback during
# interpreter shutdown even after a fully green test run. Keep the suite output
# clean by filtering just that teardown artifact.
warnings.filterwarnings(
    "ignore",
    message=r"Exception ignored in: <function _xla_gc_callback",
    category=pytest.PytestUnraisableExceptionWarning,
)
