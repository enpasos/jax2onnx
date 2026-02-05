# jax2onnx/plugins/jax/numpy/__init__.py

# Ensure FFT plugin metadata registers when this package is imported.
from . import fft  # noqa: F401
from . import mean  # noqa: F401
