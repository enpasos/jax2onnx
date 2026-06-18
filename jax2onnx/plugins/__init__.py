# jax2onnx/plugins/__init__.py

from jax2onnx.plugins.jax._batching_compat import ensure_batching_not_mapped_attr

ensure_batching_not_mapped_attr()
