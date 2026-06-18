# jax2onnx/plugins/jax/_batching_compat.py

"""Compatibility helpers for JAX batching internals."""

from __future__ import annotations

from typing import Any

from jax.interpreters import batching


def ensure_batching_not_mapped_attr() -> Any:
    """Expose the legacy ``batching.not_mapped`` name when JAX uses ``None``."""

    try:
        return batching.not_mapped
    except AttributeError:
        setattr(batching, "not_mapped", None)
        return None


NOT_MAPPED: Any = ensure_batching_not_mapped_attr()
