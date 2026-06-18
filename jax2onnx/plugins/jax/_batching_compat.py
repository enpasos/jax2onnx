# jax2onnx/plugins/jax/_batching_compat.py

"""Backward-compatible imports for JAX batching internals."""

from __future__ import annotations

from jax2onnx._compat.jax import (
    NOT_MAPPED,
    batching,
    ensure_batching_not_mapped_attr,
)

__all__ = ["NOT_MAPPED", "batching", "ensure_batching_not_mapped_attr"]
