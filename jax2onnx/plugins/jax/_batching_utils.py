# jax2onnx/plugins/jax/_batching_utils.py

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from jax import lax

from jax2onnx._compat.jax import (
    NOT_MAPPED,
    batching,
    definitely_equal_shape,
    ensure_batching_not_mapped_attr,
)

ensure_batching_not_mapped_attr()


def _handle_scalar_broadcasting(ndim: int, x: Any, dim: Any) -> Any:
    if dim is NOT_MAPPED or ndim == np.ndim(x):
        return x
    return lax.expand_dims(x, tuple(range(np.ndim(x), ndim)))


def broadcast_batcher_compat(
    prim: Any, args: Sequence[Any], dims: Sequence[Any], **params: Any
) -> Any:
    """Broadcasting batch rule that avoids relying on JAX's internal helper."""
    if len(args) <= 1:
        raise ValueError("broadcast_batcher_compat requires at least two arguments")

    shape, dim = next((x.shape, d) for x, d in zip(args, dims) if d is not NOT_MAPPED)
    if all(
        definitely_equal_shape(shape, x.shape) and d == dim
        for x, d in zip(args, dims)
        if np.ndim(x)
    ):
        out = prim.bind(*args, **params)
        return (out, (dim,) * len(out)) if prim.multiple_results else (out, dim)

    args = [
        batching.bdim_at_front(x, d, 1) if np.ndim(x) else x for x, d in zip(args, dims)
    ]
    ndim = max(np.ndim(x) for x in args)
    args = [_handle_scalar_broadcasting(ndim, x, d) for x, d in zip(args, dims)]
    out = prim.bind(*args, **params)
    return (out, (0,) * len(out)) if prim.multiple_results else (out, 0)
