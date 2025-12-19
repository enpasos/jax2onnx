# jax2onnx/plugins/jax/_batching_utils.py

from __future__ import annotations

from typing import Any, Sequence, Callable

import numpy as np
from jax import lax
from jax.interpreters import batching


def _resolve_definitely_equal_shape() -> Callable[[Any, Any], bool]:
    try:  # Prefer the internal helper when available (moved in newer JAX versions).
        from jax._src import core as _core_src

        return _core_src.definitely_equal_shape
    except Exception:  # pragma: no cover - fallback for older/older-stub JAX
        try:
            from jax import core as _core_public

            return _core_public.definitely_equal_shape  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - minimal fallback

            def _fallback(s1: Any, s2: Any) -> bool:
                if len(s1) != len(s2):
                    return False
                for d1, d2 in zip(s1, s2):
                    if d1 is d2:
                        continue
                    try:
                        if d1 != d2:
                            return False
                    except Exception:
                        return False
                return True

            return _fallback


_definitely_equal_shape: Callable[[Any, Any], bool] = _resolve_definitely_equal_shape()


def _handle_scalar_broadcasting(ndim: int, x: Any, dim: Any) -> Any:
    if dim is batching.not_mapped or ndim == np.ndim(x):
        return x
    return lax.expand_dims(x, tuple(range(np.ndim(x), ndim)))


def broadcast_batcher_compat(
    prim: Any, args: Sequence[Any], dims: Sequence[Any], **params: Any
):
    """Broadcasting batch rule that avoids relying on JAX's internal helper."""
    if len(args) <= 1:
        raise ValueError("broadcast_batcher_compat requires at least two arguments")

    shape, dim = next(
        (x.shape, d) for x, d in zip(args, dims) if d is not batching.not_mapped
    )
    if all(
        _definitely_equal_shape(shape, x.shape) and d == dim
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
