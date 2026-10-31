# jax2onnx/_linen_bridge.py

"""The initialized Linen-to-NNX callable used by native source recipes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class _LinenToNNXCallable:
    def __init__(self, model: Callable[..., Any], rngs: Any) -> None:
        self._model = model
        self._rngs = rngs

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, rngs=self._rngs, **kwargs)
