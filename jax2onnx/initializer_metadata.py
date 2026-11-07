# jax2onnx/initializer_metadata.py

"""Errors and export-local lifetime for captured initializer annotations."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax2onnx.converter.initializer_metadata import InitializerMetadataRegistry


class InitializerMetadataError(ValueError):
    """An accepted initializer annotation could not satisfy its contract."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


@contextmanager
def initializer_metadata_session(
    callback: Callable[[object], Mapping[str, str] | None] | None,
) -> Iterator[InitializerMetadataRegistry]:
    """Keep all provenance outside the model, for exactly one public export."""
    from jax2onnx.converter.initializer_metadata import (
        InitializerMetadataRegistry,
        _ACTIVE_REGISTRY,
    )

    if callback is not None and not callable(callback):
        raise TypeError("initializer_metadata must be callable or None")
    registry = InitializerMetadataRegistry(callback)
    token = _ACTIVE_REGISTRY.set(registry)
    try:
        yield registry
    finally:
        _ACTIVE_REGISTRY.reset(token)
