# file: jax2onnx/typing_helpers.py

from typing import Protocol, Optional, Any
from jax2onnx.to_onnx import Z
from functools import partial


class Supports2Onnx(Protocol):
    """Protocol for objects that support ONNX export via `to_onnx`."""

    def to_onnx(self, z: Z, parameters: Optional[dict] = None) -> Z: ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

    def __getattr__(self, name: str) -> Any: ...


class PartialWithOnnx:
    """A wrapper around `functools.partial` that explicitly defines `to_onnx` for MyPy."""

    def __init__(self, func: Supports2Onnx, *args: Any, **kwargs: Any) -> None:
        """Ensure MyPy correctly recognizes the `partial` function."""
        self._wrapped_func = partial(func, *args, **kwargs)  # Store wrapped function
        self.to_onnx = func.to_onnx  # Assign `to_onnx` directly

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Ensure MyPy knows this is callable."""
        return self._wrapped_func(*args, **kwargs)
