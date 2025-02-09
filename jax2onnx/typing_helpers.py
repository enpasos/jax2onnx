# file: jax2onnx/typing_helpers.py

from typing import Protocol, Any
from jax2onnx.to_onnx import Z
from functools import partial


class Supports2Onnx(Protocol):
    """Protocol for objects that support ONNX export via `to_onnx`."""

    def to_onnx(self, z: Z, **params) -> Z: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def __getattr__(self, name: str) -> Any: ...


class PartialWithOnnx:
    """A wrapper around `functools.partial` that explicitly defines `to_onnx` for MyPy."""

    def __init__(self, func: Supports2Onnx, *args: Any, **kwargs: Any) -> None:
        """Ensure MyPy correctly recognizes the `partial` function."""
        self._wrapped_func = partial(func, *args, **kwargs)

        # Ensure function has `to_onnx`
        if hasattr(func, "to_onnx"):
            self.to_onnx = func.to_onnx
        else:
            raise AttributeError(
                f"The function {func} wrapped in PartialWithOnnx does not have a `to_onnx` method."
            )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._wrapped_func(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped function."""
        return getattr(self._wrapped_func.func, name)
