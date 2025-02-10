# file: jax2onnx/typing_helpers.py
from functools import partial
from typing import Protocol, Any, Callable, Mapping, TypeVar
from types import MappingProxyType

from jax2onnx.to_onnx import Z


class Supports2Onnx(Protocol):
    """Protocol for objects that support ONNX export via `to_onnx`."""

    def to_onnx(self, z: Z, **params: Any) -> Z:
        """Exports the object to ONNX, using the provided parameters."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Makes the object callable."""

    def __getattr__(self, name: str) -> Any:
        """Handles attribute access."""


T = TypeVar("T", bound=Supports2Onnx)


class PartialWithOnnx(Supports2Onnx):
    """Wraps `functools.partial` to handle ONNX export parameters."""

    def __init__(self, func: T, *args: Any, **kwargs: Any) -> None:
        """Initializes the PartialWithOnnx wrapper."""
        self._wrapped_func = partial(func, *args, **kwargs)
        self._onnx_params: Mapping[str, Any] = MappingProxyType(kwargs)

        if hasattr(func, "to_onnx"):
            self._wrapped_to_onnx = self._to_onnx_wrapper(
                func.to_onnx
            )  # ✅ Use correct wrapping
        else:
            raise AttributeError(
                f"The function {func} wrapped in PartialWithOnnx does not have a `to_onnx` method."
            )

    def _to_onnx_wrapper(
        self,
        to_onnx_fn: Callable[
            ..., Z
        ],  # ✅ Allow flexible parameters to avoid KwArg issues
    ) -> Callable[..., Z]:
        """Wraps the ONNX function to include additional parameters."""

        def wrapped_to_onnx(z: Z, **params: Any) -> Z:
            """Combines stored and passed parameters and calls the ONNX function."""
            combined_params = {**self._onnx_params, **params}
            return to_onnx_fn(z, **combined_params)  # ✅ Expand correctly

        return wrapped_to_onnx

    def to_onnx(self, z: Z, **params: Any) -> Z:  # ✅ Matches superclass
        """Calls the wrapped `to_onnx` method."""
        return self._wrapped_to_onnx(z, **params)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Calls the wrapped partial function."""
        return self._wrapped_func(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegates attribute access to the wrapped function."""
        return getattr(self._wrapped_func.func, name)


def supports_onnx(obj: Any) -> bool:
    """Check if an object supports ONNX export."""
    return hasattr(obj, "to_onnx") and callable(obj.to_onnx)
