# jax2onnx/__init__.py

from jax2onnx.user_interface import (  # noqa: F401
    to_onnx,
    onnx_function,
    allclose,
)

# Compatibility shim for protobuf upb containers (protobuf>=5).
# The sandbox repro still expects `RepeatedCompositeContainer.clear()`,
# which existed on the legacy python implementation but was dropped from
# the upb-backed containers that ship with onnx>=1.17. Inject the helper
# so downstream code keeps working without having to special-case the
# container type.
try:  # pragma: no cover - defensive import
    from google._upb import _message as _upb_message  # type: ignore

    _RepeatedCompositeContainer = _upb_message.RepeatedCompositeContainer
except Exception:  # pragma: no cover - upb missing
    _RepeatedCompositeContainer = None

if _RepeatedCompositeContainer is not None and not hasattr(
    _RepeatedCompositeContainer, "clear"
):

    def _clear(self) -> None:
        """Match the legacy protobuf repeated field API."""
        del self[:]  # uses the slice-aware __delitem__ provided by upb

    _RepeatedCompositeContainer.clear = _clear  # type: ignore[attr-defined]
