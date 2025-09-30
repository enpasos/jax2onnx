from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - optional dependency hints only
    pass  # type: ignore[import-not-found]


def run_optional_shape_inference(model: Any) -> Any:
    """Best-effort ONNX shape inference.

    Returns the original model if ONNX is unavailable or inference fails.
    """
    try:
        import onnx  # type: ignore[import-not-found, import-error]
    except Exception:
        return model

    try:
        return onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:
        return model
