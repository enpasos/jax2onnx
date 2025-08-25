from __future__ import annotations
from typing import Any, Dict, List, Optional
import onnx

def to_onnx(
    *,
    fn: Any,
    inputs: List[Any],
    input_params: Optional[Dict[str, Any]],
    model_name: str,
    opset: int,
    enable_double_precision: bool,
    loosen_internal_shapes: bool,
    record_primitive_calls_file: Optional[str],
) -> onnx.ModelProto:
    raise NotImplementedError("converter2 (onnx_ir) not implemented yet.")
    raise NotImplementedError(
        "converter2.to_onnx (onnx_ir) not implemented yet. "
        "Call user_interface.to_onnx(..., use_onnx_ir=False) or unset JAX2ONNX_USE_ONNX_IR."
    )
