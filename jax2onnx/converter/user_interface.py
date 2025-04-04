# file: jax2onnx/converter/user_interface.py

import onnx
from typing import Any, Optional, Dict
import numpy as np
import onnxruntime as ort
from jax2onnx.converter.jax_to_onnx import to_onnx as to_onnx_implementation
from collections import defaultdict


def to_onnx(
    fn: Any,
    input_shapes: Any,
    model_name: str = "jax_model",
    opset: int = 21,
    hierarchical: bool = False,
) -> onnx.ModelProto:
    """
    User-friendly interface for converting JAX functions to ONNX.

    Args:
        fn: JAX function or module to convert.
        input_shapes: List of input shapes (tuples).
        model_name: Name of the ONNX model.
        opset: ONNX opset version.
        hierarchical: Use hierarchical ONNX functions.

    Returns:
        An ONNX ModelProto object.
    """
    return to_onnx_implementation(fn, input_shapes, model_name=model_name, opset=opset)


def save_onnx(
    fn: Any,
    input_shapes: Any,
    output_path: str = "model.onnx",
    model_name: str = "jax_model",
    opset: int = 21,
    hierarchical: bool = False,
):
    onnx_model = to_onnx(
        fn,
        input_shapes,
        model_name=model_name,
        opset=opset,
        hierarchical=hierarchical,
    )
    onnx.save_model(onnx_model, output_path)


def allclose(callable, onnx_model_path, *xs):
    # Test ONNX and JAX outputs
    session = ort.InferenceSession(onnx_model_path)

    # Prepare inputs for ONNX model
    p = {"var_" + str(i): np.array(x) for i, x in enumerate(xs)}
    onnx_output = session.run(None, p)

    # Get JAX model output
    jax_output = callable(*xs)

    # Verify outputs match
    if not isinstance(jax_output, list):
        jax_output = [jax_output]
    if not isinstance(onnx_output, list):
        onnx_output = [onnx_output]

    isOk = np.allclose(onnx_output, jax_output, rtol=1e-3, atol=1e-5)

    return (
        isOk,
        (
            "ONNX and JAX outputs match :-)"
            if isOk
            else "ONNX and JAX outputs do not match :-("
        ),
    )


class ModelExportContext:
    """
    Holds model-specific state for naming and caching.
    """

    def __init__(self, model_id: Optional[str] = None):
        self.model_id: str = model_id or "default_model"
        self.function_cache: Dict[str, Any] = {}
        self.instance_counters: Dict[str, int] = defaultdict(int)

    def next_function_name(self, base_name: str) -> str:
        """
        Generates a unique ONNX function name scoped to this model.
        E.g., TransformerBlock_1, TransformerBlock_2, ...
        """
        self.instance_counters[base_name] += 1
        return f"{base_name}_{self.instance_counters[base_name]}"
