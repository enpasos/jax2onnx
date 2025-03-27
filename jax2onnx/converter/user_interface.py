# file: jax2onnx/converter/user_interface.py

import onnx
from typing import Any
import numpy as np
import onnxruntime as ort
from jax2onnx.converter.jax_to_onnx import to_onnx as to_onnx_implementation


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
