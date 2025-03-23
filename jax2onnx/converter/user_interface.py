# jax2onnx/converter/user_interface.py

import onnx
from typing import Any
import numpy as np
import onnxruntime as ort
from jax2onnx.plugin_system import (
    import_all_plugins,
)
from jax2onnx.converter.simple_handler_conversion import to_onnx as simple_to_onnx


def save_onnx(
    fn: Any,
    input_shapes: Any,
    output_path: str = "model.onnx",
    model_name: str = "jax_model",
    opset: int = 21,
):
    onnx_model = to_onnx(
        fn,
        input_shapes,
        model_name=model_name,
        opset=opset,
    )
    onnx.save_model(onnx_model, output_path)


def to_onnx(
    fn: Any,
    input_shapes: Any,
    model_name: str = "jax_model",
    opset: int = 21,
) -> onnx.ModelProto:
    import_all_plugins()
    onnx_model = simple_to_onnx(
        fn,
        input_shapes,
        model_name=model_name,
        opset=opset,
    )
    return onnx_model


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
