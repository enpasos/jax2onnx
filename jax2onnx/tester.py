import numpy as np
import onnxruntime as ort


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
