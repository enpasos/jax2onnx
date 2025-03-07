import jax
import numpy as np
import onnxruntime as ort


def test(callable, onnx_model_path, x):

    # Test ONNX and JAX outputs
    session = ort.InferenceSession(onnx_model_path)
    onnx_output = session.run(None, {"var_0": np.array(x)})[0]

    jax_output = callable(x)

    # Verify outputs match
    return (
        np.allclose(onnx_output, jax_output, rtol=1e-3, atol=1e-5),
        "ONNX and JAX outputs do not match.",
    )
