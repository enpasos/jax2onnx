import numpy as np
import onnxruntime as ort


def allclose(callable, onnx_model_path, *xs):

    # Test ONNX and JAX outputs
    session = ort.InferenceSession(onnx_model_path)
    # iterate over inputs xs ("var_0", "var_1", ...) for (x1, x2, ...)
    # for i, x in enumerate(xs):
    #   session.set_input(i, np.array(x))

    #  onnx_output = session.run(None, {"var_0": np.array(x)})[0]

    p = {"var_" + str(i): np.array(x) for i, x in enumerate(xs)}
    onnx_output = session.run(None, p)

    jax_output = callable(*xs)

    # Verify outputs match
    # if single output, convert to list
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
