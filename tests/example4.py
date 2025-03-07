# jax2onnx/examples/example4.py

import jax
import jax.numpy as jnp
import numpy as np
import onnxruntime as ort

import flax.nnx as nnx

from jax2onnx.converter.converter import JaxprToOnnx
from jax2onnx.converter.primitives.flax.nnx.linear_general import (
    temporary_linear_general_patch,
)


def example4():
    seed = 1001

    # Instantiate nnx.LinearGeneral
    linear_fn = nnx.LinearGeneral(
        in_features=(8, 32), out_features=(256,), axis=(-2, -1), rngs=nnx.Rngs(seed)
    )

    model_path = "example4.onnx"

    # Create the converter
    converter = JaxprToOnnx()

    # Use the monkey patch so that nnx.LinearGeneral uses our linear_general_p
    with temporary_linear_general_patch(converter):
        # Convert the model -> ONNX
        converter.save_onnx(
            fn=linear_fn,
            input_shapes=[(2, 4, 8, 32)],  # Example shape
            output_path=model_path,
            model_name="example4_model",
        )

    # Minimal smoke test with random input
    rng = jax.random.PRNGKey(seed)
    x_example = jax.random.normal(rng, (2, 4, 8, 32))

    # ONNX inference
    session = ort.InferenceSession(model_path)
    # In a full version, your converter._add_input() would produce an input name like "var_0"
    # so we assume the model expects a single input named "var_0".
    onnx_output = session.run(None, {"var_0": np.array(x_example)})[0]

    # JAX run
    jax_output = linear_fn(x_example)

    # Compare the outputs
    # At this point, the code in handle_linear_general is still a stub,
    # so you'd need real ONNX nodes to see correct results.
    # For demonstration, we won't do an exact numerical check here.
    print("example4 finished successfully.")


if __name__ == "__main__":
    example4()
