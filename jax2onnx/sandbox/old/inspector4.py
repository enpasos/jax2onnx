import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort


# Define tile operation
def _my_tile(x):
    repeats_tensor = jnp.array([3, 1, 1], dtype=jnp.int32)
    tile = jnp.tile(x, repeats_tensor)
    return tile


# Prepare input
x = jnp.ones((1, 1, 8))
out = _my_tile(x)
print("JAX output shape:", out.shape)

# Load ONNX model
onnx_model_path = "docs/onnx/primitives/jnp/tile_repeats_tensor.onnx"
model = onnx.load(onnx_model_path)

# Run ONNX model
ort_session = ort.InferenceSession(onnx_model_path)
ort_inputs = {ort_session.get_inputs()[0].name: np.asarray(x)}
out_onnx = ort_session.run(None, ort_inputs)[0]

print("ONNX output shape:", out_onnx.shape)
# compare the outputs
print("Outputs are equal:", np.array_equal(out, out_onnx))
