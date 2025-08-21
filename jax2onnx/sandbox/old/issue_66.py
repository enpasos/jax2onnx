import jax
import jax.numpy as jnp
from jax2onnx import to_onnx
import onnxruntime as ort

jax.config.update("jax_enable_x64", True)


def bad_concat_model():
    a = jnp.array([1, 2, 3], dtype=jnp.int32)
    b = jnp.array([1.1, 2.2, 3.3], dtype=jnp.float32)
    return jnp.concatenate([a, b])


# Export to ONNX
onnx_model = to_onnx(bad_concat_model, inputs=[], enable_double_precision=True)
with open("bad_concat.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# Try to load in ONNX Runtime
try:
    ort.InferenceSession("bad_concat.onnx")
except Exception as e:
    print("Expected ONNXRuntime error:")
    print(e)
