import jax
import jax.numpy as jnp
from jax2onnx import to_onnx
import onnxruntime as ort

jax.config.update("jax_enable_x64", True)


def broken():
    float_arr = jnp.array([1.0, 2.0], dtype=jnp.float32)
    int_arr = jnp.array([3, 4], dtype=jnp.int32)
    concat_result = jnp.concatenate([float_arr, int_arr])
    lookup = jnp.array([100, 200, 300, 400, 500], dtype=jnp.int32)
    indices = jnp.clip(concat_result.astype(jnp.int32), 0, len(lookup) - 1)
    indexed_vals = jnp.take(lookup, indices)
    float_vals = concat_result * 1.5
    return concat_result, indexed_vals, float_vals


try:
    jax_result = broken()
    onnx_model = to_onnx(broken, inputs=[], enable_double_precision=True)
    with open("broken.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    session = ort.InferenceSession("broken.onnx")
    outputs = session.run(None, {})
    print("All tests passed!")

except Exception as e:
    error_str = str(e)
    print(f"Error: {type(e).__name__}: {error_str}")
