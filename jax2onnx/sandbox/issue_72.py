import jax
import jax.numpy as jnp
from jax2onnx import to_onnx

jax.config.update("jax_enable_x64", True)


def mismatch_where(x):
    cond = x > 0
    return jnp.where(cond, x, jnp.array([1, 2, 3], dtype=jnp.int32))
    # true branch: float64 (x)
    # false branch: int32 constant → ONNX Where will mismatch


x = jnp.array([1.0, -2.0, 3.0], dtype=jnp.float64)

onnx_model = to_onnx(
    fn=mismatch_where,
    inputs=[x],
    enable_double_precision=True,
)

with open("type_mismatch_where.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
