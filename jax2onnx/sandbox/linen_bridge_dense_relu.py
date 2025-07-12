import os
import numpy as np
import jax
import jax.numpy as jnp
import onnxruntime as ort
from flax import linen as nn
import jax2onnx


# ----------------------------------------------------------------------
# 1. Build & initialise the Linen module once (normal eager init)
# ----------------------------------------------------------------------
class LinenDenseRelu(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.features, name="dense_layer")(x)
        return nn.relu(x)


# real data sample just for initialisation
x_init = jnp.ones((1, 10), dtype=jnp.float32)

module = LinenDenseRelu(features=128)
variables = module.init(jax.random.key(0), x_init)  # <-- normal init

# ----------------------------------------------------------------------
# 2. Pull out the trained parameters as NumPy arrays
# ----------------------------------------------------------------------
kernel = np.asarray(variables["params"]["dense_layer"]["kernel"])
bias = np.asarray(variables["params"]["dense_layer"]["bias"])


# ----------------------------------------------------------------------
# 3.  Define a *pure JAX* forward that uses those constants
# ----------------------------------------------------------------------
def forward(x: jnp.ndarray) -> jnp.ndarray:
    # constants are captured from Python scope
    y = jnp.dot(x, kernel) + bias
    return jax.nn.relu(y)


# sanity-check: should equal the original module
assert np.allclose(
    forward(x_init), module.apply(variables, x_init), rtol=1e-6, atol=1e-6
)

# ----------------------------------------------------------------------
# 4.  Export to ONNX – no Flax, no initialisers, no tracer shapes
# ----------------------------------------------------------------------
print("🚀  Converting to ONNX…")
onnx_model = jax2onnx.to_onnx(
    forward, [x_init], model_name="DenseReluFrozen"  # concrete array
)

out_dir = "docs/onnx"
os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, "dense_relu_frozen.onnx")
with open(path, "wb") as f:
    f.write(onnx_model.SerializeToString())
print(f"💾  Model saved → {path}")

# quick numerical verification
sess = ort.InferenceSession(path)
onnx_out = sess.run(None, {sess.get_inputs()[0].name: np.array(x_init)})[0]
np.testing.assert_allclose(np.array(forward(x_init)), onnx_out, rtol=1e-5, atol=1e-5)
print("✅  JAX and ONNX outputs match.")
