# file: jax2onnx/sandbox/linen_bridge.py
import os
import flax.linen as nn
from flax import nnx
import jax.numpy as jnp
import numpy as np
import jax2onnx
import onnxruntime as ort


# ---------------------------------------------------------
# 1.  Define the Linen module
# ---------------------------------------------------------
class LinenModule(nn.Module):
    def setup(self):
        self.dense = nn.Dense(128)

    def __call__(self, x):
        x = self.dense(x)
        return nn.relu(x)


model = LinenModule()
model = nnx.bridge.ToNNX(model, rngs=nnx.Rngs(0))
inputs = jnp.ones((1, 10))
model = nnx.bridge.lazy_init(model, inputs)


# ---------------------------------------------------------
# 2.  Convert to ONNX
# ---------------------------------------------------------
onnx_model = jax2onnx.to_onnx(
    model,
    [inputs],
    model_name="LinenDenseRelu",
)

# ---------------------------------------------------------
# 3.  Save and quick‑test under ONNX Runtime
# ---------------------------------------------------------
out_dir = "docs/onnx"
os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, "linen_dense_relu.onnx")
with open(path, "wb") as f:
    f.write(onnx_model.SerializeToString())
print(f"💾  ONNX model written to {path}")

sess = ort.InferenceSession(path)
inp_name = sess.get_inputs()[0].name
onnx_out = sess.run(None, {inp_name: np.array(inputs)})[0]

jax_out = np.array(model(inputs))

np.testing.assert_allclose(jax_out, onnx_out, rtol=1e-6, atol=1e-6)
print("✅  JAX and ONNX outputs match.")
