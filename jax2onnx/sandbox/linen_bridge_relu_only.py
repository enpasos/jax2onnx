# file: sandbox/relu_only.py

import os
import numpy as np
import jax
import jax.numpy as jnp
import jax2onnx
import onnxruntime as ort


# --- 1) Define a pure ReLU function ---
def fwd(x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.relu(x)


# --- 2) Prepare a concrete test input ---
x0 = jnp.array([[-1.0, 0.0, 2.0]], dtype=jnp.float32)

# --- 3) Convert to ONNX ---
print("🚀 Converting ReLU-only function to ONNX...")
onnx_model = jax2onnx.to_onnx(fwd, [x0], model_name="ReluOnly")

# --- 4) Save a copy and verify it runs under ONNX Runtime ---
out_dir = "docs/onnx"
os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, "linen_bridge_relu_only.onnx")
with open(path, "wb") as f:
    f.write(onnx_model.SerializeToString())
print(f"💾 ONNX model written to {path}")

# numerical sanity-check
sess = ort.InferenceSession(path)
inp_name = sess.get_inputs()[0].name
onnx_out = sess.run(None, {inp_name: np.array(x0)})[0]
jax_out = np.array(fwd(x0))

np.testing.assert_allclose(jax_out, onnx_out, rtol=1e-6, atol=1e-6)
print("✅ Outputs match: ReLU-only pipeline is working!")
