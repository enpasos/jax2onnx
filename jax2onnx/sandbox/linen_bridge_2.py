import flax.linen as nn
from jax import numpy as jnp
import jax
import jax2onnx
import onnxruntime as ort
import numpy as np
import os

# --- Model Definition ---
class LinenModule(nn.Module):
    features: int

    def setup(self):
        # Layers are created here, once, when the model is initialized.
        self.dense = nn.Dense(features=self.features)

    def __call__(self, x):
        # The __call__ method now only uses the pre-defined layer.
        x = self.dense(x)
        # Use the basic jnp operation that jax2onnx understands.
        return jnp.maximum(0, x)

# --- Initialization ---
model = LinenModule(features=128)
inputs = jnp.ones((1, 10), dtype=jnp.float32)
variables = model.init(jax.random.key(0), inputs)
params = variables['params']

# --- Pure Inference Function ---
def pure_inference_function(learned_params, x):
    kernel = learned_params['dense']['kernel']
    bias = learned_params['dense']['bias']
    x = jnp.dot(x, kernel) + bias
    return jnp.maximum(0, x)

# --- Lambda for Conversion ---
forward_pass_to_convert = lambda x: pure_inference_function(params, x)

# --- Convert the Model ---
onnx_model = jax2onnx.to_onnx(
    forward_pass_to_convert,
    [inputs]
)

# --- Save the ONNX Model ---
output_dir = "docs/onnx"
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, "linen_bridge.onnx")
with open(model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"✅ Model converted and saved to {model_path} successfully!")


# --- Verification Step ---
print("\nVerifying model outputs...")

# 1. Get JAX output
jax_result = forward_pass_to_convert(inputs)

# 2. Get ONNX Runtime output
sess = ort.InferenceSession(model_path)
input_name = sess.get_inputs()[0].name
onnx_result = sess.run(None, {input_name: np.array(inputs)})[0]

# 3. Compare the results
np.testing.assert_allclose(jax_result, onnx_result, rtol=1e-5, atol=1e-5)

print("✅ JAX and ONNX runtime outputs match.")