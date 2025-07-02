import flax.linen as nn
import flax.nnx as nnx
from jax import numpy as jnp
import jax
import jax2onnx
import onnxruntime as ort
import numpy as np
import os

# --- Model Definition using the native NNX API ---
class NnxModule(nnx.Module):
    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs):
        self.dense = nnx.Linear(in_features, out_features, rngs=rngs)

    def __call__(self, x: jax.Array):
        x = self.dense(x)
        return nn.relu(x)

# --- Initialization & Splitting ---
model = NnxModule(in_features=10, out_features=128, rngs=nnx.Rngs(0))
inputs = jnp.ones((1, 10), dtype=jnp.float32)
graph_def, state = nnx.split(model)

# --- Define and Create the Pure Forward Pass ---
# This returns a function that takes only the input_data
forward_pass_callable = graph_def.apply(state)

# --- Convert the Model with Explicit Plugin ---
# We trace a function that returns only the first element (the output)
# of what the NNX callable returns. This bakes the parameters in as
# constants and excludes the final state from the ONNX graph's outputs.
onnx_model = jax2onnx.to_onnx(
    lambda x: forward_pass_callable(x)[0],  # <--- MODIFIED
    [inputs]
)

# --- Save and Verify ---
output_dir = "docs/onnx"
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, "nnx_model_final_optimized.onnx")
with open(model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"✅ Optimized NNX model saved to {model_path} successfully!")

# Verification Step
print("\nVerifying model outputs...")
# Select the first element of the JAX output tuple for comparison
jax_result = forward_pass_callable(inputs)[0] # <--- MODIFIED
sess = ort.InferenceSession(model_path)
input_name = sess.get_inputs()[0].name
onnx_result = sess.run(None, {input_name: np.array(inputs)})[0]

np.testing.assert_allclose(jax_result, onnx_result, rtol=1e-5, atol=1e-5)
print("✅ JAX and ONNX runtime outputs match.")