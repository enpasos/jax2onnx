import numpy as np
import onnxruntime as ort
import jax
from jax import numpy as jnp, nn


# --- (A) jax reference function, exactly as before ---
@jax.jit
def debug_dpa(q, k, v, mask):
    # 1) compute logits
    logits = jnp.einsum("btnh,bsnh->bnts", q, k) / jnp.sqrt(q.shape[-1])
    # 2) additive mask
    not_mask = ~mask
    additive = not_mask.astype(q.dtype) * -1e9
    masked = logits + additive
    # 3) softmax
    weights = nn.softmax(masked, axis=-1)
    # 4) weighted sum
    out = jnp.einsum("bnts,bsnh->btnh", weights, v)
    return out


# --- (B) exactly the same test inputs you used above ---
Q = np.array([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=np.float32)  # (1,1,1,4)
K = np.array(
    [  # (1,2,1,4)
        [
            [[1.0, 0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0, 0.0]],
        ]
    ],
    dtype=np.float32,
)
V = np.array(
    [  # (1,2,1,4)
        [
            [[10.0, 20.0, 30.0, 40.0]],
            [[50.0, 60.0, 70.0, 80.0]],
        ]
    ],
    dtype=np.float32,
)
M = np.array([[[[True, False]]]], dtype=bool)  # (1,1,1,2)

# --- (C) run jax reference to get “ground truth” ---
expected = np.array(debug_dpa(Q, K, V, M).block_until_ready())
print("JAX reference output:\n", expected)

# --- (D) load ONNX and run it in onnxruntime ---
model_path = "docs/onnx/primitives/nn/dpa_debug_one_false.onnx"
sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

# ONNX Runtime will expose the graph’s inputs in order:
input_names = [i.name for i in sess.get_inputs()]
print("ONNX graph inputs:", input_names)

# build the feed dict in the same order
feeds = {
    input_names[0]: Q,
    input_names[1]: K,
    input_names[2]: V,
    input_names[3]: M,
}

# run and grab the single output
onnx_out = sess.run(None, feeds)[0]
print("ONNX Runtime output:\n", onnx_out)

# --- (E) compare ---
diff = onnx_out - expected
print("Difference:\n", diff)
print("Max absolute diff:", np.abs(diff).max())
