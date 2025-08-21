import numpy as np
import jax.numpy as jnp
import onnx
import onnxruntime as ort
from jax import lax

# ---------------------------------------------------------------------
# load the ONNX model -------------------------------------------------
m = onnx.load("docs/onnx/primitives/lax/cond_my_new_complex_scenario_dynamic.onnx")
sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])

print("Inputs :", [i.name for i in sess.get_inputs()])
print("Outputs:", [o.name for o in sess.get_outputs()])

# ---------------------------------------------------------------------
# create a random test-batch in the same shape as the original test ---
batch = 2
x = np.random.randn(batch, 3, 4).astype("f")
y = np.random.randn(3, 4).astype("f")

# ---------------------------------------------------------------------
# run ONNX ------------------------------------------------------------
try:
    ort_out = sess.run(None, {"var_0": x, "var_1": y})
    print("ORT output shapes:", [o.shape for o in ort_out])
except Exception as e:
    print("ORT FAILED with:", e)
    raise


# ---------------------------------------------------------------------
# run the *reference* JAX function ------------------------------------
def true_branch(t):
    return (t[0] * 2 + t[1], jnp.sum(t[0]))


def false_branch(t):
    return (t[0] - t[1] * 2, jnp.mean(t[0]))


jax_out = lax.cond(jnp.all(x > 0), true_branch, false_branch, (x, y))

# ---------------------------------------------------------------------
# numeric diff --------------------------------------------------------
for i, (o_onnx, o_jax) in enumerate(zip(ort_out, jax_out)):
    print(f"out[{i}]  max abs diff =", np.max(np.abs(o_onnx - np.array(o_jax))))
