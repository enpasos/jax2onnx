import onnx
import onnxruntime as ort
import numpy as np

m = onnx.load("docs/onnx/primitives/lax/cond_my_new_complex_scenario_dynamic.onnx")
sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
print([i.name for i in sess.get_inputs()])
print([o.name for o in sess.get_outputs()])

# use the same shapes as the JAX test:
batch = 2
x = np.random.randn(batch, 3, 4).astype("f")
y = np.random.randn(3, 4).astype("f")
sess.run(None, {"var_0": x, "var_1": y})
