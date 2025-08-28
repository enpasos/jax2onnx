import onnx
import onnxruntime as ort
import numpy as np
from onnx import helper as oh

model = "docs/onnx/primitives2/nnx/linear_no_bias_f64.onnx"

# -- dump optimized graph
so = ort.SessionOptions()
so.optimized_model_filepath = model.replace(".onnx", ".optimized.onnx")
sess_opt = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
print("Optimized model written to:", so.optimized_model_filepath)

m_opt = onnx.load(so.optimized_model_filepath)
print("Ops in optimized graph:", [n.op_type for n in m_opt.graph.node])

# -- inspect Gemm B dtype in the optimized graph
gemm = next(n for n in m_opt.graph.node if n.op_type == "Gemm")
B_name = gemm.input[1]
inits = {t.name: t for t in m_opt.graph.initializer}
np_dtype_B = oh.tensor_dtype_to_np_dtype(inits[B_name].data_type)
print("Gemm B dtype (optimized):", np.dtype(np_dtype_B).name)

# -- also print the runtime input dtype strings from ORT
print("ORT input type (optimized):", sess_opt.get_inputs()[0].type)

# -- compare optimized vs NO-OPT numerics quickly
so_no = ort.SessionOptions()
so_no.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess_no = ort.InferenceSession(model, so_no, providers=["CPUExecutionProvider"])
print("ORT input type (no-opt):", sess_no.get_inputs()[0].type)

rng = np.random.default_rng(0)
x = rng.standard_normal((3, 128)).astype(np.float64)
y_opt = sess_opt.run(None, {sess_opt.get_inputs()[0].name: x})[0]
y_no = sess_no.run(None, {sess_no.get_inputs()[0].name: x})[0]
print("opt vs no-opt max diff:", float(np.max(np.abs(y_opt - y_no))))
