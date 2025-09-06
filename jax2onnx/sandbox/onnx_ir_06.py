# file: jax2onnx/sandbox/function_gemm_constant.py
from __future__ import annotations
import os
import numpy as np
import onnx
from onnx import helper as oh, numpy_helper, TensorProto

# ---------------------------------------------------------------------
# output
# ---------------------------------------------------------------------
out_dir = "docs/onnx/"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "function_gemm_constant.onnx")

# ---------------------------------------------------------------------
# function: GemmFn in custom domain "local.fn"
#   Y = Gemm(X, W, B)  with W,B embedded as Constant(value=...) nodes
# ---------------------------------------------------------------------
DOM_FN = "local.fn"  # <<< non-empty custom domain
OPSET_AI_ONNX = oh.make_operatorsetid("", 21)
OPSET_FN = oh.make_operatorsetid(DOM_FN, 1)  # version is arbitrary for user domain

N, K, M = 3, 4, 2
X_name, Y_name = "X", "Y"
W_name, B_name = "W", "B"

W_np = np.arange(K * M, dtype=np.float32).reshape(K, M) / 10.0
B_np = np.array([0.1, -0.2], dtype=np.float32)
W_t = numpy_helper.from_array(W_np, name=W_name)
B_t = numpy_helper.from_array(B_np, name=B_name)

# Constant nodes must carry exactly one 'value' attribute (spec-compliant)
const_W = oh.make_node(
    "Constant", inputs=[], outputs=[W_name], name="W_const", value=W_t
)
const_B = oh.make_node(
    "Constant", inputs=[], outputs=[B_name], name="B_const", value=B_t
)
gemm = oh.make_node(
    "Gemm", inputs=[X_name, W_name, B_name], outputs=[Y_name], name="Gemm_0"
)

gemm_fn = oh.make_function(
    domain=DOM_FN,  # <<< custom domain
    fname="GemmFn",  # op_type to be used at call site
    inputs=[X_name],
    outputs=[Y_name],
    nodes=[const_W, const_B, gemm],
    opset_imports=[
        OPSET_AI_ONNX
    ],  # the function body uses core ops from the default domain
    doc_string="Y = Gemm(X, W, B) with W,B embedded as Constant nodes",
)

# ---- NEW: add shape/type info for names used inside the function body ----
vi_X = oh.make_tensor_value_info(X_name, TensorProto.FLOAT, [N, K])
vi_W = oh.make_tensor_value_info(W_name, TensorProto.FLOAT, [K, M])
vi_B = oh.make_tensor_value_info(B_name, TensorProto.FLOAT, [M])
vi_Y = oh.make_tensor_value_info(Y_name, TensorProto.FLOAT, [N, M])

# FunctionProto supports value_info; append them so viewers can render shapes
gemm_fn.value_info.extend([vi_X, vi_W, vi_B, vi_Y])

# ---------------------------------------------------------------------
# top graph calls GemmFn once
# ---------------------------------------------------------------------
in_vi = oh.make_tensor_value_info("in_0", TensorProto.FLOAT, [N, K])
out_vi = oh.make_tensor_value_info("out_0", TensorProto.FLOAT, [N, M])

call = oh.make_node(
    "GemmFn",  # matches function name
    inputs=["in_0"],
    outputs=["out_0"],
    name="GemmFn_0",
    domain=DOM_FN,  # <<< call in the same custom domain
)

graph = oh.make_graph(
    name="TopGraph",
    nodes=[call],
    inputs=[in_vi],
    outputs=[out_vi],
    initializer=[],
)

model = oh.make_model(
    graph,
    ir_version=10,
    opset_imports=[
        OPSET_AI_ONNX,
        OPSET_FN,
    ],  # <<< import both default and custom domains
    producer_name="sandbox",
)

# attach function
model.functions.append(gemm_fn)

# validate & save
onnx.checker.check_model(model)
onnx.save(model, out_path)
print(f"✅ Wrote: {out_path}")
