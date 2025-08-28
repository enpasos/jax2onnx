# checker_tri.py
import onnx
import onnxruntime as ort
import numpy as np
from onnx import helper as oh, numpy_helper
import jax
import jax.numpy as jnp
from flax import nnx


def get_w_b(model):
    g = model.graph
    inits = {t.name: t for t in g.initializer}
    # Find the single Gemm
    gemm = next(n for n in g.node if n.op_type == "Gemm")
    _A_name, B_name = gemm.input[0], gemm.input[1]
    C_name = gemm.input[2] if len(gemm.input) >= 3 else None
    transB = 0
    for a in gemm.attribute:
        if a.name == "transB":
            transB = oh.get_attribute_value(a)
    W = numpy_helper.to_array(inits[B_name])
    if transB:
        W = W.T  # ensure shape (K,M)
    b = numpy_helper.to_array(inits[C_name]) if C_name and C_name in inits else None
    return W, b


def choose_input_shape(model, K):
    # Use model input rank, fill dynamics with small sizes like tests
    dims = model.graph.input[0].type.tensor_type.shape.dim
    dd = [
        d.dim_value if d.HasField("dim_value") and d.dim_value > 0 else 3 for d in dims
    ]
    if len(dd) >= 2:
        dd[-1] = K
    elif len(dd) == 1:
        dd[0] = K
    return tuple(dd)


def ort_run(path, x):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        path, sess_options=so, providers=["CPUExecutionProvider"]
    )
    return sess.run(None, {sess.get_inputs()[0].name: x})[0]


def stats(a, b):
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    ad = np.abs(d)
    idx = np.unravel_index(ad.argmax(), ad.shape)
    from statistics import median

    return dict(
        max=float(ad[idx]),
        mean=float(ad.mean()),
        median=float(median(ad.ravel())),
        max_idx=tuple(int(i) for i in idx),
    )


if __name__ == "__main__":
    onnx_path = "docs/onnx/primitives2/nnx/linear_high_rank_dynamic_f64.onnx"

    m = onnx.load(onnx_path)
    W, b = get_w_b(m)  # W: (K,M), b: (M,) or None
    K, M = W.shape

    xshape = choose_input_shape(m, K)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(xshape).astype(np.float64)

    # ORT (no optimizations)
    y_ort = ort_run(onnx_path, x)

    # NumPy reference: flatten -> matmul -> add bias -> reshape back
    x2 = x.reshape(-1, K)
    y2 = x2.dot(W)
    if b is not None:
        y2 = y2 + b
    y_np = y2.reshape(*xshape[:-1], M).astype(np.float64)

    # JAX reference with same W/b
    jax.config.update("jax_enable_x64", True)
    lin = nnx.Linear(K, M, rngs=nnx.Rngs(0))
    # overwrite params with ONNX weights to guarantee same values
    lin.kernel.value = jnp.asarray(W)  # keep dtype from ONNX (likely float64 here)
    if b is None:
        lin.use_bias = False
        lin.bias = None
    else:
        lin.bias.value = jnp.asarray(b)

    xj = jnp.asarray(x)  # fp64 input
    y_jax = np.asarray(lin(xj))  # back to NumPy

    print("dtypes: x", x.dtype, "W", W.dtype, "b", None if b is None else b.dtype)
    print("[ORT vs NumPy] ", stats(y_ort, y_np))
    print("[JAX vs NumPy] ", stats(y_jax, y_np))
    print("[JAX vs ORT ] ", stats(y_jax, y_ort))
