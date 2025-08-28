import onnx
import onnxruntime as ort
import numpy as np
from onnx import helper as oh, numpy_helper
import hashlib
from statistics import median


def md5_bytes(b: bytes) -> str:
    h = hashlib.md5()
    h.update(b)
    return h.hexdigest()


def tensor_md5(t: onnx.TensorProto) -> str:
    # Works for all tensor types stored in raw_data
    if t.raw_data:
        return md5_bytes(t.raw_data)
    # Fallback via NumPy
    arr = numpy_helper.to_array(t)
    return md5_bytes(arr.tobytes(order="C"))


def summarize(model_path):
    m = onnx.load(model_path)
    inits = {t.name: t for t in m.graph.initializer}
    nodes = []
    for n in m.graph.node:
        if n.op_type == "Gemm":
            attrs = {a.name: oh.get_attribute_value(a) for a in n.attribute}
        else:
            attrs = {}
        nodes.append(
            (n.name or n.op_type, n.op_type, tuple(n.input), tuple(n.output), attrs)
        )

    init_summ = {}
    for name, t in inits.items():
        dtype = oh.tensor_dtype_to_np_dtype(t.data_type)
        init_summ[name] = {
            "shape": tuple(t.dims),
            "dtype": dtype,
            "md5": tensor_md5(t),
            "min": float(numpy_helper.to_array(t).min()),
            "max": float(numpy_helper.to_array(t).max()),
        }
    return m, nodes, init_summ


def print_diff_summary(nodes_a, nodes_b, inits_a, inits_b, tag_a="OLD", tag_b="NEW"):
    print(f"\n--- NODE LIST ({tag_a}) ---")
    for it in nodes_a:
        print(it)
    print(f"\n--- NODE LIST ({tag_b}) ---")
    for it in nodes_b:
        print(it)

    only_a = set(inits_a) - set(inits_b)
    only_b = set(inits_b) - set(inits_a)
    print("\nInitializers only in A:", only_a)
    print("Initializers only in B:", only_b)

    common = set(inits_a).intersection(inits_b)
    meta_diff = []
    data_diff = []
    for k in sorted(common):
        Ia, Ib = inits_a[k], inits_b[k]
        if Ia["shape"] != Ib["shape"] or Ia["dtype"] != Ib["dtype"]:
            meta_diff.append((k, Ia["shape"], Ib["shape"], Ia["dtype"], Ib["dtype"]))
        if Ia["md5"] != Ib["md5"]:
            data_diff.append(
                (k, Ia["md5"], Ib["md5"], Ia["min"], Ia["max"], Ib["min"], Ib["max"])
            )

    if meta_diff:
        print("\n!! Initializer meta differs (shape/dtype):")
        for row in meta_diff:
            print(row)
    if data_diff:
        print("\n!! Initializer DATA differs (md5/min/max):")
        for row in data_diff:
            print(row)
    if not meta_diff and not data_diff:
        print("\nInitializers match exactly (shape, dtype, and bytes).")


def ort_run(model_path, x, optimize=False):
    so = ort.SessionOptions()
    if not optimize:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        model_path, sess_options=so, providers=["CPUExecutionProvider"]
    )
    inp = sess.get_inputs()[0]
    assert inp.type == "tensor(double)", f"Expected fp64 input, got {inp.type}"
    return sess.run(None, {inp.name: x})[0]


def choose_input_shape(model_path, override_last_dim=None, seed=0):
    m = onnx.load(model_path)
    g = m.graph
    # Prefer Gemm.B to get in_features, else fall back to model input shape
    W = None
    for n in g.node:
        if n.op_type == "Gemm":
            B_name = n.input[1]
            for t in g.initializer:
                if t.name == B_name:
                    W = numpy_helper.to_array(t)
                    break
            break
    if W is None:
        # fallback
        inp = g.input[0].type.tensor_type.shape.dim
        shp = [d.dim_value if d.dim_value > 0 else 2 for d in inp]  # 2 for dynamic
        return tuple(shp)

    K = W.shape[0]
    # Heuristic batch: (3,10,K) if rank>=3, otherwise (3,K)
    # You can customize this to match your failing tests.
    rank = len(onnx.load(model_path).graph.input[0].type.tensor_type.shape.dim)
    if rank >= 3:
        shp = (3, 10, override_last_dim or K)
    elif rank == 2:
        shp = (3, override_last_dim or K)
    else:
        shp = (override_last_dim or K,)
    return shp


def diff_stats(a, b):
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    absd = np.abs(d)
    max_idx = np.unravel_index(np.argmax(absd), absd.shape)
    return {
        "max": float(absd[max_idx]),
        "mean": float(absd.mean()),
        "median": float(median(absd.ravel())),
        "max_idx": tuple(int(i) for i in max_idx),
    }


if __name__ == "__main__":
    OLD = "docs/onnx/primitives/nnx/linear_high_rank_dynamic_f64.onnx"
    NEW = "docs/onnx/primitives2/nnx/linear_high_rank_dynamic_f64.onnx"

    mA, nodesA, initsA = summarize(OLD)
    mB, nodesB, initsB = summarize(NEW)

    print_diff_summary(nodesA, nodesB, initsA, initsB, "OLD", "NEW")

    # Build a deterministic fp64 input compatible with both
    rng = np.random.default_rng(0)
    shp = choose_input_shape(NEW)  # uses NEW to infer K; both should match
    x = rng.standard_normal(shp).astype(np.float64)

    # No-optimization runs
    y_old = ort_run(OLD, x, optimize=False)
    y_new = ort_run(NEW, x, optimize=False)
    s = diff_stats(y_old, y_new)
    print("\n[NO-OPT] OLD vs NEW:", s)

    # Optimized runs
    y_old_opt = ort_run(OLD, x, optimize=True)
    y_new_opt = ort_run(NEW, x, optimize=True)
    s_opt = diff_stats(y_old_opt, y_new_opt)
    print("[OPT]    OLD vs NEW:", s_opt)

    # Cross check old opt/no-opt and new opt/no-opt
    print("[OLD] opt vs no-opt:", diff_stats(y_old_opt, y_old))
    print("[NEW] opt vs no-opt:", diff_stats(y_new_opt, y_new))
