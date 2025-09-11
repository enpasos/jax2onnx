# tests/extra_tests/test_ir_optimizations.py
import onnx_ir as ir
from jax2onnx.converter2.ir_optimizations import optimize_graph

def V(name, dtype=ir.DataType.FLOAT, shape=()):
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))

def build_graph_with_not_tm():
    # graph IO
    x    = V("x",             ir.DataType.FLOAT, (3, 30))
    det  = V("deterministic", ir.DataType.BOOL,  ())
    ratio= V("ratio",         ir.DataType.FLOAT, ())
    y    = V("y",             ir.DataType.FLOAT, (3, 10))

    # intermediates
    a      = V("after_gemm", ir.DataType.FLOAT, (3, 20))
    b      = V("after_bn",   ir.DataType.FLOAT, (3, 20))
    not_out= V("not_out",    ir.DataType.BOOL,  ())
    d_out  = V("drop_out",   ir.DataType.FLOAT, (3, 20))
    g_out  = V("gelu_out",   ir.DataType.FLOAT, (3, 20))

    n1 = ir.Node(op_type="Gemm",               domain="", inputs=[x],                 outputs=[a],     name="Gemm_1")
    n2 = ir.Node(op_type="BatchNormalization", domain="", inputs=[a],                 outputs=[b],     name="BN_1")
    n3 = ir.Node(op_type="Not",                domain="", inputs=[det],               outputs=[not_out], name="Not_1")
    n4 = ir.Node(op_type="Dropout",            domain="", inputs=[b, ratio, not_out], outputs=[d_out], name="Drop_1")
    n5 = ir.Node(op_type="Gelu",               domain="", inputs=[d_out],             outputs=[g_out], name="Gelu_1")
    n6 = ir.Node(op_type="Gemm",               domain="", inputs=[g_out],             outputs=[y],     name="Gemm_2")

    g = ir.Graph(
        name="g",
        inputs=[x, det, ratio],
        outputs=[y],
        nodes=[n1, n2, n3, n4, n5, n6],
    )
    m = ir.Model(graph=g, ir_version=10)
    try:
        m.opset_imports = {"": 21}
    except Exception:
        pass
    return m

def _nodes(g):
    return list(getattr(g, "nodes", getattr(g, "_nodes", [])))

def _inputs(g):
    arr = getattr(g, "inputs", None) or getattr(g, "input", None) or []
    try:
        return list(arr)
    except Exception:
        return []

def test_dropout_training_mode_inlined_constant_false_and_not_removed():
    m = build_graph_with_not_tm()
    m = optimize_graph(m)
    g = m.graph
    nodes = _nodes(g)

    # Not must be gone
    assert "Not" not in [n.op_type for n in nodes]

    # Dropout must remain and its 3rd input must be "missing" (empty name)
    drops = [n for n in nodes if n.op_type == "Dropout"]
    assert len(drops) == 1
    d = drops[0]
    ins = getattr(d, "inputs", getattr(d, "input", []))
    assert len(ins) >= 3
    third = ins[2]
    third_name = getattr(third, "name", "")
    assert third_name == "", f"expected missing tm input, got {third_name!r}"

    # Unused graph input 'deterministic' must be pruned; 'x' and 'ratio' must remain
    in_names = {getattr(v, "name", "") for v in _inputs(g)}
    assert "deterministic" not in in_names
    assert "x" in in_names
    assert "ratio" in in_names
