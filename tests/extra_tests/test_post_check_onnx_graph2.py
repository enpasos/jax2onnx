# tests/extra_tests/test_post_check_onnx_graph2.py
from __future__ import annotations
import onnx_ir as ir
from jax2onnx.plugins2._post_check_onnx_graph2 import expect_graph2 as EG2

def V(name, dtype=ir.DataType.FLOAT, shape=()):
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))

def _nodes(g):
    return list(getattr(g, "nodes", getattr(g, "_nodes", [])))

def build_static_chain(B=3):
    x = V("x", ir.DataType.FLOAT, (B, 30))
    ratio = V("ratio", ir.DataType.FLOAT, ())
    y = V("y", ir.DataType.FLOAT, (B, 10))

    a = V("a", ir.DataType.FLOAT, (B, 20))
    b = V("b", ir.DataType.FLOAT, (B, 20))
    d = V("d", ir.DataType.FLOAT, (B, 20))
    g = V("g", ir.DataType.FLOAT, (B, 20))

    n1 = ir.Node(op_type="Gemm",               domain="", inputs=[x],        outputs=[a], name="Gemm_1")
    n2 = ir.Node(op_type="BatchNormalization", domain="", inputs=[a],        outputs=[b], name="BN_1")
    n3 = ir.Node(op_type="Dropout",            domain="", inputs=[b, ratio], outputs=[d], name="Drop_1")
    n4 = ir.Node(op_type="Gelu",               domain="", inputs=[d],        outputs=[g], name="Gelu_1")
    n5 = ir.Node(op_type="Gemm",               domain="", inputs=[g],        outputs=[y], name="Gemm_2")

    gr = ir.Graph(name="top", inputs=[x, ratio], outputs=[y], nodes=[n1, n2, n3, n4, n5])
    m = ir.Model(graph=gr, ir_version=10)
    try: m.opset_imports = {"": 21}
    except Exception: pass
    return m

def build_dynamic_chain():  # B symbolic
    x = V("x", ir.DataType.FLOAT, ("B", 30))
    ratio = V("ratio", ir.DataType.FLOAT, ())
    y = V("y", ir.DataType.FLOAT, ("B", 10))

    a = V("a", ir.DataType.FLOAT, ("B", 20))
    b = V("b", ir.DataType.FLOAT, ("B", 20))
    d = V("d", ir.DataType.FLOAT, ("B", 20))
    g = V("g", ir.DataType.FLOAT, ("B", 20))

    n1 = ir.Node(op_type="Gemm",               domain="", inputs=[x],        outputs=[a], name="Gemm_1")
    n2 = ir.Node(op_type="BatchNormalization", domain="", inputs=[a],        outputs=[b], name="BN_1")
    n3 = ir.Node(op_type="Dropout",            domain="", inputs=[b, ratio], outputs=[d], name="Drop_1")
    n4 = ir.Node(op_type="Gelu",               domain="", inputs=[d],        outputs=[g], name="Gelu_1")
    n5 = ir.Node(op_type="Gemm",               domain="", inputs=[g],        outputs=[y], name="Gemm_2")

    gr = ir.Graph(name="top", inputs=[x, ratio], outputs=[y], nodes=[n1, n2, n3, n4, n5])
    m = ir.Model(graph=gr, ir_version=10)
    try: m.opset_imports = {"": 21}
    except Exception: pass
    return m

def build_chain_with_dangling_input():
    m = build_static_chain()
    # add a dangling input
    det = V("deterministic", ir.DataType.BOOL, ())
    ins = getattr(m.graph, "inputs", getattr(m.graph, "input", None))
    try:
        m.graph.inputs = list(ins) + [det]
    except Exception:
        pass
    return m

def test_static_path_with_shapes_and_symbols_and_no_unused():
    m = build_static_chain(B=3)
    check = EG2(
        ["Gemm:Bx20 -> BatchNormalization:Bx20 -> Dropout:Bx20 -> Gelu:Bx20 -> Gemm:Bx10"],
        symbols={"B": None},
        must_absent=["Not"],
        no_unused_inputs=True,
    )
    assert check(m)

def test_dynamic_unknown_batch_via_question_mark():
    m = build_dynamic_chain()
    check = EG2(
        ["Gemm:?x20 -> BatchNormalization:?x20 -> Dropout:?x20 -> Gelu:?x20 -> Gemm:?x10"],
        no_unused_inputs=True,
    )
    assert check(m)

def test_no_unused_inputs_catches_dangling():
    m = build_chain_with_dangling_input()
    check = EG2(
        ["Gemm:Bx20 -> BatchNormalization:Bx20 -> Dropout:Bx20 -> Gelu:Bx20 -> Gemm:Bx10"],
        symbols={"B": None},
        no_unused_inputs=True,
    )
    assert not check(m)  # should fail because of dangling 'deterministic'

def test_function_body_search_matches():
    # top graph: trivial pass-through
    xi = V("xi", ir.DataType.FLOAT, (2, 2))
    xo = V("xo", ir.DataType.FLOAT, (2, 2))
    passt = ir.Node(op_type="Identity", domain="", inputs=[xi], outputs=[xo], name="Id")
    top = ir.Graph(name="top", inputs=[xi], outputs=[xo], nodes=[passth])
    m = ir.Model(graph=top, ir_version=10)

    # function body graph
    a = V("a", ir.DataType.FLOAT, (2, 2))
    b = V("b", ir.DataType.FLOAT, (2, 2))
    c = V("c", ir.DataType.FLOAT, (2, 2))
    r1 = ir.Node(op_type="Reshape", domain="", inputs=[a], outputs=[b], name="R1")
    ge = ir.Node(op_type="Gelu",    domain="", inputs=[b], outputs=[c], name="G")
    r2 = ir.Node(op_type="Reshape", domain="", inputs=[c], outputs=[b], name="R2")  # reuse b for simplicity
    fgraph = ir.Graph(name="fn_g", inputs=[a], outputs=[b], nodes=[r1, ge, r2])

    class _Fn: pass
    fn = _Fn()
    fn.domain = "custom"
    fn.name = "mlp_body"
    fn.graph = fgraph
    m.functions = [fn]

    check = EG2(
        ["Reshape -> Gelu -> Reshape"],  # will be found in the function body
        must_absent=["Not"],
        search_functions=True,
    )
    assert check(m)
