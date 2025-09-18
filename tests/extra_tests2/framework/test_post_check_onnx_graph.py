# tests/extra_tests/test_post_check_onnx_graph.py
from __future__ import annotations
import onnx_ir as ir
from jax2onnx.plugins2._post_check_onnx_graph import expect_graph as EG


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

    n1 = ir.Node(op_type="Gemm", domain="", inputs=[x], outputs=[a], name="Gemm_1")
    n2 = ir.Node(
        op_type="BatchNormalization", domain="", inputs=[a], outputs=[b], name="BN_1"
    )
    n3 = ir.Node(
        op_type="Dropout", domain="", inputs=[b, ratio], outputs=[d], name="Drop_1"
    )
    n4 = ir.Node(op_type="Gelu", domain="", inputs=[d], outputs=[g], name="Gelu_1")
    n5 = ir.Node(op_type="Gemm", domain="", inputs=[g], outputs=[y], name="Gemm_2")

    gr = ir.Graph(
        name="top", inputs=[x, ratio], outputs=[y], nodes=[n1, n2, n3, n4, n5]
    )
    m = ir.Model(graph=gr, ir_version=10)
    try:
        m.opset_imports = {"": 21}
    except Exception:
        pass
    return m


def build_dynamic_chain():  # B symbolic
    x = V("x", ir.DataType.FLOAT, ("B", 30))
    ratio = V("ratio", ir.DataType.FLOAT, ())
    y = V("y", ir.DataType.FLOAT, ("B", 10))

    a = V("a", ir.DataType.FLOAT, ("B", 20))
    b = V("b", ir.DataType.FLOAT, ("B", 20))
    d = V("d", ir.DataType.FLOAT, ("B", 20))
    g = V("g", ir.DataType.FLOAT, ("B", 20))

    n1 = ir.Node(op_type="Gemm", domain="", inputs=[x], outputs=[a], name="Gemm_1")
    n2 = ir.Node(
        op_type="BatchNormalization", domain="", inputs=[a], outputs=[b], name="BN_1"
    )
    n3 = ir.Node(
        op_type="Dropout", domain="", inputs=[b, ratio], outputs=[d], name="Drop_1"
    )
    n4 = ir.Node(op_type="Gelu", domain="", inputs=[d], outputs=[g], name="Gelu_1")
    n5 = ir.Node(op_type="Gemm", domain="", inputs=[g], outputs=[y], name="Gemm_2")

    gr = ir.Graph(
        name="top", inputs=[x, ratio], outputs=[y], nodes=[n1, n2, n3, n4, n5]
    )
    m = ir.Model(graph=gr, ir_version=10)
    try:
        m.opset_imports = {"": 21}
    except Exception:
        pass
    return m


def build_chain_with_dangling_input():
    m = build_static_chain()
    # add a dangling input 'deterministic' (robust across onnx_ir variants)
    det = V("deterministic", ir.DataType.BOOL, ())
    for attr in ("inputs", "input"):
        arr = getattr(m.graph, attr, None)
        if arr is None:
            continue
        # Try replace with a new list
        try:
            lst = list(arr)
            lst.append(det)
            setattr(m.graph, attr, lst)
            break
        except Exception:
            # Try in-place append if container is mutable
            try:
                arr.append(det)  # type: ignore[attr-defined]
                break
            except Exception:
                continue
    return m


def test_static_path_with_shapes_and_symbols_and_no_unused():
    m = build_static_chain(B=3)
    check = EG(
        [
            "Gemm:Bx20 -> BatchNormalization:Bx20 -> Dropout:Bx20 -> Gelu:Bx20 -> Gemm:Bx10"
        ],
        symbols={"B": None},
        must_absent=["Not"],
        no_unused_inputs=True,
    )
    assert check(m)


def test_dynamic_unknown_batch_via_question_mark():
    m = build_dynamic_chain()
    check = EG(
        [
            "Gemm:?x20 -> BatchNormalization:?x20 -> Dropout:?x20 -> Gelu:?x20 -> Gemm:?x10"
        ],
        no_unused_inputs=True,
    )
    assert check(m)


def test_no_unused_inputs_catches_dangling():
    m = build_chain_with_dangling_input()
    check = EG(
        [
            "Gemm:Bx20 -> BatchNormalization:Bx20 -> Dropout:Bx20 -> Gelu:Bx20 -> Gemm:Bx10"
        ],
        symbols={"B": None},
        no_unused_inputs=True,
    )
    assert not check(m)  # should fail because of dangling 'deterministic'


def test_function_body_search_matches():
    # top graph: trivial pass-through
    xi = V("xi", ir.DataType.FLOAT, (2, 2))
    xo = V("xo", ir.DataType.FLOAT, (2, 2))
    passt = ir.Node(op_type="Identity", domain="", inputs=[xi], outputs=[xo], name="Id")
    top = ir.Graph(name="top", inputs=[xi], outputs=[xo], nodes=[passt])
    m = ir.Model(graph=top, ir_version=10)

    # function body graph
    a = V("a", ir.DataType.FLOAT, (2, 2))
    b1 = V("b1", ir.DataType.FLOAT, (2, 2))
    c = V("c", ir.DataType.FLOAT, (2, 2))
    b2 = V("b2", ir.DataType.FLOAT, (2, 2))
    r1 = ir.Node(op_type="Reshape", domain="", inputs=[a], outputs=[b1], name="R1")
    ge = ir.Node(op_type="Gelu", domain="", inputs=[b1], outputs=[c], name="G")
    r2 = ir.Node(op_type="Reshape", domain="", inputs=[c], outputs=[b2], name="R2")
    fgraph = ir.Graph(name="fn_g", inputs=[a], outputs=[b2], nodes=[r1, ge, r2])

    class _Fn:
        pass

    fn = _Fn()
    fn.domain = "custom"
    fn.name = "mlp_body"
    fn.graph = fgraph

    # Attach to the model in a way that's compatible with different onnx_ir builds:
    # - If there's a writable "functions" container (dict or list), use it.
    # - Otherwise, set the private "_functions" attribute (GraphView reads that too).
    attached = False
    cont = getattr(m, "functions", None)
    try:
        if isinstance(cont, dict):
            cont[
                (getattr(fn, "domain", "") or "", getattr(fn, "name", "") or "", "")
            ] = fn
            attached = True
        elif isinstance(cont, list):
            cont.append(fn)
            attached = True
    except Exception:
        pass
    if not attached:
        try:
            setattr(m, "_functions", [fn])
            attached = True
        except Exception:
            pass
    assert attached, "Could not attach function body to the test Model"

    check = EG(
        ["Reshape -> Gelu -> Reshape"],  # will be found in the function body
        must_absent=["Not"],
        search_functions=True,
    )
    assert check(m)


def test_strict_symbols_reject_unknown_dims():
    # Build a chain where one edge has unknown batch (?x20)
    m = build_dynamic_chain()
    # Artificially drop shape on the Dropout->Gelu edge to simulate unknown:
    g = m.graph
    drop = next(
        (
            n
            for n in getattr(g, "nodes", getattr(g, "_nodes", []))
            if getattr(n, "op_type", "") == "Dropout"
        ),
        None,
    )
    if drop is not None:
        try:
            dout = drop.outputs[0]
            if hasattr(dout, "shape"):
                dout.shape = ir.Shape((None, 20))
        except Exception:
            pass
    check = EG(
        [
            "Gemm:Bx20 -> BatchNormalization:Bx20 -> Dropout:Bx20 -> Gelu:Bx20 -> Gemm:Bx10"
        ],
        symbols={"B": None},
        no_unused_inputs=True,
    )
    assert not check(m)  # must fail because B requires a concrete dim, not None


def test_must_absent_ignores_unreachable_nodes():
    """
    Build a valid static chain, then append a *dangling* Not node whose output
    is never used. `must_absent=["Not"]` should still pass, because the checker
    must consider only nodes reachable from graph outputs.
    """
    m = build_static_chain(B=3)

    # Grab graph references
    g = m.graph
    nodes = list(getattr(g, "nodes", getattr(g, "_nodes", [])))

    # Find the existing 'ratio' input to use as Not's input (type mismatch is fine; we don't execute)
    ratio_in = next(
        (
            vi
            for vi in getattr(g, "inputs", getattr(g, "input", []))
            if getattr(vi, "name", "") == "ratio"
        ),
        None,
    )
    assert ratio_in is not None

    # Create a Not node whose output is not consumed by anyone (dangling)
    dangling_out = V("dangling_not_out", ir.DataType.BOOL, ())
    not_node = ir.Node(
        op_type="Not",
        domain="",
        inputs=[ratio_in],
        outputs=[dangling_out],
        name="DanglingNot",
    )
    nodes.append(not_node)

    # Persist node list back to the graph
    if hasattr(g, "nodes"):
        g.nodes = nodes
    elif hasattr(g, "_nodes"):
        g._nodes = nodes
    else:
        # Last resort: try 'node'
        try:
            g.node[:] = nodes
        except Exception:
            pass

    # The main path and shapes are correct; 'Not' is unreachable → should not trip must_absent
    check = EG(
        [
            "Gemm:Bx20 -> BatchNormalization:Bx20 -> Dropout:Bx20 -> Gelu:Bx20 -> Gemm:Bx10"
        ],
        symbols={"B": None},
        must_absent=["Not"],
        no_unused_inputs=True,
    )
    assert check(m)
