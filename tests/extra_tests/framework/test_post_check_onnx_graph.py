# tests/extra_tests/framework/test_post_check_onnx_graph.py

from __future__ import annotations

import json

import numpy as np
import onnx_ir as ir
import pytest

from jax2onnx.plugins._post_check_onnx_graph import (
    expect_graph as EG,
    auto_expect_graph_spec,
    expect_graph_from_spec,
    expect_graph_from_file,
)


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


def build_branchy_transpose_reshape():
    x = V("x", ir.DataType.FLOAT, (3, 28, 28, 1))
    t = V("t", ir.DataType.FLOAT, (3, 1, 28, 28))
    shape_out = V("shape", ir.DataType.INT64, (4,))
    y = V("y", ir.DataType.FLOAT, (3, 3136))

    n1 = ir.Node(op_type="Transpose", domain="", inputs=[x], outputs=[t], name="T")
    n2 = ir.Node(op_type="Shape", domain="", inputs=[t], outputs=[shape_out], name="S")
    n3 = ir.Node(
        op_type="Reshape", domain="", inputs=[t, shape_out], outputs=[y], name="R"
    )

    gr = ir.Graph(name="top", inputs=[x], outputs=[y], nodes=[n1, n2, n3])
    m = ir.Model(graph=gr, ir_version=10)
    try:
        m.opset_imports = {"": 21}
    except Exception:
        pass
    return m


def attach_function(model: ir.Model, graph: ir.Graph, *, domain: str, name: str):
    class _Fn:
        domain: str
        name: str
        graph: ir.Graph

    fn = _Fn()
    fn.domain = domain
    fn.name = name
    fn.graph = graph

    cont = getattr(model, "functions", None)
    attached = False
    try:
        if isinstance(cont, dict):
            cont[(domain, name, "")] = fn
            attached = True
        elif isinstance(cont, list):
            cont.append(fn)
            attached = True
    except Exception:
        attached = False

    if not attached:
        existing = getattr(model, "_functions", None)
        if isinstance(existing, list):
            try:
                existing.append(fn)
            except Exception:
                setattr(model, "_functions", list(existing) + [fn])
        else:
            setattr(model, "_functions", [fn])

    return fn


def build_dropout_like_graph(ratio=0.5, training=False, use_expand=True):
    x = V("x", ir.DataType.FLOAT, ("B", 64))
    ratio_scalar = V("ratio_scalar", ir.DataType.FLOAT, ())
    ratio_scalar.const_value = np.asarray(ratio, dtype=np.float32)
    ratio_shape = V("ratio_shape", ir.DataType.INT64, (1,))
    ratio_shape.const_value = np.asarray([64], dtype=np.int64)
    ratio_expanded = V("ratio", ir.DataType.FLOAT, (64,))
    training_val = V("training", ir.DataType.BOOL, ())
    training_val.const_value = np.asarray(training, dtype=np.bool_)
    out = V("out", ir.DataType.FLOAT, ("B", 64))

    nodes = []
    if use_expand:
        nodes.append(
            ir.Node(
                op_type="Expand",
                domain="",
                inputs=[ratio_scalar, ratio_shape],
                outputs=[ratio_expanded],
                name="Expand",
            )
        )
        ratio_input = ratio_expanded
    else:
        ratio_input = ratio_scalar

    nodes.append(
        ir.Node(
            op_type="Dropout",
            domain="",
            inputs=[x, ratio_input, training_val],
            outputs=[out],
            name="Dropout",
        )
    )

    graph = ir.Graph(name="top", inputs=[x], outputs=[out], nodes=nodes)
    model = ir.Model(graph=graph, ir_version=10)
    try:
        model.opset_imports = {"": 21}
    except Exception:
        pass
    return model


def build_attributed_chain(model_format: str) -> ir.Model | object:
    """Transpose -> LeakyRelu -> Pow -> Mul -> Softmax with typed attributes.

    Pow takes 2.7 through a Constant node and a narrowing Cast to INT64, and Mul
    takes 0.5 from a Constant node. ``model_format="protobuf"`` returns the
    serialized ``ModelProto``, which is what the test harness checks.
    """
    x = V("x", ir.DataType.FLOAT, (3, 4))
    transposed = V("transposed", ir.DataType.FLOAT, (4, 3))
    activated = V("activated", ir.DataType.FLOAT, (4, 3))
    exponent_raw = V("exponent_raw", ir.DataType.DOUBLE, ())
    exponent = V("exponent", ir.DataType.INT64, ())
    powered = V("powered", ir.DataType.FLOAT, (4, 3))
    scale = V("scale", ir.DataType.FLOAT, ())
    scaled = V("scaled", ir.DataType.FLOAT, (4, 3))
    y = V("y", ir.DataType.FLOAT, (4, 3))

    def constant(value: np.ndarray, output: ir.Value, name: str) -> ir.Node:
        return ir.Node(
            "",
            "Constant",
            inputs=[],
            outputs=[output],
            name=name,
            attributes=[ir.Attr("value", ir.AttributeType.TENSOR, ir.tensor(value))],
        )

    nodes = [
        ir.Node(
            "",
            "Transpose",
            inputs=[x],
            outputs=[transposed],
            name="T",
            attributes=[ir.Attr("perm", ir.AttributeType.INTS, [1, 0])],
        ),
        ir.Node(
            "",
            "LeakyRelu",
            inputs=[transposed],
            outputs=[activated],
            name="L",
            attributes=[ir.Attr("alpha", ir.AttributeType.FLOAT, 0.2)],
        ),
        constant(np.asarray(2.7, dtype=np.float64), exponent_raw, "C_exp"),
        ir.Node(
            "",
            "Cast",
            inputs=[exponent_raw],
            outputs=[exponent],
            name="Cast_exp",
            attributes=[
                ir.Attr("to", ir.AttributeType.INT, int(ir.DataType.INT64.value))
            ],
        ),
        ir.Node("", "Pow", inputs=[activated, exponent], outputs=[powered], name="P"),
        constant(np.asarray(0.5, dtype=np.float32), scale, "C_scale"),
        ir.Node("", "Mul", inputs=[powered, scale], outputs=[scaled], name="M"),
        ir.Node(
            "",
            "Softmax",
            inputs=[scaled],
            outputs=[y],
            name="S",
            attributes=[ir.Attr("axis", ir.AttributeType.INT, 1)],
        ),
    ]
    graph = ir.Graph(
        name="top", inputs=[x], outputs=[y], nodes=nodes, opset_imports={"": 21}
    )
    model = ir.Model(graph=graph, ir_version=10)
    if model_format == "protobuf":
        return ir.serde.serialize_model(model)
    return model


MODEL_FORMATS = pytest.mark.parametrize("model_format", ["onnx_ir", "protobuf"])


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


def test_path_walks_over_shape_side_chain():
    m = build_branchy_transpose_reshape()
    check = EG(["Transpose -> Reshape"])
    assert check(m)


def test_inputs_predicate_matches_constant_after_expand():
    m = build_dropout_like_graph(ratio=0.5, training=False, use_expand=True)
    check = EG(
        [
            {
                "path": "Dropout",
                "inputs": {1: {"const": 0.5}, 2: {"const_bool": False}},
            }
        ]
    )
    assert check(m)


def test_inputs_predicate_matches_constant_after_cast():
    x = V("x", ir.DataType.FLOAT, (2,))
    ratio_raw = V("ratio_raw", ir.DataType.DOUBLE, ())
    ratio_raw.const_value = np.asarray(0.5, dtype=np.float64)
    ratio = V("ratio", ir.DataType.FLOAT, ())
    y = V("y", ir.DataType.FLOAT, (2,))

    cast = ir.Node(
        op_type="Cast",
        domain="",
        inputs=[ratio_raw],
        outputs=[ratio],
        name="CastRatio",
        attributes=[ir.Attr("to", ir.AttributeType.INT, int(ir.DataType.FLOAT.value))],
    )
    dropout = ir.Node(
        op_type="Dropout",
        domain="",
        inputs=[x, ratio],
        outputs=[y],
        name="Dropout",
    )
    graph = ir.Graph(name="top", inputs=[x], outputs=[y], nodes=[cast, dropout])
    model = ir.Model(graph=graph, ir_version=10)

    check = EG([{"path": "Dropout", "inputs": {1: {"const": 0.5}}}])

    assert check(model)


def test_inputs_predicate_const_mismatch():
    m = build_dropout_like_graph(ratio=0.3, training=False, use_expand=True)
    check = EG(
        [
            {
                "path": "Dropout",
                "inputs": {1: {"const": 0.5}},
            }
        ]
    )
    assert not check(m)


@MODEL_FORMATS
@pytest.mark.parametrize(
    ("attrs", "matches"),
    [
        ({"Softmax": {"axis": 1}}, True),
        ({"Softmax": {"axis": 0}}, False),
        ({"Transpose": {"perm": [1, 0]}}, True),
        ({"Transpose": {"perm": (0, 1)}}, False),
        # FLOAT attributes hold single precision, so 0.2 must still match.
        ({"LeakyRelu": {"alpha": 0.2}}, True),
        ({"LeakyRelu": {"alpha": 0.3}}, False),
        # Attributes left at their ONNX default are not set on the node.
        ({"Softmax": {"epsilon": 1}}, False),
    ],
)
def test_attrs_predicate_reads_node_attributes(model_format, attrs, matches):
    check = EG(
        [{"path": "Transpose -> LeakyRelu", "attrs": attrs}], explain_on_fail=False
    )
    assert check(build_attributed_chain(model_format)) == matches


@MODEL_FORMATS
def test_const_predicate_reads_constant_node_payload(model_format):
    model = build_attributed_chain(model_format)
    assert EG([{"path": "Mul", "inputs": {1: {"const": 0.5}}}])(model)
    assert not EG(
        [{"path": "Mul", "inputs": {1: {"const": 0.25}}}], explain_on_fail=False
    )(model)


@MODEL_FORMATS
def test_const_predicate_applies_cast_target_dtype(model_format):
    model = build_attributed_chain(model_format)
    # 2.7 cast to INT64 is 2, so only the cast value matches.
    assert EG([{"path": "Pow", "inputs": {1: {"const": 2}}}])(model)
    assert not EG(
        [{"path": "Pow", "inputs": {1: {"const": 2.7}}}], explain_on_fail=False
    )(model)


def test_auto_expect_graph_spec_roundtrip_static():
    model = build_static_chain(B=3)
    spec = auto_expect_graph_spec(model)
    check = expect_graph_from_spec(spec)
    assert check(model)


def test_expect_graph_from_file_roundtrip(tmp_path):
    model = build_dynamic_chain()
    spec = auto_expect_graph_spec(model)
    dest = tmp_path / "spec.json"
    dest.write_text(json.dumps(spec, indent=2))
    check = expect_graph_from_file(str(dest))
    assert check(model)


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


def test_no_unused_function_inputs_detects_dangling():
    xi = V("xi", ir.DataType.FLOAT, (2, 2))
    xo = V("xo", ir.DataType.FLOAT, (2, 2))
    passt = ir.Node(op_type="Identity", domain="", inputs=[xi], outputs=[xo], name="Id")
    top = ir.Graph(name="top", inputs=[xi], outputs=[xo], nodes=[passt])
    m = ir.Model(graph=top, ir_version=10)

    a = V("a", ir.DataType.FLOAT, (2, 2))
    det = V("deterministic", ir.DataType.BOOL, ())
    b = V("b", ir.DataType.FLOAT, (2, 2))
    body = ir.Node(op_type="Identity", domain="", inputs=[a], outputs=[b], name="Body")
    fgraph = ir.Graph(name="fn_body", inputs=[a, det], outputs=[b], nodes=[body])

    class _Fn:
        pass

    fn = _Fn()
    fn.domain = "custom"
    fn.name = "has_unused_input"
    fn.graph = fgraph

    attached = False
    cont = getattr(m, "functions", None)
    try:
        if isinstance(cont, dict):
            cont[(getattr(fn, "domain", ""), getattr(fn, "name", ""), "")] = fn
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
        [],
        search_functions=True,
        no_unused_function_inputs=True,
    )
    assert not check(m)


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


def _build_position_embedding_graph(name: str, *, with_cast: bool) -> ir.Graph:
    table = V(f"{name}_table", ir.DataType.FLOAT, (1, 4))
    start = V(f"{name}_start", ir.DataType.INT64, ())
    limit = V(f"{name}_limit", ir.DataType.INT64, ())
    delta = V(f"{name}_delta", ir.DataType.INT64, ())
    axes = V(f"{name}_axes", ir.DataType.INT64, (1,))
    expand_shape = V(f"{name}_expand_shape", ir.DataType.INT64, (2,))

    range_out = V(f"{name}_range", ir.DataType.INT64, (4,))
    unsqueeze_out = V(f"{name}_unsq", ir.DataType.INT64, (1, 4))
    expanded = V(f"{name}_expanded", ir.DataType.INT64, (1, 4))
    gather_indices = expanded

    nodes = [
        ir.Node(
            op_type="Range",
            domain="",
            inputs=[start, limit, delta],
            outputs=[range_out],
            name=f"{name}_Range",
        ),
        ir.Node(
            op_type="Unsqueeze",
            domain="",
            inputs=[range_out, axes],
            outputs=[unsqueeze_out],
            name=f"{name}_Unsqueeze",
        ),
        ir.Node(
            op_type="Expand",
            domain="",
            inputs=[unsqueeze_out, expand_shape],
            outputs=[expanded],
            name=f"{name}_Expand",
        ),
    ]

    if with_cast:
        cast_out = V(f"{name}_cast", ir.DataType.INT64, (1, 4))
        nodes.append(
            ir.Node(
                op_type="Cast",
                domain="",
                inputs=[expanded],
                outputs=[cast_out],
                name=f"{name}_Cast",
            )
        )
        gather_indices = cast_out

    gather_out = V(f"{name}_out", ir.DataType.FLOAT, (1, 4))
    nodes.append(
        ir.Node(
            op_type="Gather",
            domain="",
            inputs=[table, gather_indices],
            outputs=[gather_out],
            name=f"{name}_Gather",
        )
    )

    inputs = [table, start, limit, delta, axes, expand_shape]
    return ir.Graph(name=name, inputs=inputs, outputs=[gather_out], nodes=nodes)


def test_graph_scoped_path_matches_specific_function():
    xi = V("xi", ir.DataType.FLOAT, (2, 2))
    xo = V("xo", ir.DataType.FLOAT, (2, 2))
    passt = ir.Node(op_type="Identity", domain="", inputs=[xi], outputs=[xo], name="Id")
    top = ir.Graph(name="top", inputs=[xi], outputs=[xo], nodes=[passt])
    m = ir.Model(graph=top, ir_version=10)

    clean_graph = _build_position_embedding_graph(
        "PositionEmbedding_clean", with_cast=False
    )
    dirty_graph = _build_position_embedding_graph(
        "PositionEmbedding_dirty", with_cast=True
    )

    attach_function(m, clean_graph, domain="custom", name="PositionEmbedding_clean")
    attach_function(m, dirty_graph, domain="custom", name="PositionEmbedding_dirty")

    check = EG(
        [
            {
                "graph": "custom:PositionEmbedding_clean",
                "path": "Range -> Unsqueeze -> Expand -> Gather",
                "must_absent": ["Cast"],
            }
        ],
        search_functions=True,
    )
    assert check(m)

    check_dirty = EG(
        [
            {
                "graph": "custom:PositionEmbedding_dirty",
                "path": "Range -> Unsqueeze -> Expand -> Gather",
                "must_absent": ["Cast"],
            }
        ],
        search_functions=True,
    )
    assert not check_dirty(m)


def test_graph_filter_reports_missing_function():
    m = build_static_chain(B=2)
    check = EG(
        [
            {
                "graph": "custom:does_not_exist",
                "path": "Gemm -> BatchNormalization",
            }
        ],
        search_functions=True,
    )
    assert not check(m)
