# tests/extra_tests/test_ir_optimizations.py
from __future__ import annotations

import numpy as np
import onnx_ir as ir

# --------- Unit tests for helper functions (restored) ---------

# import the functions to test
from jax2onnx.converter.ir_optimizations import (
    _is_elem,
    _get_perm_attr,
    _perms_compose_identity,
    _has_input_name_or_obj,
    _count_consumers,
    _find_next_consumer_idx,
    optimize_graph,
)


class V:
    def __init__(self, name=None):
        self.name = name


class N:
    def __init__(self, op, inputs=(), outputs=(), attributes=()):
        self.op_type = op
        # mirror the onnx_ir shapes: tests use .inputs/.outputs lists of V
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        # attributes is a list of Attr
        self.attributes = list(attributes)


class Attr:
    def __init__(self, name, ints):
        self.name = name
        self.ints = list(ints)


def test_is_elem_lower_and_mixed():
    assert _is_elem("Relu")
    assert _is_elem("relu")
    assert _is_elem("Cast")
    assert _is_elem("castlike")
    assert not _is_elem("AveragePool")


def test_get_perm_attr_and_identity():
    t1 = N("Transpose", attributes=[Attr("perm", [0, 3, 1, 2])])
    t2 = N("Transpose", attributes=[Attr("perm", [0, 2, 3, 1])])
    p1 = _get_perm_attr(t1)
    p2 = _get_perm_attr(t2)
    assert p1 == [0, 3, 1, 2] and p2 == [0, 2, 3, 1]
    assert _perms_compose_identity(p1, p2)


def test_match_by_name_or_obj():
    a = V("a")
    b = V("b")
    n = N("Relu", inputs=[a])
    assert _has_input_name_or_obj(n, "a", None)
    assert _has_input_name_or_obj(n, None, a)
    assert not _has_input_name_or_obj(n, "b", None)
    assert not _has_input_name_or_obj(n, None, b)


def test_consumer_scan():
    v = V("x")
    nodes = [N("Transpose", outputs=[v]), N("Something"), N("Relu", inputs=[v])]
    assert _find_next_consumer_idx(nodes, 0, "x", v) == 2
    assert _count_consumers(nodes, "x", v) == 1


# --------- Integration test for constant Not removal (current) ---------


def V_ir(name, dtype=ir.DataType.FLOAT, shape=()):
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))


def build_graph_with_not_tm():
    # graph IO
    x = V_ir("x", ir.DataType.FLOAT, (3, 30))
    ratio = V_ir("ratio", ir.DataType.FLOAT, ())
    y = V_ir("y", ir.DataType.FLOAT, (3, 10))

    # keep a dangling graph input 'deterministic' so prune pass can remove it
    det = V_ir("deterministic", ir.DataType.BOOL, ())

    # intermediates
    a = V_ir("after_gemm", ir.DataType.FLOAT, (3, 20))
    b = V_ir("after_bn", ir.DataType.FLOAT, (3, 20))
    not_out = V_ir("not_out", ir.DataType.BOOL, ())
    d_out = V_ir("drop_out", ir.DataType.FLOAT, (3, 20))
    g_out = V_ir("gelu_out", ir.DataType.FLOAT, (3, 20))

    # Constant True for training-mode corridor, so Not(True) → can be inlined.
    # Make the scalar readable in a build-agnostic way: attach it directly to
    # the Value via `const_value`. The optimizer always checks this first.
    const_true = V_ir("const_true", ir.DataType.BOOL, ())
    # Attach constant payload directly; skip tricky Attr/Attributes handling.
    const_true.const_value = ir.tensor(np.asarray(True, dtype=np.bool_))
    const_node = ir.Node(
        op_type="Constant",
        domain="",
        inputs=[],
        outputs=[const_true],
        name="Const_true",
        attributes=[],  # payload is on Value.const_value
        num_outputs=1,
    )

    n1 = ir.Node(op_type="Gemm", domain="", inputs=[x], outputs=[a], name="Gemm_1")
    n2 = ir.Node(
        op_type="BatchNormalization", domain="", inputs=[a], outputs=[b], name="BN_1"
    )
    n3 = ir.Node(
        op_type="Not", domain="", inputs=[const_true], outputs=[not_out], name="Not_1"
    )
    n4 = ir.Node(
        op_type="Dropout",
        domain="",
        inputs=[b, ratio, not_out],
        outputs=[d_out],
        name="Drop_1",
    )
    n5 = ir.Node(
        op_type="Gelu", domain="", inputs=[d_out], outputs=[g_out], name="Gelu_1"
    )
    n6 = ir.Node(op_type="Gemm", domain="", inputs=[g_out], outputs=[y], name="Gemm_2")

    g = ir.Graph(
        name="g",
        inputs=[x, det, ratio],  # 'det' is intentionally unused so it can be pruned
        outputs=[y],
        nodes=[const_node, n1, n2, n3, n4, n5, n6],
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
    # read inputs from either .inputs or .input
    ins = getattr(d, "inputs", None)
    if ins is None:
        ins = getattr(d, "input", [])
    # If .input stores names instead of Values, normalize to names only
    if ins and isinstance(ins[0], str):
        third_name = ins[2]
    else:
        third = ins[2]
        third_name = getattr(third, "name", "")
    assert third_name == "", f"expected missing tm input, got {third_name!r}"

    # Unused graph input 'deterministic' must be pruned; 'x' and 'ratio' must remain
    in_names = {getattr(v, "name", "") for v in _inputs(g)}
    assert "deterministic" not in in_names
    assert "x" in in_names
    assert "ratio" in in_names


def test_prune_unused_input_not_kept_due_to_nested_graph_name_collision():
    top_in = V_ir("in_0", ir.DataType.FLOAT, (2, 4))
    det_top = V_ir("deterministic", ir.DataType.BOOL, ())
    top_out = V_ir("out", ir.DataType.FLOAT, (2, 4))

    inner_data = V_ir("payload", ir.DataType.FLOAT, (2, 4))
    inner_det = V_ir("deterministic", ir.DataType.BOOL, ())
    inner_out = V_ir("inner_out", ir.DataType.FLOAT, (2, 4))
    inner_node = ir.Node(
        op_type="Identity",
        domain="",
        inputs=[inner_det],
        outputs=[inner_out],
        name="InnerIdentity",
    )

    inner_graph = ir.Graph(
        name="inner_graph",
        inputs=[inner_data, inner_det],
        outputs=[inner_out],
        nodes=[inner_node],
    )

    call_node = ir.Node(
        op_type="CallInner",
        domain="",
        inputs=[top_in],
        outputs=[top_out],
        name="CallInner",
        attributes=[ir.Attr("body", ir.AttributeType.GRAPH, inner_graph)],
    )

    top_graph = ir.Graph(
        name="top_graph",
        inputs=[top_in, det_top],
        outputs=[top_out],
        nodes=[call_node],
    )

    model = ir.Model(graph=top_graph, ir_version=10)
    optimized = optimize_graph(model)
    input_names = {getattr(v, "name", "") for v in _inputs(optimized.graph)}

    assert "deterministic" not in input_names
    assert "in_0" in input_names
