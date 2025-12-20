# tests/extra_tests/framework/test_ir_optimizations.py

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
    _to_numpy_from_any,
    _as_scalar_bool,
    optimize_graph,
)

from onnx_ir import AttributeType as IRAttrType


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


def test_to_numpy_and_scalar_bool_from_tensor_and_attr():
    tensor = ir.tensor(np.asarray(True, dtype=np.bool_))
    arr = _to_numpy_from_any(tensor)
    assert arr is not None and arr.shape == () and arr.dtype == np.bool_
    assert bool(arr)
    attr_tensor = ir.Attr(name="value", type=IRAttrType.TENSOR, value=tensor)
    attr_arr = _to_numpy_from_any(attr_tensor)
    assert attr_arr is not None and bool(attr_arr)
    assert _as_scalar_bool(tensor) is True
    assert _as_scalar_bool(attr_tensor) is True


def test_literal_false_strings_roundtrip():
    arr = _to_numpy_from_any("false")
    assert arr is not None and arr.shape == () and arr.dtype == np.bool_
    assert bool(arr) is False
    attr_str = ir.Attr(name="value", type=IRAttrType.STRING, value="false")
    attr_arr = _to_numpy_from_any(attr_str)
    assert attr_arr is not None and bool(attr_arr) is False
    assert _as_scalar_bool(attr_str) is False


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
    assert third_name == "false_const", f"expected missing tm input, got {third_name!r}"

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


def _cast_attr(dtype: ir.DataType) -> ir.Attr:
    return ir.Attr("to", ir.AttributeType.INT, int(dtype.value))


def build_graph_with_identity_cast(dtype: ir.DataType = ir.DataType.FLOAT):
    x = V_ir("x", dtype, (2,))
    cast_out = V_ir("x_cast", dtype, (2,))
    relu_out = V_ir("y", dtype, (2,))

    cast_node = ir.Node(
        op_type="Cast",
        domain="",
        inputs=[x],
        outputs=[cast_out],
        name="Cast_identity",
        attributes=[_cast_attr(dtype)],
    )
    relu_node = ir.Node(
        op_type="Relu",
        domain="",
        inputs=[cast_out],
        outputs=[relu_out],
        name="Relu_after_cast",
    )

    g = ir.Graph(name="g", inputs=[x], outputs=[relu_out], nodes=[cast_node, relu_node])
    m = ir.Model(graph=g, ir_version=10)
    try:
        m.opset_imports = {"": 21}
    except Exception:
        pass
    return m


def test_identity_cast_removed_and_consumers_rewired():
    m = build_graph_with_identity_cast()
    m = optimize_graph(m)
    g = m.graph
    nodes = _nodes(g)
    assert [n.op_type for n in nodes] == ["Relu"]
    relu = nodes[0]
    relu_inputs = getattr(relu, "inputs", None) or getattr(relu, "input", [])
    assert relu_inputs and getattr(relu_inputs[0], "name", relu_inputs[0]) == "x"


def test_identity_cast_removed_inside_function_body():
    inner_model = build_graph_with_identity_cast()
    top_in = V_ir("top_in", ir.DataType.FLOAT, (2,))
    top_out = V_ir("top_out", ir.DataType.FLOAT, (2,))
    passthrough = ir.Node(
        op_type="Identity",
        domain="",
        inputs=[top_in],
        outputs=[top_out],
        name="TopIdentity",
    )
    top_graph = ir.Graph(
        name="top", inputs=[top_in], outputs=[top_out], nodes=[passthrough]
    )

    class Fn:
        pass

    fn = Fn()
    fn.domain = "custom"
    fn.name = "identity_cast"
    fn.graph = inner_model.graph

    model = ir.Model(graph=top_graph, ir_version=10)
    attached = False
    cont = getattr(model, "functions", None)
    try:
        if isinstance(cont, list):
            cont.append(fn)
            attached = True
        elif isinstance(cont, dict):
            cont[(fn.domain, fn.name, "")] = fn
            attached = True
    except Exception:
        attached = False
    if not attached:
        try:
            existing = getattr(model, "_functions", None)
            if isinstance(existing, list):
                existing.append(fn)
            else:
                setattr(model, "_functions", [fn])
            attached = True
        except Exception:
            attached = False
    assert attached, "Could not attach function to test model"
    try:
        model.opset_imports = {"": 21}
    except Exception:
        pass

    optimized = optimize_graph(model)
    funcs = getattr(optimized, "functions", None) or getattr(
        optimized, "_functions", None
    )
    if isinstance(funcs, dict):
        fn_graph = next(iter(funcs.values())).graph
    elif isinstance(funcs, list):
        fn_graph = funcs[0].graph
    else:
        raise AssertionError("optimized model has no function registry")
    fn_nodes = _nodes(fn_graph)
    assert [n.op_type for n in fn_nodes] == ["Relu"]


def test_identity_reshape_removed_when_target_matches_source():
    data = V_ir("in", ir.DataType.FLOAT, (3, 4))
    shape_tensor = ir.tensor(np.asarray([3, 4], dtype=np.int64))
    shape_val = ir.Value(
        name="shape",
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape((2,)),
        const_value=shape_tensor,
    )
    out_val = V_ir("out", ir.DataType.FLOAT, (3, 4))
    reshape = ir.Node(
        op_type="Reshape",
        domain="",
        inputs=[data, shape_val],
        outputs=[out_val],
        name="Reshape_identity",
    )
    graph = ir.Graph(
        name="reshape_identity",
        inputs=[data],
        outputs=[out_val],
        nodes=[reshape],
        initializers=[shape_val],
    )
    model = ir.Model(graph=graph, ir_version=10)
    optimized = optimize_graph(model)
    nodes = _nodes(optimized.graph)
    assert all(n.op_type != "Reshape" for n in nodes)
    out_names = {getattr(v, "name", "") for v in optimized.graph.outputs}
    assert "in" in out_names
