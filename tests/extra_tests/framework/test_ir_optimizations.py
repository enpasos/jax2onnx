# tests/extra_tests/framework/test_ir_optimizations.py

from __future__ import annotations

import numpy as np
import onnx_ir as ir

# --------- Integration test for constant Not removal (current) ---------


def build_graph_with_not_tm():
    # graph IO
    x = ir.val("x", ir.DataType.FLOAT, (3, 30))
    ratio = ir.val("ratio", ir.DataType.FLOAT, ())
    y = ir.val("y", ir.DataType.FLOAT, (3, 10))

    # keep a dangling graph input 'deterministic' so prune pass can remove it
    det = ir.val("deterministic", ir.DataType.BOOL, ())

    # intermediates
    a = ir.val("after_gemm", ir.DataType.FLOAT, (3, 20))
    b = ir.val("after_bn", ir.DataType.FLOAT, (3, 20))
    not_out = ir.val("not_out", ir.DataType.BOOL, ())
    d_out = ir.val("drop_out", ir.DataType.FLOAT, (3, 20))
    g_out = ir.val("gelu_out", ir.DataType.FLOAT, (3, 20))

    # Constant True for training-mode corridor, so Not(True) → can be inlined.
    # Make the scalar readable in a build-agnostic way: attach it directly to
    # the Value via `const_value`. The optimizer always checks this first.
    const_true = ir.val("const_true", ir.DataType.BOOL, ())
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


def test_dropout_training_mode_inlined_constant_false_and_not_removed():
    from jax2onnx.converter.ir_optimizations import optimize_graph

    m = build_graph_with_not_tm()
    m = optimize_graph(m)
    g = m.graph
    nodes = list(g)

    # Not must be gone
    assert "Not" not in [n.op_type for n in nodes]

    # Dropout must remain and its 3rd input must be "missing" (empty name)
    drops = [n for n in nodes if n.op_type == "Dropout"]
    assert len(drops) == 1
    d = drops[0]
    # read inputs from either .inputs or .input
    ins = d.inputs
    # If .input stores names instead of Values, normalize to names only
    if ins and isinstance(ins[0], str):
        third_name = ins[2]
    else:
        third = ins[2]
        third_name = third.name if third else ""
    assert third_name == "false_const", f"expected missing tm input, got {third_name!r}"

    # Unused graph input 'deterministic' must be pruned; 'x' and 'ratio' must remain
    in_names = {v.name for v in g.inputs}
    assert "deterministic" not in in_names
    assert "x" in in_names
    assert "ratio" in in_names


def test_prune_unused_input_not_kept_due_to_nested_graph_name_collision():
    from jax2onnx.converter.ir_optimizations import optimize_graph

    top_in = ir.val("in_0", ir.DataType.FLOAT, (2, 4))
    det_top = ir.val("deterministic", ir.DataType.BOOL, ())
    top_out = ir.val("out", ir.DataType.FLOAT, (2, 4))

    inner_data = ir.val("payload", ir.DataType.FLOAT, (2, 4))
    inner_det = ir.val("deterministic", ir.DataType.BOOL, ())
    inner_out = ir.val("inner_out", ir.DataType.FLOAT, (2, 4))
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
    input_names = {v.name for v in optimized.graph.inputs}

    assert "deterministic" not in input_names
    assert "in_0" in input_names


def _cast_attr(dtype: ir.DataType) -> ir.Attr:
    return ir.Attr("to", ir.AttributeType.INT, int(dtype.value))


def build_graph_with_identity_cast(dtype: ir.DataType = ir.DataType.FLOAT):
    x = ir.val("x", dtype, (2,))
    cast_out = ir.val("x_cast", dtype, (2,))
    relu_out = ir.val("y", dtype, (2,))

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
    from jax2onnx.converter.ir_optimizations import optimize_graph

    m = build_graph_with_identity_cast()
    m = optimize_graph(m)
    g = m.graph
    nodes = list(g)
    assert [n.op_type for n in nodes] == ["Relu"]
    relu = nodes[0]
    relu_inputs = relu.inputs
    assert (
        relu_inputs
        and (relu_inputs[0].name if hasattr(relu_inputs[0], "name") else relu_inputs[0])
        == "x"
    )


def test_identity_cast_removed_inside_function_body():
    from jax2onnx.converter.ir_optimizations import optimize_graph

    inner_model = build_graph_with_identity_cast()
    top_in = ir.val("top_in", ir.DataType.FLOAT, (2,))
    top_out = ir.val("top_out", ir.DataType.FLOAT, (2,))
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
    # Correctly adding to functions list
    if model.functions is None:
        model.functions = []

    if isinstance(model.functions, list):
        model.functions.append(fn)
    elif isinstance(model.functions, dict):
        model.functions[(fn.domain, fn.name, "")] = fn

    optimized = optimize_graph(model)
    funcs = optimized.functions

    if isinstance(funcs, dict):
        fn_graph = next(iter(funcs.values())).graph
    elif isinstance(funcs, list):
        fn_graph = funcs[0].graph
    else:
        raise AssertionError("optimized model has no function registry")
    fn_nodes = list(fn_graph)
    assert [n.op_type for n in fn_nodes] == ["Relu"]


def test_identity_reshape_removed_when_target_matches_source():
    from jax2onnx.converter.ir_optimizations import optimize_graph

    data = ir.val("in", ir.DataType.FLOAT, (3, 4))
    shape_tensor = ir.tensor(np.asarray([3, 4], dtype=np.int64))
    shape_val = ir.Value(
        name="shape",
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape((2,)),
        const_value=shape_tensor,
    )
    out_val = ir.val("out", ir.DataType.FLOAT, (3, 4))
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
    nodes = list(optimized.graph)
    assert all(n.op_type != "Reshape" for n in nodes)
    out_names = {v.name for v in optimized.graph.outputs}
    assert "in" in out_names


def test_cse_simple():
    from jax2onnx.converter.ir_optimizations import optimize_graph

    data = ir.val("in", ir.DataType.FLOAT, (3, 4))

    # Branch 1
    out1 = ir.val("out1", ir.DataType.FLOAT, (3, 4))
    node1 = ir.Node(
        op_type="Relu",
        domain="",
        inputs=[data],
        outputs=[out1],
        name="Relu1",
    )

    # Branch 2 (identical to Branch 1)
    out2 = ir.val("out2", ir.DataType.FLOAT, (3, 4))
    node2 = ir.Node(
        op_type="Relu",
        domain="",
        inputs=[data],  # Same input object
        outputs=[out2],
        name="Relu2",
    )

    # Graph outputs BOTH
    graph = ir.Graph(
        name="cse_simple",
        inputs=[data],
        outputs=[out1, out2],
        nodes=[node1, node2],
    )

    model = ir.Model(graph=graph, ir_version=10)
    optimized = optimize_graph(model)

    nodes = list(optimized.graph)
    # Should be merged but graph outputs must remain distinct objects.
    # So we expect 1 Relu and 1 Identity (to alias the second output).
    assert len(nodes) == 2
    ops = sorted([n.op_type for n in nodes])
    assert ops == ["Identity", "Relu"]

    # Outputs must be distinct objects
    outs = optimized.graph.outputs
    assert len(outs) == 2
    assert outs[0] is not outs[1]


def test_lift_constants():
    from jax2onnx.converter.ir_optimizations import optimize_graph

    # Make a graph with a Constant node in the body
    out_const = ir.val("const_out", ir.DataType.FLOAT, (2,))
    const_node = ir.Node(
        op_type="Constant",
        domain="",
        inputs=[],
        outputs=[out_const],
        name="Const1",
        attributes={
            "value": ir.Attr(
                name="value",
                type=ir.AttributeType.TENSOR,
                value=ir.tensor(np.array([1.0, 2.0], dtype=np.float32)),
            )
        },
    )

    out_identity = ir.val("out", ir.DataType.FLOAT, (2,))
    id_node = ir.Node(
        op_type="Identity",
        domain="",
        inputs=[out_const],
        outputs=[out_identity],
        name="Identity1",
    )

    graph = ir.Graph(
        name="lift_const",
        inputs=[],
        outputs=[out_identity],
        nodes=[const_node, id_node],
    )

    model = ir.Model(graph=graph, ir_version=10)
    # Check before: no initializers
    assert len(graph.initializers) == 0

    optimized = optimize_graph(model)

    # Check after: Constant node gone, Identity inputs point to initializer
    nodes = list(optimized.graph)
    assert len(nodes) == 1
    assert nodes[0].op_type == "Identity"

    assert len(optimized.graph.initializers) == 1
    init_val = list(optimized.graph.initializers.values())[0]
    # Name should be preserved or match usage
    assert init_val.name == "const_out"
    assert init_val.const_value is not None
