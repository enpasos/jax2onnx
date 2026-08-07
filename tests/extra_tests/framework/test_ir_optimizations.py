# tests/extra_tests/framework/test_ir_optimizations.py

from __future__ import annotations

import numpy as np
import onnx
import onnx_ir as ir
import pytest
from numpy.typing import NDArray
from onnx_ir import serde

# --------- Unit tests for helper functions (restored) ---------

# import the functions to test
from jax2onnx.converter.ir_optimizations import (
    _cast_roundtrip_is_value_preserving,
    _get_perm_attr,
    _has_input_name_or_obj,
    _OPTIMIZER_PASSES,
    _to_numpy_from_any,
    _as_scalar_bool,
    optimize_graph,
)

from onnx_ir import AttributeType as IRAttrType


def test_optimizer_pass_registry_declares_order_and_function_scope():
    assert [opt_pass.name for opt_pass in _OPTIMIZER_PASSES] == [
        "name_fix",
        "remove_redundant_casts",
        "remove_redundant_transpose_reduce",
        "remove_redundant_transpose_add_forests",
        "remove_redundant_transpose_pairs",
        "remove_redundant_reshape_pairs",
        "remove_identity_reshapes",
        "common_subexpression_elimination",
        "lift_constants_to_initializers",
        "rewrite_mul_sigmoid_as_swish",
        "rewrite_mul_rsqrt_as_div",
        "inline_dropout_training_mode_constants",
        "propagate_elementwise_shapes",
        "propagate_unary_shapes",
        "remove_redundant_casts_after_propagation",
        "remove_dead_nodes",
        "remove_orphan_transposes",
        "prune_unused_graph_inputs",
    ]
    assert [
        opt_pass.name
        for opt_pass in _OPTIMIZER_PASSES
        if opt_pass.function_graph_runner is not None
    ] == [
        "remove_redundant_casts",
        "remove_redundant_transpose_reduce",
        "remove_redundant_transpose_add_forests",
        "remove_redundant_transpose_pairs",
        "remove_redundant_reshape_pairs",
        "remove_identity_reshapes",
        "rewrite_mul_sigmoid_as_swish",
        "rewrite_mul_rsqrt_as_div",
        "inline_dropout_training_mode_constants",
        "propagate_elementwise_shapes",
        "propagate_unary_shapes",
        "remove_redundant_casts_after_propagation",
        "remove_orphan_transposes",
    ]


def test_get_perm_attr_and_identity():
    # Real ir.Attr required
    t1 = ir.Node(
        "",
        "Transpose",
        [],
        attributes=[ir.Attr(name="perm", type=IRAttrType.INTS, value=(0, 3, 1, 2))],
    )
    t2 = ir.Node(
        "",
        "Transpose",
        [],
        attributes=[ir.Attr(name="perm", type=IRAttrType.INTS, value=(0, 2, 3, 1))],
    )
    p1 = _get_perm_attr(t1)
    p2 = _get_perm_attr(t2)
    assert p1 == [0, 3, 1, 2] and p2 == [0, 2, 3, 1]


def test_match_by_name_or_obj():
    a = ir.Value(name="a")
    b = ir.Value(name="b")
    n = ir.Node("", "Relu", inputs=[a])

    # We must properly connect logic if needed?
    # _has_input_name_or_obj checks _node_inputs(node).
    # ir.Node inputs are stored.

    assert _has_input_name_or_obj(n, "a", None)
    assert _has_input_name_or_obj(n, None, a)
    assert not _has_input_name_or_obj(n, "b", None)
    assert not _has_input_name_or_obj(n, None, b)


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

    const_out = ir.Value(name="const_out")
    ir.Node(
        op_type="Constant",
        domain="",
        inputs=[],
        outputs=[const_out],
        name="Const_true",
        attributes=[attr_tensor],
    )
    const_arr = _to_numpy_from_any(const_out)
    assert const_arr is not None and const_arr.shape == () and bool(const_arr)
    assert _as_scalar_bool(const_out) is True


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
    m.opset_imports[""] = 21
    return m


def test_dropout_training_mode_inlined_constant_false_and_not_removed():
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
    ins = d.inputs
    # If .input stores names instead of Values, normalize to names only
    if ins and isinstance(ins[0], str):
        third_name = ins[2]
    else:
        third = ins[2]
        third_name = third.name
    assert third_name == "false_const", f"expected missing tm input, got {third_name!r}"

    # Unused graph input 'deterministic' must be pruned; 'x' and 'ratio' must remain
    in_names = {v.name for v in g.inputs}
    assert "deterministic" not in in_names
    assert "x" in in_names
    assert "ratio" in in_names


def test_prune_unused_input_not_kept_due_to_nested_graph_name_collision():
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


@pytest.mark.parametrize(
    ("source", "intermediate", "expected"),
    [
        (ir.DataType.BOOL, ir.DataType.INT8, True),
        (ir.DataType.BOOL, ir.DataType.COMPLEX64, True),
        (ir.DataType.INT8, ir.DataType.INT16, True),
        (ir.DataType.UINT8, ir.DataType.INT16, True),
        (ir.DataType.INT8, ir.DataType.UINT16, False),
        (ir.DataType.UINT8, ir.DataType.INT8, False),
        (ir.DataType.INT32, ir.DataType.DOUBLE, True),
        (ir.DataType.INT64, ir.DataType.DOUBLE, False),
        (ir.DataType.FLOAT16, ir.DataType.FLOAT, True),
        (ir.DataType.BFLOAT16, ir.DataType.FLOAT, True),
        (ir.DataType.FLOAT, ir.DataType.FLOAT16, False),
        (ir.DataType.FLOAT, ir.DataType.DOUBLE, True),
        (ir.DataType.FLOAT, ir.DataType.COMPLEX64, True),
        (ir.DataType.COMPLEX64, ir.DataType.COMPLEX128, True),
        (ir.DataType.COMPLEX128, ir.DataType.COMPLEX64, False),
        (ir.DataType.FLOAT8E4M3FN, ir.DataType.FLOAT16, False),
        (ir.DataType.STRING, ir.DataType.INT64, False),
    ],
)
def test_cast_roundtrip_value_preservation_by_source_domain(
    source: ir.DataType, intermediate: ir.DataType, expected: bool
) -> None:
    assert (
        _cast_roundtrip_is_value_preserving(int(source.value), int(intermediate.value))
        is expected
    )


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
    m.opset_imports[""] = 21
    return m


def test_identity_cast_removed_and_consumers_rewired():
    m = build_graph_with_identity_cast()
    m = optimize_graph(m)
    g = m.graph
    nodes = list(g)
    assert [n.op_type for n in nodes] == ["Relu"]
    relu = nodes[0]

    assert relu.inputs[0].name == "x"


def test_lossy_cast_roundtrip_is_preserved_and_quantizes_values() -> None:
    ort = pytest.importorskip("onnxruntime")
    x = ir.val("x", ir.DataType.FLOAT, (4,))
    narrowed = ir.val("narrowed", ir.DataType.FLOAT16, (4,))
    y = ir.val("y", ir.DataType.FLOAT, (4,))
    cast_to_f16 = ir.Node(
        op_type="Cast",
        domain="",
        inputs=[x],
        outputs=[narrowed],
        name="Cast_to_f16",
        attributes=[_cast_attr(ir.DataType.FLOAT16)],
    )
    cast_to_f32 = ir.Node(
        op_type="Cast",
        domain="",
        inputs=[narrowed],
        outputs=[y],
        name="Cast_back_to_f32",
        attributes=[_cast_attr(ir.DataType.FLOAT)],
    )
    graph = ir.Graph(
        name="lossy_cast_roundtrip",
        inputs=[x],
        outputs=[y],
        nodes=[cast_to_f16, cast_to_f32],
    )
    model = ir.Model(graph=graph, ir_version=10)
    model.opset_imports[""] = 21

    optimized = optimize_graph(model)

    assert [node.name for node in optimized.graph] == [
        "Cast_to_f16",
        "Cast_back_to_f32",
    ]
    input_values: NDArray[np.float32] = np.asarray(
        [1.0001, -3.1415927, 1.0e-8, 123.456], dtype=np.float32
    )
    quantized: NDArray[np.float32] = input_values.astype(np.float16).astype(np.float32)
    assert np.any(quantized != input_values)

    model_proto = serde.serialize_model(optimized)
    session = ort.InferenceSession(
        model_proto.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    result = session.run(None, {"x": input_values})[0]
    np.testing.assert_array_equal(result, quantized)


def test_lossy_integer_float_roundtrip_is_preserved_above_exact_range() -> None:
    ort = pytest.importorskip("onnxruntime")
    x = ir.val("x", ir.DataType.INT64, (2,))
    rounded = ir.val("rounded", ir.DataType.DOUBLE, (2,))
    y = ir.val("y", ir.DataType.INT64, (2,))
    cast_to_f64 = ir.Node(
        op_type="Cast",
        domain="",
        inputs=[x],
        outputs=[rounded],
        name="Cast_to_f64",
        attributes=[_cast_attr(ir.DataType.DOUBLE)],
    )
    cast_back_to_i64 = ir.Node(
        op_type="Cast",
        domain="",
        inputs=[rounded],
        outputs=[y],
        name="Cast_back_to_i64",
        attributes=[_cast_attr(ir.DataType.INT64)],
    )
    graph = ir.Graph(
        name="lossy_integer_float_roundtrip",
        inputs=[x],
        outputs=[y],
        nodes=[cast_to_f64, cast_back_to_i64],
    )
    model = ir.Model(graph=graph, ir_version=10)
    model.opset_imports[""] = 21

    optimized = optimize_graph(model)

    assert [node.name for node in optimized.graph] == [
        "Cast_to_f64",
        "Cast_back_to_i64",
    ]
    outside_exact_f64_range = 2**53 + 1
    input_values: NDArray[np.int64] = np.asarray(
        [outside_exact_f64_range, -outside_exact_f64_range], dtype=np.int64
    )
    rounded_values: NDArray[np.int64] = input_values.astype(np.float64).astype(np.int64)
    assert rounded_values.tolist() == [2**53, -(2**53)]

    model_proto = serde.serialize_model(optimized)
    session = ort.InferenceSession(
        model_proto.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    result = session.run(None, {"x": input_values})[0]
    np.testing.assert_array_equal(result, rounded_values)


def test_lossless_widening_cast_roundtrip_is_removed() -> None:
    x = ir.val("x", ir.DataType.FLOAT16, (2,))
    widened = ir.val("widened", ir.DataType.FLOAT, (2,))
    y = ir.val("y", ir.DataType.FLOAT16, (2,))
    cast_to_f32 = ir.Node(
        op_type="Cast",
        domain="",
        inputs=[x],
        outputs=[widened],
        name="Cast_to_f32",
        attributes=[_cast_attr(ir.DataType.FLOAT)],
    )
    cast_back_to_f16 = ir.Node(
        op_type="Cast",
        domain="",
        inputs=[widened],
        outputs=[y],
        name="Cast_back_to_f16",
        attributes=[_cast_attr(ir.DataType.FLOAT16)],
    )
    graph = ir.Graph(
        name="lossless_cast_roundtrip",
        inputs=[x],
        outputs=[y],
        nodes=[cast_to_f32, cast_back_to_f16],
    )
    model = ir.Model(graph=graph, ir_version=10)
    model.opset_imports[""] = 21

    optimized = optimize_graph(model)

    assert list(optimized.graph) == []
    assert [output.name for output in optimized.graph.outputs] == ["x"]


def _integer_range_cast_roundtrip_model(
    *, start: int, limit: int, delta: int, expand: bool
) -> ir.Model:
    def _const(name: str, value: np.ndarray) -> ir.Value:
        return ir.val(
            name,
            ir.DataType.INT64,
            value.shape,
            const_value=ir.tensor(value),
        )

    start_value = _const("start", np.asarray(start, dtype=np.int64))
    limit_value = _const("limit", np.asarray([limit], dtype=np.int64))
    squeeze_axes = _const("squeeze_axes", np.asarray([0], dtype=np.int64))
    delta_value = _const("delta", np.asarray(delta, dtype=np.int64))
    scalar_limit = ir.val("scalar_limit", ir.DataType.INT64, ())
    squeeze = ir.Node(
        op_type="Squeeze",
        domain="",
        inputs=[limit_value, squeeze_axes],
        outputs=[scalar_limit],
        name="Squeeze_limit",
    )
    length = max(0, len(range(start, limit, delta)))
    range_output = ir.val("range", ir.DataType.INT64, (length,))
    range_node = ir.Node(
        op_type="Range",
        domain="",
        inputs=[start_value, scalar_limit, delta_value],
        outputs=[range_output],
        name="Range_values",
    )
    nodes = [squeeze, range_node]
    initializers = [start_value, limit_value, squeeze_axes, delta_value]
    cast_input = range_output
    if expand:
        unsqueeze_axes = _const("unsqueeze_axes", np.asarray([0], dtype=np.int64))
        expanded_shape = _const(
            "expanded_shape", np.asarray([1, length], dtype=np.int64)
        )
        unsqueezed = ir.val("unsqueezed", ir.DataType.INT64, (1, length))
        expanded = ir.val("expanded", ir.DataType.INT64, (1, length))
        nodes.extend(
            [
                ir.Node(
                    op_type="Unsqueeze",
                    domain="",
                    inputs=[range_output, unsqueeze_axes],
                    outputs=[unsqueezed],
                    name="Unsqueeze_range",
                ),
                ir.Node(
                    op_type="Expand",
                    domain="",
                    inputs=[unsqueezed, expanded_shape],
                    outputs=[expanded],
                    name="Expand_range",
                ),
            ]
        )
        initializers.extend([unsqueeze_axes, expanded_shape])
        cast_input = expanded

    narrowed = ir.val("narrowed", ir.DataType.INT32, cast_input.shape)
    restored = ir.val("restored", ir.DataType.INT64, cast_input.shape)
    nodes.extend(
        [
            ir.Node(
                op_type="Cast",
                domain="",
                inputs=[cast_input],
                outputs=[narrowed],
                name="Cast_to_i32",
                attributes=[_cast_attr(ir.DataType.INT32)],
            ),
            ir.Node(
                op_type="Cast",
                domain="",
                inputs=[narrowed],
                outputs=[restored],
                name="Cast_back_to_i64",
                attributes=[_cast_attr(ir.DataType.INT64)],
            ),
        ]
    )
    graph = ir.Graph(
        name="integer_range_cast_roundtrip",
        inputs=[],
        outputs=[restored],
        nodes=nodes,
        initializers=initializers,
    )
    model = ir.Model(graph=graph, ir_version=10)
    model.opset_imports[""] = 21
    return model


@pytest.mark.parametrize(
    ("start", "limit", "delta"),
    [(0, 1024, 1), (7, -5, -3), (5, 5, 1), (0, 5, -1)],
)
def test_known_integer_range_narrowing_roundtrip_is_removed(
    start: int, limit: int, delta: int
) -> None:
    model = _integer_range_cast_roundtrip_model(
        start=start, limit=limit, delta=delta, expand=True
    )

    optimized = optimize_graph(model)

    assert "Cast" not in [node.op_type for node in optimized.graph]


@pytest.mark.parametrize(
    ("start", "limit", "delta"),
    [
        (2**31, 2**31 + 2, 1),
        (-(2**31) - 1, -(2**31) - 3, -1),
    ],
)
def test_out_of_range_integer_range_narrowing_roundtrip_is_preserved(
    start: int, limit: int, delta: int
) -> None:
    model = _integer_range_cast_roundtrip_model(
        start=start, limit=limit, delta=delta, expand=False
    )

    optimized = optimize_graph(model)

    assert [node.name for node in optimized.graph if node.op_type == "Cast"] == [
        "Cast_to_i32",
        "Cast_back_to_i64",
    ]


def test_dynamic_integer_range_narrowing_roundtrip_is_preserved() -> None:
    start = ir.val(
        "start",
        ir.DataType.INT64,
        (),
        const_value=ir.tensor(np.asarray(0, dtype=np.int64)),
    )
    limit = ir.val("limit", ir.DataType.INT64, ())
    delta = ir.val(
        "delta",
        ir.DataType.INT64,
        (),
        const_value=ir.tensor(np.asarray(1, dtype=np.int64)),
    )
    range_output = ir.val("range", ir.DataType.INT64, (None,))
    narrowed = ir.val("narrowed", ir.DataType.INT32, (None,))
    restored = ir.val("restored", ir.DataType.INT64, (None,))
    graph = ir.Graph(
        name="dynamic_integer_range_cast_roundtrip",
        inputs=[limit],
        outputs=[restored],
        initializers=[start, delta],
        nodes=[
            ir.Node(
                op_type="Range",
                domain="",
                inputs=[start, limit, delta],
                outputs=[range_output],
                name="Range_values",
            ),
            ir.Node(
                op_type="Cast",
                domain="",
                inputs=[range_output],
                outputs=[narrowed],
                name="Cast_to_i32",
                attributes=[_cast_attr(ir.DataType.INT32)],
            ),
            ir.Node(
                op_type="Cast",
                domain="",
                inputs=[narrowed],
                outputs=[restored],
                name="Cast_back_to_i64",
                attributes=[_cast_attr(ir.DataType.INT64)],
            ),
        ],
    )
    model = ir.Model(graph=graph, ir_version=10)
    model.opset_imports[""] = 21

    optimized = optimize_graph(model)

    assert [node.name for node in optimized.graph if node.op_type == "Cast"] == [
        "Cast_to_i32",
        "Cast_back_to_i64",
    ]


def test_integer_range_roundtrip_keeps_observed_intermediate_cast() -> None:
    model = _integer_range_cast_roundtrip_model(start=0, limit=4, delta=1, expand=False)
    first_cast = next(node for node in model.graph if node.name == "Cast_to_i32")
    narrowed = first_cast.outputs[0]
    assert narrowed is not None
    model.graph.outputs.insert(0, narrowed)

    optimized = optimize_graph(model)

    assert [node.name for node in optimized.graph if node.op_type == "Cast"] == [
        "Cast_to_i32"
    ]
    onnx.checker.check_model(serde.serialize_model(optimized), full_check=True)


def test_integer_range_roundtrip_keeps_nested_capture_cast() -> None:
    model = _integer_range_cast_roundtrip_model(start=0, limit=4, delta=1, expand=False)
    nodes = list(model.graph)
    first_cast = next(node for node in nodes if node.name == "Cast_to_i32")
    narrowed = first_cast.outputs[0]
    restored = model.graph.outputs[0]
    assert narrowed is not None

    condition = ir.val(
        "condition",
        ir.DataType.BOOL,
        (),
        const_value=ir.tensor(np.asarray(True, dtype=np.bool_)),
    )

    def _branch(name: str) -> ir.Graph:
        branch_output = ir.val(f"{name}_out", ir.DataType.INT32, narrowed.shape)
        return ir.Graph(
            name=name,
            inputs=[],
            outputs=[branch_output],
            nodes=[
                ir.Node(
                    op_type="Identity",
                    domain="",
                    inputs=[narrowed],
                    outputs=[branch_output],
                    name=f"{name}_capture",
                )
            ],
        )

    if_output = ir.val("if_output", ir.DataType.INT32, narrowed.shape)
    if_node = ir.Node(
        op_type="If",
        domain="",
        inputs=[condition],
        outputs=[if_output],
        name="If_capture",
        attributes=[
            ir.Attr("then_branch", ir.AttributeType.GRAPH, _branch("then")),
            ir.Attr("else_branch", ir.AttributeType.GRAPH, _branch("else")),
        ],
    )
    assert restored is model.graph.outputs[0]
    assert condition.name is not None
    model.graph.append(if_node)
    model.graph.initializers[condition.name] = condition
    model.graph.outputs.append(if_output)

    optimized = optimize_graph(model)

    assert [node.name for node in optimized.graph if node.op_type == "Cast"] == [
        "Cast_to_i32"
    ]
    onnx.checker.check_model(serde.serialize_model(optimized), full_check=True)


def test_custom_domain_value_transform_does_not_enable_range_cast_folding() -> None:
    model = _integer_range_cast_roundtrip_model(start=0, limit=4, delta=1, expand=False)
    nodes = list(model.graph)
    range_node = next(node for node in nodes if node.name == "Range_values")
    first_cast = next(node for node in nodes if node.name == "Cast_to_i32")
    range_output = range_node.outputs[0]
    narrowed_input = first_cast.inputs[0]
    assert range_output is not None and narrowed_input is range_output

    transformed = ir.val("custom_transformed", ir.DataType.INT64, range_output.shape)
    custom_identity = ir.Node(
        op_type="Identity",
        domain="custom",
        inputs=[range_output],
        outputs=[transformed],
        name="CustomIdentity",
    )
    first_cast.replace_input_with(0, transformed)
    model.graph.insert_before(first_cast, custom_identity)

    optimized = optimize_graph(model)

    assert [node.name for node in optimized.graph if node.op_type == "Cast"] == [
        "Cast_to_i32",
        "Cast_back_to_i64",
    ]


@pytest.mark.parametrize("node_name", ["Range_values", "Squeeze_limit"])
def test_custom_domain_range_path_does_not_enable_cast_folding(
    node_name: str,
) -> None:
    model = _integer_range_cast_roundtrip_model(start=0, limit=4, delta=1, expand=False)
    node = next(candidate for candidate in model.graph if candidate.name == node_name)
    node.domain = "custom"

    optimized = optimize_graph(model)

    assert [node.name for node in optimized.graph if node.op_type == "Cast"] == [
        "Cast_to_i32",
        "Cast_back_to_i64",
    ]


@pytest.mark.parametrize("node_name", ["Cast_to_i32", "Cast_back_to_i64"])
def test_custom_domain_cast_is_never_treated_as_standard_cast(node_name: str) -> None:
    model = _integer_range_cast_roundtrip_model(start=0, limit=4, delta=1, expand=False)
    node = next(candidate for candidate in model.graph if candidate.name == node_name)
    node.domain = "custom"

    optimized = optimize_graph(model)

    assert [node.name for node in optimized.graph if node.op_type == "Cast"] == [
        "Cast_to_i32",
        "Cast_back_to_i64",
    ]


def test_identity_cast_removed_inside_function_body():
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

    fn = ir.Function(
        domain="custom",
        name="identity_cast",
        graph=inner_model.graph,
        attributes=(),
    )

    model = ir.Model(graph=top_graph, ir_version=10, functions=[fn])
    model.opset_imports[""] = 21

    optimized = optimize_graph(model)
    assert [n.op_type for n in optimized.functions["custom", "identity_cast", ""]] == [
        "Relu"
    ]


def test_identity_reshape_removed_when_target_matches_source():
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
    nodes = optimized.graph
    assert all(n.op_type != "Reshape" for n in nodes)
    out_names = {v.name for v in optimized.graph.outputs}
    assert "in" in out_names


def test_cse_simple():
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

    node3 = ir.Node(
        op_type="Identity",
        domain="",
        inputs=[out2],
        outputs=[ir.val("out3", ir.DataType.FLOAT, (3, 4))],
        name="Identity1",
    )

    # Graph outputs BOTH
    graph = ir.Graph(
        name="cse_simple",
        inputs=[data],
        outputs=[out1, node3.outputs[0]],
        nodes=[node1, node2, node3],
    )

    model = ir.Model(graph=graph, ir_version=10)
    optimized = optimize_graph(model)

    nodes = optimized.graph
    # Should be merged
    assert len(nodes) == 2
    assert nodes[0].op_type == "Relu"
    assert nodes[1].op_type == "Identity"

    # ONNX graph outputs cannot share the same Value object, so both must remain
    outs = optimized.graph.outputs
    assert len(outs) == 2
    assert outs[0] is not outs[1]


def test_lift_constants():
    # Make a graph with a Constant node in the body
    out_const = V_ir("const_out", ir.DataType.FLOAT, (2,))
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
    nodes = optimized.graph
    assert len(nodes) == 1
    assert nodes[0].op_type == "Identity"

    assert len(optimized.graph.initializers) == 1
    init_val = list(optimized.graph.initializers.values())[0]
    # Name should be preserved or match usage
    assert init_val.name == "const_out"
    assert init_val.const_value is not None


def test_reshape_fold_skips_non_isolated_chain_with_shape_consumer():
    x = ir.val("in", ir.DataType.FLOAT, (4, 32))
    x3d = ir.val("x3d", ir.DataType.FLOAT, (2, 2, 32))
    max_out = ir.val("max_out", ir.DataType.FLOAT, (2, 2, 32))
    shape_out = ir.val("shape_out", ir.DataType.INT64, (3,))
    y = ir.val("out", ir.DataType.FLOAT, (4, 32))

    shape_up = ir.Value(
        name="shape_up",
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape((3,)),
        const_value=ir.tensor(np.asarray([2, 2, 32], dtype=np.int64)),
    )
    shape_down = ir.Value(
        name="shape_down",
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape((2,)),
        const_value=ir.tensor(np.asarray([-1, 32], dtype=np.int64)),
    )
    zero = ir.Value(
        name="zero",
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape(()),
        const_value=ir.tensor(np.asarray(0.0, dtype=np.float32)),
    )

    reshape_up = ir.Node(
        op_type="Reshape",
        domain="",
        inputs=[x, shape_up],
        outputs=[x3d],
        name="Reshape_up",
    )
    max_mid = ir.Node(
        op_type="Max",
        domain="",
        inputs=[x3d, zero],
        outputs=[max_out],
        name="Max_mid",
    )
    # Extra consumer keeps the elementwise middle node non-isolated.
    shape_mid = ir.Node(
        op_type="Shape",
        domain="",
        inputs=[max_out],
        outputs=[shape_out],
        name="Shape_mid",
    )
    reshape_down = ir.Node(
        op_type="Reshape",
        domain="",
        inputs=[max_out, shape_down],
        outputs=[y],
        name="Reshape_down",
    )

    graph = ir.Graph(
        name="reshape_non_isolated",
        inputs=[x],
        outputs=[y],
        nodes=[reshape_up, max_mid, shape_mid, reshape_down],
        initializers=[shape_up, shape_down, zero],
    )
    model = ir.Model(graph=graph, ir_version=10)
    model.opset_imports[""] = 21

    optimized = optimize_graph(model)
    nodes = list(optimized.graph)
    op_types = [node.op_type for node in nodes]

    # The Reshape -> Max -> Reshape chain must not be folded because Max has
    # an extra Shape consumer.
    assert op_types.count("Reshape") == 2
    max_node = next(node for node in nodes if node.name == "Max_mid")
    assert max_node.inputs[0].name == "x3d"
