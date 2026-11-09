# tests/extra_tests/converter/test_elementwise_layout_folding.py

"""Layout folding must preserve every elementwise operand's index mapping."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnx_ir as ir
import onnxruntime as ort
import pytest

from jax2onnx import to_onnx
from jax2onnx.converter.ir_optimizations import (
    optimize_graph,
    remove_redundant_reshape_pairs_ir,
    remove_redundant_transpose_pairs_ir,
)


_LAYOUT_CASES = [
    pytest.param("Reshape", (2, 3), (6,), [1, 2, 3, 4, 5, 6], id="reshape-invalid"),
    pytest.param("Reshape", (2, 2, 3), (3, 2, 2), [[1], [10]], id="reshape-silent"),
    pytest.param("Transpose", (2, 3), (3, 2), [1, 10], id="transpose-invalid"),
    pytest.param("Transpose", (2, 2), (2, 2), [1, 10], id="transpose-silent"),
]


def _layout_model(
    layout: str,
    op_type: str,
    source_shape: tuple[int, ...],
    view_shape: tuple[int, ...],
    side: np.ndarray,
    *,
    variadic: bool = False,
    transpose_side: bool = False,
) -> onnx.ModelProto:
    def value(name: str, shape: tuple[int, ...]) -> onnx.ValueInfoProto:
        return helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)

    initializers = [numpy_helper.from_array(side, name="side")]
    if layout == "Reshape":
        initializers.extend(
            numpy_helper.from_array(np.asarray(shape, dtype=np.int64), name=name)
            for name, shape in (("shape1", view_shape), ("shape2", source_shape))
        )
        first = helper.make_node(layout, ["x", "shape1"], ["view"])
        last = helper.make_node(layout, ["combined", "shape2"], ["y"])
    else:
        first = helper.make_node(layout, ["x"], ["view"], perm=[1, 0])
        last = helper.make_node(layout, ["combined"], ["y"], perm=[1, 0])
    nodes = [first]
    side_name = "side"
    if transpose_side:
        nodes.append(
            helper.make_node("Transpose", ["side"], ["side_view"], perm=[1, 0])
        )
        side_name = "side_view"
    operands = ["view", side_name]
    if variadic:
        initializers.append(
            numpy_helper.from_array(np.asarray(0, dtype=np.float32), "zero")
        )
        # The non-scalar side operand comes third, beyond the first neighbor.
        operands.insert(1, "zero")
    nodes.extend([helper.make_node(op_type, operands, ["combined"]), last])
    graph = helper.make_graph(
        nodes,
        "elementwise_layout",
        [value("x", source_shape)],
        [value("y", source_shape)],
        initializer=initializers,
        value_info=[value("view", view_shape), value("combined", view_shape)],
    )
    return helper.make_model(
        graph, ir_version=10, opset_imports=[helper.make_opsetid("", 18)]
    )


def _optimize(
    model: onnx.ModelProto, layout: str, whole_optimizer: bool
) -> onnx.ModelProto:
    model_ir = ir.from_proto(model)
    if whole_optimizer:
        model_ir = optimize_graph(model_ir)
    elif layout == "Reshape":
        remove_redundant_reshape_pairs_ir(model_ir.graph)
    else:
        remove_redundant_transpose_pairs_ir(model_ir.graph)
    return ir.to_proto(model_ir)


def _assert_matches(
    model: onnx.ModelProto, x: np.ndarray, expected: np.ndarray
) -> None:
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    (actual,) = session.run(None, {"x": x})
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("op_type", ["Max", "Min"])
@pytest.mark.parametrize("layout,source_shape,view_shape,side_values", _LAYOUT_CASES)
@pytest.mark.parametrize("whole_optimizer", [False, True], ids=["pass", "optimizer"])
@pytest.mark.parametrize("variadic", [False, True], ids=["binary", "variadic"])
def test_layout_folding_preserves_broadcast_operands(
    op_type, layout, source_shape, view_shape, side_values, whole_optimizer, variadic
) -> None:
    x = np.zeros(source_shape, dtype=np.float32)
    side = np.asarray(side_values, dtype=np.float32) * (1 if op_type == "Max" else -1)
    op = np.maximum if op_type == "Max" else np.minimum
    view = x.reshape(view_shape) if layout == "Reshape" else x.T
    combined = op(view, side)
    expected = combined.reshape(source_shape) if layout == "Reshape" else combined.T
    model = _layout_model(
        layout, op_type, source_shape, view_shape, side, variadic=variadic
    )
    _assert_matches(model, x, expected)

    optimized = _optimize(model, layout, whole_optimizer)

    _assert_matches(optimized, x, expected)
    assert [node.op_type for node in optimized.graph.node] == [layout, op_type, layout]
    assert len(
        next(node for node in optimized.graph.node if node.op_type == op_type).input
    ) == (3 if variadic else 2)


@pytest.mark.parametrize("op_type", ["Max", "Min"])
@pytest.mark.parametrize("layout,source_shape,view_shape,side_values", _LAYOUT_CASES)
def test_public_export_preserves_broadcast_operand_layout(
    op_type, layout, source_shape, view_shape, side_values
) -> None:
    side = np.asarray(side_values, dtype=np.float32) * (1 if op_type == "Max" else -1)

    def target(x):
        view = jnp.reshape(x, view_shape) if layout == "Reshape" else jnp.transpose(x)
        combined = (
            jnp.maximum(view, side) if op_type == "Max" else jnp.minimum(view, side)
        )
        return (
            jnp.reshape(combined, source_shape)
            if layout == "Reshape"
            else jnp.transpose(combined)
        )

    x = np.zeros(source_shape, dtype=np.float32)
    expected = np.asarray(target(x))

    model = to_onnx(
        target,
        [jax.ShapeDtypeStruct(source_shape, jnp.float32)],
        opset=18,
        input_names=["x"],
    )

    _assert_matches(model, x, expected)
    assert sum(node.op_type == layout for node in model.graph.node) == 2


@pytest.mark.parametrize("op_type", ["Max", "Min"])
@pytest.mark.parametrize("whole_optimizer", [False, True], ids=["pass", "optimizer"])
def test_transpose_folding_still_proves_all_operand_layouts(
    op_type, whole_optimizer
) -> None:
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    side = np.asarray([[5, 3, 1], [4, 2, 0]], dtype=np.float32)
    op = np.maximum if op_type == "Max" else np.minimum
    model = _layout_model(
        "Transpose", op_type, x.shape, (3, 2), side, transpose_side=True
    )
    _assert_matches(model, x, op(x, side))

    optimized = _optimize(model, "Transpose", whole_optimizer)

    _assert_matches(optimized, x, op(x, side))
    assert [node.op_type for node in optimized.graph.node] == [op_type]


@pytest.mark.parametrize("layout", ["Reshape", "Transpose"])
@pytest.mark.parametrize("whole_optimizer", [False, True], ids=["pass", "optimizer"])
def test_layout_folding_still_accepts_scalar_clip_bounds(
    layout, whole_optimizer
) -> None:
    x = np.arange(-3, 3, dtype=np.float32).reshape(2, 3)
    view_shape = (6,) if layout == "Reshape" else (3, 2)
    model = _layout_model(
        layout, "Clip", x.shape, view_shape, np.asarray(0, dtype=np.float32)
    )

    optimized = _optimize(model, layout, whole_optimizer)

    _assert_matches(optimized, x, np.maximum(x, 0))
    assert [node.op_type for node in optimized.graph.node] == ["Clip"]
