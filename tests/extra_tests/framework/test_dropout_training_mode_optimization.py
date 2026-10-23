# tests/extra_tests/framework/test_dropout_training_mode_optimization.py

"""Dropout optional-input cleanup preserves owned values and real dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest

from jax2onnx.converter.ir_optimizations import (
    inline_dropout_training_mode_constants_ir,
    optimize_graph,
)
from jax2onnx.converter.precision_policy import PrecisionPolicy


_DATA = np.asarray([1, -2, 3, -4], dtype=np.float32)


@dataclass
class _Fixture:
    model: ir.Model
    graph: ir.Graph
    not_node: ir.Node | None
    training: ir.Value
    dropouts: list[ir.Node]


def _value(name: str, dtype: ir.DataType, shape: tuple[int, ...] = ()) -> ir.Value:
    return ir.val(name, dtype=dtype, shape=shape)


def _constant(name: str, data: Any, *, node: bool) -> tuple[ir.Value, list[ir.Node]]:
    array = np.asarray(data)
    tensor = ir.tensor(array)
    value = _value(name, ir.DataType.from_numpy(array.dtype), array.shape)
    if node:
        return value, [
            ir.Node(
                "",
                "Constant",
                [],
                outputs=[value],
                attributes=[ir.AttrTensor("value", tensor)],
            )
        ]
    value.const_value = tensor
    return value, []


def _basic(kind: str = "initializer", shared: str = "none") -> _Fixture:
    x = _value("in_0", ir.DataType.FLOAT, (4,))
    ratio, _ = _constant("ratio", np.float32(0.5), node=False)
    nodes: list[ir.Node] = []
    initializers = [ratio]
    inputs = [x]
    if kind in {"dynamic", "overridable", "computed"}:
        flag = _value("runtime", ir.DataType.BOOL)
        inputs.append(flag)
        if kind == "overridable":
            flag.const_value = ir.tensor(np.asarray(True))
            initializers.append(flag)
        if kind == "computed":
            computed = _value("computed_flag", ir.DataType.BOOL)
            nodes.append(ir.Node("", "Identity", [flag], outputs=[computed]))
            flag = computed
    else:
        flag, const_nodes = _constant(
            "literal", np.asarray(kind != "not-false"), node=kind == "constant"
        )
        nodes.extend(const_nodes)
        if not const_nodes:
            initializers.append(flag)
    training = _value("training", ir.DataType.BOOL)
    training.metadata_props["preserve"] = "observable training flag"
    not_node = ir.Node("", "Not", [flag], outputs=[training])
    if kind == "direct-false":
        training, _ = _constant("direct_false", np.asarray(False), node=False)
        initializers.append(training)
        not_node = None
    else:
        nodes.append(not_node)
    dropouts = []
    outputs = []
    for index in range(2 if shared == "two-dropouts" else 1):
        y = _value(f"y_{index}", ir.DataType.FLOAT, (4,))
        mask = _value(f"mask_{index}", ir.DataType.BOOL, (4,))
        dropout = ir.Node("", "Dropout", [x, ratio, training], outputs=[y, mask])
        nodes.append(dropout)
        dropouts.append(dropout)
        outputs.extend((y, mask))
    if shared in {"identity", "and"}:
        observed = _value("observed", ir.DataType.BOOL)
        nodes.append(
            ir.Node(
                "",
                "Identity" if shared == "identity" else "And",
                [training] if shared == "identity" else [training, flag],
                outputs=[observed],
            )
        )
        outputs.append(observed)
    elif shared == "output":
        outputs.append(training)
    graph = ir.Graph(
        inputs,
        outputs,
        nodes=nodes,
        initializers=initializers,
        name="dropout_body",
        opset_imports={"": 23},
    )
    return _Fixture(ir.Model(graph, ir_version=10), graph, not_node, training, dropouts)


def _run(model: ir.Model, flags: dict[str, bool] | None = None) -> list[np.ndarray]:
    proto = ir.to_proto(model)
    onnx.checker.check_model(proto, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        proto.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    feed = {
        "in_0": _DATA,
        **{name: np.asarray(value) for name, value in (flags or {}).items()},
    }
    return session.run(None, feed)


def _inference_pair(before: list[np.ndarray], after: list[np.ndarray]) -> None:
    for expected, actual in zip(before, after, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(after[0], _DATA)
    np.testing.assert_array_equal(after[1], np.ones(_DATA.shape, dtype=np.bool_))


@pytest.mark.parametrize("kind", ["initializer", "constant"])
@pytest.mark.parametrize(
    "shared", ["none", "two-dropouts", "identity", "and", "output"]
)
def test_direct_cleanup_omits_only_dropout_input_and_preserves_other_uses(
    kind: str, shared: str
) -> None:
    fixture = _basic(kind, shared)
    before = _run(fixture.model)
    old_outputs = tuple(fixture.graph.outputs)
    old_names = tuple(value.name for value in old_outputs)
    old_initializers = tuple(fixture.graph.initializers)
    inline_dropout_training_mode_constants_ir(fixture.graph)
    assert all(
        len(node.inputs) == 3 and node.inputs[2] is None for node in fixture.dropouts
    )
    assert all(
        node.input[2] == ""
        for node in ir.to_proto(fixture.model).graph.node
        if node.op_type == "Dropout"
    )
    assert tuple(fixture.graph.outputs) == old_outputs
    assert tuple(value.name for value in fixture.graph.outputs) == old_names
    assert tuple(fixture.graph.initializers) == old_initializers
    observed = shared in {"identity", "and", "output"}
    assert fixture.not_node is not None
    assert (fixture.not_node.graph is fixture.graph) == observed
    if observed:
        assert fixture.training.producer() is fixture.not_node
        assert fixture.training.dtype == ir.DataType.BOOL
        assert fixture.training.shape == ir.Shape(())
        assert fixture.training.metadata_props["preserve"] == "observable training flag"
        if shared != "output":
            assert list(fixture.graph)[-1].inputs[0] is fixture.training
    after = _run(fixture.model)
    _inference_pair(before, after)
    if shared == "two-dropouts":
        np.testing.assert_array_equal(after[2], _DATA)
        assert after[3].all()
    elif observed:
        assert after[-1].shape == () and not after[-1].item()


@pytest.mark.parametrize("policy", [PrecisionPolicy.LEGACY, PrecisionPolicy.SOURCE])
@pytest.mark.parametrize("scope", ["root", "function"])
@pytest.mark.parametrize("kind", ["initializer", "constant"])
def test_real_pipeline_and_function_constants_remain_valid(
    policy: PrecisionPolicy, scope: str, kind: str
) -> None:
    fixture = _basic(kind)
    function = None
    if scope == "function":
        # Functions own Constant nodes, never initializers.
        for value in list(fixture.graph.initializers.values()):
            fixture.graph.initializers.pop(value.name)
            fixture.graph.insert_before(
                fixture.graph[0],
                ir.Node(
                    "",
                    "Constant",
                    [],
                    outputs=[value],
                    attributes=[ir.AttrTensor("value", value.const_value)],
                ),
            )
        unused = _value("unused_formal", ir.DataType.BOOL)
        fixture.graph.inputs.append(unused)
        function = ir.Function(
            "local", "DropoutBody", graph=fixture.graph, attributes=()
        )
        x = _value("in_0", ir.DataType.FLOAT, (4,))
        runtime = _value("unused_runtime", ir.DataType.BOOL)
        outputs = [
            _value("result", ir.DataType.FLOAT, (4,)),
            _value("mask", ir.DataType.BOOL, (4,)),
        ]
        top = ir.Graph(
            [x, runtime],
            outputs,
            nodes=[ir.Node("local", "DropoutBody", [x, runtime], outputs=outputs)],
            name="root",
            opset_imports={"": 23, "local": 1},
        )
        fixture.model = ir.Model(top, ir_version=10, functions=[function])
    flags = {"unused_runtime": True} if function is not None else {}
    before = _run(fixture.model, flags)
    function_inputs = tuple(fixture.graph.inputs)
    function_outputs = tuple(fixture.graph.outputs)
    optimize_graph(fixture.model, precision_policy=policy)
    assert len(fixture.dropouts[0].inputs) == 2 or fixture.dropouts[0].inputs[2] is None
    assert not any(node.op_type == "Not" for node in fixture.graph)
    if function is not None:
        assert tuple(fixture.graph.inputs) == function_inputs
        assert tuple(fixture.graph.outputs) == function_outputs
        assert not fixture.graph.initializers
        proto_function = ir.to_proto(fixture.model).functions[0]
        assert list(proto_function.input) == ["in_0", "unused_formal"]
        assert len(proto_function.output) == 2
        assert any(node.op_type == "Constant" for node in proto_function.node)
    _inference_pair(before, _run(fixture.model, flags))


@pytest.mark.parametrize(
    "kind", ["dynamic", "overridable", "computed", "not-false", "direct-false"]
)
def test_unproven_or_already_direct_controls_keep_their_original_binding(
    kind: str,
) -> None:
    fixture = _basic(kind)
    binding = fixture.dropouts[0].inputs[2]
    onnx.checker.check_model(ir.to_proto(fixture.model), full_check=True)
    flags = (True, False) if kind in {"dynamic", "overridable", "computed"} else (True,)
    before = {
        flag: _run(
            fixture.model,
            {"runtime": flag} if kind in {"dynamic", "overridable", "computed"} else {},
        )
        for flag in flags
    }
    inline_dropout_training_mode_constants_ir(fixture.graph)
    assert fixture.dropouts[0].inputs[2] is binding
    if fixture.not_node is not None:
        assert fixture.not_node.graph is fixture.graph
    onnx.checker.check_model(ir.to_proto(fixture.model), full_check=True)
    for flag in flags:
        after = _run(
            fixture.model,
            {"runtime": flag} if kind in {"dynamic", "overridable", "computed"} else {},
        )
        training = (
            not flag
            if kind in {"dynamic", "overridable", "computed"}
            else kind == "not-false"
        )
        for data, mask in (before[flag], after):
            assert data.dtype == np.float32 and mask.dtype == np.bool_
            assert data.shape == mask.shape == _DATA.shape
            assert np.isfinite(data).all()
            np.testing.assert_array_equal(
                data,
                np.where(
                    mask, _DATA * np.float32(2) if training else _DATA, np.float32(0)
                ),
            )
            if not training:
                assert mask.all()
        if not training:
            _inference_pair(before[flag], after)


def test_generic_boolean_graph_is_not_a_dropout_rewrite() -> None:
    fixture = _basic(shared="and")
    for dropout in fixture.dropouts:
        fixture.graph.remove(dropout)
    fixture.graph.outputs[:] = [list(fixture.graph)[-1].outputs[0]]
    before = ir.to_proto(fixture.model).SerializeToString()
    expected = _run(fixture.model)
    onnx.checker.check_model(ir.to_proto(fixture.model), full_check=True)
    inline_dropout_training_mode_constants_ir(fixture.graph)
    assert ir.to_proto(fixture.model).SerializeToString() == before
    onnx.checker.check_model(ir.to_proto(fixture.model), full_check=True)
    actual = _run(fixture.model)
    assert not actual[0].item()
    np.testing.assert_array_equal(actual[0], expected[0])


def _nested(
    scope: str, local: bool
) -> tuple[_Fixture | None, list[_Fixture], ir.Model]:
    x = _value("in_0", ir.DataType.FLOAT, (4,))
    ratio, _ = _constant("ratio", np.float32(0.5), node=False)
    nodes: list[ir.Node] = []
    inputs = [x]
    initializers = [ratio]
    parent = None
    outputs: list[ir.Value] = []
    training = None
    if not local:
        parent = _basic()
        x = parent.graph.inputs[0]
        ratio = parent.graph.initializers["ratio"]
        nodes = list(parent.graph)
        initializers = list(parent.graph.initializers.values())
        inputs = [x]
        outputs = list(parent.graph.outputs)
        training = parent.training
    children = []

    def child(prefix: str) -> ir.Graph:
        body_nodes: list[ir.Node] = []
        body_inputs: list[ir.Value] = []
        data = x
        condition = None
        if scope == "Loop":
            iteration = _value(prefix + "_iteration", ir.DataType.INT64)
            condition = _value(prefix + "_condition", ir.DataType.BOOL)
            data = _value(prefix + "_carry", ir.DataType.FLOAT, (4,))
            body_inputs = [iteration, condition, data]
        not_node = None
        control = training
        if local:
            flag, constant_nodes = _constant(
                prefix + "_true", np.asarray(True), node=True
            )
            body_nodes.extend(constant_nodes)
            control = _value(prefix + "_training", ir.DataType.BOOL)
            not_node = ir.Node("", "Not", [flag], outputs=[control])
            body_nodes.append(not_node)
        assert control is not None
        y = _value(prefix + "_y", ir.DataType.FLOAT, (4,))
        mask = _value(prefix + "_mask", ir.DataType.BOOL, (4,))
        dropout = ir.Node("", "Dropout", [data, ratio, control], outputs=[y, mask])
        body_nodes.append(dropout)
        if scope == "If":
            observed = _value(prefix + "_observed", ir.DataType.BOOL)
            body_nodes.append(ir.Node("", "Identity", [control], outputs=[observed]))
            body_outputs = [y, mask, observed]
        else:
            assert condition is not None
            body_outputs = [condition, y, mask]
        graph = ir.Graph(
            body_inputs,
            body_outputs,
            nodes=body_nodes,
            name=prefix,
            opset_imports={"": 23},
        )
        children.append(_Fixture(None, graph, not_node, control, [dropout]))
        return graph

    y = _value("nested_y", ir.DataType.FLOAT, (4,))
    if scope == "If":
        choose = _value("choose", ir.DataType.BOOL)
        inputs.append(choose)
        mask = _value("nested_mask", ir.DataType.BOOL, (4,))
        observed = _value("nested_observed", ir.DataType.BOOL)
        then, otherwise = child("then"), child("else")
        nodes.append(
            ir.Node(
                "",
                "If",
                [choose],
                outputs=[y, mask, observed],
                attributes=[
                    ir.AttrGraph("then_branch", then),
                    ir.AttrGraph("else_branch", otherwise),
                ],
            )
        )
        outputs.extend((y, mask, observed))
    else:
        count, _ = _constant("trip_count", np.int64(2), node=False)
        condition, _ = _constant("condition", np.asarray(True), node=False)
        initializers.extend((count, condition))
        mask = _value("nested_masks", ir.DataType.BOOL, (2, 4))
        body = child("body")
        nodes.append(
            ir.Node(
                "",
                "Loop",
                [count, condition, x],
                outputs=[y, mask],
                attributes=[ir.AttrGraph("body", body)],
            )
        )
        outputs.extend((y, mask))
    if parent is None:
        graph = ir.Graph(
            inputs,
            outputs,
            nodes=nodes,
            initializers=initializers,
            name="nested_root",
            opset_imports={"": 23},
        )
    else:
        graph = parent.graph
        graph.inputs.extend(inputs[1:])
        graph.outputs.extend(outputs[len(graph.outputs) :])
        for initializer in initializers:
            if initializer.name not in graph.initializers:
                graph.register_initializer(initializer)
        for node in nodes:
            if node.graph is None:
                graph.append(node)
    model = ir.Model(graph, ir_version=10)
    for fixture in children:
        fixture.model = model
    if parent is not None:
        parent.model, parent.graph = model, graph
    return parent, children, model


@pytest.mark.parametrize("policy", [PrecisionPolicy.LEGACY, PrecisionPolicy.SOURCE])
@pytest.mark.parametrize("scope", ["If", "Loop"])
@pytest.mark.parametrize("local", [False, True])
def test_nested_capture_ownership_and_current_nonrecursive_pipeline(
    policy: PrecisionPolicy, scope: str, local: bool
) -> None:
    parent, children, model = _nested(scope, local)
    flags = {"choose": True} if scope == "If" else {}
    before = _run(model, flags)
    optimize_graph(model, precision_policy=policy)
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    if parent is not None:
        assert (
            len(parent.dropouts[0].inputs) == 2 or parent.dropouts[0].inputs[2] is None
        )
        assert parent.not_node.graph is model.graph
        assert parent.training.uses()
    for fixture in children:
        assert fixture.dropouts[0].inputs[2] is fixture.training
        if local:
            assert fixture.not_node.graph is fixture.graph
        # A child with an enclosing producer must not match it by name or value.
        inline_dropout_training_mode_constants_ir(fixture.graph)
        assert (fixture.dropouts[0].inputs[2] is None) == local
        if local:
            assert (fixture.not_node.graph is fixture.graph) == (scope == "If")
    after = _run(model, flags)
    for expected, actual in zip(before, after, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(after[0], _DATA)
    assert after[1].all()
    if scope == "If":
        assert not after[-1].item()
        for expected, actual in zip(
            before, _run(model, {"choose": False}), strict=True
        ):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("custom_operator", ["Dropout", "Not"])
def test_custom_domain_names_do_not_authorize_standard_operator_rewrites(
    custom_operator: str,
) -> None:
    fixture = _basic()
    if custom_operator == "Not":
        formal = _value("flag", ir.DataType.BOOL)
        output = _value("result", ir.DataType.BOOL)
        body = ir.Graph(
            [formal],
            [output],
            nodes=[ir.Node("", "Not", [formal], outputs=[output])],
            opset_imports={"": 23},
        )
        fixture.not_node.domain = "custom"
    else:
        x = _value("data", ir.DataType.FLOAT, (4,))
        ratio = _value("rate", ir.DataType.FLOAT)
        training = _value("mode", ir.DataType.BOOL)
        y, mask = _value("data_out", ir.DataType.FLOAT, (4,)), _value(
            "mask_out", ir.DataType.BOOL, (4,)
        )
        body = ir.Graph(
            [x, ratio, training],
            [y, mask],
            nodes=[ir.Node("", "Dropout", [x, ratio, training], outputs=[y, mask])],
            opset_imports={"": 23},
        )
        fixture.dropouts[0].domain = "custom"
    function = ir.Function(
        domain="custom", name=custom_operator, graph=body, attributes=()
    )
    fixture.model = ir.Model(fixture.graph, ir_version=10, functions=[function])
    fixture.model.opset_imports["custom"] = 1
    before = _run(fixture.model)
    binding = fixture.dropouts[0].inputs[2]
    inline_dropout_training_mode_constants_ir(fixture.graph)
    assert fixture.dropouts[0].inputs[2] is binding
    assert fixture.not_node.graph is fixture.graph
    _inference_pair(before, _run(fixture.model))
