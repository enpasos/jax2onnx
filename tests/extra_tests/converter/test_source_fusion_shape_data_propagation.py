# tests/extra_tests/converter/test_source_fusion_shape_data_propagation.py

"""Real inline protobuf shape controls stay exact in private fusion proofs."""

from copy import deepcopy
from unittest.mock import patch

import onnx
import onnx.inliner
import pytest

from jax2onnx import _deployment_fusion_bridge as bridge
from jax2onnx._deployment_fusions import FusionUnfinished


def _vi(name, dtype, shape):
    return onnx.helper.make_tensor_value_info(name, dtype, shape)


def _constant(tag, output, dtype, shape, values):
    return onnx.helper.make_node(
        "Constant",
        [],
        [output],
        name=f"jax2onnx_src_{tag:06d}",
        value=onnx.helper.make_tensor(output + "_value", dtype, shape, values),
    )


def _fixture():
    float32, int64 = onnx.TensorProto.FLOAT, onnx.TensorProto.INT64
    roles = {}

    def node(tag, op, inputs, outputs, **attributes):
        roles[f"jax2onnx_src_{tag:06d}"] = (op, "", "root")
        return onnx.helper.make_node(
            op, inputs, outputs, name=f"jax2onnx_src_{tag:06d}", **attributes
        )

    def branch(fused, negative):
        nodes = [
            _constant(1, "weights", float32, [2, 2], [1, 0, 0, 1]),
            _constant(
                2,
                "bias",
                float32,
                [2],
                [0.25, -0.5] if not negative else [-0.75, 0.125],
            ),
            _constant(3, "one", int64, [1], [1]),
            _constant(4, "two", int64, [1], [2]),
            node(5, "Concat", ["one", "two"], ["control"], axis=0),
            node(6, "Reshape", ["bias", "control"], ["row"]),
            node(7, "Expand", ["row", "control"], ["expanded"]),
        ]
        if fused:
            nodes.append(
                node(
                    8,
                    "Gemm",
                    ["left", "weights", "expanded"],
                    ["result"],
                    alpha=1.0,
                    beta=1.0,
                    transA=0,
                    transB=0,
                )
            )
        else:
            nodes.extend(
                [
                    node(8, "MatMul", ["left", "weights"], ["matrix"]),
                    node(9, "Add", ["matrix", "expanded"], ["result"]),
                ]
            )
        graph = onnx.helper.make_graph(
            nodes,
            "same_branch",
            [],
            [_vi("result", float32, ["B", 2])],
            value_info=[_vi("row", float32, [1, 2]), _vi("expanded", float32, [1, 2])],
        )
        # Match SOURCE's unique per-primitive roles and branch temporaries.
        # Lexical reuse is valid ONNX, but collides in standard data propagation.
        prefix, offset = ("negative_", 100) if negative else ("positive_", 0)
        names = {name: prefix + name for item in graph.node for name in item.output}
        for item in graph.node:
            tag = int(item.name.rsplit("_", 1)[1]) + offset
            item.name = f"jax2onnx_src_{tag:06d}"
            roles[item.name] = (item.op_type, "", "root")
            inputs = [names.get(name, name) for name in item.input]
            outputs = [names[name] for name in item.output]
            del item.input[:]
            item.input.extend(inputs)
            del item.output[:]
            item.output.extend(outputs)
            for attribute in item.attribute:
                if attribute.type == onnx.AttributeProto.TENSOR:
                    attribute.t.name = prefix + attribute.t.name
        for value in (*graph.output, *graph.value_info):
            value.name = names[value.name]
        graph.name = prefix + "branch"
        return graph

    def model(fused):
        conditional = node(
            20,
            "If",
            ["predicate"],
            ["result"],
            then_branch=branch(fused, False),
            else_branch=branch(fused, True),
        )
        function = onnx.helper.make_function(
            "test.shape_control",
            "Repeated",
            ["left", "predicate"],
            ["result"],
            [conditional],
            [onnx.helper.make_opsetid("", 23)],
            value_info=[
                _vi("left", float32, ["B", 2]),
                _vi("predicate", onnx.TensorProto.BOOL, []),
                _vi("result", float32, ["B", 2]),
            ],
        )
        calls = [
            onnx.helper.make_node(
                "Repeated",
                [f"data{index}", f"predicate{index}"],
                [f"out{index}"],
                domain=function.domain,
                name=f"call{index}",
            )
            for index in range(2)
        ]
        inputs = [
            value
            for index in range(2)
            for value in (
                _vi(f"data{index}", float32, ["B", 2]),
                _vi(f"predicate{index}", onnx.TensorProto.BOOL, []),
            )
        ]
        root = onnx.helper.make_graph(
            calls,
            "repeated",
            inputs,
            [_vi(f"out{index}", float32, ["B", 2]) for index in range(2)],
        )
        return onnx.helper.make_model(
            root,
            functions=[function],
            ir_version=10,
            opset_imports=[
                onnx.helper.make_opsetid("", 23),
                onnx.helper.make_opsetid(function.domain, 1),
            ],
        )

    original, optimized = model(False), model(True)
    for tag in range(1, 5):
        roles[f"jax2onnx_src_{tag:06d}"] = ("Constant", "", "root")
    # Source role is MatMul even where the actual optimized node is Gemm.
    roles["jax2onnx_src_000008"] = ("MatMul", "", "root")
    roles["jax2onnx_src_000108"] = ("MatMul", "", "root")
    return original, optimized, roles


def _branches(model):
    for node in model.graph.node:
        if node.op_type == "If":
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    yield attribute.g


def _shape(value):
    return tuple(
        (
            dimension.dim_value
            if dimension.HasField("dim_value")
            else dimension.dim_param or None
        )
        for dimension in value.type.tensor_type.shape.dim
    )


def _prove_unchanged(original, optimized, roles):
    before = original.SerializeToString(), optimized.SerializeToString()
    try:
        return bridge.prove_deployment_fusions(original, optimized, roles)
    finally:
        assert (original.SerializeToString(), optimized.SerializeToString()) == before


def test_real_repeated_inline_shape_controls_reach_exact_bridge_descriptors():
    original, optimized, roles = _fixture()
    inline = onnx.inliner.inline_local_functions(deepcopy(original))
    branches = tuple(_branches(inline))
    assert len(branches) == 4
    assert all(
        not {
            node.output[0]
            for node in branch.node
            if node.op_type in {"Reshape", "Expand"}
        }.intersection(value.name for value in branch.value_info)
        for branch in branches
    )
    ordinary = onnx.shape_inference.infer_shapes(deepcopy(inline), strict_mode=True)
    for branch in _branches(ordinary):
        rows = [
            value
            for value in branch.value_info
            if any(
                value.name in node.output
                for node in branch.node
                if node.op_type == "Expand"
            )
        ]
        assert len(rows) == 1
        # A fresh computed unknown is never identified with the input symbol B.
        assert all(dimension != "B" for dimension in _shape(rows[0]))
    actual_engine = bridge.expand_proven_fusions
    observed = []

    def capture(source, target, *arguments, **keywords):
        for graph in (source, target):
            assert [graph.types[name].shape for name in ("data0", "data1")] == [
                ("B", 2),
                ("B", 2),
            ]
            children = [
                attribute.value
                for node in graph.nodes
                if node.op == "If"
                for attribute in node.attributes.values()
                if attribute.kind == "GRAPH"
            ]
            assert len(children) == 4
            for child in children:
                outputs = [
                    name
                    for node in child.nodes
                    if node.op in {"Reshape", "Expand"}
                    for name in node.outputs
                ]
                assert len(outputs) == 2
                assert all(
                    child.types[name].dtype == onnx.TensorProto.FLOAT
                    and child.types[name].shape == (1, 2)
                    for name in outputs
                )
            observed.append(graph)
        return actual_engine(source, target, *arguments, **keywords)

    with patch.object(bridge, "expand_proven_fusions", capture):
        virtual, kinds = _prove_unchanged(original, optimized, roles)
    assert len(observed) == 2 and kinds == ("Gemm",) * 4
    assert all(
        node.op_type != "Gemm" for branch in _branches(virtual) for node in branch.node
    )
    assert [_shape(value) for value in original.graph.input[::2]] == [("B", 2)] * 2


@pytest.mark.parametrize(
    "fault",
    [
        "known-shape",
        "known-dtype",
        "internal-declaration",
        "symbol",
        "shape-control",
        "bias-words",
    ],
)
def test_real_shape_propagation_preserves_contradictions_and_exact_bytes(fault):
    original, optimized, roles = _fixture()
    if fault == "known-shape":
        original.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 3
    elif fault == "known-dtype":
        original.graph.output[0].type.tensor_type.elem_type = onnx.TensorProto.INT64
    elif fault == "internal-declaration":
        branch = next(
            attribute.g
            for attribute in original.functions[0].node[0].attribute
            if attribute.name == "then_branch"
        )
        row = next(value for value in branch.value_info if value.name == "positive_row")
        row.type.tensor_type.shape.dim[1].dim_value = 3
    elif fault == "symbol":
        optimized.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "Other"
    else:
        branch = next(
            attribute.g
            for attribute in optimized.functions[0].node[0].attribute
            if attribute.name == "then_branch"
        )
        constant = next(
            node
            for node in branch.node
            if node.output[0]
            == ("positive_two" if fault == "shape-control" else "positive_bias")
        )
        tensor = constant.attribute[0].t
        if fault == "shape-control":
            tensor.int64_data[0] = 1
        else:
            tensor.float_data[0] = 1.25
    with pytest.raises((onnx.shape_inference.InferenceError, FusionUnfinished)):
        _prove_unchanged(original, optimized, roles)
