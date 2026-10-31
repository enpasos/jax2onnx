# tests/extra_tests/converter/test_source_final_constant_origin.py

"""Final matrix/Cast equality retains exact lexical Constant definitions."""

from copy import deepcopy

import onnx
import pytest

from jax2onnx.converter.source_deployment import (
    DeploymentRequirements,
    ort_cpu_deployment_target,
)
from jax2onnx.precision import _deployment_graph_boundary


def _vi(name, dtype=onnx.TensorProto.FLOAT, shape=(2, 2)):
    return onnx.helper.make_tensor_value_info(name, dtype, shape)


def _node(tag, op, inputs, outputs, **attrs):
    return onnx.helper.make_node(
        op, inputs, outputs, name=f"jax2onnx_src_{tag:06d}", **attrs
    )


def _constant(
    tag, output, dtype=onnx.TensorProto.FLOAT, shape=(2, 2), values=(1, 2, 3, 4)
):
    return _node(
        tag,
        "Constant",
        [],
        [output],
        value=onnx.helper.make_tensor("payload", dtype, shape, values),
    )


def _requirements():
    # A declarative structural fixture; no runtime/provider acceptance is claimed.
    return DeploymentRequirements(
        ort_cpu_deployment_target("1.31.0", "cpython-312"), ()
    )


def _pair(fault=None):
    leaves = []
    for index, tag in enumerate((8, 10)):
        matrix, result = f"matrix{index}", f"result{index}"
        leaves.append(
            onnx.helper.make_graph(
                [
                    _node(tag, "MatMul", ["data", "weights"], [matrix]),
                    _node(
                        tag + 1, "Cast", [matrix], [result], to=onnx.TensorProto.DOUBLE
                    ),
                ],
                f"leaf{index}",
                [],
                [_vi(result, onnx.TensorProto.DOUBLE)],
                value_info=[_vi("weights")],
            )
        )
    inner = _node(
        21,
        "If",
        ["inner_predicate"],
        ["inner_result"],
        then_branch=leaves[0],
        else_branch=leaves[1],
    )
    outer_then = onnx.helper.make_graph(
        [inner], "outer_then", [], [_vi("inner_result", onnx.TensorProto.DOUBLE)]
    )
    if fault == "parent-layout":
        # A computed local shadows the parent Constant; its layout must not be
        # erased by passing an inherited stripped Constant identity.
        outer_then.node.insert(
            0, _node(5, "Transpose", ["other_weights"], ["weights"], perm=[1, 0])
        )
    elif fault == "initializer":
        outer_then.initializer.append(
            onnx.helper.make_tensor(
                "weights", onnx.TensorProto.FLOAT, [2, 2], [1, 2, 3, 4]
            )
        )
    outer_else = onnx.helper.make_graph(
        [_node(12, "Cast", ["data"], ["fallback"], to=onnx.TensorProto.DOUBLE)],
        "outer_else",
        [],
        [_vi("fallback", onnx.TensorProto.DOUBLE)],
    )
    root = onnx.helper.make_graph(
        [
            _constant(2, "weights"),
            _constant(4, "other_weights"),
            _node(
                20,
                "If",
                ["predicate"],
                ["out"],
                then_branch=outer_then,
                else_branch=outer_else,
            ),
        ],
        "static",
        [
            _vi("data"),
            _vi("predicate", onnx.TensorProto.BOOL, ()),
            _vi("inner_predicate", onnx.TensorProto.BOOL, ()),
        ],
        [_vi("out", onnx.TensorProto.DOUBLE)],
    )
    original = onnx.helper.make_model(
        root, ir_version=10, opset_imports=[onnx.helper.make_opsetid("", 23)]
    )
    optimized = deepcopy(original)
    outer = next(
        attribute.g
        for attribute in optimized.graph.node[-1].attribute
        if attribute.name == "then_branch"
    )
    nested = next(node for node in outer.node if node.op_type == "If")
    for index, attribute in enumerate(nested.attribute):
        branch = attribute.g
        clone = _constant(4 if fault == "role" else 2, f"local_weights{index}")
        clone.name = "cloned_" + clone.name
        clone.attribute[0].t.name = f"different_tensor_name{index}"
        if fault == "words":
            clone.attribute[0].t.float_data[0] = 9
        elif fault == "dtype":
            clone.attribute[0].t.CopyFrom(
                onnx.helper.make_tensor(
                    "changed_dtype", onnx.TensorProto.DOUBLE, [2, 2], [1, 2, 3, 4]
                )
            )
        elif fault == "shape":
            clone.attribute[0].t.dims[:] = [1, 4]
        branch.node.insert(0, clone)
        matrix = next(node for node in branch.node if node.op_type == "MatMul")
        matrix.input[1] = clone.output[0]
    return original, optimized


def _prove_unchanged(original, optimized):
    before = original.SerializeToString(), optimized.SerializeToString()
    try:
        return _deployment_graph_boundary(original, optimized, _requirements())
    finally:
        assert (original.SerializeToString(), optimized.SerializeToString()) == before


def test_final_contract_matches_two_scope_capture_and_same_source_local_clone():
    original, optimized = _pair()
    assert (
        _prove_unchanged(original, optimized)
        == "matmul-transpose-and-original-casts-v1"
    )


@pytest.mark.parametrize(
    "fault", ["words", "dtype", "shape", "role", "parent-layout", "initializer"]
)
def test_final_constant_origin_rejects_typed_storage_role_and_lexical_shadow(fault):
    original, optimized = _pair(fault)
    with pytest.raises((RuntimeError, onnx.shape_inference.InferenceError)):
        _prove_unchanged(original, optimized)


def test_loop_formal_shadows_same_named_parent_constant():
    body = onnx.helper.make_graph(
        [
            _node(8, "MatMul", ["data", "weights"], ["matrix"]),
            _node(9, "Cast", ["matrix"], ["result"], to=onnx.TensorProto.DOUBLE),
        ],
        "body",
        [
            _vi("iteration", onnx.TensorProto.INT64, ()),
            _vi("condition", onnx.TensorProto.BOOL, ()),
            _vi("weights"),
            _vi("data"),
        ],
        [
            _vi("condition", onnx.TensorProto.BOOL, ()),
            _vi("weights"),
            _vi("data"),
            _vi("result", onnx.TensorProto.DOUBLE),
        ],
    )
    root = onnx.helper.make_graph(
        [
            _constant(2, "weights"),
            _constant(31, "trip_count", onnx.TensorProto.INT64, (), [2]),
            _node(
                30,
                "Loop",
                ["trip_count", "condition", "dynamic_weights", "data"],
                ["final_weights", "final_data", "scan"],
                body=body,
            ),
        ],
        "formal",
        [
            _vi("data"),
            _vi("dynamic_weights"),
            _vi("condition", onnx.TensorProto.BOOL, ()),
        ],
        [_vi("scan", onnx.TensorProto.DOUBLE, (None, 2, 2))],
    )
    original = onnx.helper.make_model(
        root, ir_version=10, opset_imports=[onnx.helper.make_opsetid("", 23)]
    )
    optimized = deepcopy(original)
    target_body = optimized.graph.node[-1].attribute[0].g
    clone = _constant(2, "weights_local")
    clone.name = "cloned_" + clone.name
    target_body.node.insert(0, clone)
    target_body.node[1].input[1] = "weights_local"
    with pytest.raises((RuntimeError, onnx.shape_inference.InferenceError)):
        _prove_unchanged(original, optimized)
