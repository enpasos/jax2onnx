# tests/extra_tests/converter/test_source_deployment_bindings.py

"""Actual IR identities survive clones and disambiguate identical lexical names."""

from dataclasses import replace

import onnx_ir as ir
import pytest

from jax2onnx import user_interface as ui
from jax2onnx.converter import source_deployment_bindings as binding
from jax2onnx.converter.source_deployment import (
    DEPLOYMENT_METADATA_KEY,
    MatrixPrecisionRequirement,
    _deployment_child_scope,
    _deployment_function_scope,
    deployment_requirements_from_json,
    deployment_requirements_to_json,
)
from tests.extra_tests.converter.test_source_deployment_target import _target


def _binding_fixture():
    left, right = [
        ir.val(name, ir.DataType.FLOAT, [2, 2]) for name in ("left", "right")
    ]
    predicate = ir.val("predicate", ir.DataType.BOOL, [])
    records, originals = [], []

    def body(inputs):
        matrix = ir.val("matrix", ir.DataType.FLOAT, [2, 2])
        logical = ir.val("result", ir.DataType.FLOAT, [2, 2])
        graph = ir.Graph(
            inputs,
            [logical],
            nodes=[
                ir.Node("", "MatMul", [left, right], outputs=[matrix]),
                ir.Node("", "Identity", [matrix], outputs=[logical]),
            ],
            name="same",
            opset_imports={"": 23},
        )
        requirement = MatrixPrecisionRequirement(
            "same",
            "dot_general",
            "result",
            ("HIGHEST",) * 2,
            ("float32",) * 3,
            None,
            "ort-cpu-x86-fp32-sgemm-v1",
            matrix_output="matrix",
        )
        records.append(binding.record_matrix_site(requirement, logical, matrix))
        originals.append((logical, matrix))
        return graph

    root = body([left, right, predicate])
    branches = [body([]), body([])]
    conditional = ir.val("conditional", ir.DataType.FLOAT, [2, 2])
    root.append(
        ir.Node(
            "",
            "If",
            [predicate],
            outputs=[conditional],
            attributes=[
                ir.AttrGraph("then_branch", branches[0]),
                ir.AttrGraph("else_branch", branches[1]),
            ],
        )
    )
    root.outputs.append(conditional)
    # Exercise the same two public shallow clones used by body/root construction.
    root = root.clone(allow_outer_scope_values=True).clone(
        allow_outer_scope_values=True
    )
    model = ir.Model(root, ir_version=10)
    binding.attach_matrix_bindings(model, _target(), records)
    return model, records, originals


def test_actual_matrix_bindings_survive_clone_and_custom_root_names():
    model, records, originals = _binding_fixture()
    requirements, resolved = binding.resolve_matrix_bindings(model)
    assert len(resolved) == len(records) == 3
    assert len({id(matrix.graph) for _, matrix in resolved}) == 3
    assert all(
        logical is not old_logical and matrix is not old_matrix
        for (logical, matrix), (old_logical, old_matrix) in zip(
            resolved, originals, strict=True
        )
    )
    assert all(
        (logical.name, matrix.name) == ("result", "matrix")
        for logical, matrix in resolved
    )
    ui._apply_custom_io_names_on_ir(
        model,
        input_names=["custom_left", "custom_right", "choice"],
        output_names=["answer", "conditional_answer"],
        positional_input_count=3,
    )
    renamed, observed = binding.resolve_matrix_bindings(model)
    assert observed == resolved
    assert [site.output for site in renamed.matrix_precision] == [
        "answer",
        "result",
        "result",
    ]
    assert all(site.matrix_output == "matrix" for site in renamed.matrix_precision)
    assert renamed.target == requirements.target
    ui._finalize_source_deployment(model)
    final, actual = binding.resolve_matrix_bindings(model)
    if_node = next(node for node in model.graph if node.op_type == "If")
    assert [site.matrix_scope for site in final.matrix_precision] == [
        "graph:same",
        _deployment_child_scope("graph:same", if_node.name, "then_branch", 0),
        _deployment_child_scope("graph:same", if_node.name, "else_branch", 0),
    ]
    for site, (logical, matrix) in zip(final.matrix_precision, actual, strict=True):
        assert site.output == logical.name and site.matrix_output == matrix.name
        assert site.matrix_tag == matrix.producer().name
        assert logical.graph is matrix.graph
    proto = ir.to_proto(model)
    assert (
        deployment_requirements_from_json(
            dict((item.key, item.value) for item in proto.metadata_props)[
                DEPLOYMENT_METADATA_KEY
            ]
        )
        == final
    )
    serialized = proto.SerializeToString()
    assert all(
        key.encode() not in serialized
        for key in (
            binding._GRAPH,
            binding._PRODUCER,
            binding._VALUES,
            binding._REGISTRY,
        )
    )


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "wrong-owner", "stale-producer"]
)
def test_actual_matrix_binding_rejects_invalid_identity(fault):
    model, _, _ = _binding_fixture()
    _, values = binding.resolve_matrix_bindings(model)
    root_matrix, branch_matrix = values[0][1], values[1][1]
    if fault == "missing":
        del root_matrix.meta[binding._VALUES]
    elif fault == "duplicate":
        root_matrix.meta[binding._VALUES] *= 2
    elif fault == "wrong-owner":
        branch_matrix.meta[binding._VALUES] = root_matrix.meta[binding._VALUES]
    else:
        root_matrix.producer().meta[binding._PRODUCER] = object()
    with pytest.raises(ValueError, match="Deployment"):
        ui._finalize_source_deployment(model)


def test_actual_matrix_binding_rejects_changed_metadata_and_nonlayout_descendant():
    model, _, _ = _binding_fixture()
    declared, values = binding.resolve_matrix_bindings(model)
    model.metadata_props[DEPLOYMENT_METADATA_KEY] = deployment_requirements_to_json(
        replace(
            declared, matrix_precision=tuple(reversed(declared.matrix_precision[:-1]))
        )
    )
    with pytest.raises(ValueError, match="binding registry"):
        ui._finalize_source_deployment(model)
    model.metadata_props[DEPLOYMENT_METADATA_KEY] = deployment_requirements_to_json(
        declared
    )
    values[0][0].producer().domain = "foreign"
    with pytest.raises(ValueError, match="layout-only"):
        ui._finalize_source_deployment(model)


def test_lexical_scope_encodes_function_overload_and_full_containment():
    ordinary = _deployment_function_scope("example", "F", "")
    overloaded = _deployment_function_scope("example", "F", "variant")
    assert ordinary == "function:example:F" and overloaded != ordinary
    assert _deployment_function_scope(
        "example:F", "G", ""
    ) != _deployment_function_scope("example", "F:G", "")
    root = "graph:same"
    first = _deployment_child_scope(root, "jax2onnx_src_000001", "body", 0)
    assert first != _deployment_child_scope(root, "jax2onnx_src_000002", "body", 0)
    assert first != _deployment_child_scope(root, "jax2onnx_src_000001", "body", 1)
    assert first != _deployment_child_scope(root, "jax2onnx_src_000001", "other", 0)
    assert _deployment_child_scope(first, "jax2onnx_src_000003", "body", 0).startswith(
        first + "/"
    )
