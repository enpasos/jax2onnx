# tests/extra_tests/converter/test_source_repeated_function_if_fusion.py

"""Real repeated Function/If fusion regression; ROOT retains actual failures."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx.converter.precision_policy import PrecisionPolicy, active_precision_policy
from jax2onnx.precision import (
    _configure_deployment_session,
    _observe_deployment_session,
    get_precision_report,
)
from tests.extra_tests.converter.test_source_deployment_target import (
    _assert_runtime_observation,
    _target,
)
from tests.extra_tests.converter.test_source_ordinary_matrix import (
    _restore_settings as _restore_settings,
)

_DIMENSIONS = (((1,), (0,)), ((), ()))


def _positive(left: Any) -> Any:
    rounded = jax.lax.convert_element_type(left, np.float32)
    weights = np.asarray([[1.0, 2.0], [-1.0, 0.5]], np.float32)
    bias = np.asarray([0.25, -0.5], np.float32)
    product = jax.lax.dot_general(
        rounded, weights, _DIMENSIONS, precision=jax.lax.Precision.HIGHEST
    )
    return jax.lax.convert_element_type(product + bias, np.float64)


def _negative(left: Any) -> Any:
    rounded = jax.lax.convert_element_type(left, np.float32)
    weights = np.asarray([[-0.5, 1.0], [2.0, -1.0]], np.float32)
    bias = np.asarray([-0.75, 0.125], np.float32)
    product = jax.lax.dot_general(
        rounded, weights, _DIMENSIONS, precision=jax.lax.Precision.HIGHEST
    )
    return jax.lax.convert_element_type(product + bias, np.float64)


@onnx_function(name="RepeatedConditionalGemm")
def _named_conditional(left: Any, predicate: Any) -> Any:
    return jax.lax.cond(predicate, _positive, _negative, left)


def _same(actual: Any, expected: Any) -> None:
    assert len(actual) == len(expected) == 6
    for observed, reference in zip(actual, expected, strict=True):
        assert observed.dtype == reference.dtype and observed.shape == reference.shape
        assert observed.tobytes() == reference.tobytes(), "bitwise native/CPU mismatch"


def _native(function: Any, values: Any, directory: Path) -> Any:
    directory.mkdir()
    compiled = jax.jit(function)
    references, events = [], []
    original = tuple(value.tobytes() for value in values)
    for role, call in (
        ("eager", function),
        ("fresh-jit", compiled),
        ("warm-jit", compiled),
    ):
        assert active_precision_policy() is PrecisionPolicy.LEGACY
        host = tuple(value.copy() for value in values)
        arguments = tuple(jnp.asarray(value) for value in host)
        events.append({"role": role, "status": "started"})
        (directory / "native-events.json").write_text(json.dumps(events, indent=2))
        returned = call(*arguments)
        actual = tuple(np.asarray(value).copy() for value in jax.tree.leaves(returned))
        np.savez(
            directory / f"native-{role}.npz",
            **{f"output{i}": value for i, value in enumerate(actual)},
        )
        events[-1]["status"] = "returned"
        (directory / "native-events.json").write_text(json.dumps(events, indent=2))
        assert tuple(value.tobytes() for value in host) == original
        assert tuple(np.asarray(value).tobytes() for value in arguments) == original
        references.append(actual)
    for reference in references:
        _same(reference, references[0])
    assert tuple(value.tobytes() for value in values) == original
    return tuple(references)


def _nodes(owner: Any, scope: tuple[str, ...] = ()) -> Any:
    for node in owner.node:
        yield scope, node
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from _nodes(attribute.g, (*scope, node.name, attribute.name))
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for index, graph in enumerate(attribute.graphs):
                    yield from _nodes(
                        graph, (*scope, node.name, attribute.name, str(index))
                    )


def _original_bindings(model: Any) -> Any:
    definitions = {
        (item.domain, item.name, item.overload): item for item in model.functions
    }
    assert (
        len(definitions) == 1
    ), "same-avals calls must share one actual Function definition"
    key, definition = next(iter(definitions.items()))
    calls = [
        node
        for node in model.graph.node
        if (node.domain, node.op_type, node.overload) in definitions
    ]
    assert len(calls) == 2 and all(
        len(node.input) == 2 and len(node.output) == 1 for node in calls
    )
    assert len(definition.input) == 2 and len(definition.output) == 1
    inputs = tuple(value.name for value in model.graph.input)
    assert len(inputs) == 4
    assert tuple(value.type.tensor_type.elem_type for value in model.graph.input) == (
        onnx.TensorProto.DOUBLE,
        onnx.TensorProto.BOOL,
        onnx.TensorProto.DOUBLE,
        onnx.TensorProto.BOOL,
    )
    assert all(value.type.tensor_type.HasField("shape") for value in model.graph.input)
    assert tuple(
        tuple(dimension.dim_value for dimension in value.type.tensor_type.shape.dim)
        for value in model.graph.input
    ) == ((2, 2), (), (2, 2), ())
    assert all(
        dimension.HasField("dim_value")
        for value in model.graph.input
        for dimension in value.type.tensor_type.shape.dim
    )
    producers = {name: node for node in model.graph.node for name in node.output}

    def origin(name: str, local: Any = producers) -> str:
        seen = set()
        while name in local and local[name].op_type == "Identity":
            assert name not in seen
            seen.add(name)
            node = local[name]
            assert not node.domain and not node.overload and not node.attribute
            assert len(node.input) == len(node.output) == 1
            name = node.input[0]
        return name

    by_binding = {tuple(origin(name) for name in node.input): node for node in calls}
    assert set(by_binding) == {(inputs[0], inputs[1]), (inputs[2], inputs[3])}
    outputs = tuple(origin(value.name) for value in model.graph.output)
    assert outputs == (
        by_binding[(inputs[0], inputs[1])].output[0],
        by_binding[(inputs[2], inputs[3])].output[0],
        inputs[0],
        inputs[2],
        inputs[1],
        inputs[3],
    ), "selected results or direct sibling outputs changed call binding"
    conditionals = [node for _, node in _nodes(definition) if node.op_type == "If"]
    assert len(conditionals) == 1
    conditional = conditionals[0]
    branches = {attribute.name: attribute.g for attribute in conditional.attribute}
    assert set(branches) == {"then_branch", "else_branch"}
    parent_nodes = {name: node for node in definition.node for name in node.output}
    predicate = conditional.input[0]
    seen = set()
    while predicate in parent_nodes:
        assert predicate not in seen
        seen.add(predicate)
        node = parent_nodes[predicate]
        assert node.op_type in {"Identity", "Cast"} and len(node.input) == 1
        predicate = node.input[0]
    assert (
        predicate == definition.input[1]
    ), "If predicate is not its actual BOOL formal"
    branch_evidence = {}
    for label, branch in branches.items():
        local = {name: node for node in branch.node for name in node.output}
        matrices = [node for node in branch.node if node.op_type == "MatMul"]
        assert len(matrices) == 1
        matrix = matrices[0]
        rounding = local[origin(matrix.input[0], local)]
        assert (
            rounding.op_type == "Cast"
            and origin(rounding.input[0], local) == definition.input[0]
        )
        assert (
            len(rounding.attribute) == 1
            and rounding.attribute[0].name == "to"
            and rounding.attribute[0].i == onnx.TensorProto.FLOAT
        )
        widening = local[origin(branch.output[0].name, local)]
        assert widening.op_type == "Cast" and len(widening.attribute) == 1
        assert (
            widening.attribute[0].name == "to"
            and widening.attribute[0].i == onnx.TensorProto.DOUBLE
        )
        addition = local[origin(widening.input[0], local)]
        assert addition.op_type == "Add" and len(addition.input) == 2
        assert matrix.output[0] in tuple(origin(name, local) for name in addition.input)
        branch_evidence[label] = {
            "graph": branch.name,
            "matrix": matrix.name,
            "rounding_cast": rounding.name,
            "widening_cast": widening.name,
        }
    return {
        "function_identifier": key,
        "function_formals": list(definition.input),
        "calls": [
            {
                "node": node.name,
                "actual_inputs": list(node.input),
                "outputs": list(node.output),
            }
            for node in calls
        ],
        "root_inputs": inputs,
        "If": conditional.name,
        "branches": branch_evidence,
        "direct_output_provenance": outputs[2:],
    }


@pytest.mark.parametrize("optimization", ("disabled", "all"))
def test_same_named_function_nested_if_keeps_actual_gemm_call_bindings(
    optimization: str,
    tmp_path: Path,
) -> None:
    def function(
        first: Any, first_predicate: Any, second: Any, second_predicate: Any
    ) -> Any:
        return (
            _named_conditional(first, first_predicate),
            _named_conditional(second, second_predicate),
            first,
            second,
            first_predicate,
            second_predicate,
        )

    values = (
        np.asarray([[1 + 2**-40, 2 + 2**-39], [-3 - 2**-41, 4 + 2**-38]], np.float64),
        np.asarray(True, np.bool_),
        np.asarray([[5 + 2**-40, -6 - 2**-39], [7 + 2**-41, -8 - 2**-38]], np.float64),
        np.asarray(False, np.bool_),
    )
    snapshots = tuple(value.tobytes() for value in values)
    np.savez(
        tmp_path / "original-inputs.npz",
        **{f"input{i}": value for i, value in enumerate(values)},
    )
    assert all(
        not np.array_equal(value, value.astype(np.float32).astype(np.float64))
        for value in (values[0], values[2])
    )
    with jax.enable_x64(True):
        references = _native(function, values, tmp_path / "native")
        model = ui._to_onnx(
            function,
            [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in values],
            precision_policy=PrecisionPolicy.SOURCE,
            deployment_target=_target(),
            opset=23,
        )
    original = model.SerializeToString()
    (tmp_path / "original-model.onnx").write_bytes(original)
    try:
        bindings = _original_bindings(model)
        bindings["original_artifact_sha256"] = hashlib.sha256(original).hexdigest()
        (tmp_path / "safe-original-bindings.json").write_text(
            json.dumps(bindings, indent=2)
        )
        report = get_precision_report(model)
        assert report is not None and report.deployment_requirements is not None
        assert report.deployment_requirements.matrix_precision
        path = tmp_path / f"optimized-{optimization}.onnx"
        options = ort.SessionOptions()
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            if optimization == "disabled"
            else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        _configure_deployment_session(options, report.deployment_requirements, path)
        session = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        session.disable_fallback()
        optimized = onnx.load_model(path)
        structure = [
            {
                "scope": scope,
                "name": node.name,
                "domain": node.domain,
                "op": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
            }
            for scope, node in _nodes(optimized.graph)
        ]
        (tmp_path / "safe-optimized-structure.json").write_text(
            json.dumps(structure, indent=2)
        )
        observation = _observe_deployment_session(model, session, ort, path)
        (tmp_path / "runtime-observation.json").write_text(
            json.dumps(asdict(observation), indent=2)
        )
        gemms = [item for item in structure if item["op"] == "Gemm"]
        if optimization == "all":
            assert any(
                item["scope"] for item in gemms
            ), "actual nested Gemm witness is required"
        assert observation.boundary_contract == (
            "matmul-gemm-quickgelu-and-original-casts-v2"
            if gemms
            else "matmul-transpose-and-original-casts-v1"
        )
        matched = _assert_runtime_observation(
            report.deployment_requirements,
            observation,
            optimization=optimization,
        )
        if matched:
            actual = tuple(
                session.run(
                    None,
                    dict(
                        zip(
                            (value.name for value in model.graph.input),
                            values,
                            strict=True,
                        )
                    ),
                )
            )
            np.savez(
                tmp_path / "actual-cpu-outputs.npz",
                **{f"output{i}": value for i, value in enumerate(actual)},
            )
            for reference in references:
                _same(actual, reference)
        else:
            assert not (tmp_path / "actual-cpu-outputs.npz").exists()
        (tmp_path / "runtime-target-gate.json").write_text(
            json.dumps(
                dict(
                    target_matched=matched,
                    target_gate="MATCHED" if matched else "REJECTED",
                    runtime="VALIDATED" if matched else "FAILED",
                    numerics="VALIDATED" if matched else "UNVERIFIED",
                    actual_target=asdict(observation.target),
                ),
                indent=2,
            )
        )
    finally:
        assert model.SerializeToString() == original
        assert tuple(value.tobytes() for value in values) == snapshots
        assert active_precision_policy() is PrecisionPolicy.LEGACY
