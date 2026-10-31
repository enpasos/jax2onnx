# tests/extra_tests/converter/test_source_normalization.py

"""Serialized source export of model-chosen FP64 variance accumulation."""

from collections.abc import Iterator
from pathlib import Path

import jax
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function
from tests.extra_tests.converter.precision_numerical_support import (
    normalization_input,
    normalization_reference,
    selected_variance_normalization,
)
from tests.extra_tests.converter.precision_test_support import export_source_dtypes


@onnx_function
def _normalization_function(x):
    return selected_variance_normalization(x)


def _call_normalization_function(x):
    return _normalization_function(x)


def _conditional_normalization(x, selector):
    return jax.lax.cond(
        selector > np.float32(0),
        selected_variance_normalization,
        lambda value: selected_variance_normalization(-value),
        x,
    )


@pytest.fixture(autouse=True)
def _restore_config() -> Iterator[None]:
    old_x64 = bool(jax.config.x64_enabled)
    old_platforms = jax.config.jax_platforms
    try:
        jax.config.update("jax_enable_x64", True)
        yield
    finally:
        jax.config.update("jax_enable_x64", old_x64)
        jax.config.update("jax_platforms", old_platforms)


def _all_nodes(model):
    def walk(nodes):
        for node in nodes:
            yield node
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    yield from walk(attribute.g.node)

    yield from walk(model.graph.node)
    for function in model.functions:
        yield from walk(function.node)


@pytest.mark.parametrize("boundary", ["inline", "function", "cond"])
@pytest.mark.parametrize("optimize", [False, True])
def test_source_normalization_retains_selected_double_sum(
    tmp_path: Path, boundary: str, optimize: bool
) -> None:
    x = normalization_input()
    fn = {
        "inline": selected_variance_normalization,
        "function": _call_normalization_function,
        "cond": _conditional_normalization,
    }[boundary]
    specs = [jax.ShapeDtypeStruct(x.shape, x.dtype)]
    if boundary == "cond":
        specs.append(jax.ShapeDtypeStruct((), np.float32))
    model = export_source_dtypes(fn, specs)
    path = tmp_path / "source_normalization.onnx"
    onnx.save_model(model, path)
    model = onnx.load_model(path)
    onnx.checker.check_model(model, full_check=True)
    assert all(
        value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        for value in model.graph.output
    )
    if boundary == "function":
        assert model.functions
    if boundary == "cond":
        assert any(node.op_type == "If" for node in model.graph.node)

    # The wider sum consumes an explicitly converted FP32 square. Keeping only
    # the output dtype, or widening after the sum, cannot satisfy this witness.
    nodes = list(_all_nodes(model))
    producer = {output: node for node in nodes for output in node.output}
    wider_sums = []
    for node in nodes:
        if node.op_type != "ReduceSum":
            continue
        operand = producer.get(node.input[0])
        if operand is None or operand.op_type != "Cast":
            continue
        if any(
            a.name == "to" and a.i == onnx.TensorProto.DOUBLE for a in operand.attribute
        ):
            assert producer[operand.input[0]].op_type == "Mul"
            wider_sums.append(node)
    assert wider_sums

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimize
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    session = ort.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )
    for selector in [np.float32(1), np.float32(-1)] if boundary == "cond" else [None]:
        inputs = [x] if selector is None else [x, np.asarray(selector)]
        expected = normalization_reference(
            x if selector is None or selector > 0 else -x
        )
        jax_result = fn(*inputs)
        actual = session.run(
            None,
            {
                spec.name: value
                for spec, value in zip(session.get_inputs(), inputs, strict=True)
            },
        )
        for result in (jax_result, actual):
            for value, reference in zip(result[:3], expected[:3], strict=True):
                assert value.dtype == np.float32
                np.testing.assert_array_equal(value, reference)
            np.testing.assert_array_equal(result[2], [[np.float32(8388609)], [0]])
            np.testing.assert_allclose(result[3], expected[3], rtol=3e-7, atol=1e-7)
            np.testing.assert_array_equal(result[3][1], np.zeros(16, dtype=np.float32))
