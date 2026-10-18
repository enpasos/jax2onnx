# tests/extra_tests/converter/test_source_arithmetic_fusion.py

"""Source operation boundaries stay explicit until an alternative is audited."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from jax2onnx import to_onnx
from tests.extra_tests.converter.precision_test_support import export_source_dtypes


@pytest.fixture(autouse=True)
def _restore_config() -> Iterator[None]:
    previous = jax.config.x64_enabled
    try:
        jax.config.update("jax_enable_x64", True)
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _saved_session(
    model: onnx.ModelProto, tmp_path: Path, optimize: bool
) -> ort.InferenceSession:
    path = tmp_path / "source_arithmetic.onnx"
    onnx.save_model(model, path)
    reloaded = onnx.load_model(path)
    onnx.checker.check_model(reloaded, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimize
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    return ort.InferenceSession(
        reloaded.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


@pytest.mark.parametrize("divisor", [np.float32(2), np.float32(2.00001)])
@pytest.mark.parametrize("optimize", [False, True], ids=["ort-off", "ort-on"])
def test_source_add_div_keeps_exact_divisor_and_rounded_addition(
    divisor: np.float32, optimize: bool, tmp_path: Path
) -> None:
    def model(x: jax.Array, y: jax.Array) -> tuple[jax.Array, jax.Array]:
        added = x + y
        return added, added / divisor

    x = np.asarray([1, 2**24, np.finfo(np.float32).max], dtype=np.float32)
    y = np.asarray([1, 1, np.finfo(np.float32).max], dtype=np.float32)
    exported = export_source_dtypes(
        model, [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in (x, y)]
    )
    nodes = {node.op_type: node for node in exported.graph.node}
    assert "Mean" not in nodes
    assert nodes["Div"].input[0] == nodes["Add"].output[0]
    assert exported.graph.output[0].name == nodes["Add"].output[0]
    initializers = {
        value.name: onnx.numpy_helper.to_array(value)
        for value in exported.graph.initializer
    }
    np.testing.assert_array_equal(initializers[nodes["Div"].input[1]], divisor)
    expected = model(jnp.asarray(x), jnp.asarray(y))
    session = _saved_session(exported, tmp_path, optimize)
    actual = session.run(
        None,
        {session.get_inputs()[0].name: x, session.get_inputs()[1].name: y},
    )
    for result, reference in zip(actual, expected, strict=True):
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, reference)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_source_square_reduction_consumes_rounded_product(dtype: Any) -> None:
    def model(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        square = x * x
        return square, jax.lax.reduce_sum(square, axes=(0,))

    x = np.full(8, 2**-13, dtype=dtype)
    exported = export_source_dtypes(model, [jax.ShapeDtypeStruct(x.shape, x.dtype)])
    nodes = {node.op_type: node for node in exported.graph.node}
    assert "ReduceSumSquare" not in nodes
    assert nodes["ReduceSum"].input[0] == nodes["Mul"].output[0]
    assert exported.graph.output[0].name == nodes["Mul"].output[0]
    onnx.checker.check_model(exported, full_check=True)
    expected = model(jnp.asarray(x))
    actual = ReferenceEvaluator(exported).run(None, {exported.graph.input[0].name: x})
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, reference)
    # ORT's separate known FP16 rounding discrepancy remains a diagnostic gate;
    # valid unfused structure alone does not certify that target's numerics.


def _normalization(x: jax.Array, *, use_numpy: bool) -> tuple[jax.Array, ...]:
    square = x * x
    total = jax.lax.reduce_sum(square, axes=(1,))
    root = jnp.sqrt(total) if use_numpy else jax.lax.sqrt(total)
    return square, total, root, x / root[:, None]


@pytest.mark.parametrize("use_numpy", [False, True], ids=["lax-sqrt", "numpy-sqrt"])
@pytest.mark.parametrize("optimize", [False, True], ids=["ort-off", "ort-on"])
def test_source_square_sum_sqrt_div_keeps_operation_boundaries(
    use_numpy: bool, optimize: bool, tmp_path: Path
) -> None:
    x = np.asarray([[3, 4], [0, 0], [1e20, 1e20]], dtype=np.float32)

    def model(value: jax.Array) -> tuple[jax.Array, ...]:
        return _normalization(value, use_numpy=use_numpy)

    exported = export_source_dtypes(model, [jax.ShapeDtypeStruct(x.shape, x.dtype)])
    nodes = {node.op_type: node for node in exported.graph.node}
    assert {"Mul", "ReduceSum", "Sqrt", "Div"} <= nodes.keys()
    assert not {"ReduceSumSquare", "ReduceL2", "LpNormalization"} & nodes.keys()
    assert nodes["ReduceSum"].input[0] == nodes["Mul"].output[0]
    assert nodes["Sqrt"].input[0] == nodes["ReduceSum"].output[0]
    session = _saved_session(exported, tmp_path, optimize)
    actual = session.run(None, {session.get_inputs()[0].name: x})
    expected = model(jnp.asarray(x))
    for result, reference in zip(actual, expected, strict=True):
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, reference)
    # Ordinary division preserves 0/0 -> NaN. A normalized zero row producing
    # zero instead would silently change the source operation's behavior.
    assert np.isnan(actual[-1][1]).all()


@pytest.mark.parametrize("kind", ["mean", "normalization"])
def test_source_and_legacy_fusion_choices_are_isolated(kind: str) -> None:
    if kind == "mean":

        def model(x: jax.Array) -> jax.Array:
            return (x + x) / np.float32(2)

        source_ops = {"Add", "Div"}
        legacy_op = "Mean"
    else:

        def model(x: jax.Array) -> jax.Array:
            return _normalization(x, use_numpy=False)[-1]

        source_ops = {"Mul", "ReduceSum", "Sqrt", "Div"}
        legacy_op = "LpNormalization"

    specs = [jax.ShapeDtypeStruct((1, 2), jnp.float32)]
    before = export_source_dtypes(model, specs)
    legacy = to_onnx(model, specs, enable_double_precision=False)
    after = export_source_dtypes(model, specs)
    assert legacy_op in {node.op_type for node in legacy.graph.node}
    for exported in (before, after):
        operations = {node.op_type for node in exported.graph.node}
        assert source_ops <= operations
        assert legacy_op not in operations
