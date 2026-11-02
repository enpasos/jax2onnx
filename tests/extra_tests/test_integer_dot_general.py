# tests/extra_tests/test_integer_dot_general.py

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from jax2onnx import to_onnx


def _native_reference(
    function: Callable[..., jax.Array],
    inputs: list[np.ndarray],
    dtype: type = np.int32,
) -> np.ndarray:
    args = [jnp.asarray(value) for value in inputs]
    eager = np.asarray(function(*args))

    # A new wrapper gives each reference a fresh JIT cache, then exercises it warm.
    def call(*values: jax.Array) -> jax.Array:
        return function(*values)

    compiled = jax.jit(call)
    fresh = np.asarray(compiled(*args))
    warm = np.asarray(compiled(*args))
    for result in (eager, fresh, warm):
        assert result.dtype == dtype
        np.testing.assert_array_equal(result, eager)
    return eager


def _run_model(
    model: onnx.ModelProto, inputs: list[np.ndarray], *, optimized: bool = True
) -> np.ndarray:
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimized
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    (actual,) = session.run(
        None,
        {meta.name: value for meta, value in zip(session.get_inputs(), inputs)},
    )
    return np.asarray(actual)


def _run(
    function: Callable[..., jax.Array],
    inputs: list[np.ndarray],
    *,
    enable_double_precision: bool = False,
    optimized: bool = True,
    input_specs: list[jax.ShapeDtypeStruct] | None = None,
    expect_integer_matmul: bool = True,
    expect_contract_padding: bool = False,
    opset: int = 23,
) -> tuple[onnx.ModelProto, np.ndarray]:
    model = to_onnx(
        function,
        inputs if input_specs is None else input_specs,
        enable_double_precision=enable_double_precision,
        opset=opset,
    )
    assert (
        any(node.op_type == "MatMulInteger" for node in model.graph.node)
        == expect_integer_matmul
    )
    if not expect_integer_matmul:
        assert any(node.op_type == "ConstantOfShape" for node in model.graph.node)
    assert sum(node.op_type == "Pad" for node in model.graph.node) == (
        2 if expect_contract_padding else 0
    )
    assert not any(node.op_type in {"Gemm", "Einsum"} for node in model.graph.node)
    return model, _run_model(model, inputs, optimized=optimized)


def _matrix_dot(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    return jax.lax.dot_general(
        lhs,
        rhs,
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.int32,
    )


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("enable_double_precision", [False, True])
def test_int8_dot_preserves_odd_integer_above_float32_exact_range(
    optimized: bool, enable_double_precision: bool
) -> None:
    lhs = np.full((1, 1041), 127, dtype=np.int8)
    rhs = np.full((1041, 1), 127, dtype=np.int8)
    expected = _native_reference(_matrix_dot, [lhs, rhs])
    assert expected.dtype == np.int32
    assert expected.item() == 16_790_289
    assert int(np.float32(expected.item())) != expected.item()

    model, actual = _run(
        _matrix_dot,
        [lhs, rhs],
        enable_double_precision=enable_double_precision,
        optimized=optimized,
    )

    assert model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.INT32
    assert actual.dtype == np.int32
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("lhs_dtype", [np.int8, np.uint8])
@pytest.mark.parametrize("rhs_dtype", [np.int8, np.uint8])
@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("opset", [10, 23])
def test_integer_dot_signedness_and_extreme_values(
    lhs_dtype: type, rhs_dtype: type, optimized: bool, opset: int
) -> None:
    def values(dtype: type) -> np.ndarray:
        if dtype == np.int8:
            return np.asarray([-128, -127, -1, 0, 1, 126, 127], dtype=dtype)
        return np.asarray([0, 1, 2, 127, 128, 254, 255], dtype=dtype)

    lhs = np.stack([values(lhs_dtype), values(lhs_dtype)[::-1]])
    rhs = np.stack([values(rhs_dtype), values(rhs_dtype)[::-1]], axis=1)
    expected = lhs.astype(np.int32) @ rhs.astype(np.int32)
    native = _native_reference(_matrix_dot, [lhs, rhs])
    np.testing.assert_array_equal(native, expected)

    model, actual = _run(_matrix_dot, [lhs, rhs], optimized=optimized, opset=opset)
    assert actual.dtype == np.int32
    np.testing.assert_array_equal(actual, expected)
    if lhs_dtype == np.uint8 and rhs_dtype == np.int8:
        # U8S8 can saturate adjacent INT16 pair sums on non-VNNI x86 CPUs,
        # although a VNNI machine produces the correct result for this witness.
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
        types = {
            value.name: value.type.tensor_type.elem_type
            for value in (*inferred.graph.input, *inferred.graph.value_info)
        }
        types.update(
            (tensor.name, tensor.data_type) for tensor in inferred.graph.initializer
        )
        integer_dot = next(
            node for node in model.graph.node if node.op_type == "MatMulInteger"
        )
        assert types[integer_dot.input[0]] == onnx.TensorProto.UINT8
        assert types[integer_dot.input[1]] == onnx.TensorProto.UINT8


@pytest.mark.parametrize(
    ("lhs_shape", "rhs_shape", "dimension_numbers"),
    [
        ((4, 3), (2, 4), (((0,), (1,)), ((), ()))),
        ((2, 3, 4, 5), (2, 4, 5, 6), (((2, 3), (1, 2)), ((0,), (0,)))),
        ((3, 2, 4), (4, 5, 2), (((2,), (0,)), ((1,), (2,)))),
        ((2, 3, 4, 5), (5, 2, 6, 4), (((3, 2), (0, 3)), ((0,), (1,)))),
        ((2, 3, 4), (4, 5, 6), (((2,), (0,)), ((), ()))),
        ((3,), (3,), (((0,), (0,)), ((), ()))),
        ((2, 3), (2, 3), (((0, 1), (0, 1)), ((), ()))),
        ((3,), (4,), (((), ()), ((), ()))),
        ((2, 3), (2, 4), (((), ()), ((0,), (0,)))),
        ((2, 0), (0, 3), (((1,), (0,)), ((), ()))),
    ],
    ids=[
        "transpose_both",
        "batched_multiple_contracts",
        "different_batch_axes",
        "permuted_contract_order",
        "multiple_free_axes",
        "vector_scalar_result",
        "full_contraction",
        "outer_product",
        "batched_outer_product",
        "empty_contraction",
    ],
)
def test_integer_dot_general_axis_groups(
    lhs_shape: tuple[int, ...],
    rhs_shape: tuple[int, ...],
    dimension_numbers: tuple,
) -> None:
    lhs = (np.arange(np.prod(lhs_shape)).reshape(lhs_shape) % 17 - 8).astype(np.int8)
    rhs = (np.arange(np.prod(rhs_shape)).reshape(rhs_shape) % 13 - 6).astype(np.int8)

    def dot(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
        return jax.lax.dot_general(
            lhs, rhs, dimension_numbers, preferred_element_type=jnp.int32
        )

    expected = _native_reference(dot, [lhs, rhs])
    zero_contraction = any(lhs_shape[axis] == 0 for axis in dimension_numbers[0][0])
    _, actual = _run(dot, [lhs, rhs], expect_integer_matmul=not zero_contraction)
    assert actual.dtype == np.int32
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize(
    ("lhs_shape", "rhs_shape", "dimension_numbers"),
    [
        ((2, 0), (0, 3), (((1,), (0,)), ((), ()))),
        ((0,), (0,), (((0,), (0,)), ((), ()))),
        ((0, 0), (0, 3), (((1,), (0,)), ((), ()))),
        ((2, 3, 0, 4), (2, 0, 4, 5), (((2, 3), (1, 2)), ((0,), (0,)))),
    ],
    ids=["matrix", "scalar", "empty_free_dimension", "multiple_contracts"],
)
def test_zero_contraction_uses_int32_zeros_without_integer_kernel(
    optimized: bool,
    lhs_shape: tuple[int, ...],
    rhs_shape: tuple[int, ...],
    dimension_numbers: tuple,
) -> None:
    lhs = np.empty(lhs_shape, dtype=np.int8)
    rhs = np.empty(rhs_shape, dtype=np.int8)

    def dot(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
        return jax.lax.dot_general(
            lhs, rhs, dimension_numbers, preferred_element_type=jnp.int32
        )

    expected = _native_reference(dot, [lhs, rhs])
    np.testing.assert_array_equal(expected, np.zeros(expected.shape, dtype=np.int32))
    model, actual = _run(
        dot, [lhs, rhs], optimized=optimized, expect_integer_matmul=False
    )
    for result in (
        actual,
        *(_run_model(model, [lhs, rhs], optimized=optimized) for _ in range(3)),
    ):
        assert result.dtype == np.int32
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("optimized", [False, True])
def test_zero_contraction_preserves_symbolic_output_dimensions(optimized: bool) -> None:
    def dot(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
        return jax.lax.dot_general(
            lhs,
            rhs,
            (((2,), (1,)), ((0,), (0,))),
            preferred_element_type=jnp.int32,
        )

    model = None
    for batch, left, right in ((1, 2, 3), (3, 4, 5)):
        lhs = np.empty((batch, left, 0), dtype=np.int8)
        rhs = np.empty((batch, 0, right), dtype=np.int8)
        expected = _native_reference(dot, [lhs, rhs])
        if model is None:
            model, actual = _run(
                dot,
                [lhs, rhs],
                optimized=optimized,
                expect_integer_matmul=False,
                input_specs=[
                    jax.ShapeDtypeStruct(("B", "M", 0), jnp.int8),
                    jax.ShapeDtypeStruct(("B", 0, "N"), jnp.int8),
                ],
            )
        else:
            actual = _run_model(model, [lhs, rhs], optimized=optimized)
        assert actual.shape == (batch, left, right)
        assert actual.dtype == np.int32
        np.testing.assert_array_equal(actual, expected)


def test_integer_dot_preserves_symbolic_batch_and_free_dimensions() -> None:
    def dot(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
        return jax.lax.dot_general(
            lhs,
            rhs,
            (((3,), (1,)), ((0,), (0,))),
            preferred_element_type=jnp.int32,
        )

    model = None
    for batch, free in ((1, 2), (3, 4)):
        lhs = (np.arange(batch * free * 12).reshape(batch, free, 3, 4) % 17 - 8).astype(
            np.int8
        )
        rhs = (np.arange(batch * 20).reshape(batch, 4, 5) % 13 - 6).astype(np.int8)
        expected = _native_reference(dot, [lhs, rhs])
        if model is None:
            model, actual = _run(
                dot,
                [lhs, rhs],
                input_specs=[
                    jax.ShapeDtypeStruct(("B", "M", 3, 4), jnp.int8),
                    jax.ShapeDtypeStruct(("B", 4, 5), jnp.int8),
                ],
            )
        else:
            actual = _run_model(model, [lhs, rhs])
        assert actual.shape == (batch, free, 3, 5)
        assert actual.dtype == np.int32
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize(
    ("lhs_dtype", "rhs_dtype"),
    [
        (np.int8, np.int8),
        (np.uint8, np.uint8),
        (np.uint8, np.int8),
        (np.int8, np.uint8),
    ],
)
@pytest.mark.parametrize("opset", [11, 23])
def test_symbolic_contract_expression_includes_zero_and_exact_large_sum(
    optimized: bool, lhs_dtype: type, rhs_dtype: type, opset: int
) -> None:
    model = None
    # K is positive, while the actual contraction dimension K - 1 can be zero.
    # The final size also makes the exact sum odd and larger than 2**24.
    for k in (1, 2, 4, 1042):
        lhs = np.full((2, k - 1), 127, dtype=lhs_dtype)
        rhs = np.full((k - 1, 3), 127, dtype=rhs_dtype)
        expected = _native_reference(_matrix_dot, [lhs, rhs])
        np.testing.assert_array_equal(
            expected, np.full((2, 3), (k - 1) * 127**2, dtype=np.int32)
        )
        if model is None:
            model, actual = _run(
                _matrix_dot,
                [lhs, rhs],
                optimized=optimized,
                expect_contract_padding=True,
                opset=opset,
                input_specs=[
                    jax.ShapeDtypeStruct((2, "K - 1"), lhs_dtype),
                    jax.ShapeDtypeStruct(("K - 1", 3), rhs_dtype),
                ],
            )
        else:
            actual = _run_model(model, [lhs, rhs], optimized=optimized)
        assert actual.dtype == np.int32
        assert actual.shape == (2, 3)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("opset", [8, 9])
def test_integer_dot_rejects_opsets_without_matmul_integer(opset: int) -> None:
    lhs = np.ones((2, 3), dtype=np.int8)
    rhs = np.ones((3, 4), dtype=np.int8)
    with pytest.raises(NotImplementedError, match="opset >= 10.*MatMulInteger"):
        to_onnx(_matrix_dot, [lhs, rhs], opset=opset)


@pytest.mark.parametrize("opset", [10, 11, 13])
@pytest.mark.parametrize("grouped", [False, True])
def test_positive_integer_contractions_preserve_requested_older_opset(
    opset: int, grouped: bool
) -> None:
    lhs_shape = (2, 3, 4) if grouped else (6, 4)
    rhs_shape = (4, 5, 6) if grouped else (4, 30)
    lhs = (np.arange(24).reshape(lhs_shape) % 17 - 8).astype(np.int8)
    rhs = (np.arange(120).reshape(rhs_shape) % 13 - 6).astype(np.int8)

    def dot(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
        return jax.lax.dot_general(
            lhs,
            rhs,
            ((((2,) if grouped else (1,)), (0,)), ((), ())),
            preferred_element_type=jnp.int32,
        )

    expected = _native_reference(dot, [lhs, rhs])
    model, actual = _run(dot, [lhs, rhs], opset=opset)
    assert (
        next(item.version for item in model.opset_import if item.domain == "") == opset
    )
    assert all(
        attribute.name != "allowzero"
        for node in model.graph.node
        if node.op_type == "Reshape"
        for attribute in node.attribute
    )
    np.testing.assert_array_equal(actual, expected)


def test_symbolic_integer_contraction_reports_pad_opset_requirement() -> None:
    with pytest.raises(NotImplementedError, match="opset >= 11 for Pad"):
        to_onnx(
            _matrix_dot,
            [
                jax.ShapeDtypeStruct((2, "K - 1"), jnp.int8),
                jax.ShapeDtypeStruct(("K - 1", 3), jnp.int8),
            ],
            opset=10,
        )


@pytest.mark.parametrize("opset", [10, 11, 13])
def test_zero_free_dimension_reports_grouped_reshape_opset_requirement(
    opset: int,
) -> None:
    def dot(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
        return jax.lax.dot_general(
            lhs, rhs, (((2,), (0,)), ((), ())), preferred_element_type=jnp.int32
        )

    with pytest.raises(NotImplementedError, match="grouped-reshape.*opset >= 14"):
        to_onnx(
            dot,
            [np.empty((2, 0, 4), dtype=np.int8), np.ones((4, 3), dtype=np.int8)],
            opset=opset,
        )


def test_zero_free_dimension_grouped_reshape_works_at_opset14() -> None:
    lhs = np.empty((2, 0, 4), dtype=np.int8)
    rhs = np.ones((4, 3), dtype=np.int8)

    def dot(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
        return jax.lax.dot_general(
            lhs, rhs, (((2,), (0,)), ((), ())), preferred_element_type=jnp.int32
        )

    expected = _native_reference(dot, [lhs, rhs])
    _, actual = _run(dot, [lhs, rhs], opset=14)
    assert actual.dtype == np.int32
    assert actual.shape == (2, 0, 3)
    np.testing.assert_array_equal(actual, expected)


def test_integer_dot_composes_with_float32_scales() -> None:
    lhs = np.asarray([[-127, 63, 1], [5, -11, 127]], dtype=np.int8)
    rhs = np.asarray([[127, -4], [-31, 127], [9, -127]], dtype=np.int8)
    lhs_scale = np.asarray([[0.125], [0.25]], dtype=np.float32)
    rhs_scale = np.asarray([[0.5, 0.0625]], dtype=np.float32)

    def scaled_dot(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
        return (
            _matrix_dot(lhs, rhs).astype(jnp.float32)
            * jnp.asarray(lhs_scale)
            * jnp.asarray(rhs_scale)
        )

    model, actual = _run(scaled_dot, [lhs, rhs])
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    integer_dot = next(
        node for node in inferred.graph.node if node.op_type == "MatMulInteger"
    )
    values = {value.name: value for value in inferred.graph.value_info}
    assert (
        values[integer_dot.output[0]].type.tensor_type.elem_type
        == onnx.TensorProto.INT32
    )
    assert model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert actual.dtype == np.float32
    expected = (lhs.astype(np.int32) @ rhs.astype(np.int32)).astype(np.float32)
    expected = expected * lhs_scale * rhs_scale
    native = _native_reference(scaled_dot, [lhs, rhs], dtype=np.float32)
    np.testing.assert_array_equal(native, expected)
    np.testing.assert_array_equal(actual, expected)
