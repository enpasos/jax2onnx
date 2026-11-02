# tests/extra_tests/test_precision_control_flow.py

"""Source-typed constants, loop state and exact integer reductions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from jax2onnx import to_onnx
from jax2onnx.plugins.jax.lax.reduce_window_sum import reduce_window_sum


def _assert_exact(actual: Any, expected: Any) -> None:
    actual_leaves, actual_tree = jax.tree_util.tree_flatten(actual)
    expected_leaves, expected_tree = jax.tree_util.tree_flatten(expected)
    assert actual_tree == expected_tree
    for got, wanted in zip(actual_leaves, expected_leaves):
        got, wanted = np.asarray(got), np.asarray(wanted)
        assert got.shape == wanted.shape
        assert got.dtype == wanted.dtype
        np.testing.assert_array_equal(got, wanted)
        if np.issubdtype(wanted.dtype, np.floating):
            zeros = wanted == 0
            np.testing.assert_array_equal(
                np.signbit(got[zeros]), np.signbit(wanted[zeros])
            )


def _check_native_and_ort(
    fn: Callable[..., Any],
    args: Sequence[np.ndarray],
    expected: Any,
    *,
    inputs: Sequence[Any] | None = None,
) -> onnx.ModelProto:
    native_args = tuple(jnp.asarray(arg) for arg in args)
    _assert_exact(fn(*native_args), expected)
    compiled = jax.jit(fn)
    _assert_exact(compiled(*native_args), expected)
    _assert_exact(compiled(*native_args), expected)
    model = to_onnx(fn, inputs=list(inputs if inputs is not None else args))
    onnx.checker.check_model(model, full_check=True)
    for optimization in (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    ):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = optimization
        session = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        outputs = session.run(
            None, {meta.name: arg for meta, arg in zip(session.get_inputs(), args)}
        )
        expected_leaves, expected_tree = jax.tree_util.tree_flatten(expected)
        assert len(outputs) == len(expected_leaves)
        _assert_exact(jax.tree_util.tree_unflatten(expected_tree, outputs), expected)
    return model


def test_where_dynamic_fp64_result_keeps_shared_fp32_constant() -> None:
    constant = np.asarray([1.0000001, 16_777_216.0], dtype=np.float32)

    def fn(mask: jax.Array, x: jax.Array) -> tuple[Any, ...]:
        return jnp.where(mask, constant, x), constant, constant + jnp.float32(1)

    mask = np.asarray([[True], [False]], dtype=np.bool_)
    x = np.asarray(
        [[1.0000000000000002, 16_777_217], [2.25, 16_777_219]], dtype=np.float64
    )
    expected = (
        np.where(mask, constant.astype(np.float64), x),
        constant,
        constant + np.float32(1),
    )
    with jax.enable_x64():
        model = _check_native_and_ort(
            fn,
            [mask, x],
            expected,
            inputs=[
                jax.ShapeDtypeStruct(("B", 1), np.bool_),
                jax.ShapeDtypeStruct(("B", 2), np.float64),
            ],
        )
    where_nodes = [node for node in model.graph.node if node.op_type == "Where"]
    assert len(where_nodes) == 1
    assert model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.DOUBLE
    assert model.graph.output[1].type.tensor_type.elem_type == onnx.TensorProto.FLOAT


@pytest.mark.parametrize("enable_x64", [False, True], ids=["int32", "int64"])
def test_while_literal_counter_keeps_native_integer_width(enable_x64: bool) -> None:
    def fn() -> Any:
        return jax.lax.while_loop(lambda v: v < 5, lambda v: v + 1, 0)

    expected = np.asarray(5, dtype=np.int64 if enable_x64 else np.int32)
    with jax.enable_x64(enable_x64):
        _check_native_and_ort(fn, [], expected)


@pytest.mark.parametrize("dtype", [np.int32, np.int64], ids=["int32", "int64"])
@pytest.mark.parametrize("with_xs", [False, True], ids=["no-xs", "with-xs"])
def test_scan_source_integer_carry_and_outputs(dtype: Any, with_xs: bool) -> None:
    initial = np.asarray(np.iinfo(dtype).max - 1, dtype=dtype)
    xs = np.asarray([1, 2, 3], dtype=dtype)

    def fn(carry: jax.Array, values: jax.Array) -> Any:
        def body(c: jax.Array, value: jax.Array) -> Any:
            result = c + value
            return result, (result, result.astype(jnp.int32))

        if with_xs:
            return jax.lax.scan(body, carry, values)
        return jax.lax.scan(
            lambda c, _: body(c, jnp.asarray(2, dtype=dtype)), carry, None, length=3
        )

    increments = [1, 2, 3] if with_xs else [2, 2, 2]
    values = []
    value = initial.item()
    width = np.iinfo(dtype).bits
    for increment in increments:
        value = (value + increment + 2 ** (width - 1)) % 2**width - 2 ** (width - 1)
        values.append(value)
    ys = np.asarray(values, dtype=dtype)
    expected = (ys[-1], (ys, ys.astype(np.int32)))
    with jax.enable_x64():
        _check_native_and_ort(fn, [initial, xs], expected)


@pytest.mark.parametrize("dtype", [np.int32, np.int64], ids=["int32", "int64"])
def test_integer_window_sum_odd_values_and_wraparound(dtype: Any) -> None:
    maximum = np.iinfo(dtype).max
    x = np.asarray([2**24 + 1, 0, maximum, 2, -3], dtype=dtype)

    def fn(value: jax.Array) -> jax.Array:
        return reduce_window_sum(value, window_dimensions=(2,))

    width = np.iinfo(dtype).bits
    expected = np.asarray(
        [
            (int(a) + int(b) + 2 ** (width - 1)) % 2**width - 2 ** (width - 1)
            for a, b in zip(x[:-1], x[1:])
        ],
        dtype=dtype,
    )
    with jax.enable_x64():
        model = _check_native_and_ort(fn, [x], expected)
    assert not any(node.op_type == "Conv" for node in model.graph.node)
    assert any(node.op_type == "CumSum" for node in model.graph.node)


def test_integer_window_sum_padding_strides_and_dilations() -> None:
    x = np.asarray([2**24 + 1, -2, 3, 5], dtype=np.int64)

    def fn(value: jax.Array) -> jax.Array:
        return reduce_window_sum(
            value,
            window_dimensions=(2,),
            window_strides=(2,),
            padding=((2, 1),),
            base_dilation=(2,),
            window_dilation=(2,),
        )

    # Dilated/padded input: [0, 0, 2**24+1, 0, -2, 0, 3, 0, 5, 0].
    expected = np.asarray([2**24 + 1, 2**24 - 1, 1, 8], dtype=np.int64)
    with jax.enable_x64():
        _check_native_and_ort(fn, [x], expected)


def test_integer_window_sum_dynamic_2d() -> None:
    x = np.asarray([[2**24 + 1, -2, 3], [5, 7, -11]], dtype=np.int64)

    def fn(value: jax.Array) -> jax.Array:
        return reduce_window_sum(
            value, window_dimensions=(2, 2), padding=((0, 0), (1, 0))
        )

    expected = np.asarray([[2**24 + 6, 2**24 + 11, -3]], dtype=np.int64)
    with jax.enable_x64():
        _check_native_and_ort(
            fn,
            [x],
            expected,
            inputs=[jax.ShapeDtypeStruct(("N", 3), np.int64)],
        )


def test_integer_window_sum_int64_odd_above_fp64_exact_range() -> None:
    x = np.asarray([2**53 + 1, 0, 2**53 + 3, -2], dtype=np.int64)

    def fn(value: jax.Array) -> jax.Array:
        return reduce_window_sum(value, window_dimensions=(2,))

    expected = np.asarray([2**53 + 1, 2**53 + 3, 2**53 + 1], dtype=np.int64)
    with jax.enable_x64():
        _check_native_and_ort(fn, [x], expected)


@pytest.mark.parametrize("dtype", [np.uint32, np.uint64], ids=["uint32", "uint64"])
def test_integer_window_sum_unsigned_wraparound(dtype: Any) -> None:
    maximum = np.iinfo(dtype).max
    x = np.asarray([2**24 + 1, 0, maximum, 2], dtype=dtype)

    def fn(value: jax.Array) -> jax.Array:
        return reduce_window_sum(value, window_dimensions=(2,))

    expected = np.asarray([2**24 + 1, maximum, 1], dtype=dtype)
    with jax.enable_x64():
        _check_native_and_ort(fn, [x], expected)


def test_integer_window_sum_empty_output() -> None:
    x = np.asarray([17], dtype=np.int64)

    def fn(value: jax.Array) -> jax.Array:
        return reduce_window_sum(value, window_dimensions=(2,))

    with jax.enable_x64():
        _check_native_and_ort(fn, [x], np.empty((0,), dtype=np.int64))


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint32, np.uint64])
def test_integer_reduce_sum_wraparound(dtype: Any) -> None:
    info = np.iinfo(dtype)
    x = np.asarray([info.max, 1], dtype=dtype)
    expected = np.asarray(info.min, dtype=dtype)

    def source(value: jax.Array) -> jax.Array:
        return jax.lax.reduce_sum(value, axes=(0,))

    with jax.enable_x64():
        _check_native_and_ort(source, [x], expected)


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.uint16])
def test_integer_reduce_sum_small_types_preserve_shared_input(dtype: Any) -> None:
    info = np.iinfo(dtype)
    x = np.asarray([info.max, 2], dtype=dtype)

    def source(value: jax.Array) -> Any:
        return jax.lax.reduce_sum(value, axes=(0,)), value

    expected = (np.asarray(info.min + 1, dtype=dtype), x)
    with jax.enable_x64():
        _check_native_and_ort(source, [x], expected)


def test_integer_reduce_sum_multiple_axes_keepdims_and_exact_large_values() -> None:
    x = np.asarray(
        [2**53 + 1, np.iinfo(np.int64).max, 2, -3, 4, 7, -11, 13], dtype=np.int64
    ).reshape(2, 2, 2)

    def source(value: jax.Array) -> Any:
        return (
            jnp.sum(value, axis=(-3, -1), keepdims=True),
            jnp.sum(value, axis=0),
            jnp.sum(value, axis=None),
            jnp.sum(value, axis=()),
        )

    expected = (
        np.sum(x, axis=(-3, -1), keepdims=True),
        np.sum(x, axis=0),
        np.sum(x, axis=None),
        np.sum(x, axis=()),
    )
    with jax.enable_x64():
        model = _check_native_and_ort(source, [x], expected)
    assert any(node.op_type == "CumSum" for node in model.graph.node)
    assert not any(node.op_type == "ReduceSum" for node in model.graph.node)


@pytest.mark.parametrize(
    "shape,axis,keepdims,input_shape",
    [
        ((2, 0, 3), 1, True, ("B", "N", 3)),
        ((0, 2), 1, False, (0, 2)),
        ((0, 2), None, False, ("N", 2)),
    ],
    ids=["dynamic-empty-axis", "empty-retained-axis", "dynamic-empty-all-axes"],
)
def test_integer_reduce_sum_empty_dimensions(
    shape: tuple[int, ...],
    axis: int | None,
    keepdims: bool,
    input_shape: tuple[Any, ...],
) -> None:
    x = np.empty(shape, dtype=np.int64)

    def source(value: jax.Array) -> jax.Array:
        return jnp.sum(value, axis=axis, keepdims=keepdims)

    expected = np.sum(x, axis=axis, keepdims=keepdims)
    with jax.enable_x64():
        _check_native_and_ort(
            source,
            [x],
            expected,
            inputs=[jax.ShapeDtypeStruct(input_shape, np.int64)],
        )


def test_integer_reduce_sum_retains_abs_and_square_wrapping() -> None:
    info = np.iinfo(np.int64)
    x = np.asarray([info.min, info.max, 1], dtype=np.int64)

    def source(value: jax.Array) -> Any:
        return (
            jax.lax.reduce_sum(jax.lax.abs(value), axes=(0,)),
            jnp.sum(value * value),
        )

    expected = (np.asarray(0, dtype=np.int64), np.asarray(2, dtype=np.int64))
    with jax.enable_x64():
        model = _check_native_and_ort(source, [x], expected)
    assert not any(
        node.op_type in {"ReduceL1", "ReduceSumSquare"} for node in model.graph.node
    )
