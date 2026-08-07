# tests/extra_tests/framework/test_opset_sensitive_reduction_axes.py

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest
from numpy.typing import NDArray
from onnx import numpy_helper

from jax2onnx import to_onnx


ReductionFn = Callable[[jax.Array], jax.Array]


_REDUCTION_CASES: tuple[tuple[str, str, ReductionFn], ...] = (
    ("jnp_reduce_l1", "ReduceL1", lambda x: jnp.sum(jnp.abs(x), axis=1)),
    (
        "jnp_reduce_sum_square",
        "ReduceSumSquare",
        lambda x: jnp.sum(jnp.square(x), axis=1),
    ),
    (
        "jnp_reduce_l2",
        "ReduceL2",
        lambda x: jnp.sqrt(jnp.sum(jnp.square(x), axis=1)),
    ),
    (
        "jnp_reduce_log_sum",
        "ReduceLogSum",
        lambda x: jnp.log(jnp.sum(x, axis=1)),
    ),
    (
        "jnp_reduce_log_sum_exp",
        "ReduceLogSumExp",
        lambda x: jnp.log(jnp.sum(jnp.exp(x), axis=1)),
    ),
    (
        "lax_reduce_l1",
        "ReduceL1",
        lambda x: jax.lax.reduce_sum(jax.lax.abs(x), axes=(1,)),
    ),
    (
        "lax_reduce_sum_square",
        "ReduceSumSquare",
        lambda x: jax.lax.reduce_sum(jax.lax.mul(x, x), axes=(1,)),
    ),
    (
        "lax_reduce_l2",
        "ReduceL2",
        lambda x: jax.lax.sqrt(jax.lax.reduce_sum(jax.lax.mul(x, x), axes=(1,))),
    ),
    (
        "lax_reduce_log_sum",
        "ReduceLogSum",
        lambda x: jax.lax.log(jax.lax.reduce_sum(x, axes=(1,))),
    ),
    (
        "lax_reduce_log_sum_exp",
        "ReduceLogSumExp",
        lambda x: jax.lax.log(jax.lax.reduce_sum(jax.lax.exp(x), axes=(1,))),
    ),
    (
        "jaxnn_logsumexp",
        "ReduceLogSumExp",
        lambda x: jax.nn.logsumexp(x, axis=1),
    ),
    (
        "jaxnn_logmeanexp",
        "ReduceLogSumExp",
        lambda x: jax.nn.logmeanexp(x, axis=1),
    ),
)


@pytest.mark.parametrize("opset", [17, 18])
@pytest.mark.parametrize(
    ("case_name", "op_type", "fn"),
    _REDUCTION_CASES,
    ids=[case[0] for case in _REDUCTION_CASES],
)
def test_reduction_axes_follow_opset_schema_and_preserve_subset_axis(
    case_name: str,
    op_type: str,
    fn: ReductionFn,
    opset: int,
) -> None:
    x: NDArray[np.float32] = np.linspace(0.25, 2.0, 24, dtype=np.float32).reshape(
        2, 3, 4
    )
    model = to_onnx(fn, [x], opset=opset, model_name=f"{case_name}_{opset}")

    matching_nodes = [node for node in model.graph.node if node.op_type == op_type]
    assert len(matching_nodes) == 1
    reduction = matching_nodes[0]

    axes_attributes = [
        attribute for attribute in reduction.attribute if attribute.name == "axes"
    ]
    if opset == 17:
        assert len(reduction.input) == 1
        assert len(axes_attributes) == 1
        assert list(axes_attributes[0].ints) == [1]
    else:
        assert len(reduction.input) == 2
        assert not axes_attributes
        axes_initializer = next(
            initializer
            for initializer in model.graph.initializer
            if initializer.name == reduction.input[1]
        )
        np.testing.assert_array_equal(
            numpy_helper.to_array(axes_initializer), np.asarray([1], dtype=np.int64)
        )

    onnx.checker.check_model(model, full_check=True)
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    actual = session.run(None, {model.graph.input[0].name: x})[0]
    np.testing.assert_allclose(
        actual,
        np.asarray(fn(jnp.asarray(x))),
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.parametrize("opset", [17, 18, 21])
@pytest.mark.parametrize(
    ("case_name", "fn"),
    [
        ("lax_reduce_or", lambda x: jax.lax.reduce_or(x, axes=(0,))),
        ("jnp_any", lambda x: jnp.any(x, axis=0)),
    ],
)
def test_boolean_or_reduction_is_empty_dimension_safe(
    case_name: str,
    fn: ReductionFn,
    opset: int,
) -> None:
    symbolic_input = jax.ShapeDtypeStruct(
        jax.export.symbolic_shape("B"),
        jnp.bool_,
    )
    model = to_onnx(
        fn,
        [symbolic_input],
        opset=opset,
        model_name=f"{case_name}_empty_safe_{opset}",
    )

    reductions = [node for node in model.graph.node if node.op_type == "ReduceSum"]
    assert len(reductions) == 1
    assert all(node.op_type != "ReduceMax" for node in model.graph.node)
    reduction = reductions[0]
    assert len(reduction.input) == 2
    assert all(attribute.name != "axes" for attribute in reduction.attribute)
    axes_initializer = next(
        initializer
        for initializer in model.graph.initializer
        if initializer.name == reduction.input[1]
    )
    np.testing.assert_array_equal(
        numpy_helper.to_array(axes_initializer),
        np.asarray([0], dtype=np.int64),
    )

    onnx.checker.check_model(model, full_check=True)
    session = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    boolean_inputs: tuple[NDArray[np.bool_], ...] = (
        np.empty((0,), dtype=np.bool_),
        np.asarray([False, True, False], dtype=np.bool_),
        np.asarray([False, False], dtype=np.bool_),
    )
    for values in boolean_inputs:
        actual = session.run(None, {input_name: values})[0]
        expected = np.asarray(fn(jnp.asarray(values)))
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("opset", [17, 18, 21])
def test_jnp_all_is_empty_dimension_safe(opset: int) -> None:
    def fn(x: jax.Array) -> jax.Array:
        return jnp.all(x, axis=0)

    symbolic_input = jax.ShapeDtypeStruct(
        jax.export.symbolic_shape("B"),
        jnp.bool_,
    )
    model = to_onnx(fn, [symbolic_input], opset=opset)

    onnx.checker.check_model(model, full_check=True)
    session = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    boolean_inputs: tuple[NDArray[np.bool_], ...] = (
        np.empty((0,), dtype=np.bool_),
        np.asarray([True, True], dtype=np.bool_),
        np.asarray([True, False], dtype=np.bool_),
    )
    for values in boolean_inputs:
        actual = session.run(None, {input_name: values})[0]
        expected = np.asarray(fn(jnp.asarray(values)))
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("opset", [17, 18, 21])
@pytest.mark.parametrize("dtype", [np.dtype(np.int32), np.dtype(np.float32)])
@pytest.mark.parametrize(
    ("case_name", "fn", "values"),
    [
        ("jnp_any", lambda x: jnp.any(x, axis=0), [1, -1]),
        ("jnp_all", lambda x: jnp.all(x, axis=0), [0, -1]),
    ],
)
def test_jnp_boolean_reductions_normalize_numeric_truth_values(
    case_name: str,
    fn: ReductionFn,
    values: list[int],
    dtype: np.dtype[Any],
    opset: int,
) -> None:
    inputs = np.asarray(values, dtype=dtype)
    model = to_onnx(
        fn,
        [inputs],
        opset=opset,
        model_name=f"{case_name}_numeric_truth_{opset}",
    )

    onnx.checker.check_model(model, full_check=True)
    session = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    actual = session.run(None, {session.get_inputs()[0].name: inputs})[0]
    expected = np.asarray(fn(jnp.asarray(inputs)))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("opset", [17, 21])
@pytest.mark.parametrize("fn", [jnp.any, jnp.all])
def test_jnp_boolean_reduction_empty_axes_booleanizes_numeric_input(
    fn: Callable[..., jax.Array],
    opset: int,
) -> None:
    inputs: NDArray[np.int32] = np.asarray([-1, 0, 2], dtype=np.int32)

    def reduce_no_axes(x: jax.Array) -> jax.Array:
        return fn(x, axis=())

    model = to_onnx(reduce_no_axes, [inputs], opset=opset)

    onnx.checker.check_model(model, full_check=True)
    session = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    actual = session.run(None, {session.get_inputs()[0].name: inputs})[0]
    expected = np.asarray(reduce_no_axes(jnp.asarray(inputs)))
    assert actual.dtype == np.bool_
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("opset", [17, 18, 21])
@pytest.mark.parametrize(
    ("case_name", "fn"),
    [
        (
            "lax_abs",
            lambda x: jax.lax.reduce_sum(jax.lax.abs(x), axes=()),
        ),
        (
            "lax_square",
            lambda x: jax.lax.reduce_sum(jax.lax.mul(x, x), axes=()),
        ),
        ("jnp_abs", lambda x: jnp.sum(jnp.abs(x), axis=())),
        ("jnp_square", lambda x: jnp.sum(jnp.square(x), axis=())),
    ],
)
def test_optimized_sum_with_empty_axes_preserves_producer_result(
    case_name: str,
    fn: ReductionFn,
    opset: int,
) -> None:
    inputs = np.asarray([[-2.0, 3.0], [4.0, -5.0]], dtype=np.float32)
    model = to_onnx(
        fn,
        [inputs],
        opset=opset,
        model_name=f"{case_name}_empty_axes_{opset}",
    )

    onnx.checker.check_model(model, full_check=True)
    assert all(
        node.op_type not in {"ReduceL1", "ReduceSum", "ReduceSumSquare"}
        for node in model.graph.node
    )
    session = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    actual = session.run(None, {session.get_inputs()[0].name: inputs})[0]
    expected = np.asarray(fn(jnp.asarray(inputs)))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("opset", [17, 18, 21])
@pytest.mark.parametrize(
    ("case_name", "fn", "inputs"),
    [
        (
            "lax_unsigned",
            lambda x: jax.lax.reduce_sum(x, axes=()),
            np.asarray([[1, 2], [3, 4]], dtype=np.uint32),
        ),
        (
            "lax_unsigned_square",
            lambda x: jax.lax.reduce_sum(jax.lax.mul(x, x), axes=()),
            np.asarray([[1, 2], [3, 4]], dtype=np.uint32),
        ),
        (
            "jnp_unsigned_explicit_promotion",
            lambda x: jnp.sum(x, axis=(), dtype=jnp.uint32),
            np.asarray([[1, 2], [3, 4]], dtype=np.uint8),
        ),
        (
            "jnp_unsigned_unpromoted_square",
            lambda x: jnp.sum(jnp.square(x), axis=(), promote_integers=False),
            np.asarray([[1, 2], [3, 4]], dtype=np.uint8),
        ),
    ],
)
def test_sum_with_empty_axes_preserves_unsigned_output_dtype(
    case_name: str,
    fn: ReductionFn,
    inputs: NDArray[np.unsignedinteger[Any]],
    opset: int,
) -> None:
    model = to_onnx(
        fn,
        [inputs],
        opset=opset,
        model_name=f"{case_name}_empty_axes_{opset}",
    )

    onnx.checker.check_model(model, full_check=True)
    session = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    actual = session.run(None, {session.get_inputs()[0].name: inputs})[0]
    expected = np.asarray(fn(jnp.asarray(inputs)))
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "fn",
    [
        lambda x: jax.lax.reduce_or(x, axes=(0,)),
        lambda x: jax.lax.reduce_and(x, axes=(0,)),
        lambda x: jax.lax.reduce_xor(x, axes=(0,)),
    ],
)
def test_integer_lax_bitwise_reductions_are_rejected_instead_of_misconverted(
    fn: ReductionFn,
) -> None:
    with pytest.raises(NotImplementedError, match="integer bitwise"):
        to_onnx(fn, [np.asarray([1, 2], dtype=np.int32)], opset=21)
