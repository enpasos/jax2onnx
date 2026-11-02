# tests/extra_tests/test_explicit_math_accuracy.py

"""Explicit unary accuracy requests must survive tracing or fail clearly."""

import jax
import numpy as np
import onnx
import pytest

from jax2onnx import onnx_function, to_onnx
from tests.extra_tests.test_precision_contract import check


_OPERATIONS = (
    "exp",
    "exp2",
    "expm1",
    "log",
    "log1p",
    "tanh",
    "logistic",
    "sin",
    "cos",
    "sqrt",
    "rsqrt",
    "cbrt",
    "tan",
)


def _assert_source_accuracy(source, operation, accuracy):
    traced = jax.make_jaxpr(source)(jax.ShapeDtypeStruct((1,), np.float32))
    equations = [eqn for eqn in traced.jaxpr.eqns if eqn.primitive.name == operation]
    assert len(equations) == 1
    assert equations[0].params["accuracy"] is accuracy


@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize(
    "accuracy",
    [jax.lax.AccuracyMode.HIGHEST, jax.lax.Tolerance(atol=1e-6)],
    ids=["highest", "tolerance"],
)
def test_nondefault_math_accuracy_is_retained_and_rejected(operation, accuracy):
    def source(value):
        return getattr(jax.lax, operation)(value, accuracy=accuracy)

    _assert_source_accuracy(source, operation, accuracy)
    with pytest.raises(
        NotImplementedError,
        match="does not currently preserve non-default accuracy requests",
    ) as error:
        to_onnx(source, [np.asarray([1.25], dtype=np.float32)])
    assert f"{operation} accuracy=" in str(error.value)


@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize(
    "accuracy", [None, jax.lax.AccuracyMode.DEFAULT], ids=["none", "default"]
)
def test_default_math_accuracy_keeps_source_and_runtime_results(operation, accuracy):
    def source(value):
        return getattr(jax.lax, operation)(value, accuracy=accuracy)

    _assert_source_accuracy(source, operation, accuracy)
    model, reference = check(
        source, [np.asarray([1.25], dtype=np.float32)], rtol=1e-6, atol=0
    )
    assert reference.dtype == np.float32
    assert model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT


@pytest.mark.parametrize("operation", ["log", "sqrt"])
def test_math_accuracy_is_rejected_before_reduction_fusion(operation):
    accuracy = jax.lax.AccuracyMode.HIGHEST

    def source(value):
        operand = jax.lax.mul(value, value) if operation == "sqrt" else value
        reduced = jax.lax.reduce_sum(operand, axes=(0,))
        return getattr(jax.lax, operation)(reduced, accuracy=accuracy)

    _assert_source_accuracy(source, operation, accuracy)
    with pytest.raises(
        NotImplementedError,
        match="does not currently preserve non-default accuracy requests",
    ):
        to_onnx(source, [np.asarray([1.25], dtype=np.float32)])


@onnx_function
def _delayed_highest_sin(value):
    return jax.lax.sin(value, accuracy=jax.lax.AccuracyMode.HIGHEST)


def test_delayed_function_keeps_nondefault_math_accuracy_request():
    def source(value):
        return _delayed_highest_sin(value)

    _assert_source_accuracy(source, "sin", jax.lax.AccuracyMode.HIGHEST)
    with pytest.raises(
        NotImplementedError,
        match="sin accuracy=.*does not currently preserve non-default accuracy requests",
    ):
        to_onnx(source, [np.asarray([1.25], dtype=np.float32)])
