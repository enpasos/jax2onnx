# tests/extra_tests/converter/test_precision_discrete_samples.py

"""Floating sample tolerances never weaken exact integer or Boolean outputs."""

from typing import Any

import numpy as np
import onnx
import pytest

from jax2onnx.precision import (
    CheckStatus,
    SampleAcceptanceError,
    SampleCriteria,
    SourceSample,
    validate_source_samples,
)


def _identity_model(dtypes: tuple[Any, ...]) -> onnx.ModelProto:
    inputs, outputs, nodes = [], [], []
    for index, dtype in enumerate(dtypes):
        tensor_type = onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
        inputs.append(onnx.helper.make_tensor_value_info(f"x{index}", tensor_type, [1]))
        outputs.append(
            onnx.helper.make_tensor_value_info(f"y{index}", tensor_type, [1])
        )
        nodes.append(onnx.helper.make_node("Identity", [f"x{index}"], [f"y{index}"]))
    return onnx.helper.make_model(
        onnx.helper.make_graph(nodes, "discrete_samples", inputs, outputs),
        ir_version=10,
        opset_imports=[onnx.helper.make_opsetid("", 23)],
    )


def _criteria(mode: str) -> SampleCriteria:
    return (
        SampleCriteria(mode, rtol=0, atol=0)
        if mode == "tolerant"
        else SampleCriteria(mode)
    )


@pytest.mark.parametrize("mode", ["exact", "tolerant", "bitwise"])
def test_uint64_difference_of_one_cannot_round_into_sample_acceptance(
    mode: str,
) -> None:
    actual = np.asarray([2**63], dtype=np.uint64)
    expected = np.asarray([2**63 + 1], dtype=np.uint64)
    # NumPy's floating tolerance route loses this deterministic integer difference.
    assert np.allclose(actual, expected, rtol=0, atol=0)
    report = validate_source_samples(
        _identity_model((np.uint64,)),
        samples=[SourceSample((actual,), (expected,), _criteria(mode))],
        optimization="disabled",
    )
    assert report.status is CheckStatus.FAILED
    assert any(
        check.category == "runtime-execution" and check.status is CheckStatus.VALIDATED
        for check in report.samples[0].checks
    )
    with pytest.raises(SampleAcceptanceError):
        report.require_sample_acceptance()


@pytest.mark.parametrize("mode", ["exact", "tolerant", "bitwise"])
def test_boolean_sample_outputs_remain_exact_under_every_mode(mode: str) -> None:
    report = validate_source_samples(
        _identity_model((np.bool_,)),
        samples=[
            SourceSample((np.asarray([True]),), (np.asarray([False]),), _criteria(mode))
        ],
        optimization="disabled",
    )
    assert report.status is CheckStatus.FAILED
    with pytest.raises(SampleAcceptanceError):
        report.require_sample_acceptance()


@pytest.mark.parametrize("wrong_integer", [False, True])
@pytest.mark.parametrize("wrong_boolean", [False, True])
def test_mixed_outputs_apply_tolerance_only_to_floating_values(
    wrong_integer: bool, wrong_boolean: bool
) -> None:
    actual = (
        np.asarray([1.0001], np.float32),
        np.asarray([2**63], np.uint64),
        np.asarray([True], np.bool_),
    )
    expected = (
        np.asarray([1.0], np.float32),
        np.asarray([2**63 + int(wrong_integer)], np.uint64),
        np.asarray([not wrong_boolean], np.bool_),
    )
    report = validate_source_samples(
        _identity_model((np.float32, np.uint64, np.bool_)),
        samples=[
            SourceSample(
                actual, expected, SampleCriteria("tolerant", rtol=1e-3, atol=0)
            )
        ],
        optimization="all",
    )
    if wrong_integer or wrong_boolean:
        assert report.status is CheckStatus.FAILED
        with pytest.raises(SampleAcceptanceError, match="exact integer/Boolean"):
            report.require_sample_acceptance()
    else:
        assert report.status is CheckStatus.VALIDATED
        report.require_sample_acceptance()
