# tests/extra_tests/converter/test_source_facade_samples.py

"""Actual private facade artifacts pass an independent, artifact-bound CPU gate."""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import pytest

from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.precision import (
    CheckStatus,
    SampleAcceptanceError,
    SampleCriteria,
    SourceSample,
    get_precision_report,
    validate_source_samples,
)
from jax2onnx.user_interface import _to_onnx
from tests.extra_tests.converter.precision_numerical_support import (
    normalization_input,
    normalization_reference,
    selected_variance_normalization,
)


@pytest.fixture(autouse=True)
def _restore_config() -> Iterator[None]:
    previous_x64 = bool(jax.config.x64_enabled)
    previous_platforms = jax.config.jax_platforms
    previous_precision = jax.config.jax_default_matmul_precision
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_matmul_precision", None)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
        jax.config.update("jax_platforms", previous_platforms)
        jax.config.update("jax_default_matmul_precision", previous_precision)


def _artifact(
    function: Callable[..., Any],
    values: list[np.ndarray],
    expected: list[np.ndarray],
    representation: str,
    tmp_path: Path,
) -> Any:
    # An independent staged reference must also describe the actual eager JAX
    # calculation. Runtime behavior never supplies its own reference.
    eager = jax.tree.leaves(function(*(jnp.asarray(value) for value in values)))
    assert len(eager) == len(expected)
    for actual, reference in zip(eager, expected, strict=True):
        actual_array = np.asarray(actual)
        np.testing.assert_array_equal(actual_array, reference, strict=True)
        assert actual_array.tobytes() == reference.tobytes()
    artifact = _to_onnx(
        function,
        [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in values],
        precision_policy=PrecisionPolicy.SOURCE,
        return_mode=representation,
        output_path=tmp_path / "facade-sample.onnx",
        input_names=[f"input_{index}" for index in range(len(values))],
        output_names=[f"output_{index}" for index in range(len(expected))],
    )
    report = get_precision_report(artifact)
    assert report is not None
    assert {check.category: check.status for check in report.checks} == {
        "representation": CheckStatus.VALIDATED,
        "lowering": CheckStatus.VALIDATED,
        "runtime-execution": CheckStatus.UNVERIFIED,
        "numerical-acceptance": CheckStatus.UNVERIFIED,
    }
    return artifact


def _square_sum(x: jax.Array) -> tuple[jax.Array, ...]:
    square = x * x
    return square, jax.lax.reduce_sum(square, axes=(0,))


def _rounded_square_sum(x: jax.Array) -> tuple[jax.Array, ...]:
    wider = x.astype(jnp.float32)
    square = (wider * wider).astype(jnp.float16)
    total = jax.lax.reduce_sum(square.astype(jnp.float32), axes=(0,))
    return square, total.astype(jnp.float16)


@pytest.mark.parametrize("optimization", ["disabled", "all"])
@pytest.mark.parametrize("witness", ["fp16-square-sum", "weak-bias", "fp32-reciprocal"])
def test_facade_artifact_known_witnesses_obey_the_actual_sample_gate(
    witness: str, optimization: str, tmp_path: Path
) -> None:
    if witness == "fp16-square-sum":
        function = _square_sum
        values = [np.full(8, 2**-13, np.float16)]
        expected = [np.zeros(8, np.float16), np.asarray(0, np.float16)]
        representation = "ir"
    elif witness == "weak-bias":
        bias = jnp.asarray(4096.0)
        assert bias.weak_type

        def function(x: jax.Array) -> tuple[jax.Array, ...]:
            return ((x + bias).astype(jnp.float32) * np.float32(2**-20),)

        values = [np.asarray([1], np.float16)]
        rounded = np.add(values[0], np.float16(4096), dtype=np.float16)
        expected = [
            np.multiply(
                rounded.astype(np.float32), np.float32(2**-20), dtype=np.float32
            )
        ]
        representation = "file"
    else:

        def function(x: jax.Array, y: jax.Array) -> tuple[jax.Array, ...]:
            return (x * (np.float32(1) / jax.lax.sqrt(y)),)

        values = [np.asarray([3], np.float32), np.asarray([2], np.float32)]
        root = np.sqrt(values[1], dtype=np.float32)
        reciprocal = np.divide(np.float32(1), root, dtype=np.float32)
        expected = [np.multiply(values[0], reciprocal, dtype=np.float32)]
        representation = "proto"
    artifact = _artifact(function, values, expected, representation, tmp_path)
    export_record = get_precision_report(artifact)
    validation = validate_source_samples(
        artifact,
        samples=[SourceSample(values, expected)],
        optimization=optimization,
    )
    assert validation.target is not None
    assert validation.target.providers == ("CPUExecutionProvider",)
    assert validation.samples[0].checks[0].status is CheckStatus.VALIDATED
    assert validation.status in (CheckStatus.VALIDATED, CheckStatus.FAILED)
    if validation.status is CheckStatus.VALIDATED:
        # A corrected runtime can pass the unchanged independent/JAX reference.
        validation.require_sample_acceptance(artifact)
        assert [value.sha256 for value in validation.samples[0].outputs] == [
            value.sha256 for value in validation.samples[0].references
        ]
    else:
        assert validation.samples[0].checks[-1].category == "numerical-acceptance"
        with pytest.raises(SampleAcceptanceError, match="failed"):
            validation.require_sample_acceptance(artifact)
    # CPU sample evidence remains separate from export provenance.
    assert get_precision_report(artifact) == export_record


@pytest.mark.parametrize("optimization", ["disabled", "all"])
@pytest.mark.parametrize(
    ("control", "representation"),
    [
        ("scientific-variance", "file"),
        ("preferred-matrix", "ir"),
        ("low-precision-signed-zero", "proto"),
        ("explicit-half-rounding", "file"),
    ],
)
def test_facade_artifact_positive_controls_receive_only_bound_sample_acceptance(
    control: str, representation: str, optimization: str, tmp_path: Path
) -> None:
    criteria = SampleCriteria()
    if control == "scientific-variance":
        jax.config.update("jax_enable_x64", True)

        def function(x: jax.Array) -> tuple[jax.Array, ...]:
            return selected_variance_normalization(x)[:3]

        values = [normalization_input()]
        expected = list(normalization_reference(values[0])[:3])
        assert expected[2][0, 0] == np.float32(8388609)
    elif control == "preferred-matrix":

        def function(x: jax.Array, w: jax.Array) -> tuple[jax.Array, ...]:
            product = jax.lax.dot_general(
                x, w, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
            )
            return product, x, w

        values = [
            np.asarray([[2048, 1]], np.float16),
            np.asarray([[1], [1]], np.float16),
        ]
        product = (values[0].astype(np.float64) @ values[1].astype(np.float64)).astype(
            np.float32
        )
        assert product[0, 0] == 2049
        expected = [product, values[0].copy(), values[1].copy()]
    elif control == "low-precision-signed-zero":

        def function(x: jax.Array) -> tuple[jax.Array, ...]:
            return -x, x.astype(jnp.float32)

        values = [np.asarray([-0.0, 0.0, 2048, 2**-24], np.float16)]
        expected = [
            np.negative(values[0], dtype=np.float16),
            values[0].astype(np.float32),
        ]
        criteria = SampleCriteria("bitwise")
    else:
        function = _rounded_square_sum
        values = [np.full(8, 2**-13, np.float16)]
        expected = [np.zeros(8, np.float16), np.asarray(0, np.float16)]
    artifact = _artifact(function, values, expected, representation, tmp_path)
    if control == "scientific-variance":
        proto = onnx.load_model(artifact)
        assert any(
            node.op_type == "Cast"
            and any(
                attribute.name == "to" and attribute.i == onnx.TensorProto.DOUBLE
                for attribute in node.attribute
            )
            for node in proto.graph.node
        )
    validation = validate_source_samples(
        artifact,
        samples=[SourceSample(values, expected, criteria)],
        optimization=optimization,
    )
    assert validation.status is CheckStatus.VALIDATED
    validation.require_sample_acceptance(artifact)
    assert validation.target is not None
    assert validation.target.providers == ("CPUExecutionProvider",)
    assert validation.artifact_sha256
    assert [value.sha256 for value in validation.samples[0].outputs] == [
        value.sha256 for value in validation.samples[0].references
    ]
