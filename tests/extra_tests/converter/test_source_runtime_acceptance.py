# tests/extra_tests/converter/test_source_runtime_acceptance.py

"""Known diagnostics cannot pass an explicit numerical acceptance gate."""

from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.extra_tests.converter import precision_runtime_support as runtime_support
from tests.extra_tests.converter.precision_runtime_support import (
    SampleAcceptanceError,
    SampleExecutionError,
    SampleStatus,
    check_cpu_sample,
    require_sample_acceptance,
)
from tests.extra_tests.converter.precision_test_support import export_source_dtypes


@pytest.fixture(autouse=True)
def _restore_jax_config() -> Iterator[None]:
    previous_x64 = bool(jax.config.x64_enabled)
    previous_platforms = jax.config.jax_platforms
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
        jax.config.update("jax_platforms", previous_platforms)


def _square_sum(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    square = x * x
    return square, jax.lax.reduce_sum(square, axes=(0,))


def _explicit_rounding_square_sum(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    wider = x.astype(jnp.float32)
    square = (wider * wider).astype(jnp.float16)
    total = jax.lax.reduce_sum(square.astype(jnp.float32), axes=(0,))
    return square, total.astype(jnp.float16)


def _sample() -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    x = np.full(8, 2**-13, dtype=np.float16)
    expected = (np.zeros(8, dtype=np.float16), np.asarray(0, dtype=np.float16))
    for reference in (_square_sum(jnp.asarray(x)), jax.jit(_square_sum)(x)):
        for value, exact in zip(reference, expected, strict=True):
            np.testing.assert_array_equal(np.asarray(value), exact, strict=True)
    return x, expected


@pytest.mark.parametrize("optimize", [False, True])
def test_fp16_rounding_diagnostic_cannot_pass_sample_acceptance(
    optimize: bool,
) -> None:
    x, expected = _sample()
    model = export_source_dtypes(_square_sum, [jax.ShapeDtypeStruct(x.shape, x.dtype)])
    check = check_cpu_sample(model, [x], expected, optimize=optimize)
    assert check.target is not None
    assert check.target.provider == "CPUExecutionProvider"
    assert check.target.runtime_version == runtime_support.ort.__version__
    assert check.target.optimization == (
        "ORT_ENABLE_ALL" if optimize else "ORT_DISABLE_ALL"
    )
    assert check.target.intra_op_threads == check.target.inter_op_threads == 1
    assert check.outputs is not None
    if check.status is SampleStatus.ACCEPTED:
        # A future runtime correction may satisfy the same source contract.
        require_sample_acceptance(check)
        for actual, reference in zip(check.outputs, expected, strict=True):
            np.testing.assert_array_equal(actual, reference, strict=True)
    else:
        assert check.status is SampleStatus.FAILED
        np.testing.assert_array_equal(check.outputs[0], expected[0], strict=True)
        np.testing.assert_array_equal(
            check.outputs[1], np.asarray(2**-23, dtype=np.float16), strict=True
        )
        with pytest.raises(SampleAcceptanceError, match="failed: Output 1"):
            require_sample_acceptance(check)


@pytest.mark.parametrize("optimize", [False, True])
def test_explicit_rounding_candidate_passes_only_its_sample(optimize: bool) -> None:
    x, expected = _sample()
    model = export_source_dtypes(
        _explicit_rounding_square_sum, [jax.ShapeDtypeStruct(x.shape, x.dtype)]
    )
    check = check_cpu_sample(model, [x], expected, optimize=optimize)
    assert check.status is SampleStatus.ACCEPTED
    require_sample_acceptance(check)


def test_no_execution_profile_remains_unverified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, expected = _sample()
    model = export_source_dtypes(_square_sum, [jax.ShapeDtypeStruct(x.shape, x.dtype)])

    def unexpected_session(*args: Any, **kwargs: Any) -> None:
        pytest.fail("An unverified sample must not select a runtime implicitly")

    monkeypatch.setattr(runtime_support.ort, "InferenceSession", unexpected_session)
    check = check_cpu_sample(model, [x], expected, optimize=None)
    assert check.status is SampleStatus.UNVERIFIED
    assert check.target is check.outputs is None
    with pytest.raises(SampleAcceptanceError, match="unverified"):
        require_sample_acceptance(check)


def test_kernel_execution_failure_is_not_a_numerical_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, expected = _sample()
    model = export_source_dtypes(_square_sum, [jax.ShapeDtypeStruct(x.shape, x.dtype)])
    original = RuntimeError("No kernel for the requested operation/dtype")

    def missing_kernel(*args: Any, **kwargs: Any) -> None:
        raise original

    monkeypatch.setattr(runtime_support.ort, "InferenceSession", missing_kernel)
    with pytest.raises(SampleExecutionError, match="could not execute") as caught:
        check_cpu_sample(model, [x], expected, optimize=False)
    assert caught.value.__cause__ is original


@pytest.mark.parametrize("difference", ["dtype", "shape", "value"])
def test_exact_sample_contract_rejects_each_output_mismatch(difference: str) -> None:
    x = np.asarray([1], dtype=np.float32)
    model = export_source_dtypes(
        lambda value: value, [jax.ShapeDtypeStruct(x.shape, x.dtype)]
    )
    expected = {
        "dtype": x.astype(np.float64),
        "shape": x.reshape(()),
        "value": np.asarray([2], dtype=np.float32),
    }[difference]
    check = check_cpu_sample(model, [x], [expected], optimize=False)
    assert check.status is SampleStatus.FAILED
    with pytest.raises(SampleAcceptanceError, match="failed: Output 0"):
        require_sample_acceptance(check)
