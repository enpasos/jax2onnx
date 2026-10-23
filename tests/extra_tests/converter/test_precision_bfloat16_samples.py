# tests/extra_tests/converter/test_precision_bfloat16_samples.py

"""Physical BF16 sample transport preserves bits without claiming CPU kernels."""

from types import SimpleNamespace
from typing import Any
import sys

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import precision, user_interface
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.precision import (
    CheckStatus,
    SampleAcceptanceError,
    SampleCriteria,
    SourceSample,
    validate_source_samples,
)

_BF16 = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(onnx.TensorProto.BFLOAT16))
_WORDS = (0x0000, 0x8000, 0x3F80, 0xC000, 0x0001, 0x8001, 0x7FC1, 0xFFD5)


def _bits() -> np.ndarray:
    return np.asarray(_WORDS, np.uint16).view(_BF16)


def _transport(kind: str) -> onnx.ModelProto:
    """Transport-only graphs; these carry no SOURCE lowering admission."""
    shape = [] if kind == "scalar" else [8] if kind == "selected" else ["N"]
    inputs = [onnx.helper.make_tensor_value_info("x", 16, shape)]
    outputs = [onnx.helper.make_tensor_value_info("y", 16, shape)]
    nodes = [onnx.helper.make_node("Identity", ["x"], ["y"])]
    if kind == "mixed":
        inputs.append(onnx.helper.make_tensor_value_info("z", 1, shape))
        outputs.append(onnx.helper.make_tensor_value_info("w", 1, shape))
        nodes.append(onnx.helper.make_node("Identity", ["z"], ["w"]))
    elif kind == "selected":
        inputs.append(onnx.helper.make_tensor_value_info("mask", 9, [8]))
        outputs = [onnx.helper.make_tensor_value_info("y", 16, [1, "K"])]
        nodes = [
            onnx.helper.make_node("NonZero", ["mask"], ["indices"]),
            onnx.helper.make_node("Gather", ["x", "indices"], ["y"], axis=0),
        ]
    elif kind == "add":
        nodes = [onnx.helper.make_node("Add", ["x", "x"], ["y"])]
    return onnx.helper.make_model(
        onnx.helper.make_graph(nodes, "BF16-transport-only", inputs, outputs),
        ir_version=10,
        opset_imports=[onnx.helper.make_opsetid("", 23)],
    )


def _count_runs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls = []
    for name in ("run", "run_with_ort_values"):
        original = getattr(ort.InferenceSession, name)

        def counted(
            self: Any, *args: Any, _fn: Any = original, _name: str = name, **kwargs: Any
        ) -> Any:
            calls.append(_name)
            return _fn(self, *args, **kwargs)

        monkeypatch.setattr(ort.InferenceSession, name, counted)
    return calls


def _checks(report: Any) -> dict[str, CheckStatus]:
    return {check.category: check.status for check in report.samples[0].checks}


@pytest.mark.parametrize("optimization", ["disabled", "all"])
@pytest.mark.parametrize(
    "operation,criteria",
    [
        ("identity", SampleCriteria("bitwise")),
        ("identity", SampleCriteria("exact")),
        ("identity", SampleCriteria("tolerant", rtol=0, atol=0)),
        ("to-bfloat16", SampleCriteria("exact")),
        ("from-bfloat16", SampleCriteria("bitwise")),
    ],
)
def test_source_identity_and_cast_samples(
    operation: str,
    criteria: SampleCriteria,
    optimization: str,
) -> None:
    def model(x: Any) -> Any:
        if operation == "to-bfloat16":
            return jax.lax.convert_element_type(x, jnp.bfloat16)
        if operation == "from-bfloat16":
            return jax.lax.convert_element_type(x, jnp.float32)
        return x

    value = (
        _bits()
        if operation == "identity" and criteria.mode == "bitwise"
        else np.asarray([0x0000, 0x8000, 0x3F80, 0xC000], np.uint16).view(_BF16)
    )
    if operation == "to-bfloat16":
        value = np.asarray([0.0, -0.0, 1.0, -2.0], np.float32)
    reference = (
        value.copy()
        if operation == "identity"
        else np.asarray(model(jnp.asarray(value)))
    )
    proto = user_interface._to_onnx(
        model,
        [jax.ShapeDtypeStruct(value.shape, value.dtype)],
        return_mode="proto",
        opset=23,
        precision_policy=PrecisionPolicy.SOURCE,
    )
    report = validate_source_samples(
        proto,
        samples=[SourceSample((value,), (reference,), criteria)],
        optimization=optimization,
    )
    report.require_sample_acceptance()
    assert report.status is CheckStatus.VALIDATED
    assert report.samples[0].outputs[0].dtype == reference.dtype.name


@pytest.mark.parametrize("optimization", ["disabled", "all"])
@pytest.mark.parametrize(
    "case", ["mixed", "empty", "scalar", "selected", "selected-empty"]
)
def test_actual_dynamic_transport_keeps_bits_rank_and_one_execution(
    case: str,
    optimization: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _count_runs(monkeypatch)
    kind = (
        "mixed"
        if case in {"mixed", "empty"}
        else "selected" if case.startswith("selected") else "scalar"
    )
    # A noncontiguous genuine BF16 feed must retain its dtype and word order.
    backing = np.empty((8, 2), dtype=_BF16)
    backing[:, 0] = _bits()
    value = backing[:, 0]
    assert not value.flags.c_contiguous
    if case == "empty":
        value = value[:0]
    elif case == "scalar":
        value = np.asarray(0x8000, np.uint16).view(_BF16)
    feeds = {"x": value}
    expected = [value.copy()]
    if kind == "mixed":
        ordinary = np.asarray(
            [
                0,
                0x80000000,
                0x3F800000,
                0xC0000000,
                1,
                0x80000001,
                0x7FC01234,
                0xFFC05678,
            ],
            np.uint32,
        ).view(np.float32)[: value.size]
        feeds["z"] = ordinary
        expected.append(ordinary.copy())
    elif kind == "selected":
        mask = np.asarray([True, False, True, False, False, True, False, False])
        if case == "selected-empty":
            mask[:] = False
        feeds["mask"] = mask
        expected = [value[mask][None, :]]
    before = {name: array.tobytes() for name, array in feeds.items()}
    report = validate_source_samples(
        _transport(kind),
        samples=[SourceSample(feeds, expected, SampleCriteria("bitwise"))],
        optimization=optimization,
    )
    report.require_sample_acceptance()
    assert calls == ["run_with_ort_values"]
    assert {name: array.tobytes() for name, array in feeds.items()} == before
    assert report.samples[0].outputs == report.samples[0].references


@pytest.mark.parametrize(
    "wrong", ["signed-zero", "nan-payload", "finite", "shape", "dtype"]
)
def test_wrong_reference_is_a_numerical_failure_after_one_execution(
    wrong: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _count_runs(monkeypatch)
    value, reference = _bits(), _bits()
    if wrong in {"signed-zero", "nan-payload", "finite"}:
        index = {"signed-zero": 0, "nan-payload": 6, "finite": 2}[wrong]
        reference.view(np.uint16)[index] ^= 0x8000 if wrong == "signed-zero" else 1
    elif wrong == "shape":
        reference = reference.reshape(2, 4)
    else:
        reference = reference.view(np.uint16)
    report = validate_source_samples(
        _transport("identity"),
        samples=[SourceSample((value,), (reference,), SampleCriteria("bitwise"))],
        optimization="disabled",
    )
    assert calls == ["run_with_ort_values"]
    assert report.samples[0].outputs[0].shape == (8,)
    assert _checks(report)["runtime-execution"] is CheckStatus.VALIDATED
    assert _checks(report)["numerical-acceptance"] is CheckStatus.FAILED
    with pytest.raises(SampleAcceptanceError):
        report.require_sample_acceptance()


def test_uint16_feed_is_not_reinterpreted_as_bfloat16() -> None:
    report = validate_source_samples(
        _transport("identity"),
        samples=[
            SourceSample(
                (_bits().view(np.uint16),), (_bits(),), SampleCriteria("bitwise")
            )
        ],
        optimization="disabled",
    )
    assert _checks(report)["runtime-execution"] is CheckStatus.FAILED
    assert _checks(report)["numerical-acceptance"] is CheckStatus.UNVERIFIED


def test_byte_read_failure_remains_execution_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _count_runs(monkeypatch)

    def failed_copy(*args: Any) -> Any:
        raise RuntimeError("bounded CPU byte read failed")

    monkeypatch.setattr(precision.ctypes, "string_at", failed_copy)
    report = validate_source_samples(
        _transport("identity"),
        samples=[SourceSample((_bits(),), (_bits(),), SampleCriteria("bitwise"))],
        optimization="disabled",
    )
    assert calls == ["run_with_ort_values"]
    assert _checks(report)["runtime-execution"] is CheckStatus.FAILED
    assert _checks(report)["numerical-acceptance"] is CheckStatus.UNVERIFIED
    assert "bounded CPU byte read failed" in report.samples[0].checks[-2].detail


@pytest.mark.parametrize("optimization", ["disabled", "all"])
def test_missing_bfloat16_add_kernel_remains_runtime_load_failure(
    optimization: str,
) -> None:
    report = validate_source_samples(
        _transport("add"),
        samples=[SourceSample((_bits(),), (_bits(),), SampleCriteria("bitwise"))],
        optimization=optimization,
    )
    assert any(
        check.category == "runtime-load" and check.status is CheckStatus.FAILED
        for check in report.checks
    )
    assert _checks(report)["numerical-acceptance"] is CheckStatus.UNVERIFIED
    with pytest.raises(SampleAcceptanceError):
        report.require_sample_acceptance()


@pytest.mark.parametrize(
    "bad", ["tensor", "device", "dtype", "shape", "dimension-type", "bytes", "overflow"]
)
def test_invalid_bfloat16_metadata_is_rejected_before_pointer_read(bad: str) -> None:
    def forbidden_pointer() -> int:
        pytest.fail("invalid tensor metadata reached data_ptr")

    value = SimpleNamespace(
        is_tensor=lambda: bad != "tensor",
        device_name=lambda: "cuda" if bad == "device" else "cpu",
        element_type=lambda: 1 if bad == "dtype" else 16,
        shape=lambda: (
            [-1]
            if bad == "shape"
            else (
                [True]
                if bad == "dimension-type"
                else [sys.maxsize] if bad == "overflow" else [1]
            )
        ),
        tensor_size_in_bytes=lambda: (
            1 if bad == "bytes" else 2 * sys.maxsize if bad == "overflow" else 2
        ),
        data_ptr=forbidden_pointer,
    )
    with pytest.raises(ValueError):
        precision._bfloat16_cpu_array(value)


@pytest.mark.parametrize("address", [0, -1, None])
def test_nonempty_bfloat16_requires_a_valid_pointer_before_byte_copy(
    address: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_copy(*args: Any) -> Any:
        pytest.fail("invalid pointer reached the CPU byte copy")

    monkeypatch.setattr(precision.ctypes, "string_at", forbidden_copy)
    value = SimpleNamespace(
        is_tensor=lambda: True,
        device_name=lambda: "cpu",
        element_type=lambda: 16,
        shape=lambda: [1],
        tensor_size_in_bytes=lambda: 2,
        data_ptr=lambda: address,
    )
    with pytest.raises(ValueError):
        precision._bfloat16_cpu_array(value)


def test_empty_bfloat16_readback_owns_storage_without_pointer_query() -> None:
    def forbidden_pointer() -> int:
        pytest.fail("empty tensor queried data_ptr")

    value = SimpleNamespace(
        is_tensor=lambda: True,
        device_name=lambda: "cpu",
        element_type=lambda: 16,
        shape=lambda: [0, 3],
        tensor_size_in_bytes=lambda: 0,
        data_ptr=forbidden_pointer,
    )
    actual = precision._bfloat16_cpu_array(value)
    assert actual.dtype == _BF16 and actual.shape == (0, 3) and actual.flags.owndata


def test_ordinary_numpy_session_mock_needs_only_run() -> None:
    value = np.asarray([1.0, -0.0], np.float32)
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [],
            "ordinary",
            [],
            [onnx.helper.make_tensor_value_info("y", 1, [2])],
        )
    )
    calls = []

    def run(names: Any, feeds: Any) -> list[np.ndarray]:
        calls.append((names, feeds))
        return [value]

    actual = precision._run_cpu_samples(SimpleNamespace(run=run), {}, model)
    assert calls == [(None, {})]
    assert actual[0] is value
