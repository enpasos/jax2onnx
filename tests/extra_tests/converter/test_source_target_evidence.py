# tests/extra_tests/converter/test_source_target_evidence.py

"""Native target witnesses and synthetic classifier controls; no target waiver."""

from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import user_interface as ui
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.precision import (
    CheckStatus,
    PrecisionCheck,
    SampleAcceptanceError,
    SampleCriteria,
    SourceSample,
    validate_source_samples,
)
from tests.extra_tests.converter.source_target_evidence import (
    FUSED_ERROR,
    _double_transpose_matmuls,
    require_target_pair,
)


def _baseline(directory: Path, *, transpose: bool = True) -> tuple[Any, ...]:
    left = jnp.asarray([[-1, 2, 3], [4, -2, 1]], dtype=jnp.float64) / 4
    right = (
        jnp.asarray([[1, 2, -1], [2, -3, 4], [0, 1, 2], [-2, 0, 3]], dtype=jnp.float64)
        / 8
    )
    if not transpose:
        right = right.T

    def fn(a: Any, b: Any) -> Any:
        return jnp.matmul(a, b.T if transpose else b)

    eager = np.asarray(fn(left, right)).copy()
    compiled = np.asarray(jax.jit(fn)(left, right)).copy()
    assert eager.dtype == compiled.dtype == np.dtype("float64")
    assert eager.shape == compiled.shape == (2, 4)
    assert eager.tobytes() == compiled.tobytes()
    a, b = np.asarray(left), np.asarray(right.T if not transpose else right)
    expected = np.array(
        [
            [sum(float(a[i, k]) * float(b[j, k]) for k in range(3)) for j in range(4)]
            for i in range(2)
        ],
        dtype=np.float64,
    )
    assert eager.tobytes() == expected.tobytes()
    values = (np.asarray(left).copy(), np.asarray(right).copy())
    np.savez(directory / "inputs.npz", *values)
    np.savez(directory / "eager.npz", eager)
    np.savez(directory / "fresh-jit.npz", compiled)
    model = ui._to_onnx(
        fn,
        [jax.ShapeDtypeStruct(v.shape, v.dtype) for v in values],
        opset=23,
        precision_policy=PrecisionPolicy.SOURCE,
    )
    (directory / "source.onnx").write_bytes(model.SerializeToString())
    onnx.checker.check_model(model, full_check=True)
    samples = [
        SourceSample(values, (reference,), SampleCriteria("bitwise"))
        for reference in (eager, compiled)
    ]
    disabled, enabled = [
        validate_source_samples(model, samples=samples, optimization=mode)
        for mode in ("disabled", "all")
    ]
    assert all(
        report.target is not None and report.target.runtime_version == ort.__version__
        for report in (disabled, enabled)
    )
    # Every hostile classifier control first proves the actual native/artifact target pair.
    actual = require_target_pair(
        model,
        samples,
        disabled,
        enabled,
        evidence_path=directory / "actual-targets.json",
    )
    disabled.require_sample_acceptance(model)
    return model, samples, disabled, enabled, actual


def _synthetic_negative(model: Any, disabled: Any) -> tuple[Any, Any]:
    """Metadata-only classifier fixture after the actual baseline, not a measurement."""
    node = _double_transpose_matmuls(model)[0]["matmul"]
    detail = FUSED_ERROR.format(node=node)
    disabled = replace(
        disabled,
        target=replace(
            disabled.target, runtime_version="1.24.1", architecture="x86_64"
        ),
    )
    enabled = replace(
        disabled,
        target=replace(
            disabled.target,
            optimization="ORT_ENABLE_ALL",
            providers=(),
            provider_options=(),
        ),
        checks=disabled.checks
        + (PrecisionCheck("runtime-load", CheckStatus.FAILED, detail),),
        samples=(
            replace(
                disabled.samples[0],
                outputs=(),
                checks=(
                    PrecisionCheck("runtime-load", CheckStatus.FAILED, detail),
                    PrecisionCheck(
                        "numerical-acceptance",
                        CheckStatus.UNVERIFIED,
                        "The runtime could not load the artifact.",
                    ),
                ),
            ),
        ),
    )
    return disabled, enabled


def test_original_native_pair_records_acceptance_or_exact_negative_target(
    tmp_path: Path,
) -> None:
    with jax.enable_x64(True):
        model, samples, disabled, _, _ = _baseline(tmp_path)
        d, a = _synthetic_negative(model, disabled)
        receipt = require_target_pair(
            model,
            samples,
            d,
            a,
            evidence_path=tmp_path / "synthetic-classifier-only.json",
        )
        assert receipt["outcome"] == "FAILED_RUNTIME_LOAD_NUMERICS_UNVERIFIED"
        assert receipt["unexecuted_all_sample_indices"] == [1]
        assert a.target.providers == () and not a.samples[0].outputs
        with pytest.raises(SampleAcceptanceError, match="FusedMatMul"):
            a.require_sample_acceptance(model)


@pytest.mark.parametrize(
    "fault",
    [
        "version",
        "architecture",
        "artifact",
        "input",
        "reference",
        "criteria",
        "settings",
        "initialized-provider",
        "error-dtype",
        "error-node",
        "stage",
        "sample-stage",
        "output",
        "extra-check",
        "source-fused",
        "source-width",
        "disabled-failure",
        "index",
    ],
)
def test_hostile_target_metadata_never_becomes_acceptance(
    fault: str, tmp_path: Path
) -> None:
    with jax.enable_x64(True):
        model, samples, disabled, _, _ = _baseline(tmp_path)
        d, a = _synthetic_negative(model, disabled)
        require_target_pair(
            model,
            samples,
            d,
            a,
            evidence_path=tmp_path / "valid-synthetic-classifier.json",
        )
        if fault in ("version", "architecture"):
            field, value = (
                ("runtime_version", "1.24.2")
                if fault == "version"
                else ("architecture", "aarch64")
            )
            d, a = replace(d, target=replace(d.target, **{field: value})), replace(
                a, target=replace(a.target, **{field: value})
            )
        elif fault == "artifact":
            a = replace(a, artifact_sha256="0" * 64)
        elif fault in (
            "input",
            "reference",
            "criteria",
            "sample-stage",
            "output",
            "index",
        ):
            sample = a.samples[0]
            if fault == "input":
                sample = replace(
                    sample,
                    inputs=(
                        replace(sample.inputs[0], sha256="0" * 64),
                        *sample.inputs[1:],
                    ),
                )
            elif fault == "reference":
                sample = replace(
                    sample, references=(replace(sample.references[0], sha256="0" * 64),)
                )
            elif fault == "criteria":
                sample = replace(
                    sample, criteria=SampleCriteria("tolerant", rtol=1e-5, atol=1e-5)
                )
            elif fault == "sample-stage":
                sample = replace(
                    sample,
                    checks=(
                        PrecisionCheck(
                            "runtime-execution", CheckStatus.FAILED, "other failure"
                        ),
                    ),
                )
            elif fault == "index":
                sample = replace(sample, index=1)
            else:
                sample = replace(sample, outputs=d.samples[0].outputs)
            a = replace(a, samples=(sample,))
        elif fault == "settings":
            a = replace(
                a,
                target=replace(
                    a.target, session_config=(("session.set_denormal_as_zero", "1"),)
                ),
            )
        elif fault == "initialized-provider":
            a = replace(
                a, target=replace(a.target, providers=("CPUExecutionProvider",))
            )
        elif fault in ("error-dtype", "error-node", "stage"):
            check = a.checks[-1]
            if fault == "error-dtype":
                check = replace(
                    check,
                    detail=check.detail.replace("tensor(double)", "tensor(float16)"),
                )
            elif fault == "error-node":
                check = replace(
                    check, detail=FUSED_ERROR.format(node="unrelated_MatMul")
                )
            else:
                check = replace(check, category="runtime-execution")
            a = replace(a, checks=(*a.checks[:-1], check))
        elif fault == "extra-check":
            a = replace(
                a,
                checks=(
                    *a.checks,
                    PrecisionCheck("unrelated", CheckStatus.UNVERIFIED, "unrelated"),
                ),
            )
        elif fault == "disabled-failure":
            d = replace(
                d,
                samples=(
                    replace(
                        d.samples[0],
                        checks=(
                            PrecisionCheck(
                                "numerical-acceptance",
                                CheckStatus.FAILED,
                                "independent baseline failure",
                            ),
                        ),
                    ),
                    *d.samples[1:],
                ),
            )
        else:
            clone = onnx.ModelProto()
            clone.CopyFrom(model)
            if fault == "source-fused":
                clone.graph.node.append(
                    onnx.helper.make_node(
                        "FusedMatMul",
                        ["left", "right"],
                        ["hostile"],
                        domain="com.microsoft",
                    )
                )
            else:
                for port in (
                    *clone.graph.input,
                    *clone.graph.output,
                    *clone.graph.value_info,
                ):
                    if port.type.tensor_type.elem_type == onnx.TensorProto.DOUBLE:
                        port.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
            model = clone
            digest = hashlib.sha256(
                model.SerializeToString(deterministic=True)
            ).hexdigest()
            d, a = replace(d, artifact_sha256=digest, _artifact=model), replace(
                a, artifact_sha256=digest, _artifact=model
            )
        with pytest.raises((AssertionError, SampleAcceptanceError)):
            require_target_pair(
                model, samples, d, a, evidence_path=tmp_path / "hostile.json"
            )


def test_plain_double_matmul_has_no_transpose_failure_exemption(tmp_path: Path) -> None:
    with jax.enable_x64(True):
        model, samples, disabled, enabled, actual = _baseline(tmp_path, transpose=False)
        assert actual["outcome"] == "BOTH_TARGETS_NUMERICALLY_VALIDATED"
        enabled.require_sample_acceptance(model)
        with pytest.raises(AssertionError, match="adjacent to Transpose"):
            _synthetic_negative(model, disabled)
