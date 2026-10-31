# tests/extra_tests/converter/precision_runtime_support.py

"""Exact sample acceptance for the private precision experiments.

An accepted sample describes only the supplied artifact, inputs, references,
and execution settings. It is neither a model-wide support verdict nor an
automatic lowering policy. Runtime execution failures remain separate from
numerical mismatches.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import platform

import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]


class SampleStatus(Enum):
    ACCEPTED = "accepted"
    FAILED = "failed"
    UNVERIFIED = "unverified"


@dataclass(frozen=True)
class CpuSampleTarget:
    runtime_version: str
    optimization: str
    architecture: str
    provider: str = "CPUExecutionProvider"
    intra_op_threads: int = 1
    inter_op_threads: int = 1


@dataclass(frozen=True)
class CpuSampleCheck:
    status: SampleStatus
    target: CpuSampleTarget | None
    outputs: tuple[np.ndarray, ...] | None
    detail: str


class SampleExecutionError(RuntimeError):
    """The selected runtime could not load or execute the sample."""


class SampleAcceptanceError(ValueError):
    """The supplied sample has no successful numerical acceptance result."""


def check_cpu_sample(
    model: onnx.ModelProto,
    inputs: Sequence[np.ndarray],
    expected_outputs: Sequence[np.ndarray],
    *,
    optimize: bool | None,
) -> CpuSampleCheck:
    """Check exact output dtypes, shapes and values for one declared CPU sample.

    ``optimize=None`` leaves runtime acceptance unverified and executes nothing.
    Comparisons use NumPy's array equality rules, including matching NaNs; this
    is not a bit-pattern check. Callers supply independent references appropriate
    to their sample and must not reuse its acceptance for other inputs/models.
    """
    onnx.checker.check_model(model, full_check=True)
    if optimize is None:
        return CpuSampleCheck(
            SampleStatus.UNVERIFIED,
            None,
            None,
            "No CPU execution profile was selected; sample acceptance is unverified.",
        )

    level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimize
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    target = CpuSampleTarget(ort.__version__, level.name, platform.machine())
    options = ort.SessionOptions()
    options.intra_op_num_threads = target.intra_op_threads
    options.inter_op_num_threads = target.inter_op_threads
    options.graph_optimization_level = level
    try:
        session = ort.InferenceSession(
            model.SerializeToString(), options, providers=[target.provider]
        )
        if session.get_providers() != [target.provider]:
            raise RuntimeError(
                f"Expected only {target.provider}; got {session.get_providers()}"
            )
        feeds = {
            spec.name: value
            for spec, value in zip(session.get_inputs(), inputs, strict=True)
        }
        outputs = tuple(session.run(None, feeds))
    except Exception as exc:
        raise SampleExecutionError(
            f"ORT {target.runtime_version} {target.provider} "
            f"{target.optimization} could not execute the sample: {exc}"
        ) from exc

    if len(outputs) != len(expected_outputs):
        return CpuSampleCheck(
            SampleStatus.FAILED,
            target,
            outputs,
            f"Expected {len(expected_outputs)} outputs, received {len(outputs)}.",
        )
    for index, (actual, expected) in enumerate(
        zip(outputs, expected_outputs, strict=True)
    ):
        try:
            np.testing.assert_array_equal(actual, expected, strict=True)
        except AssertionError as exc:
            return CpuSampleCheck(
                SampleStatus.FAILED,
                target,
                outputs,
                f"Output {index} failed exact sample acceptance: {exc}",
            )
    return CpuSampleCheck(
        SampleStatus.ACCEPTED,
        target,
        outputs,
        "The supplied sample matches its reference dtypes, shapes, and values.",
    )


def require_sample_acceptance(check: CpuSampleCheck) -> None:
    """Reject both known failures and samples whose target was not validated."""
    if check.status is not SampleStatus.ACCEPTED:
        raise SampleAcceptanceError(f"{check.status.value}: {check.detail}")
