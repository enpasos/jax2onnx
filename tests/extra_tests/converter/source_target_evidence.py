# tests/extra_tests/converter/source_target_evidence.py

"""Exact observed CPU target boundaries; a failed target never gains acceptance."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest

from jax2onnx.precision import (
    CheckStatus,
    SampleAcceptanceError,
    SampleCriteria,
    SourceSample,
    SourceSampleEvidence,
    SourceValidationReport,
    _array_evidence,
)

# Observed 2026-10-21 on original FP64 einsum, ordered symbolic contraction,
# and Eqx vmap artifacts. Independent actual 37-observation audit:
# 6ced5aac6509fbce57f0fcebbc094b240b232ad62ed46d1d4d5e7d15c7a0ff16.
# This ledger describes a negative target, not numerical acceptance.
OBSERVED_FP64_TARGET = {
    "runtime_version": "1.24.1",
    "architecture": "x86_64",
    "optimization": "ORT_ENABLE_ALL",
    "requested_providers": ("CPUExecutionProvider",),
    "actual_initialized_providers": (),
    "outcome": "FAILED_RUNTIME_LOAD_NUMERICS_UNVERIFIED",
}
FUSED_ERROR = (
    "[ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Failed to find kernel for "
    "com.microsoft.FusedMatMul(1) (node:'{node}/MatmulTransposeFusion/' "
    "ep:'CPUExecutionProvider'). Op with name ({node}/MatmulTransposeFusion/) "
    "domain (com.microsoft) and type (FusedMatMul) kernel is not supported in "
    "CPUExecutionProvider. Encountered following errors: (Kernel found kernel "
    "in the supported version range (node_version: 1). However the types are "
    "incompatible. This op has been implemented only for the following types "
    "(tensor(float),), but the node in the model has the following type "
    "(tensor(double))\n)"
)


def report_evidence(report: SourceValidationReport) -> dict[str, Any]:
    return {
        "artifact_sha256": report.artifact_sha256,
        "target": None if report.target is None else asdict(report.target),
        "status": report.status.value,
        "checks": [asdict(check) for check in report.checks],
        "samples": [asdict(sample) for sample in report.samples],
        "scope": report.scope,
    }


def _request_identity(index: int, sample: SourceSample) -> str:
    assert not isinstance(
        sample.inputs, Mapping
    ), "These owners use ordered input tuples"
    assert sample.expected_outputs is not None
    inputs = tuple(
        _array_evidence(str(i), np.asarray(v)) for i, v in enumerate(sample.inputs)
    )
    references = tuple(
        _array_evidence(str(i), np.asarray(v))
        for i, v in enumerate(sample.expected_outputs)
    )
    return SourceSampleEvidence(
        index, inputs, references, (), sample.criteria, ()
    ).sample_sha256


def _double_transpose_matmuls(model: onnx.ModelProto) -> list[dict[str, Any]]:
    """Check the original serialized graph, before any provider optimization."""

    def nodes(message: Any) -> Any:
        if isinstance(message, onnx.NodeProto):
            yield message
        for descriptor, value in message.ListFields():
            if descriptor.message_type is not None:
                for child in value if descriptor.is_repeated else (value,):
                    yield from nodes(child)

    assert not any(
        n.op_type == "FusedMatMul" for n in nodes(model)
    ), "Original artifact must not contain FusedMatMul"
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    types = {
        v.name: v.type.tensor_type.elem_type
        for v in (
            *inferred.graph.input,
            *inferred.graph.output,
            *inferred.graph.value_info,
        )
    }
    types.update({v.name: v.data_type for v in inferred.graph.initializer})
    producers = {name: n for n in inferred.graph.node for name in n.output}
    witnesses = []
    for node in inferred.graph.node:
        if node.op_type != "MatMul" or node.domain not in ("", "ai.onnx"):
            continue
        if not all(
            types.get(name) == onnx.TensorProto.DOUBLE
            for name in (*node.input, *node.output)
        ):
            continue
        transposes = [
            producers[name]
            for name in node.input
            if name in producers
            and producers[name].op_type == "Transpose"
            and producers[name].domain in ("", "ai.onnx")
        ]
        transposes += [
            n
            for n in inferred.graph.node
            if n.op_type == "Transpose"
            and n.domain in ("", "ai.onnx")
            and any(x in node.output for x in n.input)
        ]
        if transposes:
            witnesses.append(
                {
                    "matmul": node.name,
                    "onnx_dtype": "DOUBLE",
                    "transposes": [n.name for n in transposes],
                }
            )
    assert witnesses, "Need original standard DOUBLE MatMul adjacent to Transpose"
    return witnesses


def require_target_pair(
    model: onnx.ModelProto,
    samples: Sequence[SourceSample],
    disabled: SourceValidationReport,
    enabled: SourceValidationReport,
    *,
    evidence_path: Path,
) -> dict[str, Any]:
    """Require two accepted targets or the exact observed negative ALL boundary.

    Callers use the unchanged validator and identical samples for both modes.
    A constructor failure records only the first requested sample; later ALL
    samples remain unexecuted. Their acceptance is never inferred from DISABLED.
    """
    evidence = {
        "disabled": report_evidence(disabled),
        "all": report_evidence(enabled),
        "scope": "Original default CPU requests; negative target coverage is not numerical acceptance.",
    }
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
    assert samples and all(s.criteria == SampleCriteria("bitwise") for s in samples)
    identities = [_request_identity(i, sample) for i, sample in enumerate(samples)]
    evidence["requested_sample_sha256"] = identities
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
    digest = hashlib.sha256(model.SerializeToString(deterministic=True)).hexdigest()
    assert (
        disabled.artifact_sha256 == enabled.artifact_sha256 == digest
    ), "Artifact identity differs"
    assert [
        s.sample_sha256 for s in disabled.samples
    ] == identities, "Disabled sample identities differ"
    assert [s.sample_sha256 for s in enabled.samples] == identities[
        : len(enabled.samples)
    ], "ALL sample identities differ"
    assert [s.index for s in disabled.samples] == list(
        range(len(samples))
    ), "Disabled sample indices differ"
    assert [s.index for s in enabled.samples] == list(
        range(len(enabled.samples))
    ), "ALL sample indices differ"
    assert disabled.target is not None and enabled.target is not None
    for target in (disabled.target, enabled.target):
        assert (
            target.intra_op_threads,
            target.inter_op_threads,
            target.execution_mode,
            target.session_config,
        ) == (1, 1, "ORT_SEQUENTIAL", ()), "Unexpected CPU session settings"
    assert disabled.target.optimization == "ORT_DISABLE_ALL"
    assert enabled.target.optimization == "ORT_ENABLE_ALL"
    assert (disabled.target.runtime_version, disabled.target.architecture) == (
        enabled.target.runtime_version,
        enabled.target.architecture,
    )
    assert disabled.target.providers == ("CPUExecutionProvider",)
    assert disabled.target.provider_options == (("CPUExecutionProvider", ()),)
    disabled.require_sample_acceptance(model)
    for sample in disabled.samples:
        assert [(v.dtype, v.shape, v.sha256) for v in sample.outputs] == [
            (v.dtype, v.shape, v.sha256) for v in sample.references
        ], "Disabled bitwise outputs differ from references"
    if enabled.status is CheckStatus.VALIDATED:
        assert len(enabled.samples) == len(samples)
        assert enabled.target.providers == disabled.target.providers
        assert enabled.target.provider_options == disabled.target.provider_options
        enabled.require_sample_acceptance(model)
        evidence["outcome"] = "BOTH_TARGETS_NUMERICALLY_VALIDATED"
    else:
        assert enabled.status is CheckStatus.FAILED
        assert (enabled.target.runtime_version, enabled.target.architecture) == (
            OBSERVED_FP64_TARGET["runtime_version"],
            OBSERVED_FP64_TARGET["architecture"],
        ), "Unobserved target failure"
        assert (
            enabled.target.providers == () and enabled.target.provider_options == ()
        ), "Load failure must retain uninitialized provider metadata"
        witnesses = _double_transpose_matmuls(model)
        assert [(c.category, c.status) for c in enabled.checks] == [
            ("artifact-load", CheckStatus.VALIDATED),
            ("onnx-schema", CheckStatus.VALIDATED),
            ("runtime-load", CheckStatus.FAILED),
        ], "Unexpected failure stage"
        detail = enabled.checks[-1].detail
        assert detail in [
            FUSED_ERROR.format(node=w["matmul"]) for w in witnesses
        ], "Not the exact observed FusedMatMul DOUBLE CPU error"
        assert len(enabled.samples) == 1 and not enabled.samples[0].outputs
        assert [
            (c.category, c.status, c.detail) for c in enabled.samples[0].checks
        ] == [
            ("runtime-load", CheckStatus.FAILED, detail),
            (
                "numerical-acceptance",
                CheckStatus.UNVERIFIED,
                "The runtime could not load the artifact.",
            ),
        ]
        with pytest.raises(SampleAcceptanceError, match="FusedMatMul"):
            enabled.require_sample_acceptance(model)
        evidence.update(
            {
                "outcome": OBSERVED_FP64_TARGET["outcome"],
                "observed_target": OBSERVED_FP64_TARGET,
                "source_provenance": witnesses,
                "raising_gate": "SampleAcceptanceError",
                "unexecuted_all_sample_indices": list(range(1, len(samples))),
            }
        )
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence
