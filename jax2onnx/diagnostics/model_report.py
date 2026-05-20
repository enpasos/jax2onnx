# jax2onnx/diagnostics/model_report.py

"""Structured diagnostics for exported ONNX models."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import importlib
from importlib import metadata
import json
import os
import platform
from pathlib import Path
from typing import Any, Callable, Literal, TypeAlias

import onnx
import numpy as np
from onnx import TensorProto


ModelReportInput: TypeAlias = onnx.ModelProto | bytes | str | os.PathLike[str]
RuntimeInputs: TypeAlias = Mapping[str, object]
ExpectedOutputs: TypeAlias = Mapping[str, object] | Sequence[object]
FindingSeverity: TypeAlias = Literal["info", "warning", "error"]
DiagnosticStatus: TypeAlias = Literal["pass", "warning", "error"]
ReportGatePolicy: TypeAlias = Literal["errors", "warnings"]
ReportOutputFormat: TypeAlias = Literal["markdown", "json"]
TargetName: TypeAlias = Literal["ort-cpu", "ort-web", "ort-mobile"]
EvaluationStatus: TypeAlias = Literal["pass", "fail", "not_evaluated"]
ShapeDim = int | str | None
_WEB_MODEL_SIZE_WARNING_BYTES = 10 * 1024 * 1024
_MOBILE_MODEL_SIZE_WARNING_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class ValidationResult:
    """Result of a validation step that should not abort report generation."""

    passed: bool
    error: str | None = None


@dataclass(frozen=True)
class OpsetImportInfo:
    """One ONNX opset import entry."""

    domain: str
    version: int


@dataclass(frozen=True)
class TensorInfo:
    """Name, dtype, and shape metadata for a model value."""

    name: str
    elem_type: int | None
    elem_type_name: str
    shape: tuple[ShapeDim, ...] | None


@dataclass(frozen=True)
class InitializerInfo:
    """Metadata for one graph initializer."""

    name: str
    elem_type: int
    elem_type_name: str
    shape: tuple[int, ...]
    byte_size: int


@dataclass(frozen=True)
class OperatorCount:
    """Number of nodes using a given operator."""

    domain: str
    op_type: str
    count: int


@dataclass(frozen=True)
class DTypeCount:
    """Number of graph values or initializers using a given dtype."""

    elem_type: int | None
    elem_type_name: str
    count: int


@dataclass(frozen=True)
class ReportFinding:
    """Interpreted diagnostic finding derived from model facts."""

    severity: FindingSeverity
    message: str


@dataclass(frozen=True)
class DiagnosticSummary:
    """Aggregate status across model findings and optional target diagnostics."""

    status: DiagnosticStatus
    error_count: int
    warning_count: int
    info_count: int


@dataclass(frozen=True)
class PackageVersion:
    """Installed package version captured for reproducible diagnostics."""

    name: str
    version: str | None


@dataclass(frozen=True)
class EnvironmentInfo:
    """Toolchain versions relevant to export and runtime diagnostics."""

    python_version: str
    package_versions: tuple[PackageVersion, ...]


@dataclass(frozen=True)
class TargetDiagnostic:
    """Warning-only diagnostics for a deployment target profile."""

    target: TargetName
    findings: tuple[ReportFinding, ...]


@dataclass(frozen=True)
class RuntimeTensorSample:
    """Observed runtime tensor metadata for supplied inputs or produced outputs."""

    name: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class RuntimeDimensionBinding:
    """Concrete values observed for one symbolic public runtime dimension."""

    dim_param: str
    values: tuple[int, ...]
    sources: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeOutputComparison:
    """Numeric comparison for one ONNX Runtime output."""

    output_name: str
    passed: bool
    max_abs_diff: float | None = None
    max_rel_diff: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class RuntimeExecutionResult:
    """Optional ONNX Runtime CPU smoke-test result."""

    evaluated: bool
    passed: bool | None = None
    error: str | None = None
    providers: tuple[str, ...] = ()
    output_count: int | None = None
    input_samples: tuple[RuntimeTensorSample, ...] = ()
    output_samples: tuple[RuntimeTensorSample, ...] = ()
    dimension_bindings: tuple[RuntimeDimensionBinding, ...] = ()
    output_comparisons: tuple[RuntimeOutputComparison, ...] = ()
    rtol: float | None = None
    atol: float | None = None


@dataclass(frozen=True)
class ModelReport:
    """Deployment-readiness facts derived from an ONNX model."""

    model_name: str
    ir_version: int
    producer_name: str
    producer_version: str
    model_size_bytes: int
    opset_imports: tuple[OpsetImportInfo, ...]
    checker: ValidationResult
    strict_shape_inference: ValidationResult
    findings: tuple[ReportFinding, ...]
    diagnostic_summary: DiagnosticSummary
    environment: EnvironmentInfo
    node_count: int
    function_count: int
    operator_counts: tuple[OperatorCount, ...]
    inputs: tuple[TensorInfo, ...]
    outputs: tuple[TensorInfo, ...]
    initializers: tuple[InitializerInfo, ...]
    dtype_counts: tuple[DTypeCount, ...]
    runtime_execution: RuntimeExecutionResult
    target_diagnostics: tuple[TargetDiagnostic, ...]


@dataclass(frozen=True)
class JaxExportAnalysis:
    """ONNX model plus diagnostics produced from a JAX export."""

    model: onnx.ModelProto
    report: ModelReport


@dataclass(frozen=True)
class RuntimeProfile:
    """Named sample inputs used to validate one runtime shape profile."""

    name: str
    sample_inputs: tuple[object, ...]


@dataclass(frozen=True)
class RuntimeProfileReport:
    """Report generated for one named runtime profile."""

    name: str
    report: ModelReport


@dataclass(frozen=True)
class JaxExportProfileAnalysis:
    """ONNX model plus diagnostics for multiple runtime profiles."""

    model: onnx.ModelProto
    profile_reports: tuple[RuntimeProfileReport, ...]


@dataclass(frozen=True)
class RuntimeProfileSummary:
    """Compact diagnostic summary for one runtime profile."""

    name: str
    status: DiagnosticStatus
    error_count: int
    warning_count: int
    runtime_status: EvaluationStatus
    numeric_parity_status: EvaluationStatus
    dimension_bindings: tuple[RuntimeDimensionBinding, ...]
    output_samples: tuple[RuntimeTensorSample, ...]
    output_comparisons: tuple[RuntimeOutputComparison, ...]
    max_abs_diff: float | None
    max_rel_diff: float | None


@dataclass(frozen=True)
class ProfileAnalysisSummary:
    """Aggregate diagnostic summary across runtime profiles."""

    model_name: str
    profile_count: int
    status: DiagnosticStatus
    error_count: int
    warning_count: int
    runtime_pass_count: int
    runtime_fail_count: int
    runtime_not_evaluated_count: int
    numeric_parity_pass_count: int
    numeric_parity_fail_count: int
    numeric_parity_not_evaluated_count: int
    profiles: tuple[RuntimeProfileSummary, ...]


@dataclass(frozen=True)
class ReportGateResult:
    """CI-style pass/fail interpretation of a ``ModelReport``."""

    passed: bool
    fail_on: ReportGatePolicy
    status: DiagnosticStatus
    error_count: int
    warning_count: int
    message: str


def analyze_model(
    model: ModelReportInput,
    *,
    runtime_inputs: RuntimeInputs | None = None,
    expected_outputs: ExpectedOutputs | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    targets: Sequence[TargetName] = (),
) -> ModelReport:
    """Return structured diagnostics for an ONNX model.

    The report is intentionally non-fatal: checker and strict shape-inference
    failures are captured in ``ValidationResult`` values so callers can still
    inspect the model inventory.
    """

    model_proto = _load_model(model)
    serialized = model_proto.SerializeToString()
    checker = _check_model(model_proto)
    shape_inference = _infer_shapes_strict(model_proto)
    inventory_model = (
        shape_inference.model if shape_inference.model is not None else model_proto
    )
    operator_counter = _operator_counter(model_proto)
    dtype_counter = _dtype_counter(inventory_model)
    runtime_execution = _run_runtime_smoke_test(
        model_proto,
        runtime_inputs,
        expected_outputs,
        rtol,
        atol,
    )

    report = ModelReport(
        model_name=str(inventory_model.graph.name),
        ir_version=int(inventory_model.ir_version),
        producer_name=str(inventory_model.producer_name),
        producer_version=str(inventory_model.producer_version),
        model_size_bytes=len(serialized),
        opset_imports=_opset_imports(inventory_model),
        checker=checker,
        strict_shape_inference=shape_inference.result,
        findings=(),
        diagnostic_summary=DiagnosticSummary("pass", 0, 0, 0),
        environment=_environment_info(),
        node_count=sum(operator_counter.values()),
        function_count=len(inventory_model.functions),
        operator_counts=_operator_counts(operator_counter),
        inputs=tuple(_tensor_info(value) for value in inventory_model.graph.input),
        outputs=tuple(_tensor_info(value) for value in inventory_model.graph.output),
        initializers=tuple(
            _initializer_info(initializer)
            for initializer in inventory_model.graph.initializer
        ),
        dtype_counts=_dtype_counts(dtype_counter),
        runtime_execution=runtime_execution,
        target_diagnostics=(),
    )
    report = replace(report, findings=_report_findings(report))
    report = replace(report, target_diagnostics=_target_diagnostics(report, targets))
    return replace(report, diagnostic_summary=_diagnostic_summary(report))


def analyze_jax_export(
    fn: Callable[..., Any],
    inputs: Sequence[Any],
    *,
    sample_inputs: Sequence[object] | None = None,
    runtime_inputs: RuntimeInputs | None = None,
    expected_outputs: ExpectedOutputs | None = None,
    model_name: str = "jax_model",
    opset: int = 23,
    input_params: Mapping[str, object] | None = None,
    enable_double_precision: bool = False,
    record_primitive_calls_file: str | None = None,
    inputs_as_nchw: Sequence[int] | None = None,
    outputs_as_nchw: Sequence[int] | None = None,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    targets: Sequence[TargetName] = (),
) -> JaxExportAnalysis:
    """Export a JAX callable and immediately build an internal model report.

    ``sample_inputs`` are optional. When supplied, they are mapped to the
    exported graph input names for an ONNX Runtime CPU smoke test, and the JAX
    callable is evaluated once to derive named expected outputs for parity.
    Supplying ``runtime_inputs`` or ``expected_outputs`` overrides the derived
    values independently.
    """

    from jax2onnx.user_interface import to_onnx

    model = to_onnx(
        fn,
        inputs,
        input_params=input_params,
        model_name=model_name,
        opset=opset,
        enable_double_precision=enable_double_precision,
        record_primitive_calls_file=record_primitive_calls_file,
        return_mode="proto",
        inputs_as_nchw=inputs_as_nchw,
        outputs_as_nchw=outputs_as_nchw,
        input_names=input_names,
        output_names=output_names,
    )
    derived_runtime_inputs = runtime_inputs
    if derived_runtime_inputs is None and sample_inputs is not None:
        derived_runtime_inputs = _runtime_inputs_from_sample_inputs(
            model,
            sample_inputs,
        )

    derived_expected_outputs = expected_outputs
    if derived_expected_outputs is None and sample_inputs is not None:
        derived_expected_outputs = _expected_outputs_from_jax_reference(
            model,
            fn,
            sample_inputs,
        )

    report = analyze_model(
        model,
        runtime_inputs=derived_runtime_inputs,
        expected_outputs=derived_expected_outputs,
        rtol=rtol,
        atol=atol,
        targets=targets,
    )
    return JaxExportAnalysis(model=model, report=report)


def analyze_jax_export_profiles(
    fn: Callable[..., Any],
    inputs: Sequence[Any],
    *,
    profiles: Sequence[RuntimeProfile],
    model_name: str = "jax_model",
    opset: int = 23,
    input_params: Mapping[str, object] | None = None,
    enable_double_precision: bool = False,
    record_primitive_calls_file: str | None = None,
    inputs_as_nchw: Sequence[int] | None = None,
    outputs_as_nchw: Sequence[int] | None = None,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    targets: Sequence[TargetName] = (),
) -> JaxExportProfileAnalysis:
    """Export once and build reports for several runtime sample profiles."""

    from jax2onnx.user_interface import to_onnx

    _validate_runtime_profiles(profiles)
    model = to_onnx(
        fn,
        inputs,
        input_params=input_params,
        model_name=model_name,
        opset=opset,
        enable_double_precision=enable_double_precision,
        record_primitive_calls_file=record_primitive_calls_file,
        return_mode="proto",
        inputs_as_nchw=inputs_as_nchw,
        outputs_as_nchw=outputs_as_nchw,
        input_names=input_names,
        output_names=output_names,
    )
    profile_reports = tuple(
        RuntimeProfileReport(
            name=profile.name,
            report=analyze_model(
                model,
                runtime_inputs=_runtime_inputs_from_sample_inputs(
                    model,
                    profile.sample_inputs,
                ),
                expected_outputs=_expected_outputs_from_jax_reference(
                    model,
                    fn,
                    profile.sample_inputs,
                ),
                rtol=rtol,
                atol=atol,
                targets=targets,
            ),
        )
        for profile in profiles
    )
    return JaxExportProfileAnalysis(model=model, profile_reports=profile_reports)


def summarize_profile_analysis(
    analysis: JaxExportProfileAnalysis,
) -> ProfileAnalysisSummary:
    """Return compact aggregate diagnostics across runtime profiles."""

    profile_summaries = tuple(
        _runtime_profile_summary(profile_report)
        for profile_report in analysis.profile_reports
    )
    return ProfileAnalysisSummary(
        model_name=str(analysis.model.graph.name),
        profile_count=len(profile_summaries),
        status=_aggregate_status(
            tuple(profile.status for profile in profile_summaries)
        ),
        error_count=sum(profile.error_count for profile in profile_summaries),
        warning_count=sum(profile.warning_count for profile in profile_summaries),
        runtime_pass_count=_count_profile_status(
            profile_summaries,
            "runtime_status",
            "pass",
        ),
        runtime_fail_count=_count_profile_status(
            profile_summaries,
            "runtime_status",
            "fail",
        ),
        runtime_not_evaluated_count=_count_profile_status(
            profile_summaries,
            "runtime_status",
            "not_evaluated",
        ),
        numeric_parity_pass_count=_count_profile_status(
            profile_summaries,
            "numeric_parity_status",
            "pass",
        ),
        numeric_parity_fail_count=_count_profile_status(
            profile_summaries,
            "numeric_parity_status",
            "fail",
        ),
        numeric_parity_not_evaluated_count=_count_profile_status(
            profile_summaries,
            "numeric_parity_status",
            "not_evaluated",
        ),
        profiles=profile_summaries,
    )


def evaluate_model_report_gate(
    report: ModelReport,
    *,
    fail_on: ReportGatePolicy = "errors",
) -> ReportGateResult:
    """Return a simple pass/fail decision for automation around a report.

    ``fail_on="errors"`` accepts reports with warnings. ``fail_on="warnings"``
    requires a completely clean report.
    """

    if fail_on not in {"errors", "warnings"}:
        raise ValueError("fail_on must be 'errors' or 'warnings'.")

    summary = report.diagnostic_summary
    if summary.error_count:
        return ReportGateResult(
            passed=False,
            fail_on=fail_on,
            status=summary.status,
            error_count=summary.error_count,
            warning_count=summary.warning_count,
            message=f"Model report gate failed: {summary.error_count} error(s) found.",
        )
    if fail_on == "warnings" and summary.warning_count:
        return ReportGateResult(
            passed=False,
            fail_on=fail_on,
            status=summary.status,
            error_count=summary.error_count,
            warning_count=summary.warning_count,
            message=(
                "Model report gate failed: "
                f"{summary.warning_count} warning(s) found."
            ),
        )

    message = "Model report gate passed."
    if summary.warning_count:
        message = f"Model report gate passed with {summary.warning_count} warning(s)."
    return ReportGateResult(
        passed=True,
        fail_on=fail_on,
        status=summary.status,
        error_count=summary.error_count,
        warning_count=summary.warning_count,
        message=message,
    )


def format_model_report_markdown(report: ModelReport) -> str:
    """Render a ``ModelReport`` as a compact Markdown diagnostics report."""

    lines: list[str] = [
        f"# Model Report: {report.model_name or '<unnamed>'}",
        "",
        "## Status",
        "",
        f"- Overall: {report.diagnostic_summary.status}",
        f"- Errors: {report.diagnostic_summary.error_count}",
        f"- Warnings: {report.diagnostic_summary.warning_count}",
        f"- Info: {report.diagnostic_summary.info_count}",
        "",
        "## Validation",
        "",
        f"- ONNX checker: {_status_text(report.checker)}",
        f"- Strict shape inference: {_status_text(report.strict_shape_inference)}",
        "",
        "## Findings",
        "",
    ]
    lines.extend(_finding_lines(report.findings))
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Model size: {_format_bytes(report.model_size_bytes)}",
            f"- ONNX IR version: {report.ir_version}",
            f"- Producer: {_producer_text(report)}",
            f"- Opset imports: {_opset_text(report.opset_imports)}",
            f"- Nodes: {report.node_count}",
            f"- Functions: {report.function_count}",
            "",
            "## Environment",
            "",
        ]
    )
    lines.extend(_environment_table(report.environment))
    lines.extend(
        [
            "",
            "## Runtime Smoke Test",
            "",
            f"- ONNX Runtime CPU: {_runtime_status_text(report.runtime_execution)}",
            f"- Numeric parity: {_runtime_parity_text(report.runtime_execution)}",
        ]
    )
    if report.runtime_execution.input_samples:
        lines.extend(["", "## Runtime Inputs", ""])
        lines.extend(_runtime_tensor_table(report.runtime_execution.input_samples))
    if report.runtime_execution.output_samples:
        lines.extend(["", "## Runtime Outputs", ""])
        lines.extend(_runtime_tensor_table(report.runtime_execution.output_samples))
    if report.runtime_execution.dimension_bindings:
        lines.extend(["", "## Runtime Dynamic Dimensions", ""])
        lines.extend(
            _runtime_dimension_bindings_table(
                report.runtime_execution.dimension_bindings
            )
        )
    if report.target_diagnostics:
        lines.extend(["", "## Target Diagnostics", ""])
        lines.extend(_target_diagnostics_table(report.target_diagnostics))
    if report.runtime_execution.output_comparisons:
        lines.extend(["", "## Runtime Output Comparisons", ""])
        lines.extend(
            _markdown_table(
                ("Output", "Status", "Max Abs Diff", "Max Rel Diff"),
                (
                    (
                        item.output_name,
                        "pass" if item.passed else "fail",
                        _float_text(item.max_abs_diff),
                        _float_text(item.max_rel_diff),
                    )
                    for item in report.runtime_execution.output_comparisons
                ),
                empty_text="No runtime output comparisons.",
            )
        )
    lines.extend(["", "## Operators", ""])
    lines.extend(
        _markdown_table(
            ("Domain", "Op", "Count"),
            (
                (_domain_text(item.domain), item.op_type, str(item.count))
                for item in report.operator_counts
            ),
            empty_text="No operators.",
        )
    )
    lines.extend(["", "## Inputs", ""])
    lines.extend(_tensor_table(report.inputs, empty_text="No inputs."))
    lines.extend(["", "## Outputs", ""])
    lines.extend(_tensor_table(report.outputs, empty_text="No outputs."))
    lines.extend(["", "## Initializers", ""])
    lines.extend(
        _markdown_table(
            ("Name", "Dtype", "Shape", "Size"),
            (
                (
                    item.name,
                    item.elem_type_name,
                    _shape_text(item.shape),
                    _format_bytes(item.byte_size),
                )
                for item in report.initializers
            ),
            empty_text="No initializers.",
        )
    )
    lines.extend(["", "## Dtypes", ""])
    lines.extend(
        _markdown_table(
            ("Dtype", "Count"),
            ((item.elem_type_name, str(item.count)) for item in report.dtype_counts),
            empty_text="No dtype metadata.",
        )
    )
    return "\n".join(lines).rstrip() + "\n"


def format_profile_analysis_markdown(analysis: JaxExportProfileAnalysis) -> str:
    """Render a compact Markdown report across runtime profiles."""

    summary = summarize_profile_analysis(analysis)
    lines: list[str] = [
        f"# Runtime Profile Report: {summary.model_name or '<unnamed>'}",
        "",
        "## Status",
        "",
        f"- Overall: {summary.status}",
        f"- Profiles: {summary.profile_count}",
        f"- Errors: {summary.error_count}",
        f"- Warnings: {summary.warning_count}",
        (
            "- Runtime: "
            f"{summary.runtime_pass_count} pass, "
            f"{summary.runtime_fail_count} fail, "
            f"{summary.runtime_not_evaluated_count} not evaluated"
        ),
        (
            "- Numeric parity: "
            f"{summary.numeric_parity_pass_count} pass, "
            f"{summary.numeric_parity_fail_count} fail, "
            f"{summary.numeric_parity_not_evaluated_count} not evaluated"
        ),
        "",
        "## Profiles",
        "",
    ]
    lines.extend(
        _markdown_table(
            (
                "Profile",
                "Status",
                "Runtime",
                "Parity",
                "Dynamic Dims",
                "Outputs",
                "Max Abs Diff",
                "Max Rel Diff",
            ),
            (
                (
                    profile.name,
                    profile.status,
                    _evaluation_status_text(profile.runtime_status),
                    _evaluation_status_text(profile.numeric_parity_status),
                    _profile_dimension_bindings_text(profile),
                    _profile_output_samples_text(profile),
                    _float_text(profile.max_abs_diff),
                    _float_text(profile.max_rel_diff),
                )
                for profile in summary.profiles
            ),
            empty_text="No runtime profiles.",
        )
    )
    return "\n".join(lines).rstrip() + "\n"


def model_report_to_dict(report: ModelReport) -> dict[str, Any]:
    """Return a JSON-serializable dictionary representation of a report."""

    return asdict(report)


def profile_analysis_to_dict(analysis: JaxExportProfileAnalysis) -> dict[str, Any]:
    """Return a JSON-serializable dictionary for a multi-profile analysis."""

    return {
        "summary": asdict(summarize_profile_analysis(analysis)),
        "profile_reports": [
            {
                "name": profile_report.name,
                "report": model_report_to_dict(profile_report.report),
            }
            for profile_report in analysis.profile_reports
        ],
    }


def format_model_report_json(report: ModelReport, *, indent: int | None = 2) -> str:
    """Render a ``ModelReport`` as stable JSON for internal tooling."""

    return json.dumps(model_report_to_dict(report), indent=indent) + "\n"


def format_profile_analysis_json(
    analysis: JaxExportProfileAnalysis,
    *,
    indent: int | None = 2,
) -> str:
    """Render a multi-profile analysis as stable JSON for internal tooling."""

    return json.dumps(profile_analysis_to_dict(analysis), indent=indent) + "\n"


def write_model_report(
    report: ModelReport,
    path: str | os.PathLike[str],
    *,
    output_format: ReportOutputFormat | None = None,
    json_indent: int | None = 2,
) -> Path:
    """Write a report as Markdown or JSON and return the written path."""

    output_path = Path(path)
    selected_format = output_format or _infer_report_output_format(output_path)
    if selected_format == "markdown":
        content = format_model_report_markdown(report)
    elif selected_format == "json":
        content = format_model_report_json(report, indent=json_indent)
    else:
        raise ValueError("output_format must be 'markdown' or 'json'.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def write_profile_analysis_report(
    analysis: JaxExportProfileAnalysis,
    path: str | os.PathLike[str],
    *,
    output_format: ReportOutputFormat | None = None,
    json_indent: int | None = 2,
) -> Path:
    """Write a multi-profile report as Markdown or JSON."""

    output_path = Path(path)
    selected_format = output_format or _infer_report_output_format(output_path)
    if selected_format == "markdown":
        content = format_profile_analysis_markdown(analysis)
    elif selected_format == "json":
        content = format_profile_analysis_json(analysis, indent=json_indent)
    else:
        raise ValueError("output_format must be 'markdown' or 'json'.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


@dataclass(frozen=True)
class _ShapeInferenceOutcome:
    result: ValidationResult
    model: onnx.ModelProto | None


def _runtime_profile_summary(
    profile_report: RuntimeProfileReport,
) -> RuntimeProfileSummary:
    report = profile_report.report
    output_comparisons = report.runtime_execution.output_comparisons
    return RuntimeProfileSummary(
        name=profile_report.name,
        status=report.diagnostic_summary.status,
        error_count=report.diagnostic_summary.error_count,
        warning_count=report.diagnostic_summary.warning_count,
        runtime_status=_runtime_evaluation_status(report.runtime_execution),
        numeric_parity_status=_numeric_parity_evaluation_status(
            report.runtime_execution
        ),
        dimension_bindings=report.runtime_execution.dimension_bindings,
        output_samples=report.runtime_execution.output_samples,
        output_comparisons=output_comparisons,
        max_abs_diff=_max_output_comparison_value(
            output_comparisons,
            "max_abs_diff",
        ),
        max_rel_diff=_max_output_comparison_value(
            output_comparisons,
            "max_rel_diff",
        ),
    )


def _runtime_evaluation_status(
    runtime_execution: RuntimeExecutionResult,
) -> EvaluationStatus:
    if not runtime_execution.evaluated:
        return "not_evaluated"
    if runtime_execution.passed:
        return "pass"
    return "fail"


def _numeric_parity_evaluation_status(
    runtime_execution: RuntimeExecutionResult,
) -> EvaluationStatus:
    if not runtime_execution.output_comparisons:
        return "not_evaluated"
    if all(comparison.passed for comparison in runtime_execution.output_comparisons):
        return "pass"
    return "fail"


def _max_output_comparison_value(
    comparisons: tuple[RuntimeOutputComparison, ...],
    field_name: Literal["max_abs_diff", "max_rel_diff"],
) -> float | None:
    if field_name == "max_abs_diff":
        candidate_values = (comparison.max_abs_diff for comparison in comparisons)
    else:
        candidate_values = (comparison.max_rel_diff for comparison in comparisons)
    values = tuple(value for value in candidate_values if value is not None)
    if not values:
        return None
    return max(values)


def _aggregate_status(statuses: tuple[DiagnosticStatus, ...]) -> DiagnosticStatus:
    if "error" in statuses:
        return "error"
    if "warning" in statuses:
        return "warning"
    return "pass"


def _count_profile_status(
    profiles: tuple[RuntimeProfileSummary, ...],
    field_name: Literal["runtime_status", "numeric_parity_status"],
    status: EvaluationStatus,
) -> int:
    return sum(1 for profile in profiles if getattr(profile, field_name) == status)


def _infer_report_output_format(path: Path) -> ReportOutputFormat:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix == ".json":
        return "json"
    raise ValueError(
        "Cannot infer report output format from file suffix; use "
        "output_format='markdown' or output_format='json'."
    )


def _validate_runtime_profiles(profiles: Sequence[RuntimeProfile]) -> None:
    if not profiles:
        raise ValueError("profiles must contain at least one RuntimeProfile.")
    names = [profile.name for profile in profiles]
    if any(not name for name in names):
        raise ValueError("RuntimeProfile names must be non-empty.")
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise ValueError(
            "RuntimeProfile names must be unique; duplicates: "
            + _join_limited(tuple(duplicate_names))
            + "."
        )


def _runtime_inputs_from_sample_inputs(
    model: onnx.ModelProto,
    sample_inputs: Sequence[object],
) -> RuntimeInputs:
    input_names = _graph_input_names(model)
    if len(sample_inputs) != len(input_names):
        raise ValueError(
            "sample_inputs length must match the exported model graph inputs "
            f"when runtime_inputs are not provided; got {len(sample_inputs)} "
            f"sample input(s) for {len(input_names)} graph input(s)."
        )
    return {
        input_name: np.asarray(sample_input)
        for input_name, sample_input in zip(
            input_names,
            sample_inputs,
            strict=True,
        )
    }


def _expected_outputs_from_jax_reference(
    model: onnx.ModelProto,
    fn: Callable[..., Any],
    sample_inputs: Sequence[object],
) -> ExpectedOutputs:
    reference_outputs = fn(*sample_inputs)
    if isinstance(reference_outputs, Mapping):
        return {
            str(output_name): np.asarray(output_value)
            for output_name, output_value in reference_outputs.items()
        }

    output_values = _reference_output_values(reference_outputs)
    output_names = _graph_output_names(model)
    if len(output_values) != len(output_names):
        raise ValueError(
            "JAX reference output count must match the exported model outputs "
            f"when expected_outputs are not provided; got {len(output_values)} "
            f"reference output(s) for {len(output_names)} graph output(s)."
        )
    return {
        output_name: np.asarray(output_value)
        for output_name, output_value in zip(
            output_names,
            output_values,
            strict=True,
        )
    }


def _reference_output_values(reference_outputs: object) -> tuple[object, ...]:
    if isinstance(reference_outputs, (tuple, list)):
        return tuple(reference_outputs)
    return (reference_outputs,)


def _graph_input_names(model: onnx.ModelProto) -> tuple[str, ...]:
    return tuple(
        value.name or f"input_{index}" for index, value in enumerate(model.graph.input)
    )


def _graph_output_names(model: onnx.ModelProto) -> tuple[str, ...]:
    return tuple(
        value.name or f"output_{index}"
        for index, value in enumerate(model.graph.output)
    )


def _environment_info() -> EnvironmentInfo:
    package_names = (
        "jax2onnx",
        "jax",
        "numpy",
        "onnx",
        "onnxruntime",
        "onnx-ir",
    )
    return EnvironmentInfo(
        python_version=platform.python_version(),
        package_versions=tuple(
            PackageVersion(name=name, version=_package_version(name))
            for name in package_names
        ),
    )


def _package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _load_model(model: object) -> onnx.ModelProto:
    if isinstance(model, onnx.ModelProto):
        return model
    if isinstance(model, bytes):
        model_proto = onnx.ModelProto()
        model_proto.ParseFromString(model)
        return model_proto
    if isinstance(model, (str, os.PathLike)):
        return onnx.load(Path(model))
    raise TypeError(f"Unsupported model input type: {type(model)}")


def _check_model(model: onnx.ModelProto) -> ValidationResult:
    try:
        onnx.checker.check_model(model)
    except Exception as exc:  # noqa: BLE001 - preserve report generation.
        return ValidationResult(False, _format_error(exc))
    return ValidationResult(True)


def _infer_shapes_strict(model: onnx.ModelProto) -> _ShapeInferenceOutcome:
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except Exception as exc:  # noqa: BLE001 - preserve report generation.
        return _ShapeInferenceOutcome(ValidationResult(False, _format_error(exc)), None)
    return _ShapeInferenceOutcome(ValidationResult(True), inferred)


def _run_runtime_smoke_test(
    model: onnx.ModelProto,
    runtime_inputs: RuntimeInputs | None,
    expected_outputs: ExpectedOutputs | None,
    rtol: float,
    atol: float,
) -> RuntimeExecutionResult:
    if runtime_inputs is None:
        return RuntimeExecutionResult(evaluated=False)

    input_samples = _runtime_input_samples(model, runtime_inputs)
    input_dimension_bindings = _runtime_dimension_bindings(model, input_samples)
    try:
        ort = importlib.import_module("onnxruntime")
        inference_session = getattr(ort, "InferenceSession")
        session = inference_session(
            model.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        outputs = session.run(None, dict(runtime_inputs))
    except Exception as exc:  # noqa: BLE001 - preserve report generation.
        return RuntimeExecutionResult(
            evaluated=True,
            passed=False,
            error=_format_error(exc),
            input_samples=input_samples,
            dimension_bindings=input_dimension_bindings,
        )

    output_names = _graph_output_names(model)
    output_samples = _runtime_output_samples(output_names, outputs)
    dimension_bindings = _runtime_dimension_bindings(
        model,
        input_samples,
        output_samples,
    )
    output_comparisons = _compare_runtime_outputs(
        output_names,
        outputs,
        expected_outputs,
        rtol,
        atol,
    )
    passed = all(item.passed for item in output_comparisons)
    if not output_comparisons:
        passed = True
    return RuntimeExecutionResult(
        evaluated=True,
        passed=passed,
        providers=tuple(str(provider) for provider in session.get_providers()),
        output_count=len(outputs),
        input_samples=input_samples,
        output_samples=output_samples,
        dimension_bindings=dimension_bindings,
        output_comparisons=output_comparisons,
        rtol=rtol if output_comparisons else None,
        atol=atol if output_comparisons else None,
    )


def _runtime_input_samples(
    model: onnx.ModelProto,
    runtime_inputs: RuntimeInputs,
) -> tuple[RuntimeTensorSample, ...]:
    graph_input_names = _graph_input_names(model)
    graph_input_name_set = set(graph_input_names)
    samples = [
        _runtime_tensor_sample(input_name, runtime_inputs[input_name])
        for input_name in graph_input_names
        if input_name in runtime_inputs
    ]
    extra_input_names = sorted(
        str(input_name)
        for input_name in runtime_inputs
        if str(input_name) not in graph_input_name_set
    )
    samples.extend(
        _runtime_tensor_sample(input_name, runtime_inputs[input_name])
        for input_name in extra_input_names
    )
    return tuple(samples)


def _runtime_output_samples(
    output_names: tuple[str, ...],
    outputs: Sequence[object],
) -> tuple[RuntimeTensorSample, ...]:
    return tuple(
        _runtime_tensor_sample(
            output_names[index] if index < len(output_names) else f"output_{index}",
            output,
        )
        for index, output in enumerate(outputs)
    )


def _runtime_tensor_sample(name: str, value: object) -> RuntimeTensorSample:
    value_array = np.asarray(value)
    return RuntimeTensorSample(
        name=name,
        dtype=str(value_array.dtype),
        shape=tuple(int(dim) for dim in value_array.shape),
    )


def _runtime_dimension_bindings(
    model: onnx.ModelProto,
    input_samples: tuple[RuntimeTensorSample, ...],
    output_samples: tuple[RuntimeTensorSample, ...] = (),
) -> tuple[RuntimeDimensionBinding, ...]:
    observations: dict[str, list[tuple[int, str]]] = {}
    _collect_runtime_dimension_observations(
        model.graph.input,
        input_samples,
        "input",
        observations,
    )
    _collect_runtime_dimension_observations(
        model.graph.output,
        output_samples,
        "output",
        observations,
    )
    return tuple(
        RuntimeDimensionBinding(
            dim_param=dim_param,
            values=tuple(sorted({value for value, _source in observations[dim_param]})),
            sources=tuple(source for _value, source in observations[dim_param]),
        )
        for dim_param in sorted(observations)
    )


def _collect_runtime_dimension_observations(
    value_infos: Iterable[onnx.ValueInfoProto],
    samples: tuple[RuntimeTensorSample, ...],
    fallback_prefix: str,
    observations: dict[str, list[tuple[int, str]]],
) -> None:
    samples_by_name = {sample.name: sample for sample in samples}
    for index, value_info in enumerate(value_infos):
        value_name = value_info.name or f"{fallback_prefix}_{index}"
        sample = samples_by_name.get(value_name)
        if sample is None or not value_info.type.HasField("tensor_type"):
            continue

        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue

        symbolic_shape = _shape_tuple(tensor_type.shape)
        for axis, dim in enumerate(symbolic_shape):
            if not isinstance(dim, str) or axis >= len(sample.shape):
                continue
            observations.setdefault(dim, []).append(
                (sample.shape[axis], f"{sample.name}[{axis}]")
            )


def _compare_runtime_outputs(
    output_names: tuple[str, ...],
    outputs: Sequence[object],
    expected_outputs: ExpectedOutputs | None,
    rtol: float,
    atol: float,
) -> tuple[RuntimeOutputComparison, ...]:
    if expected_outputs is None:
        return ()
    expected_by_name = _expected_outputs_by_name(output_names, expected_outputs)
    if isinstance(expected_by_name, RuntimeOutputComparison):
        return (expected_by_name,)

    comparisons: list[RuntimeOutputComparison] = []
    for index, output_name in enumerate(output_names):
        if index >= len(outputs):
            comparisons.append(
                RuntimeOutputComparison(
                    output_name=output_name,
                    passed=False,
                    error="Runtime did not produce this output.",
                )
            )
            continue
        if output_name not in expected_by_name:
            comparisons.append(
                RuntimeOutputComparison(
                    output_name=output_name,
                    passed=False,
                    error="Expected output was not provided.",
                )
            )
            continue
        comparisons.append(
            _compare_one_output(
                output_name,
                outputs[index],
                expected_by_name[output_name],
                rtol,
                atol,
            )
        )
    return tuple(comparisons)


def _expected_outputs_by_name(
    output_names: tuple[str, ...],
    expected_outputs: ExpectedOutputs,
) -> dict[str, object] | RuntimeOutputComparison:
    if isinstance(expected_outputs, Mapping):
        return {str(name): value for name, value in expected_outputs.items()}

    if len(expected_outputs) != len(output_names):
        return RuntimeOutputComparison(
            output_name="<all>",
            passed=False,
            error=(
                f"Expected {len(output_names)} output(s), got "
                f"{len(expected_outputs)} expected value(s)."
            ),
        )
    return dict(zip(output_names, expected_outputs, strict=True))


def _compare_one_output(
    output_name: str,
    got: object,
    expected: object,
    rtol: float,
    atol: float,
) -> RuntimeOutputComparison:
    got_arr = np.asarray(got)
    expected_arr = np.asarray(expected)
    if got_arr.shape != expected_arr.shape:
        return RuntimeOutputComparison(
            output_name=output_name,
            passed=False,
            error=f"Shape mismatch: got {got_arr.shape}, expected {expected_arr.shape}.",
        )

    if _is_numeric_array(got_arr) and _is_numeric_array(expected_arr):
        got_float = got_arr.astype(np.float64)
        expected_float = expected_arr.astype(np.float64)
        abs_diff = np.abs(got_float - expected_float)
        max_abs_diff = float(np.max(abs_diff)) if abs_diff.size else 0.0
        denominator = np.maximum(np.abs(expected_float), np.finfo(np.float64).tiny)
        rel_diff = abs_diff / denominator
        max_rel_diff = float(np.max(rel_diff)) if rel_diff.size else 0.0
        passed = bool(np.allclose(got_float, expected_float, rtol=rtol, atol=atol))
        return RuntimeOutputComparison(
            output_name=output_name,
            passed=passed,
            max_abs_diff=max_abs_diff,
            max_rel_diff=max_rel_diff,
        )

    passed = bool(np.array_equal(got_arr, expected_arr))
    return RuntimeOutputComparison(
        output_name=output_name,
        passed=passed,
        error=None if passed else "Non-numeric output mismatch.",
    )


def _is_numeric_array(value: np.ndarray[Any, np.dtype[Any]]) -> bool:
    return bool(np.issubdtype(value.dtype, np.number))


def _format_error(exc: Exception) -> str:
    message = str(exc)
    if message:
        return f"{exc.__class__.__name__}: {message}"
    return exc.__class__.__name__


def _opset_imports(model: onnx.ModelProto) -> tuple[OpsetImportInfo, ...]:
    imports = [
        OpsetImportInfo(domain=str(entry.domain), version=int(entry.version))
        for entry in model.opset_import
    ]
    return tuple(sorted(imports, key=lambda item: (item.domain, item.version)))


def _operator_counter(model: onnx.ModelProto) -> Counter[tuple[str, str]]:
    counter: Counter[tuple[str, str]] = Counter()
    _collect_graph_operators(model.graph, counter)
    for function in model.functions:
        for node in function.node:
            _collect_node_operator(node, counter)
    return counter


def _collect_graph_operators(
    graph: onnx.GraphProto,
    counter: Counter[tuple[str, str]],
) -> None:
    for node in graph.node:
        _collect_node_operator(node, counter)


def _collect_node_operator(
    node: onnx.NodeProto,
    counter: Counter[tuple[str, str]],
) -> None:
    counter[(str(node.domain), str(node.op_type))] += 1
    for attr in node.attribute:
        if attr.HasField("g"):
            _collect_graph_operators(attr.g, counter)
        for graph in attr.graphs:
            _collect_graph_operators(graph, counter)


def _operator_counts(
    counter: Counter[tuple[str, str]],
) -> tuple[OperatorCount, ...]:
    return tuple(
        OperatorCount(domain=domain, op_type=op_type, count=count)
        for (domain, op_type), count in sorted(counter.items())
    )


def _tensor_info(value_info: onnx.ValueInfoProto) -> TensorInfo:
    if not value_info.type.HasField("tensor_type"):
        return TensorInfo(
            name=str(value_info.name),
            elem_type=None,
            elem_type_name="UNDEFINED",
            shape=None,
        )

    tensor_type = value_info.type.tensor_type
    elem_type = int(tensor_type.elem_type) if tensor_type.elem_type else None
    shape = _shape_tuple(tensor_type.shape) if tensor_type.HasField("shape") else None
    return TensorInfo(
        name=str(value_info.name),
        elem_type=elem_type,
        elem_type_name=_elem_type_name(elem_type),
        shape=shape,
    )


def _shape_tuple(shape: onnx.TensorShapeProto) -> tuple[ShapeDim, ...]:
    dims: list[ShapeDim] = []
    for dim in shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(str(dim.dim_param))
        else:
            dims.append(None)
    return tuple(dims)


def _initializer_info(initializer: onnx.TensorProto) -> InitializerInfo:
    elem_type = int(initializer.data_type)
    return InitializerInfo(
        name=str(initializer.name),
        elem_type=elem_type,
        elem_type_name=_elem_type_name(elem_type),
        shape=tuple(int(dim) for dim in initializer.dims),
        byte_size=int(initializer.ByteSize()),
    )


def _dtype_counter(model: onnx.ModelProto) -> Counter[int | None]:
    counter: Counter[int | None] = Counter()
    _collect_graph_dtypes(model.graph, counter)
    return counter


def _collect_graph_dtypes(
    graph: onnx.GraphProto,
    counter: Counter[int | None],
) -> None:
    for value_info in graph.input:
        _collect_value_info_dtype(value_info, counter)
    for value_info in graph.output:
        _collect_value_info_dtype(value_info, counter)
    for value_info in graph.value_info:
        _collect_value_info_dtype(value_info, counter)
    for initializer in graph.initializer:
        counter[int(initializer.data_type)] += 1

    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField("t"):
                counter[int(attr.t.data_type)] += 1
            for tensor in attr.tensors:
                counter[int(tensor.data_type)] += 1
            if attr.HasField("sparse_tensor"):
                counter[int(attr.sparse_tensor.values.data_type)] += 1
            for sparse_tensor in attr.sparse_tensors:
                counter[int(sparse_tensor.values.data_type)] += 1
            if attr.HasField("g"):
                _collect_graph_dtypes(attr.g, counter)
            for graph_attr in attr.graphs:
                _collect_graph_dtypes(graph_attr, counter)


def _collect_value_info_dtype(
    value_info: onnx.ValueInfoProto,
    counter: Counter[int | None],
) -> None:
    if not value_info.type.HasField("tensor_type"):
        counter[None] += 1
        return
    elem_type = int(value_info.type.tensor_type.elem_type)
    counter[elem_type if elem_type else None] += 1


def _dtype_counts(counter: Counter[int | None]) -> tuple[DTypeCount, ...]:
    return tuple(
        DTypeCount(
            elem_type=elem_type,
            elem_type_name=_elem_type_name(elem_type),
            count=count,
        )
        for elem_type, count in sorted(
            counter.items(), key=lambda item: _elem_type_name(item[0])
        )
    )


def _elem_type_name(elem_type: int | None) -> str:
    if elem_type is None:
        return "UNDEFINED"
    try:
        return str(TensorProto.DataType.Name(elem_type))
    except ValueError:
        return f"UNKNOWN({elem_type})"


def _report_findings(report: ModelReport) -> tuple[ReportFinding, ...]:
    findings: list[ReportFinding] = []

    if report.checker.passed and report.strict_shape_inference.passed:
        findings.append(ReportFinding("info", "Basic ONNX validation passed."))
    else:
        if not report.checker.passed:
            findings.append(
                ReportFinding(
                    "error",
                    "ONNX checker failed"
                    + (f": {report.checker.error}" if report.checker.error else "."),
                )
            )
        if not report.strict_shape_inference.passed:
            findings.append(
                ReportFinding(
                    "error",
                    "Strict shape inference failed"
                    + (
                        f": {report.strict_shape_inference.error}"
                        if report.strict_shape_inference.error
                        else "."
                    ),
                )
            )

    dynamic_dims = _public_dynamic_dimensions(report)
    if dynamic_dims:
        findings.append(
            ReportFinding(
                "info",
                "Public I/O uses dynamic dimensions: "
                + _join_limited(dynamic_dims)
                + ".",
            )
        )

    unknown_shape_values = _public_values_with_unknown_shapes(report)
    if unknown_shape_values:
        findings.append(
            ReportFinding(
                "warning",
                "Public I/O has unknown shape metadata for: "
                + _join_limited(unknown_shape_values)
                + ".",
            )
        )

    undefined_dtype_count = sum(
        item.count for item in report.dtype_counts if item.elem_type is None
    )
    if undefined_dtype_count:
        findings.append(
            ReportFinding(
                "warning",
                f"{undefined_dtype_count} graph value(s) have undefined dtype metadata.",
            )
        )

    custom_domains = _custom_operator_domains(report)
    if custom_domains:
        findings.append(
            ReportFinding(
                "warning",
                "Model uses non-standard operator domains: "
                + _join_limited(custom_domains)
                + ".",
            )
        )

    if report.function_count:
        findings.append(
            ReportFinding(
                "warning",
                f"Model defines {report.function_count} ONNX Function(s); "
                "target runtimes may need function support.",
            )
        )
    else:
        findings.append(
            ReportFinding("info", "The model does not define ONNX Functions.")
        )

    if report.initializers:
        findings.append(
            ReportFinding(
                "info",
                f"The graph has {len(report.initializers)} initializer(s).",
            )
        )
    else:
        findings.append(ReportFinding("info", "The graph has no initializers."))

    if report.runtime_execution.evaluated and report.runtime_execution.passed:
        findings.append(
            ReportFinding(
                "info",
                "ONNX Runtime CPU execution passed with supplied sample inputs.",
            )
        )
        if report.runtime_execution.output_comparisons:
            findings.append(
                ReportFinding(
                    "info",
                    "ONNX Runtime CPU numeric parity passed for "
                    f"{len(report.runtime_execution.output_comparisons)} output(s).",
                )
            )
        if report.runtime_execution.dimension_bindings:
            conflicts = _conflicting_runtime_dimension_bindings(
                report.runtime_execution.dimension_bindings
            )
            if conflicts:
                findings.append(
                    ReportFinding(
                        "warning",
                        "Runtime sample has conflicting symbolic dimension values: "
                        + _runtime_dimension_bindings_text(conflicts)
                        + ".",
                    )
                )
            else:
                findings.append(
                    ReportFinding(
                        "info",
                        "Runtime sample resolves dynamic dimensions: "
                        + _runtime_dimension_bindings_text(
                            report.runtime_execution.dimension_bindings
                        )
                        + ".",
                    )
                )
        findings.append(
            ReportFinding(
                "info",
                "Target-specific compatibility beyond this CPU smoke test is not "
                "evaluated by this report.",
            )
        )
    elif report.runtime_execution.evaluated:
        failed_outputs = tuple(
            item.output_name
            for item in report.runtime_execution.output_comparisons
            if not item.passed
        )
        findings.append(
            ReportFinding(
                "error",
                "ONNX Runtime CPU execution failed with supplied sample inputs"
                + (
                    f": {report.runtime_execution.error}"
                    if report.runtime_execution.error
                    else (
                        f"; numeric parity failed for {_join_limited(failed_outputs)}."
                        if failed_outputs
                        else "."
                    )
                ),
            )
        )
    else:
        findings.append(
            ReportFinding(
                "info",
                "ONNX Runtime execution is not evaluated without sample inputs.",
            )
        )
        findings.append(
            ReportFinding(
                "info",
                "Target runtime compatibility is not evaluated by this report.",
            )
        )

    return tuple(findings)


def _target_diagnostics(
    report: ModelReport,
    targets: Sequence[TargetName],
) -> tuple[TargetDiagnostic, ...]:
    return tuple(
        TargetDiagnostic(target, _target_findings(report, target)) for target in targets
    )


def _diagnostic_summary(report: ModelReport) -> DiagnosticSummary:
    findings = _all_findings(report)
    error_count = sum(1 for finding in findings if finding.severity == "error")
    warning_count = sum(1 for finding in findings if finding.severity == "warning")
    info_count = sum(1 for finding in findings if finding.severity == "info")
    if error_count:
        status: DiagnosticStatus = "error"
    elif warning_count:
        status = "warning"
    else:
        status = "pass"
    return DiagnosticSummary(
        status=status,
        error_count=error_count,
        warning_count=warning_count,
        info_count=info_count,
    )


def _all_findings(report: ModelReport) -> tuple[ReportFinding, ...]:
    target_findings = tuple(
        finding
        for diagnostic in report.target_diagnostics
        for finding in diagnostic.findings
    )
    return (*report.findings, *target_findings)


def _target_findings(
    report: ModelReport,
    target: TargetName,
) -> tuple[ReportFinding, ...]:
    if target == "ort-cpu":
        return _ort_cpu_findings(report)
    if target == "ort-web":
        return _ort_web_findings(report)
    if target == "ort-mobile":
        return _ort_mobile_findings(report)
    raise ValueError(f"Unsupported target profile: {target}")


def _ort_cpu_findings(report: ModelReport) -> tuple[ReportFinding, ...]:
    findings = _target_base_findings(report)
    if report.runtime_execution.evaluated and report.runtime_execution.passed:
        findings.append(ReportFinding("info", "ONNX Runtime CPU smoke test passed."))
    elif report.runtime_execution.evaluated:
        findings.append(
            ReportFinding(
                "error",
                "ONNX Runtime CPU smoke test failed"
                + (
                    f": {report.runtime_execution.error}"
                    if report.runtime_execution.error
                    else "."
                ),
            )
        )
    else:
        findings.append(
            ReportFinding(
                "warning",
                "Provide runtime_inputs to execute an ONNX Runtime CPU smoke test.",
            )
        )
    return _finalize_target_findings(findings)


def _ort_web_findings(report: ModelReport) -> tuple[ReportFinding, ...]:
    findings = _target_base_findings(report)
    findings.extend(_custom_domain_target_findings(report))
    findings.extend(_function_target_findings(report))
    findings.extend(
        _dtype_portability_findings(
            report,
            "Review dtype support in ONNX Runtime Web before deployment.",
        )
    )
    if report.model_size_bytes > _WEB_MODEL_SIZE_WARNING_BYTES:
        findings.append(
            ReportFinding(
                "warning",
                "Model size may be high for web delivery: "
                f"{_format_bytes(report.model_size_bytes)}.",
            )
        )
    findings.append(
        ReportFinding(
            "warning",
            "Warning-only profile: no ONNX Runtime Web execution is performed by "
            "this report.",
        )
    )
    return _finalize_target_findings(findings)


def _ort_mobile_findings(report: ModelReport) -> tuple[ReportFinding, ...]:
    findings = _target_base_findings(report)
    findings.extend(_custom_domain_target_findings(report))
    findings.extend(_function_target_findings(report))
    findings.extend(
        _dtype_portability_findings(
            report,
            "Review dtype support in the selected mobile runtime build before deployment.",
        )
    )
    dynamic_dims = _public_dynamic_dimensions(report)
    if dynamic_dims:
        findings.append(
            ReportFinding(
                "warning",
                "Review dynamic public I/O dimensions for mobile packaging: "
                + _join_limited(dynamic_dims)
                + ".",
            )
        )
    if report.model_size_bytes > _MOBILE_MODEL_SIZE_WARNING_BYTES:
        findings.append(
            ReportFinding(
                "warning",
                "Model size may be high for mobile packaging: "
                f"{_format_bytes(report.model_size_bytes)}.",
            )
        )
    findings.append(
        ReportFinding(
            "warning",
            "Warning-only profile: no ONNX Runtime mobile execution is performed "
            "by this report.",
        )
    )
    return _finalize_target_findings(findings)


def _target_base_findings(report: ModelReport) -> list[ReportFinding]:
    findings: list[ReportFinding] = []
    if report.checker.passed and report.strict_shape_inference.passed:
        findings.append(ReportFinding("info", "Basic ONNX validation passed."))
    else:
        findings.append(
            ReportFinding(
                "error",
                "Fix ONNX checker or strict shape inference failures before "
                "target-specific evaluation.",
            )
        )
    return findings


def _custom_domain_target_findings(report: ModelReport) -> tuple[ReportFinding, ...]:
    custom_domains = _custom_operator_domains(report)
    if not custom_domains:
        return ()
    return (
        ReportFinding(
            "warning",
            "Non-standard operator domains require explicit target runtime support: "
            + _join_limited(custom_domains)
            + ".",
        ),
    )


def _function_target_findings(report: ModelReport) -> tuple[ReportFinding, ...]:
    if report.function_count == 0:
        return ()
    return (
        ReportFinding(
            "warning",
            f"Model defines {report.function_count} ONNX Function(s); verify target "
            "runtime support.",
        ),
    )


def _dtype_portability_findings(
    report: ModelReport,
    message: str,
) -> tuple[ReportFinding, ...]:
    less_portable = {
        item.elem_type_name
        for item in report.dtype_counts
        if item.elem_type_name in {"BFLOAT16", "DOUBLE", "FLOAT16"}
    }
    if not less_portable:
        return ()
    return (
        ReportFinding(
            "warning",
            message
            + " Dtypes present: "
            + _join_limited(tuple(sorted(less_portable)))
            + ".",
        ),
    )


def _finalize_target_findings(
    findings: list[ReportFinding],
) -> tuple[ReportFinding, ...]:
    if all(finding.severity == "info" for finding in findings):
        findings.append(
            ReportFinding(
                "info",
                "No target-specific warnings were found by static diagnostics; "
                "this is not a compatibility guarantee.",
            )
        )
    return tuple(findings)


def _status_text(result: ValidationResult) -> str:
    if result.passed:
        return "pass"
    if result.error:
        return f"fail ({result.error})"
    return "fail"


def _runtime_status_text(result: RuntimeExecutionResult) -> str:
    if not result.evaluated:
        return "not evaluated"
    if result.passed:
        details: list[str] = []
        if result.providers:
            details.append("providers: " + ", ".join(result.providers))
        if result.output_count is not None:
            details.append(f"outputs: {result.output_count}")
        return "pass" if not details else "pass (" + "; ".join(details) + ")"
    if result.error:
        return f"fail ({result.error})"
    return "fail"


def _runtime_parity_text(result: RuntimeExecutionResult) -> str:
    if not result.output_comparisons:
        return "not evaluated"
    status = (
        "pass" if all(item.passed for item in result.output_comparisons) else "fail"
    )
    tolerance = ""
    if result.rtol is not None and result.atol is not None:
        tolerance = f" (rtol={result.rtol:g}, atol={result.atol:g})"
    return status + tolerance


def _format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size / (1024 * 1024):.1f} MiB"


def _producer_text(report: ModelReport) -> str:
    if report.producer_name and report.producer_version:
        return f"{report.producer_name} {report.producer_version}"
    if report.producer_name:
        return report.producer_name
    return "<unknown>"


def _opset_text(imports: tuple[OpsetImportInfo, ...]) -> str:
    if not imports:
        return "<none>"
    return ", ".join(f"{_domain_text(item.domain)}={item.version}" for item in imports)


def _domain_text(domain: str) -> str:
    return domain or "ai.onnx"


def _tensor_table(
    values: tuple[TensorInfo, ...],
    *,
    empty_text: str,
) -> list[str]:
    return _markdown_table(
        ("Name", "Dtype", "Shape"),
        (
            (value.name, value.elem_type_name, _shape_text(value.shape))
            for value in values
        ),
        empty_text=empty_text,
    )


def _runtime_tensor_table(values: tuple[RuntimeTensorSample, ...]) -> list[str]:
    return _markdown_table(
        ("Name", "Dtype", "Shape"),
        ((value.name, value.dtype, _shape_text(value.shape)) for value in values),
        empty_text="No runtime tensor metadata.",
    )


def _runtime_dimension_bindings_table(
    bindings: tuple[RuntimeDimensionBinding, ...],
) -> list[str]:
    return _markdown_table(
        ("Dim", "Value", "Sources"),
        (
            (
                binding.dim_param,
                _runtime_dimension_binding_value_text(binding),
                _join_limited(binding.sources),
            )
            for binding in bindings
        ),
        empty_text="No runtime dynamic dimension metadata.",
    )


def _runtime_dimension_binding_value_text(
    binding: RuntimeDimensionBinding,
) -> str:
    if len(binding.values) == 1:
        return str(binding.values[0])
    return "conflict: " + ", ".join(str(value) for value in binding.values)


def _runtime_dimension_bindings_text(
    bindings: tuple[RuntimeDimensionBinding, ...],
) -> str:
    return _join_limited(
        tuple(
            f"{binding.dim_param}={_runtime_dimension_binding_value_text(binding)}"
            for binding in bindings
        )
    )


def _profile_dimension_bindings_text(profile: RuntimeProfileSummary) -> str:
    if not profile.dimension_bindings:
        return "-"
    return _runtime_dimension_bindings_text(profile.dimension_bindings)


def _profile_output_samples_text(profile: RuntimeProfileSummary) -> str:
    if not profile.output_samples:
        return "-"
    return _join_limited(
        tuple(
            f"{sample.name}: {_shape_text(sample.shape)}"
            for sample in profile.output_samples
        )
    )


def _evaluation_status_text(status: EvaluationStatus) -> str:
    if status == "not_evaluated":
        return "not evaluated"
    return status


def _conflicting_runtime_dimension_bindings(
    bindings: tuple[RuntimeDimensionBinding, ...],
) -> tuple[RuntimeDimensionBinding, ...]:
    return tuple(binding for binding in bindings if len(binding.values) > 1)


def _shape_text(shape: tuple[ShapeDim, ...] | tuple[int, ...] | None) -> str:
    if shape is None:
        return "<unknown>"
    if not shape:
        return "scalar"
    return " x ".join("?" if dim is None else str(dim) for dim in shape)


def _markdown_table(
    headers: tuple[str, ...],
    rows: Iterable[tuple[str, ...]],
    *,
    empty_text: str,
) -> list[str]:
    materialized = [tuple(str(cell) for cell in row) for row in rows]
    if not materialized:
        return [empty_text]
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in materialized),
    ]


def _target_diagnostics_table(
    target_diagnostics: tuple[TargetDiagnostic, ...],
) -> list[str]:
    return _markdown_table(
        ("Target", "Severity", "Finding"),
        (
            (diagnostic.target, finding.severity.upper(), finding.message)
            for diagnostic in target_diagnostics
            for finding in diagnostic.findings
        ),
        empty_text="No target diagnostics.",
    )


def _environment_table(environment: EnvironmentInfo) -> list[str]:
    rows = [("python", environment.python_version)]
    rows.extend(
        (item.name, item.version if item.version is not None else "<not installed>")
        for item in environment.package_versions
    )
    return _markdown_table(
        ("Component", "Version"),
        rows,
        empty_text="No environment metadata.",
    )


def _float_text(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6g}"


def _finding_lines(findings: tuple[ReportFinding, ...]) -> list[str]:
    if not findings:
        return ["No findings."]
    return [f"- {finding.severity.upper()}: {finding.message}" for finding in findings]


def _public_dynamic_dimensions(report: ModelReport) -> tuple[str, ...]:
    dims: set[str] = set()
    for value in (*report.inputs, *report.outputs):
        if value.shape is None:
            continue
        for dim in value.shape:
            if isinstance(dim, str):
                dims.add(dim)
    return tuple(sorted(dims))


def _public_values_with_unknown_shapes(report: ModelReport) -> tuple[str, ...]:
    values: list[str] = []
    for value in (*report.inputs, *report.outputs):
        if value.shape is None or any(dim is None for dim in value.shape):
            values.append(value.name or "<unnamed>")
    return tuple(values)


def _custom_operator_domains(report: ModelReport) -> tuple[str, ...]:
    domains = {
        item.domain
        for item in report.operator_counts
        if item.domain not in {"", "ai.onnx"}
    }
    return tuple(sorted(domains))


def _join_limited(values: tuple[str, ...], limit: int = 5) -> str:
    visible = values[:limit]
    suffix = "" if len(values) <= limit else f", ... (+{len(values) - limit} more)"
    return ", ".join(visible) + suffix
