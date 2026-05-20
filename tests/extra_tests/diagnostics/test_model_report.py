# tests/extra_tests/diagnostics/test_model_report.py

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from onnx import TensorProto, helper

from jax2onnx.diagnostics import (
    OperatorCount,
    RuntimeProfile,
    analyze_jax_export,
    analyze_jax_export_profiles,
    analyze_model,
    evaluate_model_report_gate,
    format_model_report_json,
    format_model_report_markdown,
    format_profile_analysis_json,
    format_profile_analysis_markdown,
    model_report_to_dict,
    profile_analysis_to_dict,
    summarize_profile_analysis,
    write_model_report,
    write_profile_analysis_report,
)
from jax2onnx.user_interface import to_onnx


def _add(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    return lhs + rhs


def _tanh(x: jax.Array) -> jax.Array:
    return jnp.tanh(x)


def _sum_and_tanh(lhs: jax.Array, rhs: jax.Array) -> tuple[jax.Array, jax.Array]:
    return lhs + rhs, jnp.tanh(lhs)


def test_analyze_model_reports_exported_model_inventory() -> None:
    model = to_onnx(
        _add,
        [
            jax.ShapeDtypeStruct(("B", 3), jnp.float32),
            jax.ShapeDtypeStruct(("B", 3), jnp.float32),
        ],
        model_name="diagnostics_add",
        input_names=["lhs", "rhs"],
        output_names=["sum"],
        return_mode="proto",
    )

    report = analyze_model(model)

    assert report.model_name == "diagnostics_add"
    assert report.model_size_bytes > 0
    assert report.checker.passed
    assert report.checker.error is None
    assert report.strict_shape_inference.passed
    assert report.strict_shape_inference.error is None
    assert report.diagnostic_summary.status == "pass"
    assert report.diagnostic_summary.error_count == 0
    assert report.diagnostic_summary.warning_count == 0
    assert report.environment.python_version
    package_versions = {
        entry.name: entry.version for entry in report.environment.package_versions
    }
    assert {"jax2onnx", "jax", "numpy", "onnx", "onnxruntime", "onnx-ir"} <= set(
        package_versions
    )
    assert package_versions["jax2onnx"]
    assert package_versions["jax"]
    assert package_versions["onnx"]
    assert any(
        finding.message == "Basic ONNX validation passed."
        for finding in report.findings
    )
    assert any(
        finding.message == "Public I/O uses dynamic dimensions: B."
        for finding in report.findings
    )
    assert not report.runtime_execution.evaluated
    assert ("", 23) in {(entry.domain, entry.version) for entry in report.opset_imports}
    assert report.node_count == sum(entry.count for entry in report.operator_counts)
    assert OperatorCount(domain="", op_type="Add", count=1) in report.operator_counts

    assert [
        (value.name, value.elem_type_name, value.shape) for value in report.inputs
    ] == [
        ("lhs", "FLOAT", ("B", 3)),
        ("rhs", "FLOAT", ("B", 3)),
    ]
    assert [
        (value.name, value.elem_type_name, value.shape) for value in report.outputs
    ] == [
        ("sum", "FLOAT", ("B", 3)),
    ]
    assert any(entry.elem_type == TensorProto.FLOAT for entry in report.dtype_counts)


def test_analyze_model_accepts_serialized_bytes_and_paths(tmp_path) -> None:
    model = to_onnx(
        _tanh,
        [jax.ShapeDtypeStruct((2,), jnp.float32)],
        model_name="diagnostics_tanh",
        return_mode="proto",
    )
    serialized = model.SerializeToString()
    model_path = tmp_path / "diagnostics_tanh.onnx"
    model_path.write_bytes(serialized)

    bytes_report = analyze_model(serialized)
    path_report = analyze_model(model_path)

    assert bytes_report.model_name == "diagnostics_tanh"
    assert path_report.model_name == "diagnostics_tanh"
    assert bytes_report.model_size_bytes == len(serialized)
    assert path_report.model_size_bytes == len(serialized)


def test_analyze_model_captures_checker_failures_without_raising() -> None:
    graph = helper.make_graph(
        nodes=[helper.make_node("Add", ["x"], ["y"], name="bad_add")],
        name="invalid_add_graph",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 23)],
    )

    report = analyze_model(model)

    assert not report.checker.passed
    assert report.checker.error is not None
    assert report.diagnostic_summary.status == "error"
    assert report.diagnostic_summary.error_count >= 1
    assert "Add" in report.checker.error
    assert any(
        finding.severity == "error" and "ONNX checker failed" in finding.message
        for finding in report.findings
    )
    assert report.model_name == "invalid_add_graph"
    assert report.node_count == 1
    assert OperatorCount(domain="", op_type="Add", count=1) in report.operator_counts
    assert [
        (value.name, value.elem_type_name, value.shape) for value in report.inputs
    ] == [
        ("x", "FLOAT", (2,)),
    ]


def test_format_model_report_markdown_is_readable() -> None:
    model = to_onnx(
        _add,
        [
            jax.ShapeDtypeStruct(("B", 3), jnp.float32),
            jax.ShapeDtypeStruct(("B", 3), jnp.float32),
        ],
        model_name="diagnostics_add",
        input_names=["lhs", "rhs"],
        output_names=["sum"],
        return_mode="proto",
    )

    markdown = format_model_report_markdown(analyze_model(model))

    assert "# Model Report: diagnostics_add" in markdown
    assert "## Status" in markdown
    assert "- Overall: pass" in markdown
    assert "- Errors: 0" in markdown
    assert "- ONNX checker: pass" in markdown
    assert "- Strict shape inference: pass" in markdown
    assert "## Findings" in markdown
    assert "## Environment" in markdown
    assert "| Component | Version |" in markdown
    assert "| jax2onnx |" in markdown
    assert "- INFO: Basic ONNX validation passed." in markdown
    assert "- INFO: Public I/O uses dynamic dimensions: B." in markdown
    assert "- Opset imports: ai.onnx=23" in markdown
    assert "| ai.onnx | Add | 1 |" in markdown
    assert "| lhs | FLOAT | B x 3 |" in markdown
    assert "| sum | FLOAT | B x 3 |" in markdown
    assert "Target runtime compatibility is not evaluated" in markdown


def test_model_report_json_is_machine_readable() -> None:
    model = to_onnx(
        _add,
        [
            jax.ShapeDtypeStruct(("B", 3), jnp.float32),
            jax.ShapeDtypeStruct(("B", 3), jnp.float32),
        ],
        model_name="diagnostics_add_json",
        input_names=["lhs", "rhs"],
        output_names=["sum"],
        return_mode="proto",
    )

    report = analyze_model(model, targets=("ort-web",))
    payload = model_report_to_dict(report)
    decoded = json.loads(format_model_report_json(report))

    assert payload["model_name"] == "diagnostics_add_json"
    assert decoded["model_name"] == "diagnostics_add_json"
    assert decoded["diagnostic_summary"]["status"] == "warning"
    assert decoded["environment"]["python_version"]
    assert decoded["checker"]["passed"] is True
    assert decoded["inputs"][0]["shape"] == ["B", 3]
    assert decoded["target_diagnostics"][0]["target"] == "ort-web"
    assert decoded["target_diagnostics"][0]["findings"]


def test_analyze_jax_export_builds_report_from_sample_inputs() -> None:
    lhs = np.ones((2, 3), dtype=np.float32)
    rhs = np.full((2, 3), 2.0, dtype=np.float32)

    analysis = analyze_jax_export(
        _add,
        [
            jax.ShapeDtypeStruct(("B", 3), jnp.float32),
            jax.ShapeDtypeStruct(("B", 3), jnp.float32),
        ],
        sample_inputs=(lhs, rhs),
        model_name="diagnostics_jax_export",
        input_names=["lhs", "rhs"],
        output_names=["sum"],
        targets=("ort-cpu",),
    )

    assert analysis.model.graph.name == "diagnostics_jax_export"
    assert analysis.report.model_name == "diagnostics_jax_export"
    assert analysis.report.runtime_execution.evaluated
    assert analysis.report.runtime_execution.passed
    assert [
        (sample.name, sample.dtype, sample.shape)
        for sample in analysis.report.runtime_execution.input_samples
    ] == [
        ("lhs", "float32", (2, 3)),
        ("rhs", "float32", (2, 3)),
    ]
    assert [
        (sample.name, sample.dtype, sample.shape)
        for sample in analysis.report.runtime_execution.output_samples
    ] == [
        ("sum", "float32", (2, 3)),
    ]
    assert [
        (binding.dim_param, binding.values, binding.sources)
        for binding in analysis.report.runtime_execution.dimension_bindings
    ] == [
        ("B", (2,), ("lhs[0]", "rhs[0]", "sum[0]")),
    ]
    assert len(analysis.report.runtime_execution.output_comparisons) == 1
    comparison = analysis.report.runtime_execution.output_comparisons[0]
    assert comparison.output_name == "sum"
    assert comparison.passed
    assert comparison.max_abs_diff == 0.0
    assert any(
        finding.message == "ONNX Runtime CPU smoke test passed."
        for finding in analysis.report.target_diagnostics[0].findings
    )


def test_analyze_jax_export_maps_multiple_reference_outputs() -> None:
    lhs = np.ones((2, 3), dtype=np.float32)
    rhs = np.full((2, 3), 2.0, dtype=np.float32)

    analysis = analyze_jax_export(
        _sum_and_tanh,
        [
            jax.ShapeDtypeStruct(("B", 3), jnp.float32),
            jax.ShapeDtypeStruct(("B", 3), jnp.float32),
        ],
        sample_inputs=(lhs, rhs),
        model_name="diagnostics_jax_export_multi_output",
        input_names=["lhs", "rhs"],
        output_names=["sum", "activation"],
    )

    assert analysis.report.runtime_execution.passed
    assert [
        comparison.output_name
        for comparison in analysis.report.runtime_execution.output_comparisons
    ] == ["sum", "activation"]
    assert all(
        comparison.passed
        for comparison in analysis.report.runtime_execution.output_comparisons
    )


def test_analyze_jax_export_rejects_ambiguous_sample_input_mapping() -> None:
    lhs = np.ones((2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="sample_inputs length"):
        analyze_jax_export(
            _add,
            [
                jax.ShapeDtypeStruct(("B", 3), jnp.float32),
                jax.ShapeDtypeStruct(("B", 3), jnp.float32),
            ],
            sample_inputs=(lhs,),
            model_name="diagnostics_jax_export_bad_samples",
            input_names=["lhs", "rhs"],
            output_names=["sum"],
        )


def test_analyze_jax_export_profiles_exports_once_and_checks_each_profile() -> None:
    profile_small = RuntimeProfile(
        name="batch_2",
        sample_inputs=(
            np.ones((2, 3), dtype=np.float32),
            np.full((2, 3), 2.0, dtype=np.float32),
        ),
    )
    profile_large = RuntimeProfile(
        name="batch_5",
        sample_inputs=(
            np.ones((5, 3), dtype=np.float32),
            np.full((5, 3), 2.0, dtype=np.float32),
        ),
    )

    analysis = analyze_jax_export_profiles(
        _add,
        [
            jax.ShapeDtypeStruct(("B", 3), jnp.float32),
            jax.ShapeDtypeStruct(("B", 3), jnp.float32),
        ],
        profiles=(profile_small, profile_large),
        model_name="diagnostics_jax_export_profiles",
        input_names=["lhs", "rhs"],
        output_names=["sum"],
        targets=("ort-cpu",),
    )

    assert analysis.model.graph.name == "diagnostics_jax_export_profiles"
    assert [profile_report.name for profile_report in analysis.profile_reports] == [
        "batch_2",
        "batch_5",
    ]
    assert all(
        profile_report.report.runtime_execution.passed
        for profile_report in analysis.profile_reports
    )
    assert [
        profile_report.report.runtime_execution.dimension_bindings[0].values
        for profile_report in analysis.profile_reports
    ] == [(2,), (5,)]
    assert [
        profile_report.report.runtime_execution.output_samples[0].shape
        for profile_report in analysis.profile_reports
    ] == [(2, 3), (5, 3)]


def test_profile_analysis_summary_markdown_and_json_compare_profiles(
    tmp_path,
) -> None:
    analysis = analyze_jax_export_profiles(
        _add,
        [
            jax.ShapeDtypeStruct(("B", 3), jnp.float32),
            jax.ShapeDtypeStruct(("B", 3), jnp.float32),
        ],
        profiles=(
            RuntimeProfile(
                name="batch_2",
                sample_inputs=(
                    np.ones((2, 3), dtype=np.float32),
                    np.full((2, 3), 2.0, dtype=np.float32),
                ),
            ),
            RuntimeProfile(
                name="batch_5",
                sample_inputs=(
                    np.ones((5, 3), dtype=np.float32),
                    np.full((5, 3), 2.0, dtype=np.float32),
                ),
            ),
        ),
        model_name="diagnostics_profile_report",
        input_names=["lhs", "rhs"],
        output_names=["sum"],
        targets=("ort-cpu",),
    )

    summary = summarize_profile_analysis(analysis)
    markdown = format_profile_analysis_markdown(analysis)
    payload = profile_analysis_to_dict(analysis)
    decoded = json.loads(format_profile_analysis_json(analysis))
    markdown_path = write_profile_analysis_report(
        analysis,
        tmp_path / "profiles.md",
    )
    json_path = write_profile_analysis_report(analysis, tmp_path / "profiles.json")

    assert summary.model_name == "diagnostics_profile_report"
    assert summary.profile_count == 2
    assert summary.status == "pass"
    assert summary.error_count == 0
    assert summary.warning_count == 0
    assert summary.runtime_pass_count == 2
    assert summary.runtime_fail_count == 0
    assert summary.numeric_parity_pass_count == 2
    assert summary.numeric_parity_fail_count == 0
    assert [
        (
            profile.name,
            profile.runtime_status,
            profile.numeric_parity_status,
            profile.dimension_bindings[0].values,
            profile.output_samples[0].shape,
            profile.max_abs_diff,
            profile.max_rel_diff,
        )
        for profile in summary.profiles
    ] == [
        ("batch_2", "pass", "pass", (2,), (2, 3), 0.0, 0.0),
        ("batch_5", "pass", "pass", (5,), (5, 3), 0.0, 0.0),
    ]
    assert "# Runtime Profile Report: diagnostics_profile_report" in markdown
    assert "- Profiles: 2" in markdown
    assert "- Runtime: 2 pass, 0 fail, 0 not evaluated" in markdown
    assert "| batch_2 | pass | pass | pass | B=2 | sum: 2 x 3 | 0 | 0 |" in markdown
    assert "| batch_5 | pass | pass | pass | B=5 | sum: 5 x 3 | 0 | 0 |" in markdown
    assert payload["summary"]["runtime_pass_count"] == 2
    assert decoded["summary"]["profiles"][1]["dimension_bindings"][0]["values"] == [5]
    assert decoded["profile_reports"][0]["report"]["model_name"] == (
        "diagnostics_profile_report"
    )
    assert markdown_path.read_text(encoding="utf-8").startswith(
        "# Runtime Profile Report:"
    )
    assert (
        json.loads(json_path.read_text(encoding="utf-8"))["summary"]["profile_count"]
        == 2
    )


def test_analyze_jax_export_profiles_requires_unique_profiles() -> None:
    profile = RuntimeProfile(
        name="duplicate",
        sample_inputs=(
            np.ones((2, 3), dtype=np.float32),
            np.full((2, 3), 2.0, dtype=np.float32),
        ),
    )

    with pytest.raises(ValueError, match="unique"):
        analyze_jax_export_profiles(
            _add,
            [
                jax.ShapeDtypeStruct(("B", 3), jnp.float32),
                jax.ShapeDtypeStruct(("B", 3), jnp.float32),
            ],
            profiles=(profile, profile),
            model_name="diagnostics_jax_export_duplicate_profiles",
            input_names=["lhs", "rhs"],
            output_names=["sum"],
        )


def test_write_model_report_persists_markdown_and_json(tmp_path) -> None:
    model = to_onnx(
        _add,
        [
            jax.ShapeDtypeStruct(("B", 3), jnp.float32),
            jax.ShapeDtypeStruct(("B", 3), jnp.float32),
        ],
        model_name="diagnostics_write_report",
        input_names=["lhs", "rhs"],
        output_names=["sum"],
        return_mode="proto",
    )
    report = analyze_model(model, targets=("ort-web",))

    markdown_path = write_model_report(report, tmp_path / "nested" / "report.md")
    json_path = write_model_report(report, tmp_path / "report.json")
    explicit_markdown_path = write_model_report(
        report,
        tmp_path / "report.txt",
        output_format="markdown",
    )

    assert markdown_path == tmp_path / "nested" / "report.md"
    assert "# Model Report: diagnostics_write_report" in markdown_path.read_text(
        encoding="utf-8"
    )
    decoded = json.loads(json_path.read_text(encoding="utf-8"))
    assert decoded["model_name"] == "diagnostics_write_report"
    assert decoded["diagnostic_summary"]["status"] == "warning"
    assert explicit_markdown_path.read_text(encoding="utf-8").startswith(
        "# Model Report:"
    )


def test_write_model_report_requires_known_suffix_or_explicit_format(
    tmp_path,
) -> None:
    model = to_onnx(
        _tanh,
        [jax.ShapeDtypeStruct((2,), jnp.float32)],
        model_name="diagnostics_unknown_report_suffix",
        return_mode="proto",
    )
    report = analyze_model(model)

    with pytest.raises(ValueError, match="Cannot infer report output format"):
        write_model_report(report, tmp_path / "report.txt")


def test_evaluate_model_report_gate_allows_warnings_by_default() -> None:
    model = to_onnx(
        _add,
        [
            jax.ShapeDtypeStruct(("B", 3), jnp.float32),
            jax.ShapeDtypeStruct(("B", 3), jnp.float32),
        ],
        model_name="diagnostics_gate_warning",
        input_names=["lhs", "rhs"],
        output_names=["sum"],
        return_mode="proto",
    )

    report = analyze_model(model, targets=("ort-web",))
    gate = evaluate_model_report_gate(report)

    assert report.diagnostic_summary.status == "warning"
    assert gate.passed
    assert gate.fail_on == "errors"
    assert gate.error_count == 0
    assert gate.warning_count >= 1
    assert "passed with" in gate.message


def test_evaluate_model_report_gate_can_fail_on_warnings() -> None:
    model = to_onnx(
        _add,
        [
            jax.ShapeDtypeStruct(("B", 3), jnp.float32),
            jax.ShapeDtypeStruct(("B", 3), jnp.float32),
        ],
        model_name="diagnostics_gate_strict_warning",
        input_names=["lhs", "rhs"],
        output_names=["sum"],
        return_mode="proto",
    )

    report = analyze_model(model, targets=("ort-web",))
    gate = evaluate_model_report_gate(report, fail_on="warnings")

    assert report.diagnostic_summary.status == "warning"
    assert not gate.passed
    assert gate.fail_on == "warnings"
    assert gate.error_count == 0
    assert gate.warning_count >= 1
    assert "warning(s) found" in gate.message


def test_evaluate_model_report_gate_fails_on_errors() -> None:
    graph = helper.make_graph(
        nodes=[helper.make_node("Add", ["x"], ["y"], name="bad_add")],
        name="invalid_gate_graph",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 23)],
    )

    report = analyze_model(model)
    gate = evaluate_model_report_gate(report)

    assert report.diagnostic_summary.status == "error"
    assert not gate.passed
    assert gate.fail_on == "errors"
    assert gate.error_count >= 1
    assert "error(s) found" in gate.message


def test_analyze_model_can_run_onnxruntime_cpu_smoke_test() -> None:
    model = to_onnx(
        _add,
        [
            jax.ShapeDtypeStruct(("B", 3), jnp.float32),
            jax.ShapeDtypeStruct(("B", 3), jnp.float32),
        ],
        model_name="diagnostics_add_runtime",
        input_names=["lhs", "rhs"],
        output_names=["sum"],
        return_mode="proto",
    )

    report = analyze_model(
        model,
        runtime_inputs={
            "lhs": np.ones((2, 3), dtype=np.float32),
            "rhs": np.full((2, 3), 2.0, dtype=np.float32),
        },
    )
    markdown = format_model_report_markdown(report)
    decoded = json.loads(format_model_report_json(report))

    assert report.runtime_execution.evaluated
    assert report.runtime_execution.passed
    assert report.runtime_execution.output_count == 1
    assert [
        (sample.name, sample.dtype, sample.shape)
        for sample in report.runtime_execution.input_samples
    ] == [
        ("lhs", "float32", (2, 3)),
        ("rhs", "float32", (2, 3)),
    ]
    assert [
        (sample.name, sample.dtype, sample.shape)
        for sample in report.runtime_execution.output_samples
    ] == [
        ("sum", "float32", (2, 3)),
    ]
    assert [
        (binding.dim_param, binding.values, binding.sources)
        for binding in report.runtime_execution.dimension_bindings
    ] == [
        ("B", (2,), ("lhs[0]", "rhs[0]", "sum[0]")),
    ]
    assert decoded["runtime_execution"]["input_samples"][0] == {
        "name": "lhs",
        "dtype": "float32",
        "shape": [2, 3],
    }
    assert decoded["runtime_execution"]["dimension_bindings"][0] == {
        "dim_param": "B",
        "values": [2],
        "sources": ["lhs[0]", "rhs[0]", "sum[0]"],
    }
    assert "CPUExecutionProvider" in report.runtime_execution.providers
    assert any(
        finding.message
        == "ONNX Runtime CPU execution passed with supplied sample inputs."
        for finding in report.findings
    )
    assert any(
        finding.message == "Runtime sample resolves dynamic dimensions: B=2."
        for finding in report.findings
    )
    assert "- ONNX Runtime CPU: pass" in markdown
    assert "## Runtime Inputs" in markdown
    assert "| lhs | float32 | 2 x 3 |" in markdown
    assert "## Runtime Outputs" in markdown
    assert "| sum | float32 | 2 x 3 |" in markdown
    assert "## Runtime Dynamic Dimensions" in markdown
    assert "| B | 2 | lhs[0], rhs[0], sum[0] |" in markdown
    assert "ONNX Runtime CPU execution passed with supplied sample inputs" in markdown


def test_analyze_model_can_compare_expected_runtime_outputs() -> None:
    model = to_onnx(
        _add,
        [
            jax.ShapeDtypeStruct(("B", 3), jnp.float32),
            jax.ShapeDtypeStruct(("B", 3), jnp.float32),
        ],
        model_name="diagnostics_add_parity",
        input_names=["lhs", "rhs"],
        output_names=["sum"],
        return_mode="proto",
    )
    lhs = np.ones((2, 3), dtype=np.float32)
    rhs = np.full((2, 3), 2.0, dtype=np.float32)

    report = analyze_model(
        model,
        runtime_inputs={"lhs": lhs, "rhs": rhs},
        expected_outputs={"sum": lhs + rhs},
    )
    markdown = format_model_report_markdown(report)

    assert report.runtime_execution.passed
    assert len(report.runtime_execution.output_comparisons) == 1
    comparison = report.runtime_execution.output_comparisons[0]
    assert comparison.output_name == "sum"
    assert comparison.passed
    assert comparison.max_abs_diff == 0.0
    assert comparison.max_rel_diff == 0.0
    assert any(
        finding.message == "ONNX Runtime CPU numeric parity passed for 1 output(s)."
        for finding in report.findings
    )
    assert "- Numeric parity: pass" in markdown
    assert "| sum | pass | 0 | 0 |" in markdown


def test_analyze_model_reports_warning_only_target_diagnostics() -> None:
    model = to_onnx(
        _add,
        [
            jax.ShapeDtypeStruct(("B", 3), jnp.float32),
            jax.ShapeDtypeStruct(("B", 3), jnp.float32),
        ],
        model_name="diagnostics_add_targets",
        input_names=["lhs", "rhs"],
        output_names=["sum"],
        return_mode="proto",
    )
    lhs = np.ones((2, 3), dtype=np.float32)
    rhs = np.full((2, 3), 2.0, dtype=np.float32)

    report = analyze_model(
        model,
        runtime_inputs={"lhs": lhs, "rhs": rhs},
        expected_outputs={"sum": lhs + rhs},
        targets=("ort-cpu", "ort-web", "ort-mobile"),
    )
    markdown = format_model_report_markdown(report)

    assert [diagnostic.target for diagnostic in report.target_diagnostics] == [
        "ort-cpu",
        "ort-web",
        "ort-mobile",
    ]
    assert report.diagnostic_summary.status == "warning"
    assert report.diagnostic_summary.warning_count >= 1
    assert any(
        finding.message == "ONNX Runtime CPU smoke test passed."
        for finding in report.target_diagnostics[0].findings
    )
    assert any(
        finding.severity == "warning"
        and "dynamic public I/O dimensions" in finding.message
        for finding in report.target_diagnostics[2].findings
    )
    assert "## Target Diagnostics" in markdown
    assert "| ort-cpu | INFO | ONNX Runtime CPU smoke test passed. |" in markdown
    assert "| ort-mobile | WARNING | Review dynamic public I/O dimensions" in markdown


def test_analyze_model_reports_failed_expected_output_comparison() -> None:
    model = to_onnx(
        _add,
        [
            jax.ShapeDtypeStruct((2, 3), jnp.float32),
            jax.ShapeDtypeStruct((2, 3), jnp.float32),
        ],
        model_name="diagnostics_add_parity_failure",
        input_names=["lhs", "rhs"],
        output_names=["sum"],
        return_mode="proto",
    )
    lhs = np.ones((2, 3), dtype=np.float32)
    rhs = np.full((2, 3), 2.0, dtype=np.float32)

    report = analyze_model(
        model,
        runtime_inputs={"lhs": lhs, "rhs": rhs},
        expected_outputs={"sum": np.zeros((2, 3), dtype=np.float32)},
    )
    markdown = format_model_report_markdown(report)

    assert report.runtime_execution.evaluated
    assert report.runtime_execution.passed is False
    assert report.diagnostic_summary.status == "error"
    assert report.diagnostic_summary.error_count >= 1
    comparison = report.runtime_execution.output_comparisons[0]
    assert comparison.output_name == "sum"
    assert not comparison.passed
    assert comparison.max_abs_diff == 3.0
    assert any(
        finding.severity == "error"
        and "numeric parity failed for sum" in finding.message
        for finding in report.findings
    )
    assert "- ONNX Runtime CPU: fail" in markdown
    assert "- Numeric parity: fail" in markdown
    assert "| sum | fail | 3 |" in markdown


def test_analyze_model_reports_target_diagnostics() -> None:
    model = to_onnx(
        _add,
        [
            jax.ShapeDtypeStruct(("B", 3), jnp.float32),
            jax.ShapeDtypeStruct(("B", 3), jnp.float32),
        ],
        model_name="diagnostics_add_targets",
        input_names=["lhs", "rhs"],
        output_names=["sum"],
        return_mode="proto",
    )
    lhs = np.ones((2, 3), dtype=np.float32)
    rhs = np.full((2, 3), 2.0, dtype=np.float32)

    report = analyze_model(
        model,
        runtime_inputs={"lhs": lhs, "rhs": rhs},
        expected_outputs={"sum": lhs + rhs},
        targets=("ort-cpu", "ort-web", "ort-mobile"),
    )
    diagnostics = {
        diagnostic.target: diagnostic for diagnostic in report.target_diagnostics
    }
    markdown = format_model_report_markdown(report)

    assert set(diagnostics) == {"ort-cpu", "ort-web", "ort-mobile"}
    assert all(
        finding.severity == "info" for finding in diagnostics["ort-cpu"].findings
    )
    assert any(
        finding.message == "ONNX Runtime CPU smoke test passed."
        for finding in diagnostics["ort-cpu"].findings
    )
    assert any(
        finding.severity == "warning" and "Warning-only profile" in finding.message
        for finding in diagnostics["ort-web"].findings
    )
    assert any(
        finding.severity == "warning" and "Warning-only profile" in finding.message
        for finding in diagnostics["ort-mobile"].findings
    )
    assert "| ort-cpu | INFO | ONNX Runtime CPU smoke test passed. |" in markdown
    assert "| ort-web | WARNING | Warning-only profile" in markdown
    assert "| ort-mobile | WARNING | Warning-only profile" in markdown
